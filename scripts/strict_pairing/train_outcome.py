"""Train four-outcome selective repair; no measured robot success is inferred from accuracy."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train import load
from outcome import OutcomeTCN,LABELS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--min-pairs',type=int,default=64);p.add_argument('--min-sequences',type=int,default=10)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.steps<1 or args.min_pairs<12 or args.min_sequences<6:p.error('Require positive steps, >=12 pairs, >=6 sequences')
    torch.manual_seed(42);torch.set_num_threads(4);rng=np.random.default_rng(42)
    d,rows=load(args.run,args.min_pairs,args.min_sequences,all_outcomes=True)
    seq=np.unique(d['sequence']);rng.shuffle(seq);n=max(1,len(seq)//5)
    groups={'test':seq[:n],'validation':seq[n:2*n],'train':seq[2*n:]}
    ids={k:np.flatnonzero(np.isin(d['sequence'],v)) for k,v in groups.items()}
    counts={k:np.bincount(d['outcome'][i],minlength=4).tolist() for k,i in ids.items()}
    if min(counts['train'])==0:
        raise ValueError(f'Four-outcome training requires all four classes in training episodes: {counts}. Collect more; do not fabricate labels.')
    norm={}
    for k in ['state','belief','delta_belief']:
        mean=d[k][ids['train']].mean(0);std=np.maximum(d[k][ids['train']].std(0),1e-4)
        norm[k]={'mean':mean,'std':std};d[k]=(d[k]-mean)/std
    x={k:torch.as_tensor(v,device=args.device,dtype=torch.long if k=='outcome' else torch.float32) for k,v in d.items() if k!='sequence'}
    config=dict(state_dim=x['state'].shape[1],summary_dim=x['belief'].shape[1],action_dim=7,
        task_dim=x['task'].shape[1],chunk_len=10,hidden_dim=128,cond_dim=128,num_blocks=3,dropout=.05,output_scale=.02)
    model=OutcomeTCN(config).to(args.device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    def forward(i):return model(*(x[k][i] for k in ['state','belief','action','delta_belief','task','meta','mask']))
    def average(v,m):return (v*m).sum()/m.sum().clamp_min(1)
    def losses(h,i):
        y=x['outcome'][i];mask=x['mask'][i]
        # Only successful repairs supervise nonzero deltas. Preserve/harmful shrink to zero.
        repair=(y==0).float().unsqueeze(-1)*mask
        conservative=((y==1)|(y==3)).float().unsqueeze(-1)*mask
        reg=average(((h['delta']-x['target'][i])/.003).square().mean(-1),repair)
        shrink=average((h['delta']/.003).square().mean(-1),conservative)
        cls=F.cross_entropy(h['outcome_logits'],y)
        # Unresolved remains a separate outcome, not a no-repair label.
        risk=F.binary_cross_entropy_with_logits(h['risk_logit'],(y==3).float())
        return cls+risk+reg+.1*shrink
    def evaluate(i):
        model.eval()
        with torch.no_grad():
            h=forward(i);y=x['outcome'][i];pred=h['outcome_logits'].argmax(-1)
            matrix=torch.bincount(4*y+pred,minlength=16).reshape(4,4)
            return {'loss':float(losses(h,i)), 'accuracy':float((pred==y).float().mean()),
                'confusion_matrix':matrix.cpu().tolist(),
                'harmful_brier':float((h['risk_logit'].sigmoid()-(y==3).float()).square().mean())}
    # Class-weighted sampling applies ONLY to train; held-out distributions untouched.
    ratios=np.array([.4,.3,.15,.15]);train=ids['train']
    weights=ratios[d['outcome'][train]]/np.array(counts['train'])[d['outcome'][train]];weights/=weights.sum()
    best=float('inf');saved=None;history=[];selected=0
    for step in range(1,args.steps+1):
        model.train();i=rng.choice(train,size=min(64,len(train)),replace=True,p=weights)
        loss=losses(forward(i),i);opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0 or step==args.steps:
            val=evaluate(ids['validation']);history.append(dict(step=step,**val));print(json.dumps(history[-1]),flush=True)
            if val['loss']<best:
                best=val['loss'];selected=step;saved={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(saved);test=evaluate(ids['test']);args.output.mkdir(parents=True)
    torch.save(dict(state_dict=saved,config=config,normalization=norm,labels=LABELS,
        thresholds_validated=False,max_action_delta=min(float(r['max_action_delta']) for r in rows) if all(r['max_action_delta']>0 for r in rows) else None),args.output/'outcome_tcn.pt')
    report=dict(labels=LABELS,class_counts=counts,split_sequences={k:v.tolist() for k,v in groups.items()},
        selected_step=selected,test=test,history=history,robot_success_evaluated=False,thresholds_validated=False,
        limitations=['Shared frozen cache goal is not raw historical vision; no fabricated vision history.',
            'Outcome supervision describes measured policy-generated candidates; TCN deltas require new matched rollouts to validate outcome calibration.',
            'ABBA repetitions are determinism checks, not independent estimates of success probability.',
            'Unresolved asks caller to replan/hold; no robot executor or replanner is launched by this trainer.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(test,indent=2))

if __name__=='__main__':main()
