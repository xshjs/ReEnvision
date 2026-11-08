"""Masked TCN regression on strictly measured, positive-return belief pairs.

Offline correction prediction only: held-out robot success requires new rollouts.
No threshold lowering or synthetic ordering if measured data are insufficient.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
from paths import defi_root
sys.path.insert(0,str(defi_root()/'Step3_DeFI'))
from policy_evaluation.train_directional_sensitivity_tcn import DirectionalSensitivityTCN
from strict_core import accept_pair,first_success_return


def task_vector(task):
    digest=hashlib.sha256(task.encode()).digest()
    return (np.frombuffer(digest,dtype=np.uint8).astype(np.float32)/127.5-1)


def load(run,min_pairs,min_sequences):
    if (run/'fatal.json').exists():raise ValueError('Fatal replay error: dataset not approved')
    if not (run/'collection_summary.json').exists():raise ValueError('Collection has not completed successfully')
    rows=[json.loads(s) for s in (run/'pairs.jsonl').read_text().splitlines()]
    good=[r for r in rows if r['accepted']]
    if len(good)<min_pairs:raise ValueError(f'Only {len(good)} accepted pairs; require {min_pairs}. Collect more, do not weaken return filter.')
    seqs=sorted({r['sequence'] for r in good})
    if len(seqs)<min_sequences:raise ValueError(f'Only {len(seqs)} independent sequences; require {min_sequences}')
    out={k:[] for k in ['state','belief','delta_belief','action','target','task','meta','mask','sequence']}
    for r in good:
        gap=r.get('min_return_gap',.01)
        if not accept_pair(r['returns_prior'],r['returns_post'],gap):raise ValueError('Invalid return evidence')
        with np.load(run/r['file']) as z:
            if not np.array_equal(z['returns_prior'],r['returns_prior']) or not np.array_equal(z['returns_post'],r['returns_post']):
                raise ValueError('Manifest/NPZ return mismatch')
            if not accept_pair(z['returns_prior'],z['returns_post'],gap):raise ValueError('Invalid measured return pair')
            for side in ['prior','post']:
                hits=z['hits_'+side]
                if len(hits)>r.get('return_horizon',r['horizon']):raise ValueError('Return trace exceeds evaluation budget')
                if not np.allclose(z['returns_'+side],first_success_return(hits),atol=1e-9,rtol=0):
                    raise ValueError('Recorded return differs from actual success-event trace')
            if r.get('full_episode_return',False) and r['return_horizon']!=r['episode_deadline_steps']-r['executed_steps_before_pair']:
                raise ValueError('Full-episode flag conflicts with actual rollout budget')
            a=z['action_prior'];b=z['action_post'];n=len(a)
            if n!=r['horizon'] or n>10 or a.shape!=b.shape or a.shape[1]!=7:raise ValueError('Action alignment mismatch')
            def pad(x):return np.pad(x,((0,10-n),(0,0))).astype(np.float32)
            out['state'].append(z['state']);out['belief'].append(z['prior'])
            out['delta_belief'].append(z['posterior']-z['prior'])
            out['action'].append(pad(a));out['target'].append(pad(b-a))
            out['task'].append(task_vector(r['task']))
            out['meta'].append([r['suffix_start']/10,n/10,1])
            valid=r.get('valid_suffix_steps',n)
            if not 1<=valid<=n:raise ValueError('Invalid measured suffix mask')
            out['mask'].append(np.arange(10)<valid);out['sequence'].append(r['sequence'])
    return {k:np.asarray(v) for k,v in out.items()},rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--device',default='cuda:0')
    p.add_argument('--min-pairs',type=int,default=64);p.add_argument('--min-sequences',type=int,default=10)
    p.add_argument('--allow-truncated-return',action='store_true',help='Explicit historical/truncated-return ablation only')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.min_sequences<6 or a.min_pairs<12 or a.steps<1:p.error('At least 6 sequences, 12 pairs, positive steps required')
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise SystemExit('CUDA unavailable; explicit --device cpu allowed')
    torch.set_num_threads(4);torch.manual_seed(42);rng=np.random.default_rng(42)
    d,allrows=load(a.run,a.min_pairs,a.min_sequences)
    if not a.allow_truncated_return and any(not r.get('full_episode_return',False) for r in allrows if r['accepted']):
        raise ValueError('Training defaults to full-episode measured returns; recollect or explicitly choose --allow-truncated-return for an ablation')
    seqs=np.unique(d['sequence']);rng.shuffle(seqs);nv=max(1,len(seqs)//5);nt=max(1,len(seqs)//5)
    groups={'test':seqs[:nt],'validation':seqs[nt:nt+nv],'train':seqs[nt+nv:]}
    ids={k:np.flatnonzero(np.isin(d['sequence'],v)) for k,v in groups.items()}
    norm={}
    for k in ['state','belief','delta_belief']:
        m=d[k][ids['train']].mean(0);s=np.maximum(d[k][ids['train']].std(0),1e-4)
        norm[k]={'mean':m,'std':s};d[k]=(d[k]-m)/s
    x={k:torch.as_tensor(v,dtype=torch.float32,device=a.device) for k,v in d.items() if k!='sequence'}
    config=dict(state_dim=x['state'].shape[1],summary_dim=x['belief'].shape[1],action_dim=7,task_dim=32,
        chunk_len=10,hidden_dim=128,cond_dim=128,num_blocks=3,dropout=.05,output_scale=.02)
    model=DirectionalSensitivityTCN(**config).to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    def forward(i):return model.forward_with_heads(x['state'][i],x['belief'][i],x['action'][i],x['delta_belief'][i],x['task'][i],x['meta'][i])
    def masked(v,mask):return (v*mask).sum()/mask.sum().clamp_min(1)
    def metrics(i):
        model.eval()
        with torch.no_grad():
            h=forward(i);target=x['target'][i];mask=x['mask'][i]
            mse=masked((h['delta']-target).square().mean(-1),mask)
            base=masked(target.square().mean(-1),mask)
            cosine=nn.functional.cosine_similarity(h['delta'],target,dim=-1)
            active=(target.norm(dim=-1)>1e-4).float()*mask
            return {'mse':float(mse),'zero_correction_mse':float(base),
                    'active_direction_cosine':float(masked(cosine,active)),'frames':int(mask.sum())}
    best=float('inf');best_step=0;weights=None;history=[]
    for step in range(1,a.steps+1):
        model.train();i=rng.choice(ids['train'],min(64,len(ids['train'])),replace=False)
        h=forward(i);target=x['target'][i];mask=x['mask'][i]
        scale=.003
        reg=masked(((h['delta']-target)/scale).square().mean(-1),mask)
        affected=(target.norm(dim=-1)>1e-4).float()
        cls=masked(nn.functional.binary_cross_entropy_with_logits(h['invalidity_logit'],affected,reduction='none'),mask)
        boundary=torch.exp(-torch.abs(target.norm(dim=-1)-1e-4)/1e-4)
        unc=masked((h['uncertainty_logit'].sigmoid()-boundary).abs(),mask)
        loss=reg+cls+.1*unc
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0 or step==a.steps:
            val=metrics(ids['validation']);history.append({'step':step,'loss':float(loss),**val})
            print(json.dumps(history[-1]),flush=True)
            if val['mse']<best:best=val['mse'];best_step=step;weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(weights);result=metrics(ids['test'])
    a.output.mkdir(parents=True)
    torch.save({'state_dict':weights,'config':config,'normalization':norm,'selected_step':best_step,
        'note':'Research checkpoint, different normalization and rebased cache; NOT drop-in old-runtime deployment'},a.output/'tcn.pt')
    report={'attempted_pairs':len(allrows),'accepted_pairs':len(d['sequence']),'split_sequences':{k:v.tolist() for k,v in groups.items()},
        'selected_step':best_step,'test':result,'history':history,'robot_success_evaluated':False,
        'protocol_versions':sorted({r.get('protocol_version',1) for r in allrows}),
        'full_episode_return':all(r.get('full_episode_return',False) for r in allrows if r['accepted']),
        'limitations':['Return ends at current-subtask success or common episode deadline, not the entire multi-subtask chain; no partial-progress reward.',
            'Positive-return selection does not prove necessity of each action edit.',
            'Fixed pooled belief-to-decoder adapter and cached observation.',
            'Short suffixes padded; padded positions excluded from all losses/metrics.',
            'No automatic replacement of old TCN; requires compatible runtime integration and rollout validation.']}
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
