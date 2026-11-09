"""Four-outcome training ONLY; export the unchanged original TCN interface."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train import load,DirectionalSensitivityTCN
from paths import defi_root
from outcome import LABELS


def runtime_helpers():
    """Reuse exact pure feature helpers without importing the full GPU evaluator."""
    root=defi_root()/'Step3_DeFI/policy_evaluation'
    ns={'np':np,'hashlib':hashlib}
    for path,names in [(root/'calvin_evaluate_with_memory_reflection.py',
            {'_pool_exec_summary_np','_coerce_runtime_vector_np','_chunk_effect_runtime_meta'}),
            (root/'train_sufficiency_mask_critic.py',{'hashed_task'})]:
        nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
        if len(nodes)!=len(names):raise ValueError('Original runtime helper layout changed')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns


def legacy_inputs(run,d,rows,config):
    helpers=runtime_helpers();pool=helpers['_pool_exec_summary_np'];coerce=helpers['_coerce_runtime_vector_np']
    size=config['summary_dim'];before=[];delta=[];states=[];tasks=[];meta=[]
    for row in rows:
        with np.load(run/row['file']) as z:
            if 'legacy_runtime_belief_before' not in z:
                raise ValueError('Recollect with current collector: actual original-runtime belief inputs missing; cached decoder goals are NOT equivalent')
            b=pool(z['legacy_runtime_belief_before'],size)
            if 'legacy_runtime_innovation' in z:change=pool(z['legacy_runtime_innovation'],size)
            else:change=pool(z['legacy_runtime_belief_after'],size)-b
            before.append(b);delta.append(change);states.append(coerce(z['state'],config['state_dim']))
        tasks.append(helpers['hashed_task'](row['task'],config['task_dim']))
        # Existing inference pads remaining suffix to chunk_len BEFORE building meta.
        meta.append(helpers['_chunk_effect_runtime_meta'](row['subtask'],config['chunk_len'],config['chunk_len']).reshape(-1))
    d.update(state=np.asarray(states),belief=np.asarray(before),delta_belief=np.asarray(delta),
             task=np.asarray(tasks),meta=np.asarray(meta))
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--initial-tcn',type=Path,required=True,help='Original directional sensitivity TCN; architecture and heads retained')
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--device',default='cuda:0')
    p.add_argument('--min-pairs',type=int,default=64);p.add_argument('--min-sequences',type=int,default=10)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.steps<1 or a.min_pairs<12 or a.min_sequences<6:p.error('Require positive steps, >=12 pairs and >=6 original episodes')
    torch.manual_seed(42);torch.set_num_threads(4);rng=np.random.default_rng(42)
    # Trusted local checkpoint only; no random substitute for unavailable old weights.
    old=torch.load(a.initial_tcn,map_location='cpu',weights_only=False)
    required=['state_dim','summary_dim','action_dim','task_dim','chunk_len']
    if not all(k in old for k in required) or 'model_state' not in old:raise ValueError('Expected original directional TCN checkpoint')
    defaults=dict(hidden_dim=128,cond_dim=256,num_blocks=3,kernel_size=3,dropout=.05,output_scale=1.)
    config={k:old[k] for k in required};config.update({k:old.get(k,v) for k,v in defaults.items()})
    if config['action_dim']!=7 or config['chunk_len']!=10:raise ValueError('Collector supports 10x7 cache only')
    d,rows=load(a.run,a.min_pairs,a.min_sequences,all_outcomes=True)
    d=legacy_inputs(a.run,d,rows,config)
    if any(not row.get('episode_key') for row in rows):raise ValueError('Recollect data with stable episode keys before claiming held-out evaluation')
    epkeys=np.asarray([r['episode_key'] for r in rows])
    seq=np.unique(epkeys)
    if len(seq)<a.min_sequences:raise ValueError('Insufficient independent original episodes (duplicate sequence IDs do not count)')
    rng.shuffle(seq);n=max(1,len(seq)//5)
    groups={'test':seq[:n],'validation':seq[n:2*n],'train':seq[2*n:]}
    ids={k:np.flatnonzero(np.isin(epkeys,v)) for k,v in groups.items()}
    counts={k:np.bincount(d['outcome'][i],minlength=4).tolist() for k,i in ids.items()}
    if min(counts['train'])==0:raise ValueError(f'Missing training outcome classes: {counts}; collect more real pairs')
    normalization={}
    for k in ['state','belief','delta_belief']:
        mean=d[k][ids['train']].mean(0);std=np.maximum(d[k][ids['train']].std(0),1e-4)
        normalization[k]=(mean,std);d[k]=(d[k]-mean)/std
    x={k:torch.as_tensor(v,device=a.device,dtype=torch.long if k=='outcome' else torch.float32) for k,v in d.items() if k!='sequence'}
    model=DirectionalSensitivityTCN(**config).to(a.device);model.load_state_dict(old['model_state'],strict=True)
    # Extra classification/risk heads exist only while training; NEVER exported.
    aux=nn.Linear(config['hidden_dim'],5).to(a.device)
    for module in [model.validity_proj,model.uncertainty_proj]:module.requires_grad_(False)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]+list(aux.parameters()),lr=1e-4,weight_decay=1e-4)
    def average(v,m):return (v*m).sum()/m.sum().clamp_min(1)
    def forward(i):
        features=model._features(*(x[k][i] for k in ['state','belief','action','delta_belief','task','meta']))
        length=torch.tensor([rows[j]['horizon'] for j in i],device=a.device)
        known=(torch.arange(10,device=a.device)[None,:]<length[:,None]).float()
        pooled=(features*known[:,None,:]).sum(-1)/known.sum(-1,keepdim=True).clamp_min(1)
        return model.out_proj(features).transpose(1,2)*model.output_scale,aux(pooled)
    def objective(pred,heads,i):
        y=x['outcome'][i];mask=x['mask'][i]
        repair=(y==0).float()[:,None]*mask
        preserve=(y==1).float()[:,None]*mask;harmful=(y==3).float()[:,None]*mask
        reg=average(((pred-x['target'][i])/.003).square().mean(-1),repair)
        zero=(pred/.003).square().mean(-1)
        # Unresolved has no known successful direction: auxiliary loss only, NOT zero target.
        return reg+.1*average(zero,preserve)+average(zero,harmful)+F.cross_entropy(heads[:,:4],y)+F.binary_cross_entropy_with_logits(heads[:,4],(y==3).float())
    def metrics(i):
        model.eval();aux.eval()
        with torch.no_grad():
            pred,heads=forward(i);y=x['outcome'][i];repair=(y==0).float()[:,None]*x['mask'][i]
            return dict(loss=float(objective(pred,heads,i)),repair_mse=float(average((pred-x['target'][i]).square().mean(-1),repair)),
                auxiliary_accuracy=float((heads[:,:4].argmax(-1)==y).float().mean()),
                harmful_delta_norm=float(average(pred.norm(dim=-1),(y==3).float()[:,None]*x['mask'][i])),
                preserve_delta_norm=float(average(pred.norm(dim=-1),(y==1).float()[:,None]*x['mask'][i])))
    train=ids['train'];ratios=np.array([.4,.3,.15,.15]);weights=ratios[d['outcome'][train]]/np.array(counts['train'])[d['outcome'][train]];weights/=weights.sum()
    best=float('inf');saved=None;selected=0;history=[]
    for step in range(1,a.steps+1):
        model.train();aux.train();i=rng.choice(train,size=min(64,len(train)),replace=True,p=weights)
        pred,heads=forward(i);loss=objective(pred,heads,i);opt.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(list(model.parameters())+list(aux.parameters()),1);opt.step()
        if step%100==0 or step==a.steps:
            result=metrics(ids['validation']);history.append(dict(step=step,**result));print(json.dumps(history[-1]),flush=True)
            if result['loss']<best:
                best=result['loss'];selected=step
                saved=({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                       {k:v.detach().cpu().clone() for k,v in aux.state_dict().items()})
    model.load_state_dict(saved[0]);aux.load_state_dict(saved[1]);test=metrics(ids['test'])
    a.output.mkdir(parents=True)
    exported=dict(old);exported.update(config);exported['model_state']=saved[0]
    for key,prefix in [('state','state'),('belief','h'),('delta_belief','delta_h')]:
        exported[prefix+'_mean'],exported[prefix+'_std']=normalization[key]
    exported['training_method']='four_outcome_auxiliary_original_inference_unchanged'
    torch.save(exported,a.output/'directional_sensitivity_tcn.pt')
    report=dict(initial_tcn=str(a.initial_tcn),export='directional_sensitivity_tcn.pt',
        inference_changed=False,auxiliary_heads_exported=False,labels=LABELS,class_counts=counts,
        split_sequences={k:np.unique(d['sequence'][ids[k]]).tolist() for k in groups},
        split_episode_keys={k:v.tolist() for k,v in groups.items()},
        checkpoint_sha256=hashlib.sha256((a.output/'directional_sensitivity_tcn.pt').read_bytes()).hexdigest(),
        selected_step=selected,history=history,test=test,
        robot_success_evaluated=False,limitations=['No online four-class gate or explicit Unresolved replanner; original inference unchanged.',
            'Original validity/uncertainty head weights retained, but shared features change; validate original gating in new rollouts.',
            'Successful chunk repair does not prove action-wise causal necessity.'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(test,indent=2))

if __name__=='__main__':main()
