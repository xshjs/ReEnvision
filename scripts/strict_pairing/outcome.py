"""Sequence-level outcome supervision and conservative inference gate.

Predictions concern a proposed correction, not causal necessity of individual edits.
Thresholds are explicit research defaults, not validated deployment thresholds.
"""
import numpy as np
import torch
from torch import nn

LABELS=('repair','preserve','unresolved','harmful')

def classify_outcome(before,after):
    a=np.asarray(before);b=np.asarray(after)
    if a.ndim!=1 or a.shape!=b.shape or a.size<2:
        raise ValueError('Require repeated matched binary outcomes')
    if not np.isin(a,[0,1]).all() or not np.isin(b,[0,1]).all() or not (a==a[0]).all() or not (b==b[0]).all():
        raise ValueError('Non-binary or nondeterministic outcomes')
    return {(0,1):0,(1,1):1,(0,0):2,(1,0):3}[int(a[0]),int(b[0])]

class OutcomeTCN(nn.Module):
    def __init__(self,config):
        super().__init__()
        from train import DirectionalSensitivityTCN
        self.tcn=DirectionalSensitivityTCN(**config)
        # Heads evaluate the generated delta as well as shared context features.
        width=config['hidden_dim']+config['action_dim']
        self.outcome=nn.Linear(width,4)
        self.risk=nn.Linear(width,1)

    def forward(self,state,belief,action,delta_belief,task,meta,mask):
        features=self.tcn._features(state,belief,action,delta_belief,task,meta)
        delta=self.tcn.out_proj(features).transpose(1,2)*self.tcn.output_scale
        w=mask.unsqueeze(-1)
        pooled=(torch.cat([features.transpose(1,2),delta],-1)*w).sum(1)/w.sum(1).clamp_min(1)
        return {'delta':delta,'outcome_logits':self.outcome(pooled),'risk_logit':self.risk(pooled).squeeze(-1)}

def decide(cached,delta,probabilities,risk,*,repair_threshold=.8,risk_threshold=.1,
           confidence_threshold=.7,max_delta=.015,thresholds_validated=False):
    """Pure gate; caller must handle replan/hold before executing unresolved actions.

    Risk means harmful regression (cache success -> candidate failure), NOT any failure.
    This function does not execute a robot or silently launch a planner.
    """
    a=np.asarray(cached);d=np.asarray(delta);p=np.asarray(probabilities)
    if a.shape!=d.shape or p.shape!=(4,) or not np.isfinite(a).all() or not np.isfinite(d).all() or not np.isfinite(p).all():
        raise ValueError('Invalid gate inputs')
    if (p<0).any() or not np.isclose(p.sum(),1,atol=1e-5) or not np.isfinite(risk) or not 0<=risk<=1:
        raise ValueError('Invalid probabilities')
    if max_delta<=0 or any(not 0<=x<=1 for x in [repair_threshold,risk_threshold,confidence_threshold]):
        raise ValueError('Invalid gate controls')
    label=LABELS[int(p.argmax())]
    if not thresholds_validated:
        return dict(action=a.copy(),decision='unvalidated',requires_replan=True,execute=False)
    if label=='harmful' or risk>risk_threshold:
        return dict(action=a.copy(),decision='reject_harmful',requires_replan=False,execute=True)
    if p.max()<confidence_threshold or label=='unresolved':
        return dict(action=a.copy(),decision='replan',requires_replan=True,execute=False)
    if label=='repair' and p[0]>=repair_threshold:
        return dict(action=a+np.clip(d,-max_delta,max_delta),decision='repair',requires_replan=False,execute=True)
    if label=='preserve':
        return dict(action=a.copy(),decision='preserve',requires_replan=False,execute=True)
    return dict(action=a.copy(),decision='uncertain',requires_replan=True,execute=False)
