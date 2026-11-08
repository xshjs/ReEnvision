"""Online CALVIN cache gate; called after observation and belief update, before next action.

No simulator oracle outcomes are passed to the learned model. The oracle is used
by the evaluator only to skip interventions after a task already terminated.
"""
import inspect
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from cache_anchor import install_capture,versions,CacheMismatch
from infer_outcome import SelectiveRepair
from train import task_vector


def gate_inputs(model,payload,state,task,remaining_tasks,belief_scale=1.):
    generation=getattr(model,'_strict_live_cache_generation',None)
    cache=getattr(model,'pred_action_seq',None)
    cut=int(model.rollout_step_counter)
    if generation is None or cache is None:raise CacheMismatch('Missing original cache provenance')
    if model.training or any(p.requires_grad for p in model.parameters()):raise CacheMismatch('Policy not frozen')
    if generation['parameter_versions']!=versions(model):raise CacheMismatch('Policy weights changed')
    original=generation['actions']
    if original.shape!=cache.shape or not torch.equal(original,cache):raise CacheMismatch('Cache already edited')
    if cache.ndim!=3 or cache.shape[0]!=1 or not 1<=cut<cache.shape[1]:raise CacheMismatch('No valid unexecuted suffix')
    if not payload or payload.get('h_prior') is None or payload.get('h_posterior') is None:
        raise CacheMismatch('No current RSSM update')
    goal=generation['inputs']['goal']
    innovation=(payload['h_posterior'].detach().float()-payload['h_prior'].detach().float()).reshape(1,1,-1)
    revision=F.adaptive_avg_pool1d(innovation,goal.numel()).reshape_as(goal).to(goal)*belief_scale
    n=cache.shape[1]-cut
    def cpu(x):return x.detach().float().cpu().numpy()
    return dict(state=np.asarray(state,np.float32),belief=cpu(goal).reshape(-1),
        delta_belief=cpu(revision).reshape(-1),action=cpu(cache[0,cut:]),
        task=np.concatenate([task_vector(task),task_vector(json.dumps(remaining_tasks,ensure_ascii=False))]),
        meta=np.asarray([cut/10,n/10,1],np.float32),valid_steps=n)


@torch.inference_mode()
def apply_gate(model,result):
    """Consume explicit decision, never execute stale cache when requesting replan."""
    if result['requires_replan'] or not result['execute']:
        model.request_full_replan_next_step()
        return 'replan'
    if result['decision']=='repair':
        a=torch.as_tensor(result['action'],device=model.pred_action_seq.device,dtype=model.pred_action_seq.dtype)
        cut=int(model.rollout_step_counter)
        if a.shape!=model.pred_action_seq[0,cut:].shape or not torch.isfinite(a).all():
            raise ValueError('Gate output does not align with actual cache')
        # Direct assignment: no hidden extra mixing, smoothing, or gripper processing.
        model.pred_action_seq[0,cut:]=a
        model._outcome_edited_generation=getattr(model,'_strict_live_cache_generation',None)
        return 'repair'
    return 'preserve'


class OnlineGate:
    def __init__(self,checkpoint,log,observe_only=False,device='cpu',belief_scale=1.):
        self.predictor=SelectiveRepair(checkpoint,device)
        self.observe_only=observe_only;self.belief_scale=belief_scale
        self.log=Path(log);self.log.parent.mkdir(parents=True,exist_ok=True)
        self.hits=0;self.applied=0;self.replans=0
        if not observe_only and not self.predictor.ckpt.get('thresholds_validated',False):
            raise ValueError('Outcome checkpoint thresholds are unvalidated: use --observe-only or validate on independent matched rollouts first')
        if not observe_only and self.predictor.ckpt.get('max_action_delta') is None:
            raise ValueError('No validated measured correction bound')
        self.thresholds=self.predictor.ckpt.get('gate_thresholds',{})
        if not observe_only and set(self.thresholds)!={'repair_threshold','risk_threshold','confidence_threshold'}:
            raise ValueError('Checkpoint must contain explicitly calibrated gate_thresholds')

    def tick(self,live,remaining_tasks):
        model=live['model'];payload=live.get('joint_update_payload')
        if live.get('current_task_info'):return
        generation=getattr(model,'_strict_live_cache_generation',None)
        if generation is not None and generation is getattr(model,'_outcome_edited_generation',None):return
        if model.pred_action_seq is None or model.rollout_step_counter==0:return
        if not payload:return
        self.hits+=1
        try:
            values=gate_inputs(model,payload,live['state_after_step'],str(live['subtask']),remaining_tasks,self.belief_scale)
            result=self.predictor.predict(**values,**self.thresholds)
            actual='observe_only' if self.observe_only else apply_gate(model,result)
            self.applied+=actual=='repair';self.replans+=actual=='replan'
            row=dict(step=int(live['step']),task=str(live['subtask']),decision=result['decision'],
                actual=actual,probabilities=result['probabilities'],harmful_risk=result['harmful_risk'],
                requires_replan=result['requires_replan'],suffix_start=int(values['meta'][0]*10))
        except CacheMismatch as exc:
            row=dict(step=int(live['step']),decision='skip_unanchored',reason=str(exc))
            if not self.observe_only:model.request_full_replan_next_step();self.replans+=1
        except Exception:
            # Never silently fall back to an ungated edit after a broken inference.
            if not self.observe_only:model.request_full_replan_next_step()
            raise
        with self.log.open('a') as f:f.write(json.dumps(row)+'\n')


def install(ns,checkpoint,log,observe_only=False,device='cpu',belief_scale=1.):
    from policy_models.VPP_policy import VPP_Policy
    install_capture(VPP_Policy)
    # Defense in depth: all legacy patch calls are disabled in this launch mode.
    # The only allowed edit path is apply_gate's checked, direct cache assignment.
    VPP_Policy.patch_remaining_action_seq=lambda self,*a,**kw:dict(patched=False,reason='outcome_gate_owns_cache')
    gate=OnlineGate(checkpoint,log,observe_only,device,belief_scale)
    def tick(live):
        frame=inspect.currentframe().f_back
        while frame and frame.f_code.co_name!='evaluate_sequence_memory':frame=frame.f_back
        if frame is None:raise RuntimeError('Original remaining task chain unavailable')
        seq=[str(x) for x in frame.f_locals['eval_sequence']]
        index=int(live['subtask_index'])
        if seq[index]!=str(live['subtask']):raise ValueError('Task sequence alignment mismatch')
        gate.tick(live,seq[index:])
    ns['_outcome_online_tick']=tick
    original=ns['main']
    def main(cfg):
        cfg.log_wandb=False
        cfg.joint_belief_clean_controller=True
        cfg.dynamic_coupling_delta_action_repair_mix=0.
        cfg.dynamic_coupling_disable_replan=True
        cfg.collect_sensitivity_teacher=False
        cfg.directional_sensitivity_tcn_transport=False
        cfg.direct_action_correction_tcn_transport=False
        cfg.linear_suffix_sensitivity_transport=False
        cfg.counterfactual_suffix_transport=False
        cfg.suffix_mask_ablation_mode='off'
        return original(cfg)
    ns['main']=main
    return gate
