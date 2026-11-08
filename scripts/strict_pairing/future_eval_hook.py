"""Log frozen future ablation and compare interventions at the same method trigger."""
import functools
import hashlib
import inspect
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from future_features import install_feature_control


def install(ns,root,feature,intervention,seed):
    from policy_models.module.diffusion_extract import Diffusion_feature_extractor
    from policy_models.VPP_policy import VPP_Policy
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    install_feature_control(Diffusion_feature_extractor,feature)
    context={};depth={};counts={'feature':feature,'intervention':intervention}
    def log(name,row):
        with (root/name).open('a') as f:f.write(json.dumps(row)+'\n')
    for name in ['patch_remaining_action_seq','sensitivity_transport_remaining_action_seq','counterfactual_transport_remaining_action_seq']:
        if not hasattr(VPP_Policy,name):continue
        original=getattr(VPP_Policy,name)
        def wrap(original,name):
            @functools.wraps(original)
            def call(model,*args,**kwargs):
                key=id(model);level=depth.get(key,0);depth[key]=level+1
                cache=getattr(model,'pred_action_seq',None)
                before=cache.detach().clone() if level==0 and cache is not None else None
                cut=int(model.rollout_step_counter);start=time.perf_counter()
                try:result=original(model,*args,**kwargs)
                finally:depth[key]-=1
                if before is not None and model.pred_action_seq is not None and level==0:
                    after=model.pred_action_seq
                    if before.shape!=after.shape:raise ValueError('Unexpected cache shape change during repair')
                    change=(after-before).float();scores=change[0,cut:].norm(dim=-1)
                    indices=torch.nonzero(scores>1e-8).flatten().cpu().tolist()
                    if indices:
                        row=dict(context,method=name,suffix_start=cut,edited_indices=[cut+i for i in indices],
                            changed_steps=len(indices),remaining_steps=len(scores),
                            delta_l2=float(change.norm()),elapsed_s=time.perf_counter()-start,
                            action_steps_before=before[0,cut:].float().cpu().tolist(),
                            action_steps_after=after[0,cut:].float().cpu().tolist(),
                            annotation='coverage only; NOT localization accuracy without independently measured necessary-edit labels')
                        frame=inspect.currentframe().f_back
                        while frame and frame.f_code.co_name!='rollout_with_lang_text':frame=frame.f_back
                        if frame:
                            row.update(step=int(frame.f_locals['step']),subtask=int(frame.f_locals['subtask_index']),task=str(frame.f_locals['subtask']))
                        if intervention=='full_replan':
                            model.pred_action_seq=before
                            model.request_full_replan_next_step()
                            row['actual']='full_replan_on_same_successful_patch_trigger'
                            # Preserve caller's return schema while truthfully disabling the patch.
                            result=dict(result or {},patched=False,applied=False,replan_requested=True,
                                reason='future_ablation_matched_trigger_full_replan')
                        else:row['actual']='original_selective_repair'
                        log('interventions.jsonl',row)
                return result
            return call
        setattr(VPP_Policy,name,wrap(original,name))
    original_seq=ns['evaluate_sequence_memory'];sig=inspect.signature(original_seq)
    @functools.wraps(original_seq)
    def sequence(*args,**kwargs):
        b=sig.bind(*args,**kwargs);b.apply_defaults();v=b.arguments
        index=int(v['seq_idx']);tasks=[str(x) for x in v['eval_sequence']]
        trial_seed=(seed+index*1000003)%(2**32)
        random.seed(trial_seed);np.random.seed(trial_seed);torch.manual_seed(trial_seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(trial_seed)
        context.clear();context.update(sequence=index,tasks=tasks,seed=trial_seed)
        initial=json.dumps(v['initial_state'],sort_keys=True,default=lambda x:np.asarray(x).tolist())
        start=time.perf_counter();completed=int(original_seq(*args,**kwargs))
        log('sequences.jsonl',dict(context,initial_state_sha256=hashlib.sha256(initial.encode()).hexdigest(),
            completed=completed,total=len(tasks),success=completed==len(tasks),elapsed_s=time.perf_counter()-start))
        return completed
    ns['evaluate_sequence_memory']=sequence
    original_log=ns.get('get_log_dir')
    (root/'original_logs').mkdir(exist_ok=True)
    if original_log:ns['get_log_dir']=lambda *a,**kw:root/'original_logs'
    original_main=ns['main']
    def main(cfg):
        cfg.log_wandb=False
        cfg.seed=seed
        # All other original controller settings are kept as supplied by launcher.
        return original_main(cfg)
    ns['main']=main
