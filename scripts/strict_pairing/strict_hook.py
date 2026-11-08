"""Runtime hook for the existing CALVIN collector; original sources untouched."""
import copy
import hashlib
import inspect
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from strict_core import BulletSnapshot,RNG,PolicyRuntime,array_hash,accept_pair,rollout_remaining_chain


class CollectionComplete(BaseException):pass


def install(ns):
    root=Path(os.environ['REENVISION_STRICT_RUN']);root.mkdir(parents=True,exist_ok=True)
    rows=root/'pairs.jsonl';count=0;seen=0
    max_pairs=int(os.environ.get('STRICT_MAX_PAIRS','200'))
    every=int(os.environ.get('STRICT_EVERY','20'))
    original=ns['maybe_save_sensitivity_teacher_row']
    def record(cfg,**kw):
        nonlocal count,seen
        # Supported collector version supplies live locals; fail closed otherwise.
        frame=inspect.currentframe().f_back
        while frame and frame.f_code.co_name!='rollout_with_lang_text':frame=frame.f_back
        if frame is None:raise SystemExit('Cannot locate live rollout context')
        live=frame.f_locals;seen+=1
        if seen%every:return None
        chain_frame=frame.f_back
        while chain_frame and chain_frame.f_code.co_name!='evaluate_sequence_memory':chain_frame=chain_frame.f_back
        if chain_frame is None:raise SystemExit('Full remaining task sequence is unavailable; refusing subtask-only return')
        sequence=[str(x) for x in chain_frame.f_locals['eval_sequence']]
        sub_index=int(kw['subtask_index'])
        if not 0<=sub_index<len(sequence) or sequence[sub_index]!=kw['task']:
            raise SystemExit('Task-chain position mismatch')
        remaining_tasks=sequence[sub_index:]
        payload=live.get('joint_update_payload')
        if not payload or payload.get('h_prior') is None or payload.get('h_posterior') is None:
            print('[strict] skip: no true RSSM prior/posterior',flush=True);return None
        model=live['model'];env=live['env'];oracle=live['task_oracle'];task=kw['task']
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise SystemExit('Strict collector requires eval() and frozen weights')
        raw=env
        while hasattr(raw,'env'):raw=raw.env
        if not getattr(env,'relative_actions',False):raise SystemExit('Only relative-action CALVIN supported')
        ref=copy.deepcopy(live['start_info'])
        done=lambda info:bool(oracle.get_task_info_for_set(ref,info,{task}))
        if done(env.get_info()):return None
        ctx=getattr(model,'_last_suffix_transport_context',None)
        noise=getattr(model,'_last_denoise_initial_x',None)
        if ctx is None or noise is None:return None
        h0=payload['h_prior'].detach().float().reshape(1,-1)
        h1=payload['h_posterior'].detach().float().reshape(1,-1)
        if h0.shape!=h1.shape or h0.shape[1]%2:raise SystemExit('Invalid joint belief')
        if not torch.isfinite(h0).all() or not torch.isfinite(h1).all():return None
        half=h0.shape[1]//2
        # Explicit fixed adapter: pooled RSSM future/action halves to decoder conditions.
        # This mapping, not return labels, is the intervention being evaluated.
        shape=ctx['latent_goal_base'].shape;adim=int(model.action_intent_dim)
        def condition(h):
            f=F.adaptive_avg_pool1d(h[:,:half].unsqueeze(1),int(np.prod(shape))).reshape(shape)
            a=F.adaptive_avg_pool1d(h[:,half:].unsqueeze(1),adim).reshape(1,16,7)
            return model._make_action_intent_conditions(f,a)
        c0=condition(h0);c1=condition(h1)
        original_noise=noise.detach().clone();cached=model.pred_action_seq.detach().clone()
        start=int(model.rollout_step_counter)
        if start<1 or start>=cached.shape[1]:return None
        rng=RNG();values=[]
        context_hash=hashlib.sha256(''.join(array_hash(v.detach().cpu().numpy())
            for k,v in sorted(ctx['perceptual_emb'].items()) if torch.is_tensor(v)).encode()).hexdigest()
        try:
            with torch.inference_mode():
                for c in [c0,c1,c0]:
                    rng.restore()
                    values.append(model.denoise_actions(torch.zeros_like(c[0]),
                        {k:v.clone() if torch.is_tensor(v) else v for k,v in ctx['perceptual_emb'].items()},
                        c[0].clone(),ctx.get('latent_motion_tokens_up').clone() if torch.is_tensor(ctx.get('latent_motion_tokens_up')) else None,
                        action_intent_cond=c[1],inference=True,initial_x=original_noise.clone()).detach())
        finally:
            model._last_denoise_initial_x=original_noise;rng.restore()
        if not torch.allclose(values[0],values[2],atol=1e-7,rtol=0):
            raise SystemExit('Matched noise/conditioning does not reproduce action output')
        if not torch.equal(cached,model.pred_action_seq):raise SystemExit('Decoder modified action cache')
        a0=values[0][0,start:].cpu().numpy().copy();a1=values[1][0,start:].cpu().numpy().copy()
        if not (np.isfinite(a0).all() and np.isfinite(a1).all()):return None
        if a0.shape!=a1.shape or len(a0)==0:return None
        requested_tail=int(os.environ.get('STRICT_CONTINUATION_STEPS','-1'))
        executed_steps=int(kw['step'])+1
        if int(cfg.ep_len)-executed_steps < len(a0):
            print('[strict] skip: complete candidate suffix exceeds episode deadline',flush=True)
            return None
        first_budget=int(cfg.ep_len)-executed_steps
        chain_budget=first_budget+(len(remaining_tasks)-1)*int(cfg.ep_len)
        tail_steps=chain_budget-len(a0) if requested_tail==-1 else min(requested_tail,chain_budget-len(a0))
        gamma=float(os.environ.get('STRICT_RETURN_GAMMA','1'))
        # No action clipping: the target must match the actions actually measured.
        snapshot=BulletSnapshot(raw)
        try:
            def step(a):return env.step(torch.as_tensor(a,device=model.device))[3]
            runtime=PolicyRuntime(model)
            tail_seed=int(os.environ.get('STRICT_CONTINUATION_SEED','1701'))
            rec={0:[],1:[]}
            try:
                for side in [0,1,1,0]:
                    snapshot.restore();runtime.restore();refs={};goals={};reset_done=set()
                    def reset_policy():
                        model.reset();model.future_feature_mode='gfdm'
                        model.override_future_feature=None;model.override_action_intent=None
                        model.override_latent_motion_tokens_up=None
                    def begin_task(index):
                        refs[index]=copy.deepcopy(ref if index==0 else env.get_info())
                        if index==0:goals[index]=copy.deepcopy(live['goal'])
                        else:
                            lang=str(live['val_annotations'][remaining_tasks[index]][0])
                            goals[index]=live['lang_embeddings'].get_lang_goal(lang)
                            goals[index]['lang_text']=lang
                            reset_policy();reset_done.add(index)
                    def chain_success(index,info):
                        return bool(oracle.get_task_info_for_set(refs[index],info,{remaining_tasks[index]}))
                    def next_action(index,local):
                        if index not in reset_done:reset_policy();reset_done.add(index)
                        # Couple sampling noise by task position and local control step.
                        seed=(tail_seed+index*int(cfg.ep_len)+local)%(2**32)
                        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
                        obs=env.get_obs()
                        with torch.inference_mode():act=model.step(obs,copy.deepcopy(goals[index])).detach().clone()
                        arr=act.float().cpu().numpy().reshape(-1).copy()
                        if arr.shape!=(7,) or not np.isfinite(arr).all():raise ValueError('Invalid continuation action')
                        return arr
                    result=rollout_remaining_chain([a0,a1][side],len(remaining_tasks),first_budget,int(cfg.ep_len),
                        step,begin_task,next_action,chain_success,gamma=gamma,
                        max_steps=None if requested_tail==-1 else len(a0)+tail_steps)
                    result['endpoint']=snapshot.fingerprint();rec[side].append(result)
                for side in [0,1]:
                    a,b=rec[side]
                    if a['rewards']!=b['rewards'] or a['events']!=b['events'] or a['termination']!=b['termination'] or not np.allclose(a['endpoint'],b['endpoint'],atol=1e-7,rtol=0):
                        raise ValueError('Full-chain replay is nondeterministic')
                    if np.asarray(a['actions']).shape!=np.asarray(b['actions']).shape or not np.allclose(a['actions'],b['actions'],atol=1e-7,rtol=0):
                        raise ValueError('Full-chain action replay mismatch')
            finally:
                try:runtime.restore()
                finally:snapshot.restore()
            ret0=[r['return_value'] for r in rec[0]];ret1=[r['return_value'] for r in rec[1]]
            valid_suffix_steps=min(rec[0][0]['suffix_executed'],rec[1][0]['suffix_executed'])
            accepted=accept_pair(ret0,ret1,float(os.environ.get('STRICT_MIN_GAP','.01')))
            identity=(int(kw['sequence_index']),int(kw['subtask_index']),int(kw['step']))
            # All attempted pairs retained, including ties and regressions.
            file=root/f'pair_{count:06d}.npz'
            np.savez_compressed(file,state=np.asarray(kw['state_after'],np.float32),
                prior=h0.cpu().numpy()[0],posterior=h1.cpu().numpy()[0],
                action_prior=a0,action_post=a1,cached_suffix=cached[0,start:].cpu().numpy(),
                returns_prior=np.asarray(ret0),returns_post=np.asarray(ret1),
                rewards_prior=np.asarray(rec[0][0]['rewards']),rewards_post=np.asarray(rec[1][0]['rewards']),
                active_tasks_prior=np.asarray(rec[0][0]['active_tasks']),active_tasks_post=np.asarray(rec[1][0]['active_tasks']),
                rollout_actions_prior=np.asarray(rec[0][0]['actions'],np.float32).reshape(-1,7),
                rollout_actions_post=np.asarray(rec[1][0]['actions'],np.float32).reshape(-1,7),
                endpoint_prior=rec[0][0]['endpoint'],endpoint_post=rec[1][0]['endpoint'],physics_start=snapshot.start,
                noise=original_noise.cpu().numpy())
            info=dict(pair=count,sequence=identity[0],subtask=identity[1],step=identity[2],task=task,
                file=file.name,accepted=accepted,returns_prior=ret0,returns_post=ret1,
                snapshot_hash=array_hash(snapshot.start),noise_hash=array_hash(original_noise.cpu().numpy()),
                observation_condition_hash=context_hash,
                cache_hash=array_hash(cached.cpu().numpy()),adapter='RSSM halves pooled to future512 and action16x7',
                observation='cached policy observation; identical for both branches, not reencoded current image',
                protocol_version=4,remaining_tasks=remaining_tasks,return_gamma=gamma,
                return_definition='sum gamma^t r_t across ALL remaining ordered tasks; r_t=1 on each first task completion; terminate on all-complete or first task timeout; no partial-progress reward',
                return_scope='full_remaining_task_chain' if requested_tail==-1 else 'explicit_truncated_chain_ablation',
                episode_deadline_steps=int(cfg.ep_len),executed_steps_before_pair=executed_steps,
                full_remaining_chain_return=requested_tail==-1,
                continuation_rule='same frozen model.step rule and fixed task order; reset policy at suffix boundary and task transitions; NEVER reset physical env between tasks; first timeout ends chain',
                continuation_steps=tail_steps,continuation_requested_steps=requested_tail,
                continuation_seed=tail_seed,min_return_gap=float(os.environ.get('STRICT_MIN_GAP','.01')),
                replay='ABBA identical-state identical-RNG determinism check; NOT independent trials',
                cache_alignment='both decoded chunks use same cached cut; expected branch is rebased, not assumed equal to original cache',
                suffix_start=start,horizon=len(a0),return_horizon=len(a0)+tail_steps,
                valid_suffix_steps=valid_suffix_steps,
                completed_tasks_prior=rec[0][0]['completed_tasks'],completed_tasks_post=rec[1][0]['completed_tasks'],
                all_remaining_succeeded_prior=rec[0][0]['all_tasks_completed'],all_remaining_succeeded_post=rec[1][0]['all_tasks_completed'],
                completion_events_prior=rec[0][0]['events'],completion_events_post=rec[1][0]['events'],
                termination_prior=rec[0][0]['termination'],termination_post=rec[1][0]['termination'])
            with rows.open('a') as f:f.write(json.dumps(info)+'\n');f.flush()
            count+=1
            print(f'[strict] pair={count}/{max_pairs} R_prior={ret0[0]:.4f} R_post={ret1[0]:.4f} accepted={accepted} full_chain={requested_tail==-1} remaining_tasks={len(remaining_tasks)}',flush=True)
        except Exception as e:
            (root/'fatal.json').write_text(json.dumps({'error':repr(e),'pair':count},indent=2))
            raise SystemExit(f'Strict collection aborted: {e}') from e
        finally:snapshot.restore();snapshot.close()
        if count>=max_pairs:raise CollectionComplete()
        return None
    def guarded_record(cfg,**kw):
        try:return record(cfg,**kw)
        except Exception as e:
            (root/'fatal.json').write_text(json.dumps({'error':repr(e),'attempted':count},indent=2))
            raise SystemExit(f'Strict collection failed closed: {e}') from e
    ns['maybe_save_sensitivity_teacher_row']=guarded_record
    main=ns['main']
    def frozen_main(cfg):
        cfg.log_wandb=False
        return main(cfg)
    ns['main']=frozen_main
