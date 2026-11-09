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
from strict_core import BulletSnapshot,RNG,PolicyRuntime,array_hash,accept_binary_repair,rollout_remaining_chain
from cache_anchor import install_capture,anchored_pair,CacheMismatch
from outcome import classify_outcome,LABELS
from repair_metrics import episode_key


class CollectionComplete(BaseException):pass


def install(ns):
    from policy_models.VPP_policy import VPP_Policy
    install_capture(VPP_Policy)
    root=Path(os.environ['REENVISION_STRICT_RUN']);root.mkdir(parents=True,exist_ok=True)
    rows=root/'pairs.jsonl';count=0;seen=0;episode_counts={}
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
        epkey=episode_key(chain_frame.f_locals['initial_state'],sequence)
        if episode_counts.get(epkey,0)>=int(os.environ.get('STRICT_MAX_PAIRS_PER_EPISODE','4')):return None
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
        h0=payload['h_prior'].detach().float().reshape(1,-1)
        h1=payload['h_posterior'].detach().float().reshape(1,-1)
        if h0.shape!=h1.shape or h0.shape[1]%2:raise SystemExit('Invalid joint belief')
        if not torch.isfinite(h0).all() or not torch.isfinite(h1).all():return None
        if model.pred_action_seq is None:return None
        cached=model.pred_action_seq.detach().clone()
        start=int(model.rollout_step_counter)
        if start<1 or start>=cached.shape[1]:return None
        belief_scale=float(os.environ.get('STRICT_BELIEF_UPDATE_SCALE','1'))
        max_delta=float(os.environ.get('STRICT_MAX_ACTION_DELTA','.015'))
        try:
            pair=anchored_pair(model,h1-h0,start,belief_scale,max_delta)
        except CacheMismatch as e:
            print(f'[strict] rejected cache provenance: {e}',flush=True);return None
        sampler_inputs=pair['inputs'];original_noise=sampler_inputs['x_t'].detach().clone()
        ctx=sampler_inputs['state']
        tensors=sorted(ctx.items()) if isinstance(ctx,dict) else [('state',ctx)]
        context_hash=hashlib.sha256(''.join(array_hash(v.detach().cpu().numpy())
            for k,v in tensors if torch.is_tensor(v)).encode()).hexdigest()
        a0=pair['baseline'].cpu().numpy().copy();a1=pair['candidate'].cpu().numpy().copy()
        if not (np.isfinite(a0).all() and np.isfinite(a1).all()):return None
        if a0.shape!=a1.shape or len(a0)==0:return None
        requested_tail=int(os.environ.get('STRICT_CONTINUATION_STEPS','-1'))
        if requested_tail!=-1:raise SystemExit('Binary full-chain collection forbids truncated return horizons')
        executed_steps=int(kw['step'])+1
        if int(cfg.ep_len)-executed_steps < len(a0):
            print('[strict] skip: complete candidate suffix exceeds episode deadline',flush=True)
            return None
        first_budget=int(cfg.ep_len)-executed_steps
        chain_budget=first_budget+(len(remaining_tasks)-1)*int(cfg.ep_len)
        tail_steps=chain_budget-len(a0) if requested_tail==-1 else min(requested_tail,chain_budget-len(a0))
        gamma=float(os.environ.get('STRICT_RETURN_GAMMA','1'))
        if gamma!=1.:raise SystemExit('Binary success pairs cannot be ranked by speed/discount')
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
            accepted=accept_binary_repair(ret0,ret1)
            outcome=LABELS[classify_outcome(ret0,ret1)]
            identity=(int(kw['sequence_index']),int(kw['subtask_index']),int(kw['step']))
            # All attempted pairs retained, including ties and regressions.
            file=root/f'pair_{count:06d}.npz'
            capture_arrays={}
            # Preserve exactly the existing inference inputs for drop-in TCN training.
            if live.get('base_summary_exec') is not None:
                capture_arrays['legacy_runtime_belief_before']=np.asarray(live['base_summary_exec'],np.float32).reshape(-1)
                capture_arrays['legacy_runtime_belief_after']=h1.cpu().numpy().reshape(-1)
                innovation=payload.get('innovation')
                if torch.is_tensor(innovation):capture_arrays['legacy_runtime_innovation']=innovation.detach().float().cpu().numpy().reshape(-1)
            for name in ['sigmas','latent_motion_tokens_up','latent_plan','goal']:
                value=sampler_inputs.get(name)
                if torch.is_tensor(value):capture_arrays['cache_input_'+name]=value.detach().cpu().numpy()
            for key,value in tensors:
                if torch.is_tensor(value):capture_arrays['cache_observation_'+str(key)]=value.detach().cpu().numpy()
            for key,value in sampler_inputs.get('extra_args',{}).items():
                if torch.is_tensor(value):capture_arrays['cache_extra_'+str(key)]=value.detach().cpu().numpy()
            generation=model._strict_live_cache_generation
            capture_arrays['sampler_rng_cpu']=generation['rng'].cpu.cpu().numpy()
            for device,value in enumerate(generation['rng'].cuda or []):capture_arrays['sampler_rng_cuda_'+str(device)]=value.cpu().numpy()
            np.savez_compressed(file,state=np.asarray(kw['state_after'],np.float32),
                prior=pair['cache_goal'].cpu().numpy(),posterior=pair['candidate_goal'].cpu().numpy(),
                rssm_prior=h0.cpu().numpy()[0],rssm_posterior=h1.cpu().numpy()[0],
                action_prior=a0,action_post=a1,cached_suffix=cached[0,start:].cpu().numpy(),
                measured_correction=a1-a0,raw_candidate=pair['raw_candidate'].cpu().numpy(),
                returns_prior=np.asarray(ret0),returns_post=np.asarray(ret1),
                rewards_prior=np.asarray(rec[0][0]['rewards']),rewards_post=np.asarray(rec[1][0]['rewards']),
                task_completion_rewards_prior=np.asarray(rec[0][0]['task_completion_rewards']),
                task_completion_rewards_post=np.asarray(rec[1][0]['task_completion_rewards']),
                active_tasks_prior=np.asarray(rec[0][0]['active_tasks']),active_tasks_post=np.asarray(rec[1][0]['active_tasks']),
                rollout_actions_prior=np.asarray(rec[0][0]['actions'],np.float32).reshape(-1,7),
                rollout_actions_post=np.asarray(rec[1][0]['actions'],np.float32).reshape(-1,7),
                endpoint_prior=rec[0][0]['endpoint'],endpoint_post=rec[1][0]['endpoint'],physics_start=snapshot.start,
                noise=original_noise.cpu().numpy(),**capture_arrays)
            info=dict(pair=count,sequence=identity[0],subtask=identity[1],step=identity[2],task=task,
                episode_key=epkey,
                episode_tasks=sequence,candidate_source='belief_counterfactual',tcn_triggered=False,
                file=file.name,accepted=accepted,outcome=outcome,training_eligible=True,returns_prior=ret0,returns_post=ret1,
                completion_steps_prior=len(rec[0][0]['actions']),completion_steps_post=len(rec[1][0]['actions']),
                correction_l2=float(np.linalg.norm(a1-a0)),correction_max=float(np.abs(a1-a0).max()),
                snapshot_hash=array_hash(snapshot.start),noise_hash=array_hash(original_noise.cpu().numpy()),
                observation_condition_hash=context_hash,
                cache_hash=array_hash(cached.cpu().numpy()),adapter='actual_cached_decoder_goal + scale*Pool(RSSM posterior-prior); all other sampler conditions unchanged',
                baseline_original_cache=True,cache_reconstruction_verified=True,
                branch_labels={'prior':'original_cached_suffix','post':'final_measured_cache_anchored_candidate'},
                sampler_type=str(sampler_inputs['sampler_type']),
                cache_reconstruction_max_error=pair['reconstruction_error'],cache_reconstruction_atol=1e-6,
                belief_update_scale=belief_scale,max_action_delta=max_delta,candidate_processed_before_rollout=True,
                observation='cached policy observation; identical for both branches, not reencoded current image',
                protocol_version=5,remaining_tasks=remaining_tasks,return_gamma=1.,return_type='binary_full_remaining_chain_success',
                return_definition='1 iff ALL remaining ordered tasks complete; otherwise 0. Completion count and speed never rank pairs. Both success and both failure are ties.',
                return_scope='full_remaining_task_chain' if requested_tail==-1 else 'explicit_truncated_chain_ablation',
                episode_deadline_steps=int(cfg.ep_len),executed_steps_before_pair=executed_steps,
                full_remaining_chain_return=requested_tail==-1,
                continuation_rule='same frozen model.step rule and fixed task order; reset policy at suffix boundary and task transitions; NEVER reset physical env between tasks; first timeout ends chain',
                continuation_steps=tail_steps,continuation_requested_steps=requested_tail,
                continuation_seed=tail_seed,min_return_gap=float(os.environ.get('STRICT_MIN_GAP','.01')),
                replay='ABBA identical-state identical-RNG determinism check; NOT independent trials',
                cache_alignment='baseline is ORIGINAL cached suffix, verified against its actual generation inputs/noise/weights; target is final measured candidate minus original cache',
                suffix_start=start,horizon=len(a0),return_horizon=len(a0)+tail_steps,
                valid_suffix_steps=valid_suffix_steps,
                completed_tasks_prior=rec[0][0]['completed_tasks'],completed_tasks_post=rec[1][0]['completed_tasks'],
                all_remaining_succeeded_prior=rec[0][0]['all_tasks_completed'],all_remaining_succeeded_post=rec[1][0]['all_tasks_completed'],
                completion_events_prior=rec[0][0]['events'],completion_events_post=rec[1][0]['events'],
                termination_prior=rec[0][0]['termination'],termination_post=rec[1][0]['termination'])
            with rows.open('a') as f:f.write(json.dumps(info)+'\n');f.flush()
            count+=1
            episode_counts[epkey]=episode_counts.get(epkey,0)+1
            print(f'[strict] pair={count}/{max_pairs} G_cache={ret0[0]:.0f} G_candidate={ret1[0]:.0f} accepted={accepted} full_chain=True remaining_tasks={len(remaining_tasks)}',flush=True)
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
    log_dir=root/'original_logs';log_dir.mkdir(exist_ok=True)
    if 'get_log_dir' in ns:ns['get_log_dir']=lambda *a,**kw:log_dir
    main=ns['main']
    def frozen_main(cfg):
        cfg.log_wandb=False
        return main(cfg)
    ns['main']=frozen_main
