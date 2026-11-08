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
from strict_core import BulletSnapshot,RNG,PolicyRuntime,array_hash,accept_pair,replay_pair,continuation_budget


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
        tail_steps=continuation_budget(int(cfg.ep_len),executed_steps,len(a0),requested_tail)
        # No action clipping: the target must match the actions actually measured.
        snapshot=BulletSnapshot(raw)
        try:
            def step(a):return env.step(torch.as_tensor(a,device=model.device))[3]
            runtime=PolicyRuntime(model);tail_logs=[]
            tail_seed=int(os.environ.get('STRICT_CONTINUATION_SEED','1701'))
            def reset_branch(side):
                runtime.restore();tail_logs.append([])
            def continuation():
                # Both branches discard their suffix cache at the SAME boundary
                # and use the SAME frozen base-policy rule on their own new obs.
                model.reset();model.future_feature_mode='gfdm'
                model.override_future_feature=None;model.override_action_intent=None
                model.override_latent_motion_tokens_up=None
                for j in range(tail_steps):
                    # Common random numbers indexed by relative control step.
                    # This is not independent multi-seed return estimation.
                    seed=tail_seed+j
                    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
                    obs=env.get_obs()
                    with torch.inference_mode():
                        act=model.step(obs,copy.deepcopy(live['goal'])).detach().clone()
                    arr=act.float().cpu().numpy().reshape(-1).copy()
                    if arr.shape!=(7,) or not np.isfinite(arr).all():raise ValueError('Invalid continuation action')
                    tail_logs[-1].append(arr)
                    yield step(arr)
            rec=replay_pair(snapshot,[a0,a1],step,done,continuation=continuation,
                            reset=reset_branch,finalize=runtime.restore)
            for i,j in [(0,3),(1,2)]:
                if np.asarray(tail_logs[i]).shape!=np.asarray(tail_logs[j]).shape or not np.allclose(tail_logs[i],tail_logs[j],atol=1e-7,rtol=0):
                    raise ValueError('Identical-branch continuation action replay mismatch')
            ret0=[r[0] for r in rec[0]];ret1=[r[0] for r in rec[1]]
            valid_suffix_steps=min(len(a0),len(rec[0][0][2]),len(rec[1][0][2]))
            accepted=accept_pair(ret0,ret1,float(os.environ.get('STRICT_MIN_GAP','.01')))
            identity=(int(kw['sequence_index']),int(kw['subtask_index']),int(kw['step']))
            # All attempted pairs retained, including ties and regressions.
            file=root/f'pair_{count:06d}.npz'
            np.savez_compressed(file,state=np.asarray(kw['state_after'],np.float32),
                prior=h0.cpu().numpy()[0],posterior=h1.cpu().numpy()[0],
                action_prior=a0,action_post=a1,cached_suffix=cached[0,start:].cpu().numpy(),
                returns_prior=np.asarray(ret0),returns_post=np.asarray(ret1),
                hits_prior=np.asarray(rec[0][0][2]),hits_post=np.asarray(rec[1][0][2]),
                continuation_prior=np.asarray(tail_logs[0],np.float32).reshape(-1,7),
                continuation_post=np.asarray(tail_logs[1],np.float32).reshape(-1,7),
                endpoint_prior=rec[0][0][1],endpoint_post=rec[1][0][1],physics_start=snapshot.start,
                noise=original_noise.cpu().numpy())
            info=dict(pair=count,sequence=identity[0],subtask=identity[1],step=identity[2],task=task,
                file=file.name,accepted=accepted,returns_prior=ret0,returns_post=ret1,
                snapshot_hash=array_hash(snapshot.start),noise_hash=array_hash(original_noise.cpu().numpy()),
                observation_condition_hash=context_hash,
                cache_hash=array_hash(cached.cpu().numpy()),adapter='RSSM halves pooled to future512 and action16x7',
                observation='cached policy observation; identical for both branches, not reencoded current image',
                protocol_version=3,
                return_definition='sum gamma^t r_t until current subtask success or episode deadline; r_t=1 only at first oracle success, otherwise 0; gamma=.99; no partial-progress reward',
                return_scope='full_current_subtask_episode' if requested_tail==-1 else 'explicit_truncated_ablation',
                episode_deadline_steps=int(cfg.ep_len),executed_steps_before_pair=executed_steps,
                full_episode_return=requested_tail==-1,
                continuation_rule='reset at suffix boundary, frozen model.step on branch obs, GFDM mode, no belief controller or overrides',
                continuation_steps=tail_steps,continuation_requested_steps=requested_tail,
                continuation_seed=tail_seed,min_return_gap=float(os.environ.get('STRICT_MIN_GAP','.01')),
                replay='ABBA identical-state identical-RNG determinism check; NOT independent trials',
                cache_alignment='both decoded chunks use same cached cut; expected branch is rebased, not assumed equal to original cache',
                suffix_start=start,horizon=len(a0),return_horizon=len(a0)+tail_steps,
                valid_suffix_steps=valid_suffix_steps,
                prior_success=any(rec[0][0][2]),post_success=any(rec[1][0][2]))
            with rows.open('a') as f:f.write(json.dumps(info)+'\n');f.flush()
            count+=1
            print(f'[strict] pair={count}/{max_pairs} R_prior={ret0[0]:.4f} R_post={ret1[0]:.4f} accepted={accepted} full_episode={requested_tail==-1} return_horizon={len(a0)+tail_steps}',flush=True)
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
