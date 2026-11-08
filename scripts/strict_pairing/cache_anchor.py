"""Capture the real cache-generating sampler call and anchor edits to that cache."""
import copy
import functools
import inspect
import torch
import torch.nn.functional as F
from strict_core import RNG


class CacheMismatch(ValueError):pass


def clone_tree(x):
    if torch.is_tensor(x):return x.detach().clone()
    if isinstance(x,dict):return {k:clone_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [clone_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(clone_tree(v) for v in x)
    return copy.deepcopy(x)


def versions(model):
    return tuple((name,int(p._version)) for name,p in model.named_parameters())


def install_capture(policy_class):
    """Only record eval_forward -> denoise_actions -> sample_loop generation.

    Capture sampler RNG AFTER initial noise was drawn. Teacher/reconstruction
    calls never overwrite the provenance of the live action cache.
    """
    if getattr(policy_class.sample_loop,'_strict_capture',False):return
    original=policy_class.sample_loop;sig=inspect.signature(original)
    @functools.wraps(original)
    def wrapped(self,*args,**kwargs):
        caller=inspect.currentframe().f_back
        actual=bool(caller and caller.f_code.co_name=='denoise_actions' and caller.f_back and caller.f_back.f_code.co_name=='eval_forward')
        record=None
        if actual:
            bound=sig.bind(self,*args,**kwargs);bound.apply_defaults()
            inputs={k:v for k,v in bound.arguments.items() if k!='self'}
            record=dict(inputs=clone_tree(inputs),rng=RNG(),parameter_versions=versions(self))
        output=original(self,*args,**kwargs)
        if record is not None:
            record['actions']=output.detach().clone();self._strict_cache_generation=record
        return output
    wrapped._strict_capture=True;wrapped._strict_original=original
    policy_class.sample_loop=wrapped
    original_step=policy_class.step
    @functools.wraps(original_step)
    def step(self,*args,**kwargs):
        new_chunk=int(getattr(self,'rollout_step_counter',0))==0
        out=original_step(self,*args,**kwargs)
        if new_chunk:
            record=getattr(self,'_strict_cache_generation',None)
            cache=getattr(self,'pred_action_seq',None)
            # Shadow eval_forward calls may generate actions without committing
            # them to the live cache. Bind provenance only at actual step commit.
            if record is not None and cache is not None and record['actions'].shape==cache.shape and torch.equal(record['actions'],cache):
                self._strict_live_cache_generation=record
            else:self._strict_live_cache_generation=None
        return out
    policy_class.step=step


def anchored_pair(model,innovation,start,belief_scale=1.,max_action_delta=.015,atol=1e-6):
    """Replay actual cached conditioning, then add Pool(innovation) to its goal.

    Baseline is the original cache, NOT a resampled expected-belief action.
    Clip the candidate correction BEFORE rollout and return the measured target.
    """
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Policy must be frozen/eval')
    if max_action_delta<0 or belief_scale<0 or atol<0:raise ValueError('Invalid candidate controls')
    generation=getattr(model,'_strict_live_cache_generation',None)
    if generation is None:raise CacheMismatch('Actual cache-generation record missing')
    if generation['parameter_versions']!=versions(model):raise CacheMismatch('Weights changed since cache generation')
    cached=model.pred_action_seq.detach().clone()
    if cached.ndim!=3 or cached.shape[0]!=1 or not 1<=start<cached.shape[1]:raise CacheMismatch('Unsupported cache/cut shape')
    original=getattr(type(model).sample_loop,'_strict_original',None)
    if original is None:raise CacheMismatch('Sampler capture hook missing')
    inputs=generation['inputs'];goal=inputs['goal'];delta=innovation.detach().float().reshape(1,1,-1)
    delta=F.adaptive_avg_pool1d(delta,goal.numel()).reshape_as(goal).to(goal)*belief_scale
    if not torch.isfinite(delta).all():raise ValueError('Nonfinite belief revision')
    live_rng=RNG();outputs=[]
    try:
        with torch.inference_mode():
            for changed in [False,True,False]:
                generation['rng'].restore();call=clone_tree(inputs)
                if changed:call['goal']=call['goal']+delta
                outputs.append(original(model,**call).detach().clone())
    finally:live_rng.restore()
    for output in [generation['actions'],outputs[0],outputs[2]]:
        if output.shape!=cached.shape or not torch.isfinite(output).all():raise CacheMismatch('Invalid cache reconstruction')
        if not torch.allclose(output,cached,atol=atol,rtol=0):raise CacheMismatch('Cannot reproduce current cache; it may have been patched')
    if not torch.equal(model.pred_action_seq,cached):raise CacheMismatch('Sampler modified live cache')
    if versions(model)!=generation['parameter_versions']:raise CacheMismatch('Sampler modified weights')
    baseline=cached[0,start:].clone();decoded=outputs[1][0,start:]
    if not torch.isfinite(decoded).all():raise ValueError('Nonfinite candidate')
    raw_delta=decoded-baseline
    applied=raw_delta.clamp(-max_action_delta,max_action_delta) if max_action_delta>0 else raw_delta
    candidate=baseline+applied
    return dict(baseline=baseline,candidate=candidate,raw_candidate=decoded,correction=candidate-baseline,
        cache_goal=goal.detach().reshape(-1),candidate_goal=(goal+delta).detach().reshape(-1),
        decoder_revision=delta.detach().reshape(-1),inputs=inputs,
        reconstruction_error=max(float((x-cached).abs().max()) for x in [outputs[0],outputs[2]]))
