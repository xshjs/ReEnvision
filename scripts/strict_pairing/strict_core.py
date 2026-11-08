"""Strict branch replay primitives. No proxy returns or return-label swapping."""
import copy
import enum
import hashlib
import random
import types
import numpy as np


def array_hash(x):
    a=np.ascontiguousarray(x)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()


def first_success_return(hits, gamma=.99):
    if not 0 < gamma <= 1: raise ValueError('invalid discount')
    return next((float(gamma**i) for i,h in enumerate(hits) if h),0.)


def continuation_budget(episode_steps,executed_steps,suffix_steps,requested=-1):
    """-1 evaluates through the episode deadline; >=0 is an explicit ablation.

    Both candidate suffixes must fit the same remaining episode budget.
    """
    if episode_steps<=0 or executed_steps<0 or suffix_steps<1 or requested < -1:
        raise ValueError('Invalid rollout budget')
    remaining=episode_steps-executed_steps-suffix_steps
    if remaining<0:raise ValueError('Candidate suffix exceeds the episode deadline')
    return remaining if requested==-1 else min(requested,remaining)


def discounted_return(rewards,gamma=1.):
    if not 0<gamma<=1:raise ValueError('Invalid discount')
    r=np.asarray(rewards,dtype=float)
    if not np.isfinite(r).all():raise ValueError('Nonfinite reward')
    return float(np.sum(r*gamma**np.arange(len(r))))


def rollout_remaining_chain(suffix,task_count,first_budget,task_budget,step,begin_task,next_action,success,
                            gamma=1.,max_steps=None):
    """CALVIN chain semantics: advance without resetting env; stop on first timeout.

    Return is 1 iff ALL remaining tasks complete, otherwise 0. Completion counts
    are diagnostics only, never a preference signal. The first task
    gets only its unspent budget; future tasks each get the original task budget.
    """
    if task_count<1 or first_budget<len(suffix) or task_budget<1 or gamma!=1.:raise ValueError('Invalid binary chain budget/discount')
    budget=first_budget+(task_count-1)*task_budget
    if max_steps is not None:budget=min(budget,max_steps)
    rewards=[];events=[];active=[];actions=[];suffix_used=0;reason='all_tasks_completed'
    for task in range(task_count):
        begin_task(task);completed=False
        for local in range(first_budget if task==0 else task_budget):
            if len(rewards)>=budget:
                reason='explicit_truncation';break
            if task==0 and local<len(suffix):
                action=suffix[local].copy();suffix_used+=1
            else:action=np.asarray(next_action(task,local)).copy()
            info=step(action);hit=bool(success(task,info))
            actions.append(action);rewards.append(float(hit));active.append(task)
            if hit:
                events.append({'task_index':task,'step':len(rewards)-1});completed=True;break
        if not completed:
            if reason!='explicit_truncation':reason='task_timeout'
            break
    complete=len(events)==task_count
    binary_rewards=[0.]*len(rewards)
    if complete and binary_rewards:binary_rewards[-1]=1.
    return dict(return_value=float(complete),rewards=binary_rewards,task_completion_rewards=rewards,events=events,
        active_tasks=active,actions=actions,suffix_executed=suffix_used,termination=reason,
        completed_tasks=len(events),all_tasks_completed=len(events)==task_count,budget=budget)


def accept_pair(before,after,min_gap=.01):
    a=np.asarray(before,float);b=np.asarray(after,float)
    if a.shape!=b.shape or a.size<2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return False
    return bool(np.all(b>=a) and (b-a).mean()>=min_gap)


def accept_binary_repair(before,after):
    a=np.asarray(before,float);b=np.asarray(after,float)
    return bool(a.ndim==b.ndim==1 and a.size>=2 and a.shape==b.shape and np.all(a==0.) and np.all(b==1.))


class RNG:
    def __init__(self):
        import torch
        self.py=random.getstate();self.np=np.random.get_state()
        self.cpu=torch.get_rng_state()
        self.cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    def restore(self):
        import torch
        random.setstate(self.py);np.random.set_state(self.np)
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:torch.cuda.set_rng_state_all(self.cuda)


class PolicyRuntime:
    """Preserve root-level VPP execution caches without copying frozen weights.

    Registered neural modules/parameters/buffers stay shared and must be eval/frozen.
    This is a VPP runtime adapter, not a generic recurrent-network checkpoint.
    """
    def __init__(self,model):
        import torch
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Continuation requires a frozen eval policy')
        self.model=model;self.static=set(vars(torch.nn.Module()))
        self.memo={id(m):m for m in model.modules()}
        self.data=copy.deepcopy({k:v for k,v in vars(model).items() if k not in self.static},self.memo.copy())
    def restore(self):
        current=vars(self.model)
        for k in list(current):
            if k not in self.static:del current[k]
        current.update(copy.deepcopy(self.data,self.memo.copy()))


class BulletSnapshot:
    """Physics plus Python-side CALVIN object state, including IK and switches.

    Unknown external objects are pinned (resources/config/transforms); mutable
    calvin_env objects and their containers are copied, preserving aliasing.
    Intended for non-VR single-client CALVIN only, never model snapshots.
    """
    def __init__(self,raw):
        if raw.use_vr:raise ValueError('VR is not reproducible')
        self.raw=raw;self.p=raw.p;self.cid=raw.cid
        nodes=[];seen=set();pinned={};stack=[raw]
        while stack:
            obj=stack.pop()
            if id(obj) in seen:continue
            seen.add(id(obj))
            if isinstance(obj,dict):stack.extend(obj.values());continue
            if isinstance(obj,(list,tuple)):stack.extend(obj);continue
            if isinstance(obj,(str,bytes,int,float,bool,type(None),np.ndarray,np.generic,enum.Enum)):
                continue
            if type(obj).__module__.startswith('calvin_env') and hasattr(obj,'__dict__'):
                nodes.append(obj);stack.extend(vars(obj).values())
            elif not isinstance(obj,(np.random.Generator,np.random.RandomState)):
                pinned[id(obj)]=obj
        self.nodes=nodes;self.pinned=pinned
        memo={**pinned,**{id(o):o for o in nodes}}
        self.states=copy.deepcopy([dict(vars(o)) for o in nodes],memo)
        # Bullet restoreState does not reliably restore changeVisualShape colors.
        # CALVIN lights change these in Python during scene.step().
        self.colors={}
        for i in range(self.p.getNumBodies(physicsClientId=self.cid)):
            uid=self.p.getBodyUniqueId(i,physicsClientId=self.cid)
            for shape in self.p.getVisualShapeData(uid,physicsClientId=self.cid) or []:
                key=(uid,int(shape[1]));rgba=tuple(shape[7])
                if key in self.colors and self.colors[key]!=rgba:
                    raise ValueError('Multiple differently colored visual shapes on one link unsupported')
                self.colors[key]=rgba
        self.rng=RNG();self.physics=self.p.saveState(physicsClientId=self.cid)
        self.start=self.fingerprint()
    def fingerprint(self):
        out=[]
        for i in range(self.p.getNumBodies(physicsClientId=self.cid)):
            uid=self.p.getBodyUniqueId(i,physicsClientId=self.cid)
            pos,orn=self.p.getBasePositionAndOrientation(uid,physicsClientId=self.cid)
            lin,ang=self.p.getBaseVelocity(uid,physicsClientId=self.cid)
            out.extend([*pos,*orn,*lin,*ang])
            for j in range(self.p.getNumJoints(uid,physicsClientId=self.cid)):
                q=self.p.getJointState(uid,j,physicsClientId=self.cid)
                out.extend([q[0],q[1]])
        return np.asarray(out,dtype=np.float64)
    def restore(self):
        self.p.restoreState(stateId=self.physics,physicsClientId=self.cid)
        for (uid,link),rgba in self.colors.items():
            self.p.changeVisualShape(uid,link,rgbaColor=rgba,physicsClientId=self.cid)
        memo={**self.pinned,**{id(o):o for o in self.nodes}}
        states=copy.deepcopy(self.states,memo)
        for o,s in zip(self.nodes,states):vars(o).clear();vars(o).update(s)
        self.rng.restore()
        if not np.allclose(self.start,self.fingerprint(),atol=1e-9,rtol=0):
            raise RuntimeError('Physics snapshot restore mismatch')
    def close(self):self.p.removeState(stateUniqueId=self.physics,physicsClientId=self.cid)


def replay_pair(snapshot,actions,step,success,gamma=.99,continuation=None,reset=None,finalize=None):
    """ABBA replay, optionally followed by a common closed-loop policy.

    continuation yields subsequent infos, using its own fixed step budget.
    Stop at first continuation success (absorbing terminal); never count success twice.
    Repetitions test determinism, not independent seeds or statistical robustness.
    """
    records={0:[],1:[]}
    try:
        for side in [0,1,1,0]:
            snapshot.restore();hits=[]
            if reset is not None:reset(side)
            for a in actions[side]:
                info=step(a.copy());hits.append(bool(success(info)))
                if hits[-1]:break
            if continuation is not None and not any(hits):
                for info in continuation():
                    hit=bool(success(info));hits.append(hit)
                    if hit:break
            records[side].append((first_success_return(hits,gamma),snapshot.fingerprint(),hits))
        for side in [0,1]:
            a,b=records[side]
            if a[2]!=b[2] or not np.allclose(a[1],b[1],atol=1e-7,rtol=0):
                raise RuntimeError('Identical-action replay is nondeterministic')
        return records
    finally:
        try:
            if finalize is not None:finalize()
        finally:snapshot.restore()
