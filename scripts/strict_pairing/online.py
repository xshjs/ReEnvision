"""Integrate outcome gate into original CALVIN evaluator without editing source."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from paths import defi_root,default_checkpoint

HERE=Path(__file__).resolve().parent
DEFI=defi_root()
SOURCE=DEFI/'Step3_DeFI/policy_evaluation/calvin_evaluate_with_memory_reflection.py'


def instrument(source):
    tree=ast.parse(source);ticks=0;blocked=0
    class Replace(ast.NodeTransformer):
        active=False
        def visit_FunctionDef(self,node):
            if node.name!='rollout_with_lang_text':return node
            self.active=True;node=self.generic_visit(node);self.active=False
            return node
        def visit_If(self,node):
            nonlocal blocked
            if self.active:
                expr=ast.unparse(node.test)
                if "chunk_decision['decision']" in expr and ('repair_suffix' in expr or 'replan' in expr):
                    node.test=ast.Constant(False);blocked+=1
            return self.generic_visit(node)
        def visit_Assign(self,node):
            nonlocal ticks
            if self.active and any(isinstance(t,ast.Name) and t.id=='state_after_step' for t in node.targets):
                # Prevent an old payload leaking across steps where the RSSM update was skipped.
                return [node,*ast.parse('joint_update_payload = None').body]
            if self.active and any(isinstance(t,ast.Name) and t.id=='current_task_info' for t in node.targets):
                ticks+=1
                return [node,*ast.parse('_outcome_online_tick(locals())').body]
            return node
    Replace().visit(tree)
    if ticks!=1 or blocked<4:raise ValueError(f'Unsupported evaluator layout: ticks={ticks}, legacy branches={blocked}')
    entry=[i for i,n in enumerate(tree.body) if isinstance(n,ast.If) and '__name__' in ast.unparse(n.test)]
    if len(entry)!=1:raise ValueError('Unsupported evaluator entrypoint')
    tree.body[entry[0]:entry[0]]=ast.parse('_outcome_install(globals())').body
    ast.fix_missing_locations(tree)
    return tree


def dispatch(args):
    if not args or Path(args[0]).name!=SOURCE.name:raise SystemExit('Unexpected evaluator command')
    sys.argv=[str(SOURCE),*args[1:]]
    sys.path.insert(0,str(SOURCE.parent));sys.path.insert(0,str(DEFI/'Step3_DeFI'))
    from online_hook import install
    root=Path(os.environ['REENVISION_ONLINE_RUN'])
    holder={}
    def attach(ns):
        holder['gate']=install(ns,os.environ['REENVISION_OUTCOME_CKPT'],root/'gate.jsonl',
            os.environ.get('REENVISION_OBSERVE_ONLY')=='1',os.environ.get('REENVISION_GATE_DEVICE','cpu'),
            float(os.environ.get('STRICT_BELIEF_UPDATE_SCALE','1')))
    ns={'__name__':'__main__','__file__':str(SOURCE),'__package__':None,'_outcome_install':attach}
    exec(compile(instrument(SOURCE.read_text()),str(SOURCE),'exec'),ns)
    g=holder['gate']
    (root/'online_summary.json').write_text(json.dumps(dict(gate_calls=g.hits,repairs=g.applied,replans=g.replans,
        observe_only=g.observe_only,robot_success_evaluated=False),indent=2))
    if not g.hits:raise SystemExit('No eligible belief updates reached online gate; inspect log before claiming integration success')


def main():
    raise SystemExit('Outcome online gate disabled by request. Use the original inference launcher and directional_sensitivity_tcn.pt from train_outcome.py; no four-class online decisions.')
    if len(sys.argv)>1 and sys.argv[1]=='dispatch':return dispatch(sys.argv[2:])
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outcome-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--observe-only',action='store_true')
    p.add_argument('--checkpoint',type=Path,default=default_checkpoint(DEFI))
    p.add_argument('--belief-checkpoint',type=Path,default=DEFI/'outputs/full_rssm_plan_compat_v2_success_balanced_650x2/rssm_plan_compatibility.pt')
    p.add_argument('--action-generator-checkpoint',type=Path,default=Path('/mnt/workspace/manipulation/BridgeVLA/eval/joint_action_generator_mlp_d300_abc700_abc1500_v2/joint_action_generator_mlp.pt'))
    p.add_argument('--lora-checkpoint',type=Path,default=Path('/mnt/workspace/manipulation/BridgeVLA/eval/policy_lora_action_intent_v2/best.pt'))
    p.add_argument('--sequences',type=int,default=5);p.add_argument('--episode-steps',type=int,default=360)
    p.add_argument('--gate-device',default='cpu');p.add_argument('--belief-update-scale',type=float,default=1.)
    a=p.parse_args()
    if a.sequences<1 or a.episode_steps<1 or not 0<=a.belief_update_scale<float('inf'):p.error('Invalid limits')
    # Validate checkpoints before starting a costly GPU process. No random new heads.
    for k in ['checkpoint','belief_checkpoint','action_generator_checkpoint','lora_checkpoint','outcome_checkpoint']:
        setattr(a,k,getattr(a,k).resolve(strict=True))
    if a.output.exists():raise FileExistsError(a.output)
    import torch
    ck=torch.load(a.outcome_checkpoint,map_location='cpu',weights_only=False)
    if not {'config','normalization','state_dict','labels'}.issubset(ck):
        raise SystemExit('Legacy checkpoint is not a four-outcome TCN; retain it as base/belief model and train a new outcome checkpoint')
    if not a.observe_only and not ck.get('thresholds_validated',False):
        raise SystemExit('Gate is unvalidated; use --observe-only until independent matched rollout calibration')
    if not torch.cuda.is_available():raise SystemExit('Base policy requires a GPU terminal')
    a.output.mkdir(parents=True)
    env=os.environ.copy();env.update(REENVISION_ONLINE_RUN=str(a.output.resolve()),
        REENVISION_OUTCOME_CKPT=str(a.outcome_checkpoint),REENVISION_OBSERVE_ONLY=str(int(a.observe_only)),
        REENVISION_GATE_DEVICE=a.gate_device,STRICT_BELIEF_UPDATE_SCALE=str(a.belief_update_scale),
        STRICT_PYTHON=sys.executable,PYTHON_BIN=str(HERE/'online_dispatch.sh'),
        ACTION_MODEL_FOLDER=str(a.checkpoint.parent.parent),CHECKPOINT_PATH=str(a.checkpoint),
        JOINT_BELIEF_TRANSITION_CKPT=str(a.belief_checkpoint),JOINT_PAIR_ACTION_GENERATOR_CKPT=str(a.action_generator_checkpoint),
        NUM_SEQUENCES=str(a.sequences),EP_LEN=str(a.episode_steps),DYNAMIC_COUPLING_FIRST_PASS='1',
        DISABLE_RETRY_AFTER_FIRST_PASS='1',DYNAMIC_COUPLING_DELTA_ACTION_REPAIR_MIX='0',
        JOINT_BELIEF_CLEAN_H_UPDATE_SCALE='0.8',JOINT_BELIEF_CLEAN_CONTROLLER='1',
        LINEAR_SUFFIX_SENSITIVITY='0',DIRECTIONAL_SENSITIVITY_TCN='0',DIRECT_ACTION_CORRECTION_TCN='0',
        COLLECT_SENSITIVITY_TEACHER='0',DEFI_SUFFIX_SENSITIVITY_CACHE='0',SUFFIX_MASK_ABLATION_MODE='off',
        WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    (a.output/'manifest.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2))
    with (a.output/'online.log').open('w') as log:
        print('Online log:',a.output/'online.log',flush=True)
        result=subprocess.run(['bash',str(DEFI/'Step3_DeFI/scripts/run_calvin_joint_belief_suffix.sh'),
            '--policy_action_intent_lora_ckpt',str(a.lora_checkpoint),'--joint_pair_direct_generate_only',
            '--joint_pair_no_memory','--joint_belief_transition_direct_suffix','--joint_belief_transition_sparse_suffix'],
            env=env,stdout=log,stderr=subprocess.STDOUT)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
