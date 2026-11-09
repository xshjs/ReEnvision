"""Isolated strict-return CALVIN collection launcher."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter
from paths import defi_root,default_checkpoint

HERE=Path(__file__).resolve().parent
DEFI=defi_root()
SOURCE=DEFI/'Step3_DeFI/policy_evaluation/calvin_evaluate_with_memory_reflection.py'

def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def dispatch(args):
    # Existing launch script supplies evaluator path as its first argument.
    if not args or Path(args[0]).name!=SOURCE.name:raise SystemExit('Unexpected evaluator command')
    sys.argv=[str(SOURCE),*args[1:]]
    # Match Python's sys.path[0] when the original evaluator is run directly.
    # Several legacy sibling modules use unqualified imports.
    sys.path.insert(0,str(SOURCE.parent))
    sys.path.insert(0,str(DEFI/'Step3_DeFI'))
    from future_compat import install as install_import_compat
    install_import_compat()
    tree=ast.parse(SOURCE.read_text(),filename=str(SOURCE))
    injection=ast.parse('from strict_hook import install\ninstall(globals())').body
    indices=[i for i,n in enumerate(tree.body) if isinstance(n,ast.If) and '__name__' in ast.unparse(n.test)]
    if len(indices)!=1:raise SystemExit('Unsupported evaluator entrypoint')
    i=indices[0];tree.body[i:i]=injection;ast.fix_missing_locations(tree)
    from strict_hook import CollectionComplete
    ns={'__name__':'__main__','__file__':str(SOURCE),'__package__':None}
    try:exec(compile(tree,str(SOURCE),'exec'),ns)
    except CollectionComplete:print('[strict] requested collection count reached',flush=True)
    root=Path(os.environ['REENVISION_STRICT_RUN'])
    rows=[json.loads(s) for s in (root/'pairs.jsonl').read_text().splitlines()] if (root/'pairs.jsonl').exists() else []
    (root/'collection_summary.json').write_text(json.dumps(dict(attempted=len(rows),accepted=sum(r['accepted'] for r in rows),
        simulated=True,training_started=False,protocol_version=5,
        baseline='original_live_cache_with_verified_generation_provenance',return_type='binary_full_remaining_chain_success',
        return_scope='full_remaining_task_chain',
        both_success=sum(r.get('all_remaining_succeeded_prior',False) and r.get('all_remaining_succeeded_post',False) for r in rows),
        both_failure=sum(not r.get('all_remaining_succeeded_prior',False) and not r.get('all_remaining_succeeded_post',False) for r in rows),
        both_zero_completed=sum(r.get('completed_tasks_prior',0)==0 and r.get('completed_tasks_post',0)==0 for r in rows),
        class_counts=dict(Counter(r['outcome'] for r in rows)),
        independent_episodes=len({r['episode_key'] for r in rows}),
        evaluator_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest()),indent=2))
    if not rows:raise SystemExit('No strict pairs collected; inspect evaluator log, do not train')


def collect(args):
    if args.output.exists():raise SystemExit('Use a fresh output directory (no overwrite/resume mixing)')
    ckpt=args.checkpoint.resolve(strict=True)
    sequences=args.eval_sequences.resolve(strict=True)
    episode_list=json.loads(sequences.read_text())
    if len(episode_list)<args.sequences:raise ValueError('Requested more episodes than sequence file contains')
    components={k:getattr(args,k).resolve(strict=True) for k in ['belief_checkpoint','action_generator_checkpoint','lora_checkpoint']}
    if args.dry_run:
        print(json.dumps(dict(status='preflight_only',checkpoint=str(ckpt),eval_sequences=str(sequences),
            sequence_count=args.sequences,max_pairs=args.max_pairs,max_pairs_per_episode=args.max_pairs_per_episode,
            components={k:str(v) for k,v in components.items()},gpu_collection_started=False),indent=2));return
    import torch
    if not torch.cuda.is_available():raise SystemExit('GPU not available in this process; run this command in GPU terminal')
    args.output.mkdir(parents=True)
    env=os.environ.copy();env.update(REENVISION_STRICT_RUN=str(args.output.resolve()),
        STRICT_MAX_PAIRS=str(args.max_pairs),STRICT_EVERY=str(args.every),STRICT_MIN_GAP=str(args.min_gap),
        STRICT_MAX_PAIRS_PER_EPISODE=str(args.max_pairs_per_episode),
        STRICT_CONTINUATION_STEPS=str(args.continuation_steps),STRICT_CONTINUATION_SEED=str(args.continuation_seed),
        STRICT_RETURN_GAMMA=str(args.return_gamma),
        STRICT_BELIEF_UPDATE_SCALE=str(args.belief_update_scale),STRICT_MAX_ACTION_DELTA=str(args.max_action_delta),
        STRICT_PYTHON=sys.executable,
        PYTHON_BIN=str(HERE/'dispatch.sh'),ACTION_MODEL_FOLDER=str(ckpt.parent.parent),CHECKPOINT_PATH=str(ckpt),
        EVAL_SEQUENCES_PATH=str(sequences),JOINT_BELIEF_TRANSITION_CKPT=str(components['belief_checkpoint']),
        JOINT_PAIR_ACTION_GENERATOR_CKPT=str(components['action_generator_checkpoint']),
        NUM_SEQUENCES=str(args.sequences),EP_LEN=str(args.episode_steps),COLLECT_SENSITIVITY_TEACHER='1',
        SENSITIVITY_TEACHER_MODE='paired_counterfactual',SENSITIVITY_TEACHER_STEP_SIZE='1',
        SENSITIVITY_TEACHER_MAX_ABS_DELTA='0',DYNAMIC_COUPLING_FIRST_PASS='1',
        DISABLE_RETRY_AFTER_FIRST_PASS='1',DYNAMIC_COUPLING_DELTA_ACTION_REPAIR_MIX='0',
        JOINT_BELIEF_CLEAN_H_UPDATE_SCALE='0.8',LINEAR_SUFFIX_SENSITIVITY='0',
        DIRECTIONAL_SENSITIVITY_TCN='0',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        DEFI_SUFFIX_SENSITIVITY_CACHE='0',DIRECT_ACTION_CORRECTION_TCN='0',SUFFIX_MASK_ABLATION_MODE='off',
        XDG_CACHE_HOME='/tmp/reenvision-strict-cache',MPLCONFIGDIR='/tmp/reenvision-strict-mpl',
        HF_HOME='/tmp/reenvision-strict-hf',TRANSFORMERS_CACHE='/tmp/reenvision-strict-hf/hub',USE_TF='0')
    manifest=dict(checkpoint=str(ckpt),checkpoint_sha256=file_hash(ckpt),
        evaluator=str(SOURCE),evaluator_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        eval_sequences=str(sequences),eval_sequences_sha256=file_hash(sequences),purpose='four_outcome_training_collection_not_deployment_test',
        components={k:dict(path=str(v),sha256=file_hash(v)) for k,v in components.items()},
        cli={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with (args.output/'collect.log').open('w') as f:
        print('Collection log:',args.output/'collect.log',flush=True)
        p=subprocess.run(['bash',str(DEFI/'Step3_DeFI/scripts/run_calvin_joint_belief_suffix.sh'),
            '--policy_action_intent_lora_ckpt',str(components['lora_checkpoint']),
            '--joint_pair_direct_generate_only','--joint_pair_no_memory',
            '--joint_belief_transition_direct_suffix','--joint_belief_transition_sparse_suffix'],
            env=env,stdout=f,stderr=subprocess.STDOUT)
    raise SystemExit(p.returncode)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='dispatch':return dispatch(sys.argv[2:])
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-pairs',type=int,default=200)
    p.add_argument('--checkpoint',type=Path,default=default_checkpoint(DEFI))
    p.add_argument('--eval-sequences',type=Path,default=DEFI/'outputs/splits/outcome_router_clean_1k_train1000.json')
    p.add_argument('--max-pairs-per-episode',type=int,default=4)
    p.add_argument('--belief-checkpoint',type=Path,default=DEFI/'outputs/full_rssm_plan_compat_v2_success_balanced_650x2/rssm_plan_compatibility.pt')
    p.add_argument('--action-generator-checkpoint',type=Path,default=Path('/mnt/workspace/manipulation/BridgeVLA/eval/joint_action_generator_mlp_d300_abc700_abc1500_v2/joint_action_generator_mlp.pt'))
    p.add_argument('--lora-checkpoint',type=Path,default=Path('/mnt/workspace/manipulation/BridgeVLA/eval/policy_lora_action_intent_v2/best.pt'))
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--sequences',type=int,default=20);p.add_argument('--every',type=int,default=20)
    p.add_argument('--min-gap',type=float,default=1.,choices=[1.],help='Binary success filter: original failure 0, candidate success 1')
    p.add_argument('--belief-update-scale',type=float,default=1.)
    p.add_argument('--max-action-delta',type=float,default=.015,help='Clip candidate correction BEFORE rollout; 0 disables clipping')
    p.add_argument('--episode-steps',type=int,default=360,help='Per-task budget; current task uses only its unspent portion, future tasks each get this full budget')
    p.add_argument('--return-gamma',type=float,default=1.,choices=[1.],help='Binary full-chain success; no speed discount')
    p.add_argument('--continuation-steps',type=int,default=-1,choices=[-1],
                   help='Execute ALL remaining tasks until chain complete or first task timeout; no truncated-return mode')
    p.add_argument('--continuation-seed',type=int,default=1701)
    a=p.parse_args()
    if a.max_pairs<=0 or a.every<=0 or a.sequences<=0 or a.min_gap<=0 or a.max_pairs_per_episode<1:p.error('positive limits required')
    if a.episode_steps<1 or a.continuation_steps < -1 or not 0<=a.continuation_seed<2**32-10000:p.error('Invalid continuation budget/seed')
    if not (0<=a.belief_update_scale<float('inf') and 0<=a.max_action_delta<float('inf')):
        p.error('Belief scale and action clipping must be finite/nonnegative')
    collect(a)

if __name__=='__main__':main()
