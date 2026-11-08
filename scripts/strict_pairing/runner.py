"""Isolated strict-return CALVIN collection launcher."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from paths import defi_root,default_checkpoint

HERE=Path(__file__).resolve().parent
DEFI=defi_root()
SOURCE=DEFI/'Step3_DeFI/policy_evaluation/calvin_evaluate_with_memory_reflection.py'


def dispatch(args):
    # Existing launch script supplies evaluator path as its first argument.
    if not args or Path(args[0]).name!=SOURCE.name:raise SystemExit('Unexpected evaluator command')
    sys.argv=[str(SOURCE),*args[1:]]
    # Match Python's sys.path[0] when the original evaluator is run directly.
    # Several legacy sibling modules use unqualified imports.
    sys.path.insert(0,str(SOURCE.parent))
    sys.path.insert(0,str(DEFI/'Step3_DeFI'))
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
        simulated=True,training_started=False,protocol_version=3,
        return_scope='full_current_subtask_episode' if os.environ['STRICT_CONTINUATION_STEPS']=='-1' else 'explicit_truncated_ablation',
        both_failed=sum(not r.get('prior_success',False) and not r.get('post_success',False) for r in rows),
        evaluator_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest()),indent=2))
    if not rows:raise SystemExit('No strict pairs collected; inspect evaluator log, do not train')


def collect(args):
    import torch
    if not torch.cuda.is_available():raise SystemExit('GPU not available in this process; run this command in GPU terminal')
    if args.output.exists():raise SystemExit('Use a fresh output directory (no overwrite/resume mixing)')
    args.output.mkdir(parents=True)
    ckpt=args.checkpoint.resolve(strict=True)
    train=args.output/'train_folder';(train/'saved_models').mkdir(parents=True)
    (train/'saved_models'/ckpt.name).symlink_to(ckpt)
    env=os.environ.copy();env.update(REENVISION_STRICT_RUN=str(args.output.resolve()),
        STRICT_MAX_PAIRS=str(args.max_pairs),STRICT_EVERY=str(args.every),STRICT_MIN_GAP=str(args.min_gap),
        STRICT_CONTINUATION_STEPS=str(args.continuation_steps),STRICT_CONTINUATION_SEED=str(args.continuation_seed),
        STRICT_PYTHON=sys.executable,
        PYTHON_BIN=str(HERE/'dispatch.sh'),ACTION_MODEL_FOLDER=str(train.resolve()),CHECKPOINT_PATH=str(ckpt),
        NUM_SEQUENCES=str(args.sequences),EP_LEN=str(args.episode_steps),COLLECT_SENSITIVITY_TEACHER='1',
        SENSITIVITY_TEACHER_MODE='paired_counterfactual',SENSITIVITY_TEACHER_STEP_SIZE='1',
        SENSITIVITY_TEACHER_MAX_ABS_DELTA='0',DYNAMIC_COUPLING_FIRST_PASS='1',
        DISABLE_RETRY_AFTER_FIRST_PASS='1',DYNAMIC_COUPLING_DELTA_ACTION_REPAIR_MIX='0',
        JOINT_BELIEF_CLEAN_H_UPDATE_SCALE='0.8',LINEAR_SUFFIX_SENSITIVITY='0',
        DIRECTIONAL_SENSITIVITY_TCN='0',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        XDG_CACHE_HOME='/tmp/reenvision-strict-cache',MPLCONFIGDIR='/tmp/reenvision-strict-mpl')
    manifest=dict(checkpoint=str(ckpt),checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        evaluator=str(SOURCE),evaluator_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        cli=vars(args).copy())
    manifest['cli']['output']=str(args.output)
    manifest['cli']['checkpoint']=str(ckpt)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with (args.output/'collect.log').open('w') as f:
        print('Collection log:',args.output/'collect.log',flush=True)
        p=subprocess.run(['bash',str(DEFI/'Step3_DeFI/scripts/run_calvin_joint_belief_suffix.sh'),
            '--policy_action_intent_lora_ckpt','/mnt/workspace/manipulation/BridgeVLA/eval/policy_lora_action_intent_v2/best.pt',
            '--joint_pair_direct_generate_only','--joint_pair_no_memory',
            '--joint_belief_transition_direct_suffix','--joint_belief_transition_sparse_suffix'],
            env=env,stdout=f,stderr=subprocess.STDOUT)
    raise SystemExit(p.returncode)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='dispatch':return dispatch(sys.argv[2:])
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-pairs',type=int,default=200)
    p.add_argument('--checkpoint',type=Path,default=default_checkpoint(DEFI))
    p.add_argument('--sequences',type=int,default=20);p.add_argument('--every',type=int,default=20)
    p.add_argument('--min-gap',type=float,default=.01)
    p.add_argument('--episode-steps',type=int,default=360,help='Common current-subtask episode deadline, including actions before the branch')
    p.add_argument('--continuation-steps',type=int,default=-1,
                   help='Default -1: continue until success or original episode deadline. Nonnegative values are explicit truncated ablations.')
    p.add_argument('--continuation-seed',type=int,default=1701)
    a=p.parse_args()
    if a.max_pairs<=0 or a.every<=0 or a.sequences<=0 or a.min_gap<=0:p.error('positive limits required')
    if a.episode_steps<1 or a.continuation_steps < -1 or not 0<=a.continuation_seed<2**32-10000:p.error('Invalid continuation budget/seed')
    collect(a)

if __name__=='__main__':main()
