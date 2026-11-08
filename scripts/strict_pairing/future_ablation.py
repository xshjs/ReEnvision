"""Frozen-checkpoint feature/intervention ablation. No production source edits or training."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from paths import defi_root,default_checkpoint

HERE=Path(__file__).resolve().parent;DEFI=defi_root()
SOURCE=DEFI/'Step3_DeFI/policy_evaluation/calvin_evaluate_with_memory_reflection.py'

def paper_suffix_env():
    """Exact explicit controller values from the October paper suffix launcher."""
    return dict(DIRECTIONAL_SENSITIVITY_TCN='0',DIRECT_ACTION_CORRECTION_TCN='0',
        LINEAR_SUFFIX_SENSITIVITY='1',DYNAMIC_COUPLING_FIRST_PASS='0',DISABLE_RETRY_AFTER_FIRST_PASS='0',
        RETRY_ROLLBACK_MODE='subtask_start',RETRY_MAX_ATTEMPTS='1',DYNAMIC_COUPLING_DISABLE_REPLAN='1',
        DYNAMIC_COUPLING_DELTA_ACTION_REPAIR_MIX='0.005',JOINT_BELIEF_CLEAN_H_UPDATE_SCALE='0.8',
        JOINT_BELIEF_CLEAN_PATCH_MIN_PROB='0.35',JOINT_BELIEF_CLEAN_PATCH_MARGIN='0.00',
        LINEAR_SUFFIX_SENSITIVITY_STEP_SIZE='0.030',LINEAR_SUFFIX_SENSITIVITY_MAX_ABS_DELTA='0.015',
        LINEAR_SUFFIX_SENSITIVITY_SELECTIVE_STEP_THRESHOLD='0.002',LINEAR_SUFFIX_SENSITIVITY_MAX_PATCH_FRACTION='0.70',
        LINEAR_SUFFIX_SENSITIVITY_REPLAN_FRACTION='0.90',LINEAR_SUFFIX_SENSITIVITY_REPLAN_DELTA_H_NORM='0.0')


def dispatch(args):
    if not args or Path(args[0]).name!=SOURCE.name:raise SystemExit('Unexpected evaluator')
    sys.argv=[str(SOURCE),*args[1:]]
    sys.path.insert(0,str(SOURCE.parent));sys.path.insert(0,str(DEFI/'Step3_DeFI'))
    from future_compat import install as install_import_compat
    install_import_compat()
    from future_eval_hook import install
    tree=ast.parse(SOURCE.read_text());indices=[i for i,n in enumerate(tree.body) if isinstance(n,ast.If) and '__name__' in ast.unparse(n.test)]
    if len(indices)!=1:raise ValueError('Unsupported evaluator entrypoint')
    tree.body[indices[0]:indices[0]]=ast.parse('_future_install(globals())').body;ast.fix_missing_locations(tree)
    def attach(ns):install(ns,os.environ['FUTURE_ABLATION_RUN'],os.environ['FUTURE_FEATURE'],os.environ['FUTURE_INTERVENTION'],int(os.environ['FUTURE_SEED']))
    ns={'__name__':'__main__','__file__':str(SOURCE),'__package__':None,'_future_install':attach}
    exec(compile(tree,str(SOURCE),'exec'),ns)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='dispatch':return dispatch(sys.argv[2:])
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--sequences',type=int,default=10)
    p.add_argument('--benchmarks',nargs='+',choices=['calvin','lcalvin'],default=['calvin','lcalvin'])
    p.add_argument('--features',nargs='+',choices=['predicted','current_observed','current_slot'],default=['predicted','current_observed'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--checkpoint',type=Path,default=default_checkpoint(DEFI))
    p.add_argument('--repair-backend',choices=['paper_suffix','tcn'],default='paper_suffix',
        help='paper_suffix reproduces README controller/retry settings; tcn is an explicit separate no-retry variant')
    p.add_argument('--tcn',type=Path,default=DEFI/'outputs/directional_sensitivity_tcn_v1_success_balanced_650x2/directional_sensitivity_tcn.pt')
    p.add_argument('--calvin-sequences',type=Path,default=DEFI/'outputs/splits/outcome_router_clean_1k_heldout110.json')
    p.add_argument('--lcalvin-sequences',type=Path,default=DEFI/'outputs/splits/lcalvin_len10_heldout100.json')
    p.add_argument('--belief-checkpoint',type=Path,default=DEFI/'outputs/full_rssm_plan_compat_v2_success_balanced_650x2/rssm_plan_compatibility.pt')
    p.add_argument('--action-generator-checkpoint',type=Path,default=Path('/mnt/workspace/manipulation/BridgeVLA/eval/joint_action_generator_mlp_d300_abc700_abc1500_v2/joint_action_generator_mlp.pt'))
    p.add_argument('--lora-checkpoint',type=Path,default=Path('/mnt/workspace/manipulation/BridgeVLA/eval/policy_lora_action_intent_v2/best.pt'))
    a=p.parse_args()
    if a.sequences<1 or any(not 0<=seed<2**32 for seed in a.seeds):p.error('Invalid counts/seeds')
    if len(set(a.features))!=len(a.features) or len(set(a.seeds))!=len(a.seeds):p.error('Duplicate conditions')
    if a.output.exists():raise FileExistsError('Use a fresh output, no mixing or partial resume')
    required=['checkpoint','belief_checkpoint','action_generator_checkpoint','lora_checkpoint']
    if a.repair_backend=='tcn':required.append('tcn')
    files={k:getattr(a,k).resolve(strict=True) for k in required}
    sequence_files={k:getattr(a,k+'_sequences').resolve(strict=True) for k in a.benchmarks}
    for benchmark,path in sequence_files.items():
        rows=json.loads(path.read_text())
        if len(rows)<a.sequences:raise ValueError(f'{benchmark}: only {len(rows)} sequences available')
    cells=[]
    for benchmark in a.benchmarks:
        for seed in a.seeds:
            for feature in a.features:
                for intervention in ['selective','full_replan']:
                    cells.append(dict(benchmark=benchmark,seed=seed,feature=feature,intervention=intervention,
                        repair_backend=a.repair_backend,retry_protocol='baseline_then_one_recovery_pass' if a.repair_backend=='paper_suffix' else 'first_pass_no_retry',
                        sequence_file=str(sequence_files[benchmark]),sequence_sha256=hashlib.sha256(sequence_files[benchmark].read_bytes()).hexdigest(),
                        sequences=a.sequences,episode_steps=360 if benchmark=='calvin' else 720))
    if a.dry_run:
        print(json.dumps(dict(cells=cells,checkpoints={k:str(v) for k,v in files.items()},
            note='Plan only. No GPU evaluation performed.'),indent=2));return
    import torch
    if not torch.cuda.is_available():raise SystemExit('Use a GPU terminal; --dry-run is available on CPU')
    a.output.mkdir(parents=True)
    runtime_tcn=None;tcn_compat=dict(required=False,tcn_enabled=False)
    if a.repair_backend=='tcn':
        from future_compat import prepare_tcn_checkpoint
        runtime_tcn,tcn_compat=prepare_tcn_checkpoint(files['tcn'],a.output/'tcn_loader_compat.pt')
        runtime_tcn=runtime_tcn.resolve()
    manifest=dict(cells=cells,checkpoints={k:str(v) for k,v in files.items()},
        repair_backend=a.repair_backend,controller_overrides=paper_suffix_env() if a.repair_backend=='paper_suffix' else None,
        runtime_tcn=str(runtime_tcn) if runtime_tcn else None,tcn_loader_compatibility=tcn_compat,
        checkpoint_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in files.items()},
        evaluator_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),frozen_weights=True,
        training_performed=False,oracle_available=False,
        localization_accuracy_available=False,production_inference_modified=False,
        limitations=['Exploratory plug-in comparison with identical frozen downstream checkpoints, NOT matched retraining.',
            'current_observed uses single observed-image denoising features of the same frozen backbone; it is not an independently trained CLIP baseline.',
            'Repeating current features may shift the downstream distribution; differences alone do not prove forecasting is necessary.',
            'Full replanning replaces each actually nonzero selective patch at its original trigger; original unrelated replans remain unchanged.',
            'Patch coverage is not causal repair localization accuracy; no fake labels or nearest-demo GT oracle.'])
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for cell in cells:
        root=a.output/f"{cell['benchmark']}_s{cell['seed']}_{cell['feature']}_{cell['intervention']}"
        root.mkdir();(root/'condition.json').write_text(json.dumps(cell,indent=2))
        env=os.environ.copy();env.update(FUTURE_ABLATION_RUN=str(root.resolve()),FUTURE_FEATURE=cell['feature'],
            FUTURE_INTERVENTION=cell['intervention'],FUTURE_SEED=str(cell['seed']),STRICT_PYTHON=sys.executable,
            PYTHON_BIN=str(HERE/'future_dispatch.sh'),CHECKPOINT_PATH=str(files['checkpoint']),ACTION_MODEL_FOLDER=str(files['checkpoint'].parent.parent),
            JOINT_BELIEF_TRANSITION_CKPT=str(files['belief_checkpoint']),JOINT_PAIR_ACTION_GENERATOR_CKPT=str(files['action_generator_checkpoint']),
            DIRECTIONAL_SENSITIVITY_TCN='1',DIRECTIONAL_SENSITIVITY_TCN_CKPT=str(runtime_tcn or ''),DIRECT_ACTION_CORRECTION_TCN='0',
            LINEAR_SUFFIX_SENSITIVITY='0',DEFI_SUFFIX_SENSITIVITY_CACHE='0',COLLECT_SENSITIVITY_TEACHER='0',
            SUFFIX_MASK_ABLATION_MODE='off',DYNAMIC_COUPLING_FIRST_PASS='1',DISABLE_RETRY_AFTER_FIRST_PASS='1',
            NUM_SEQUENCES=str(a.sequences),EP_LEN=str(cell['episode_steps']),EVAL_SEQUENCES_PATH=cell['sequence_file'],
            JOINT_BELIEF_CLEAN_H_UPDATE_SCALE='0.8',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            HF_HOME='/tmp/reenvision-future-hf',TRANSFORMERS_CACHE='/tmp/reenvision-future-hf/hub',
            XDG_CACHE_HOME='/tmp/reenvision-future-cache',MPLCONFIGDIR='/tmp/reenvision-future-mpl',USE_TF='0')
        if a.repair_backend=='paper_suffix':env.update(paper_suffix_env())
        print('RUN',root,flush=True)
        with (root/'eval.log').open('w') as log:
            result=subprocess.run(['bash',str(DEFI/'Step3_DeFI/scripts/run_calvin_joint_belief_suffix.sh'),
                '--policy_action_intent_lora_ckpt',str(files['lora_checkpoint']),'--joint_pair_direct_generate_only',
                '--joint_pair_no_memory','--joint_belief_transition_direct_suffix','--joint_belief_transition_sparse_suffix'],
                env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(f'Condition failed: {root}/eval.log (exit {result.returncode}); no partial result claimed')
        if not (root/'sequences.jsonl').exists():raise SystemExit(f'No sequence results: {root}')
    from future_report import summarize
    print(json.dumps(summarize(a.output),indent=2))

if __name__=='__main__':main()
