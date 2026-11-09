"""Read-only audit of historical teachers, weights and evaluation logs for reuse."""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(root):
    datasets=[]
    for relative in ['paired_counterfactual_teacher_v2/directional_sensitivity_memory_pc_7307.npz',
            'directional_sensitivity_tcn_teacher_v2/teacher_memory.npz',
            'joint_belief_transition_v2_balanced_success/transition_memory_s1024_seq934_success_balanced_650x2_oversample.npz']:
        path=root/relative
        if not path.exists():continue
        with np.load(path,allow_pickle=False) as z:
            values,counts=np.unique(z['success'],return_counts=True)
            modes,mode_counts=np.unique(z['mode_label'],return_counts=True)
            paired=all(k in z for k in ['returns_prior','returns_post'])
            datasets.append(dict(path=str(path),rows=len(z['success']),independent_sequence_ids=len(np.unique(z['sequence_index'])),
                fields=z.files,single_success_field_counts={str(v):int(c) for v,c in zip(values,counts)},
                mode_label_counts={str(v):int(c) for v,c in zip(modes,mode_counts)},paired_measured_returns_present=paired,
                eligible_for_strict_four_outcome_training=False,
                reuse='Existing features/action-delta supervision or pretrained initialization only; paired full-chain returns and simulator/RNG provenance must be recollected',
                warning='The single success column and mode labels are not the two counterfactual outcomes'))
    weights=[]
    for relative in ['directional_sensitivity_tcn_paired_counterfactual_v1/train_pc2k_h128_scale002',
            'directional_sensitivity_tcn_teacher_v3_validity_structured/train_h128']:
        folder=root/relative
        ckpt=folder/'directional_sensitivity_tcn.pt';summary=folder/'summary.json'
        if ckpt.exists() and summary.exists():
            d=json.loads(summary.read_text())
            weights.append(dict(checkpoint=str(ckpt),training_memory=d.get('memory_npz'),
                train_rows=d.get('num_train'),validation_rows=d.get('num_val'),has_validity_head=d.get('has_validity_head'),
                original_supervision=d.get('supervision'),reuse='Warm-start only; mask/validity scores are not measured repair-success rates'))
    evaluations=[]
    folder=root/'calvin_suffix_mask_control_6x100_pc_teacher_final'
    if folder.exists():
        for condition in sorted(folder.iterdir()):
            logs=condition/'train_folder/logs'
            if not logs.exists():continue
            for run in logs.iterdir():
                path=run/'results.json'
                if not path.exists():continue
                data=json.loads(path.read_text())
                metrics={k:dict(avg_seq_len=v.get('avg_seq_len'),chain_sr=v.get('chain_sr')) for k,v in data.items()}
                evaluations.append(dict(condition=condition.name,results=str(path),metrics=metrics,
                    reuse='Historical whole-run comparator; not per-trigger matched-state counterfactual labels'))
    return dict(datasets=datasets,warm_start_weights=weights,historical_evaluations=evaluations,
        conclusion='Reuse old checkpoints/features; no audited dataset has the required new strict four-outcome labels.',
        training_started=False,new_collection_started=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/mnt/workspace/manipulation/DeFi/outputs'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    report=audit(a.root);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
