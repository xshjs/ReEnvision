"""Paired descriptive evaluation; never turn patch coverage into localization accuracy."""
import argparse
import json
from pathlib import Path
import numpy as np


def records(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def summarize(root):
    manifest=json.loads((root/'manifest.json').read_text());results=[];banks={}
    for cell in manifest['cells']:
        name=f"{cell['benchmark']}_s{cell['seed']}_{cell['feature']}_{cell['intervention']}"
        folder=root/name;rows=records(folder/'sequences.jsonl')
        bank={int(r['sequence']):r for r in rows}
        if len(bank)!=cell['sequences'] or len(bank)!=len(rows):raise ValueError(f'Incomplete/duplicate sequences: {name}')
        events=records(folder/'interventions.jsonl') if (folder/'interventions.jsonl').exists() else []
        results.append(dict(condition=name,sequences=len(rows),full_chain_success_rate=float(np.mean([r['success'] for r in rows])),
            mean_completed_tasks=float(np.mean([r['completed'] for r in rows])),
            prefix_success={str(k):float(np.mean([r['completed']>=k for r in rows])) for k in range(1,max(r['total'] for r in rows)+1)},
            mean_sequence_elapsed_s=float(np.mean([r['elapsed_s'] for r in rows])),intervention_count=len(events),
            trigger_patch_coverage=float(sum(e['changed_steps'] for e in events)/max(1,sum(e['remaining_steps'] for e in events))),
            actual_patch_count=sum(e['actual']=='original_selective_repair' for e in events),
            triggered_replan_count=sum(e['actual']=='full_replan_on_same_successful_patch_trigger' for e in events),
            warning='No repair-trigger events; cannot assess selective-repair benefit' if not events else None,
            localization_accuracy=None))
        banks[(cell['benchmark'],cell['seed'],cell['feature'],cell['intervention'])]=bank
    comparisons=[];rng=np.random.default_rng(42)
    def compare(left,right,label):
        a=banks[left];b=banks[right]
        if a.keys()!=b.keys():raise ValueError('Unpaired sequence IDs')
        for i in a:
            if any(a[i][k]!=b[i][k] for k in ['initial_state_sha256','tasks','seed']):raise ValueError('Mismatched initial state/task/seed')
        delta=np.asarray([float(a[i]['success'])-float(b[i]['success']) for i in sorted(a)])
        boot=delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(1)
        comparisons.append(dict(comparison=label,left=list(left),right=list(right),n=len(delta),
            paired_success_difference=float(delta.mean()),bootstrap_95_ci=np.quantile(boot,[.025,.975]).tolist(),
            note='Per-seed paired sequence bootstrap; small smoke sets are not conclusive'))
    for key in banks:
        bench,seed,feature,intervention=key
        if intervention=='selective':compare(key,(bench,seed,feature,'full_replan'),'selective_minus_matched_trigger_full_replan')
        other=(bench,seed,'current_observed',intervention)
        if feature=='predicted' and other in banks:compare(key,other,'predicted_minus_current_observed')
    report=dict(conditions=results,paired_comparisons=comparisons,oracle_evaluated=False,
        localization_accuracy_evaluated=False,training_performed=False,
        interpretation='Frozen plug-in ablation only; current single-frame encoding changes feature distribution. No proof of future necessity without matched retraining and independent localization labels.',
        limitations=manifest['limitations'])
    (root/'ablation_report.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    print(json.dumps(summarize(parser.parse_args().run),indent=2))
