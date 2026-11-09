"""Audit held-out four-outcome pairs, separating candidate data from TCN-trigger effects.

No inference modifications. This tool evaluates measured outcomes, never predictions
or classification accuracy. Repeated ABBA replays count as ONE paired sample.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from outcome import classify_outcome,LABELS


def episode_key(initial_state,tasks):
    canonical=json.dumps(dict(initial_state=initial_state,tasks=[str(t) for t in tasks]),
        sort_keys=True,separators=(',',':'),default=lambda x:np.asarray(x).tolist())
    return hashlib.sha256(canonical.encode()).hexdigest()


def forbidden_episodes(report):
    groups=report.get('split_episode_keys')
    if not isinstance(groups,dict) or not groups.get('train') or not groups.get('validation'):
        raise ValueError('Missing stable train/validation episode keys; numeric local sequence IDs cannot prove held-out evaluation')
    # Validation influenced checkpoint selection, so it is not an independent test.
    return set(groups['train'])|set(groups['validation'])


def validate(rows,training_report,triggered=False):
    forbidden=forbidden_episodes(training_report);seen=set();sources=set();hashes=set()
    if not rows:raise ValueError('No measured pairs')
    for r in rows:
        key=r.get('episode_key')
        if not key:raise ValueError('Pair missing stable original episode key')
        if key in forbidden:raise ValueError('Train/validation episode leaked into held-out evaluation')
        identity=(key,r.get('subtask'),r.get('step'),r.get('retry_attempt',0))
        if identity in seen:raise ValueError('Duplicate intervention; ABBA repetitions must not be counted as independent samples')
        seen.add(identity)
        if not r.get('full_remaining_chain_return') or r.get('return_gamma')!=1.:
            raise ValueError('Require binary full remaining-chain evaluation')
        label=LABELS[classify_outcome(r['returns_prior'],r['returns_post'])]
        if r.get('outcome',label)!=label:raise ValueError('Outcome conflicts with measured branch returns')
        if not r.get('branch_replay_verified') and 'ABBA' not in str(r.get('replay','')):
            raise ValueError('Missing matched-state repeated replay verification')
        sources.add(r.get('candidate_source','belief_counterfactual'))
        if triggered:
            if not r.get('tcn_triggered') or r.get('candidate_source')!='actual_tcn_postprocessed_cache':
                raise ValueError('Dataset proposals are NOT actual triggered TCN repairs')
            if not r.get('candidate_processed_before_rollout') or not r.get('inference_unchanged'):
                raise ValueError('Must replay the actual original-inference postprocessed cache')
            if not r.get('continuation_rule') or not r.get('inference_rule_sha256'):
                raise ValueError('Missing original inference/continuation provenance')
            digest=r.get('tcn_checkpoint_sha256')
            if not digest or digest!=training_report.get('checkpoint_sha256'):
                raise ValueError('Measured TCN differs from training report checkpoint')
            hashes.add(digest)
    if triggered and len(hashes)!=1:raise ValueError('Do not mix different TCN checkpoints')
    if len(sources)!=1:raise ValueError('Do not mix candidate sources in one population')


def rates(counts):
    n01=counts['repair'];n11=counts['preserve'];n00=counts['unresolved'];n10=counts['harmful']
    total=n01+n11+n00+n10
    def ratio(a,b):return a/b if b else None
    return dict(n=total,counts=dict(counts),
        four_class_proportions={k:ratio(counts[k],total) for k in LABELS},
        repair_success_rate=ratio(n01,n00+n01),preservation_rate=ratio(n11,n11+n10),
        unresolved_rate=ratio(n00,n00+n01),degradation_rate=ratio(n10,n11+n10),
        baseline_failure_denominator=n00+n01,baseline_success_denominator=n11+n10,
        baseline_success_rate=ratio(n11+n10,total),candidate_success_rate=ratio(n11+n01,total),
        net_success_gain=ratio(n01-n10,total))


def population(rows,training_report,triggered=False):
    validate(rows,training_report,triggered)
    counts=Counter({k:0 for k in LABELS});clusters=defaultdict(Counter)
    for r in rows:
        label=LABELS[classify_outcome(r['returns_prior'],r['returns_post'])]
        counts[label]+=1;clusters[r['episode_key']][label]+=1
    report=rates(counts);report['episodes']=len(clusters)
    report['population']='actual_tcn_triggered_repairs' if triggered else 'all_measured_dataset_candidates'
    # Cluster bootstrap: all decision points from one episode stay together.
    rng=np.random.default_rng(42);keys=list(clusters);samples=[]
    for _ in range(2000):
        c=Counter({k:0 for k in LABELS})
        for i in rng.integers(len(keys),size=len(keys)):c.update(clusters[keys[i]])
        samples.append(rates(c))
    names=['repair_success_rate','preservation_rate','unresolved_rate','degradation_rate','net_success_gain']
    report['episode_cluster_bootstrap_95_ci']={}
    for name in names:
        values=[s[name] for s in samples if s[name] is not None]
        report['episode_cluster_bootstrap_95_ci'][name]=np.quantile(values,[.025,.975]).tolist() if len(clusters)>=2 and values else None
    report['warning']='Few independent episodes; do not interpret as reliable deployment estimate' if len(clusters)<20 else None
    if triggered:
        scopes={r.get('evaluation_scope','unspecified') for r in rows}
        report['evaluation_scopes']=sorted(scopes)
        report['full_closed_loop_deployment_evaluated']=scopes=={'original_full_closed_loop'}
        report['evaluated_trigger_pairs']=len(rows)
        report['trigger_coverage']=None  # Requires logs of ALL eligible decisions, including no-trigger cases.
        # Common base-policy continuation estimates an individual intervention
        # effect, not performance of a full TCN controller acting repeatedly.
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-report',type=Path,required=True)
    p.add_argument('--dataset-pairs',type=Path);p.add_argument('--triggered-pairs',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.dataset_pairs and not a.triggered_pairs:p.error('Provide measured dataset and/or actual-trigger pair JSONL')
    if a.output.exists():raise FileExistsError(a.output)
    training=json.loads(a.training_report.read_text());report={}
    for name,path,triggered in [('dataset_distribution',a.dataset_pairs,False),('tcn_trigger_effect',a.triggered_pairs,True)]:
        if path:
            rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
            report[name]=population(rows,training,triggered)
    report.update(definitions={'repair_success_rate':'N01/(N00+N01)','preservation_rate':'N11/(N11+N10)',
        'unresolved_rate':'N00/(N00+N01)','degradation_rate':'N10/(N11+N10)',
        'four_class_proportions':'Each Nij / all pairs, reported separately',
        'net_success_gain':'(N01-N10)/all pairs'},
        tcn_trigger_effect_evaluated=bool(a.triggered_pairs),
        limitations=['No guarantee for each action edit or unseen episode.',
            'ABBA checks determinism; repetitions are not independent trials.',
            'Episode disjointness audited for this report; initial pretrained checkpoint data provenance requires separate audit.',
            'Per-trigger effects are not automatically full-episode deployed-controller success rates.'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
