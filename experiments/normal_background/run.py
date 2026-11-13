from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.normal_background.budget import build_budget_lookup
from experiments.normal_background.inputs import load_models
from experiments.normal_background.reference import (
    SETTINGS, _complete_mean, aggregate_events, compute_model,
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(result_paths, dataset_root, output):
    paths = [Path(path).resolve() for path in result_paths]
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory must be empty: {output}')
    models = load_models(paths, Path(dataset_root).resolve())
    if not models:
        raise ValueError('No model results supplied')
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / 'models'
    model_dir.mkdir(exist_ok=True)
    events = []
    for model, sequences in models.items():
        budget_lookup, operating = build_budget_lookup(sequences)
        event_frame, donor_frame = compute_model(model, sequences, budget_lookup)
        event_frame.to_csv(model_dir / f'{model}_events.csv', index=False)
        donor_frame.to_csv(model_dir / f'{model}_donors.csv', index=False)
        budget_lookup.reset_index().to_csv(model_dir / f'{model}_budget_thresholds.csv', index=False)
        operating.to_csv(model_dir / f'{model}_budget_operating_points.csv', index=False)
        events.append(event_frame)
        print(f'{model}: normal-background evaluation completed', flush=True)
    tables = aggregate_events(pd.concat(events, ignore_index=True))
    for name, frame in tables.items():
        frame.to_csv(output / f'{name}.csv', index=False)
    policies = ['eight_point', 'train_fixed_1pct', 'rq3_1pct']
    for source, name in [('per_event', 'three_policy_per_event'),
                         ('per_farm', 'three_policy_per_farm'), ('macro', 'three_policy_macro')]:
        tables[source][tables[source].policy.isin(policies)].to_csv(output / f'{name}.csv', index=False)
    summary = {}
    for policy, group in tables['macro'].groupby('policy'):
        summary[policy] = dict(models=len(group), complete_background_models=int(group.background_probability.notna().sum()),
            mean_observed=_complete_mean(group.observed_edr), mean_background=_complete_mean(group.background_probability),
            mean_excess=_complete_mean(group.excess_over_background),
            positive_excess=int(group.excess_over_background.gt(1e-12).sum()),
            paired_mean_observed=_complete_mean(group.paired_observed_edr),
            paired_mean_background=_complete_mean(group.paired_background_probability),
            paired_mean_excess=_complete_mean(group.paired_excess_over_background))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'three_policy_summary.json').write_text(
        json.dumps({p: summary[p] for p in policies}, indent=2), encoding='utf-8')
    manifest = dict(status='completed', models=list(models), settings=[s[0] for s in SETTINGS],
        dataset_root=str(Path(dataset_root).resolve()),
        source_results_sha256={str(p): sha(p) for p in paths},
        input_sequences={model: [e['file_name'] for e in entries] for model, entries in models.items()},
        donor_weighting='equal matched donors, each donor averages all admissible windows; support explicit',
        aggregation='operating points within event, events within farm, equal farms; full background propagates missing; paired summaries use identical support',
        code_sha256={p.name: sha(p) for p in Path(__file__).resolve().parent.glob('*.py')},
        output_sha256={str(p.relative_to(output)): sha(p) for p in output.rglob('*.csv')})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', nargs='+', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/normal_background')
    args = parser.parse_args()
    print(run(args.results, args.dataset_root, args.output))


if __name__ == '__main__':
    main()
