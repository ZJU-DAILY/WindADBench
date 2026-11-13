import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.cross_domain.aggregation import write_result_tables
from experiments.cross_domain.artifacts import atomic_json, sha256_file
from experiments.cross_domain.protocol import FULL_FLEET_WORKLOAD, target_farms, validate_target_scope
from tsad_benchmark.evaluation.care_protocol import PROTOCOL
from tsad_benchmark.report.care import is_current


def collect_results(run_dirs, output_dir):
    runs = [Path(path).resolve() for path in run_dirs]
    output = Path(output_dir).resolve()
    if not runs or len(set(runs)) != len(runs):
        raise ValueError('Source run directories must be nonempty and unique.')
    frames, inputs = [], []
    for run in runs:
        if output == run or output in run.parents or run in output.parents:
            raise ValueError('Combined results must be separate from source runs.')
        state = json.loads((run/'state.json').read_text(encoding='utf8'))
        if state.get('status') != 'completed':
            raise ValueError(f'Source run is not completed: {run}')
        path = run/'results/per_event.csv'
        frame = pd.read_csv(path, float_precision='round_trip', low_memory=False)
        if frame.empty or not is_current(frame):
            raise ValueError(f'Current CARE evaluation rows are required: {run}')
        score = frame[frame.track.eq('score')]
        if not score.score_metric_status.isin(['complete', 'not_applicable', 'no_published_scores']).all():
            raise ValueError(f'Incomplete score metrics: {run}')
        frame['source_run_path'] = str(run)
        frames.append(frame)
        inputs.append(dict(run_dir=str(run), per_event_sha256=sha256_file(path)))
    rows = pd.concat(frames, ignore_index=True)
    workload = rows.get('workload_id', pd.Series('fixed_holdout', index=rows.index)).fillna('fixed_holdout')
    if workload.nunique() != 1 or rows.protocol.nunique() != 1:
        raise ValueError('Cannot combine different transfer workloads or protocols.')
    workload_id = workload.iloc[0]
    validate_target_scope(rows, workload_id)
    expected = {(source, target, track) for source in 'ABC'
                for target in target_farms(source, workload_id) for track in ('score', 'label')}
    for model, group in rows.groupby('model'):
        actual = set(group[['source_farm', 'target_farm', 'track']].itertuples(index=False, name=None))
        if actual != expected:
            raise ValueError(f'Incomplete source/target directions for {model}.')
        for field in ('seed', 'model_params'):
            if group[field].nunique(dropna=False) != 1:
                raise ValueError(f'Mixed {field} for {model}.')
    if workload_id != FULL_FLEET_WORKLOAD and (output/'results/cross_farm_macro.csv').is_file():
        raise ValueError('Fixed-holdout collection cannot overwrite a cross-farm summary; '
                         'use a separate output directory.')
    _, directions = write_result_tables(output, rows)
    manifest = dict(status='completed', evaluation_protocol=PROTOCOL,
                    models=int(rows.model.nunique()), source_runs=len(runs),
                    per_event_rows=len(rows), direction_rows=len(directions), inputs=inputs,
                    artifact_path_base='source_run_path')
    atomic_json(manifest, output/'collection_manifest.json')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = collect_results(args.run_dirs, args.output_dir)
    print(f"Combined {result['source_runs']} source runs for {result['models']} models: {args.output_dir}")


if __name__ == '__main__':
    main()
