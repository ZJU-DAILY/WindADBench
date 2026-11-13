from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
METRIC_COLS = [
    'acc', 'point_precision', 'point_recall', 'point_f1',
    'event_precision', 'event_recall', 'event_f1', 'event_affiliation_f1',
    'range_precision', 'range_recall', 'range_f1',
    'affiliation_precision', 'affiliation_recall', 'affiliation_f1',
    'auc_pr', 'auc_roc', 'range_auc_pr', 'range_auc_roc', 'vus_pr', 'vus_roc',
    'mean_lead_time', 'mean_detection_delay', 'early_detection_rate',
    'actionable_lead_time', 'normal_point_fpr', 'false_alarms_per_turbine_day', 'mtbfa',
    'fit_time', 'infer_time', 'infer_gpu_mem', 'model_size',
]
NORMAL_COLS = {'acc', 'normal_point_fpr', 'false_alarms_per_turbine_day', 'mtbfa',
               'fit_time', 'infer_time', 'infer_gpu_mem', 'model_size'}
COST_MAP = {'inference_time': 'infer_time', 'inference_gpu_peak_memory_mb': 'infer_gpu_mem',
            'model_size_mb': 'model_size'}
FARM_META = {
    'A': {'turbines': 5, 'features': 86, 'type': 'onshore'},
    'B': {'turbines': 9, 'features': 257, 'type': 'offshore'},
    'C': {'turbines': 22, 'features': 957, 'type': 'offshore'},
}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric_row(row, workload=None, normal=False):
    output = {key: row.get(key) if not normal or key in NORMAL_COLS else None for key in METRIC_COLS}
    if workload is not None:
        output.update(m=row['model'], w=workload)
    return clean(output)


def build(results_root, evidence_path):
    results_root, evidence_path = Path(results_root).resolve(), Path(evidence_path).resolve()
    sources = {}

    def read(relative):
        path = results_root / relative
        sources[relative] = sha(path)
        return pd.read_csv(path, float_precision='round_trip')

    sys.path.insert(0, str(REPO / 'agent/_impl'))
    import final as agent_final
    evidence = agent_final.load_evidence(REPO, evidence_path)
    cells = evidence.cells
    models = evidence.models
    dims = agent_final.kb.dim_scores(cells, models)
    costs = read('in_farm/main/latest_cost_by_farm.csv').rename(columns=COST_MAP)
    macro_costs = read('in_farm/main/latest_cost_macro.csv').rename(columns=COST_MAP)
    main = read('in_farm/main/latest_main_by_farm.csv').merge(costs, on=['model', 'farm_id'], validate='one_to_one')
    main_macro = read('in_farm/main/latest_main_macro.csv').merge(macro_costs, on='model', validate='one_to_one')
    ct = read('cross_turbine/metrics_by_direction_hours.csv')
    cf = read('cross_farm/metrics_by_direction_hours.csv')
    ct_macro = read('cross_turbine/metrics_macro_hours.csv')
    cf_macro = read('cross_farm/metrics_macro_hours.csv')
    assert ct.source_farm.eq(ct.target_farm).all()
    assert cf.source_farm.ne(cf.target_farm).all()
    assert ct.time_units.eq('hours').all() and cf.time_units.eq('hours').all()
    for frame, multiplier in ((main, 3), (ct, 3), (cf, 6)):
        assert set(frame.model) == set(models) and len(frame) == len(models) * multiplier
    for frame in (main_macro, ct_macro, cf_macro):
        assert set(frame.model) == set(models) and not frame.model.duplicated().any()

    metrics = []
    raw_rows = {}
    for row in main.to_dict('records'):
        workload = 'IF-' + row['farm_id']
        metrics.append(metric_row(row, workload))
        metrics.append(metric_row(row, 'NR-' + row['farm_id'], normal=True))
        raw_rows[row['model'], workload] = row
    for frame, prefix in ((ct, 'CT-'), (cf, 'CF-')):
        for row in frame.to_dict('records'):
            workload = prefix + row['source_farm']
            if prefix == 'CF-':
                workload += '>' + row['target_farm']
            cell = cells[(cells.model == row['model']) & (cells.workload == workload)].iloc[0]
            row.update(fit_time=cell.fit_s, infer_time=cell.infer_s, infer_gpu_mem=cell.gpu_mb)
            metrics.append(metric_row(row, workload))
            raw_rows[row['model'], workload] = row
    correspondence = {
        'event_f1': 'event_f1', 'vus_pr': 'vus_pr', 'edr': 'early_detection_rate',
        'delay_h': 'mean_detection_delay', 'fa_day': 'false_alarms_per_turbine_day',
        'mtbfa_h': 'mtbfa', 'normal_fpr': 'normal_point_fpr', 'acc': 'acc',
        'fit_s': 'fit_time', 'infer_s': 'infer_time', 'gpu_mb': 'infer_gpu_mem',
    }
    evidence_max_abs_difference = 0.0
    for row in cells.to_dict('records'):
        source = raw_rows[row['model'], row['workload']]
        for field, column in correspondence.items():
            a, b = row[field], source[column]
            assert (pd.isna(a) and pd.isna(b)) or np.isclose(a, b, rtol=0, atol=1e-15), (row['model'], row['workload'], field, a, b)
            if np.isfinite(a) and np.isfinite(b):
                evidence_max_abs_difference = max(evidence_max_abs_difference, abs(a - b))

    macro = {}
    for track, frame in (('in-farm', main_macro), ('normal-reliability', main_macro),
                         ('cross-turbine', ct_macro), ('cross-farm', cf_macro)):
        macro[track] = {row['model']: metric_row(row, normal=track == 'normal-reliability')
                        for row in frame.to_dict('records')}
    registry = []
    family = cells.groupby('model').family.first()
    gpu = cells.groupby('model').cpu_only.first()
    for row in macro_costs.to_dict('records'):
        registry.append(dict(id=row['model'], family=family[row['model']], needs_gpu=not gpu[row['model']],
            model_size=row['model_size'], infer_time=row['infer_time'], fit_time=row['fit_time'], flags=[]))
    registry.sort(key=lambda row: row['id'])
    workloads = agent_final.kb.workload_table().to_dict('records')
    for row in workloads:
        subset = cells[cells.workload == row['workload']]
        row['cost_proxy'] = float((subset.eval_fit_s + subset.eval_infer_s).mean())
    normal_workloads = [dict(workload='NR-' + farm, track='normal-reliability', source=farm, target=farm)
                        for farm in ('A', 'B', 'C')]
    benchmark_workloads = workloads[:3] + normal_workloads + workloads[3:]
    cards = dims.groupby('model')[agent_final.kb.DIMS].mean().to_dict('index')
    dimension_rows = [dict(m=row['model'], w=row['workload'], a=row['accuracy'], e=row['earliness'],
                           r=row['reliability'], c=row['cost']) for row in dims.to_dict('records')]
    sources['agent/evidence_cells.csv'] = sha(evidence_path)
    data = clean(dict(registry=registry, workloads=workloads, benchmarkWorkloads=benchmark_workloads,
        dims=dimension_rows, metrics=metrics, macroMetrics=macro, cards=cards, farms=FARM_META,
        provenance=dict(result_set='care_unified_20261107', time_units='hours', source_sha256=sources,
                        evidence_max_abs_difference=evidence_max_abs_difference)))
    assert len(data['metrics']) == len(models) * 15
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, default=REPO / 'agent/evidence_cells.csv')
    parser.add_argument('--out', type=Path, default=HERE / 'data.js')
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    data = build(args.results_root, args.evidence)
    args.out.write_text('window.WINDAD = ' + json.dumps(data, allow_nan=False, separators=(',', ':')) + ';\n', encoding='utf-8')
    if args.manifest:
        manifest = dict(status='passed', output_sha256=sha(args.out),
                        source_sha256=data['provenance']['source_sha256'],
                        evidence_max_abs_difference=data['provenance']['evidence_max_abs_difference'],
                        models=len(data['registry']),
                        benchmark_workloads=len(data['benchmarkWorkloads']), agent_tasks=len(data['workloads']),
                        metric_cells=len(data['metrics']) * len(METRIC_COLS), time_units='hours')
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
