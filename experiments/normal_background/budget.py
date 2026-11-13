from __future__ import annotations

import numpy as np
import pandas as pd

from tsad_benchmark.evaluation.online_protocol import evaluate_threshold
from tsad_benchmark.evaluation.metrics import early_detection_rate

BUDGETS = (.001, .005, .01, .02)
GRID = np.linspace(0., 1., 2001)


def rank_setup(prepared, y, valid):
    y, valid = np.asarray(y), np.asarray(valid, bool)
    scores = np.where(prepared['available'], prepared['stream'], -np.inf)
    if not (len(y) == len(valid) == len(scores)) or not len(y):
        raise ValueError('Scores, truth and validity require a nonempty aligned clock')
    order = np.lexsort((np.arange(len(scores)), -scores))
    ordered = scores[order]
    effective = np.r_[np.searchsorted(-ordered, -ordered, side='left'),
                      int(np.asarray(prepared['available']).sum())]
    prefix = np.r_[0, np.cumsum(((y == 0) & valid)[order])]
    nominal = np.floor(len(scores) * GRID + 1e-12).astype(int)
    return dict(sorted_scores=ordered, effective=effective, normal_prefix=prefix,
                grid_false_positives=prefix[effective[nominal]])

def threshold_at(entry, tail):
    n = len(entry['y'])
    nominal = min(n, max(0, int(np.floor(n * tail + 1e-12))))
    threshold = float(entry['sorted_scores'][nominal]) if nominal < n else -np.inf
    return nominal, threshold

def select_operating_point(entries, budget):
    denominator = sum(np.count_nonzero(e['valid'] & (e['y'] == 0)) for e in entries)
    if not denominator:
        return dict(selected_index=None, selected_tail_fraction=np.nan,
                    normal_points=0, normal_alert_points=0,
                    realized_normal_point_fpr=np.nan, next_grid_normal_point_fpr=np.nan,
                    budget_status='undefined_no_valid_negatives')
    counts = sum(e['grid_false_positives'] for e in entries)
    assert np.all(np.diff(counts) >= 0)
    rates = counts / denominator
    selected = int(np.flatnonzero(rates <= budget + 1e-12)[-1])
    return dict(selected_index=selected, selected_tail_fraction=float(GRID[selected]),
                normal_points=int(denominator), normal_alert_points=int(counts[selected]),
                realized_normal_point_fpr=float(rates[selected]),
                next_grid_normal_point_fpr=float(rates[selected + 1]) if selected + 1 < len(GRID) else np.nan,
                budget_status='defined')


def build_budget_lookup(entries):
    for entry in entries:
        entry.update(rank_setup(entry['prepared'], entry['y'], entry['valid']))
    rows, operating = [], []
    for farm in sorted({e['farm_id'] for e in entries}):
        sequences = [e for e in entries if e['farm_id'] == farm]
        for budget in BUDGETS:
            chosen = select_operating_point(sequences, budget)
            if chosen['selected_index'] is None:
                raise ValueError(f'{farm}: no valid negative points for budget calibration')
            tail = chosen['selected_tail_fraction']
            actual_count = 0
            for entry in sequences:
                y, valid, prepared = entry['y'], entry['valid'], entry['prepared']
                nominal, threshold = threshold_at(entry, tail)
                alarms, releases = evaluate_threshold(prepared, threshold)
                fp = int(np.count_nonzero(valid & (y == 0) & alarms.astype(bool)))
                actual_count += fp
                assert int(alarms.sum()) == int(entry['effective'][nominal])
                assert not alarms[~prepared['available']].any()
                measured = early_detection_rate(y, alarms, warning_predictions=releases * valid,
                                                horizon_points=288, lead_delta_points=72)
                rows.append(dict(file_name=entry['file_name'], farm_id=farm,
                                 budget_normal_point_fpr=budget, threshold=threshold,
                                 selected_tail_fraction=tail, early_detection_rate=measured))
            assert actual_count == chosen['normal_alert_points']
            operating.append(dict(farm_id=farm, budget_normal_point_fpr=budget,
                                  grid_size=len(GRID), **chosen))
    frame = pd.DataFrame(rows)
    return frame.set_index(['file_name', 'budget_normal_point_fpr']), pd.DataFrame(operating)
