from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from tsad_benchmark.common.operating_points import TRACK1_ANOMALY_RATIOS

logger = logging.getLogger(__name__)
_POINTS_PER_DAY = 144.0
_SERIES_COL = "file_name"

_RATIO_AGG = np.nanmean


DETECTION_METRICS: List[str] = [
    "accuracy",
    "point_f1",
    "point_precision",
    "point_recall",
    "event_f1",
    "event_precision",
    "event_recall",
    "event_affiliation_f1",
    "range_f1",
    "range_precision",
    "range_recall",
    "affiliation_f1",
    "affiliation_precision",
    "affiliation_recall",
    "early_detection_rate",
    "early_detection_rate_all",
    "mean_lead_time",
    "actionable_lead_time",
    "mean_detection_delay",
]

OPERATIONAL_METRICS: List[str] = [
    "false_alarms_per_turbine_day",
    "mtbfa",
]


SCORE_METRICS: List[str] = [
    "auc_pr",
    "auc_roc",
    "range_auc_pr",
    "range_auc_roc",
    "vus_pr",
    "vus_roc",
]

EFFICIENCY_FIELDS: List[str] = [
    "fit_time",
    "fit_gpu_peak_memory_mb",
    "fit_cpu_usage_percent",
    "inference_time",
    "inference_gpu_peak_memory_mb",
    "inference_cpu_usage_percent",
    "n_params",
    "model_size_mb",
]


RANK_METRIC = "event_f1"


_RANK_PRIORITY: List[str] = [
    "event_f1",
    "affiliation_f1",
    "point_f1",
    "vus_pr",
    "auc_pr",
    "range_auc_pr",
    "vus_roc",
    "auc_roc",
]


def _pick_rank_metric(df: pd.DataFrame, override: Optional[str] = None) -> Optional[str]:
    if override and override in df.columns:
        return override
    for m in _RANK_PRIORITY:
        if m in df.columns:
            return m
    return None


def _default_metrics(df: pd.DataFrame) -> List[str]:
    return [
        m for m in (DETECTION_METRICS + OPERATIONAL_METRICS + SCORE_METRICS)
        if m in df.columns
    ]


_ANOMALY_SPAN_BINS = [0, 500, 1500, float("inf")]
_ANOMALY_SPAN_LABELS = ["short", "medium", "long"]


def _series_group_cols(
    df: pd.DataFrame,
    partition_cols: Sequence[str] = (),
) -> List[str]:
    cols: List[str] = []
    for c in partition_cols:
        if c in df.columns and c not in cols:
            cols.append(c)
    for c in ("model_name", "model_params", _SERIES_COL):
        if c in df.columns and c not in cols:
            cols.append(c)
    return cols


def collapse_ratios_per_series(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    partition_cols: Sequence[str] = (),
) -> pd.DataFrame:
    if df.empty or _SERIES_COL not in df.columns:
        return df.copy()

    keys = _series_group_cols(df, partition_cols)
    metrics = [c for c in value_cols if c in df.columns]
    if not metrics:
        return df.copy()

    work = _filter_operating_points(df)
    for c in metrics:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    meta_cols = [
        c
        for c in work.columns
        if c not in keys and c not in metrics and c != "typical_anomaly_ratio"
    ]
    agg: Dict[str, object] = {c: _RATIO_AGG for c in metrics}
    for c in meta_cols:
        agg[c] = "first"

    return work.groupby(keys, sort=False, dropna=False).agg(agg).reset_index()


def _filter_operating_points(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if 'threshold_method' in work:
        methods = set(work['threshold_method'].dropna().unique())
        if 'train_percentile_v1' in methods and 'train_available_percentile_v2' in methods:
            raise ValueError('Do not merge legacy and online-v2 runs into one report.')
    if "typical_anomaly_ratio" in work.columns and "threshold_method" in work.columns:
        ratios = pd.to_numeric(work["typical_anomaly_ratio"], errors="coerce")
        shared_grid = work["threshold_method"].isin([
            "train_percentile_v1", "train_available_percentile_v2",
        ])
        work = work[~shared_grid | ratios.isna() | ratios.isin(TRACK1_ANOMALY_RATIOS)].copy()
    return work


def aggregate_model_metrics(
    df: pd.DataFrame,
    metrics: Sequence[str],
    partition_cols: Sequence[str] = (),
) -> pd.DataFrame:
    metrics = [m for m in metrics if m in df.columns]
    if not metrics or "model_name" not in df.columns:
        return pd.DataFrame()

    collapsed = collapse_ratios_per_series(df, metrics, partition_cols)
    group_cols = ["model_name"] + [
        c for c in partition_cols if c in collapsed.columns
    ]
    from tsad_benchmark.report.care import is_current
    if is_current(df) and 'farm_id' in collapsed and 'farm_id' not in group_cols:
        collapsed = collapsed.groupby(group_cols + ['farm_id'], sort=False)[metrics].mean().reset_index()
    return collapsed.groupby(group_cols, sort=False)[metrics].mean().reset_index()


def _safe_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    out = df.copy()
    for c in present:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _is_normal_event(df: pd.DataFrame) -> pd.Series:
    if "event_label" not in df.columns:
        return pd.Series(False, index=df.index)
    return (
        df["event_label"].astype(str).str.lower().str.strip().eq("normal")
    )


def _event_points(group: pd.DataFrame) -> pd.Series:
    if "total_points" in group.columns:
        return pd.to_numeric(group["total_points"], errors="coerce")
    if "test_lens" in group.columns:
        return pd.to_numeric(group["test_lens"], errors="coerce")
    return pd.Series(np.nan, index=group.index, dtype=float)


def _total_points(group: pd.DataFrame) -> float:
    pts = _event_points(group)
    valid = pts[pts > 0]
    return float(valid.sum()) if not valid.empty else np.nan


def _total_days(group: pd.DataFrame) -> float:
    if "turbine_days" in group.columns:
        days = pd.to_numeric(group["turbine_days"], errors="coerce")
        valid = days[days >= 0]
        if not valid.empty:
            return float(valid.sum())
    points = _total_points(group)
    if pd.isna(points):
        return np.nan
    return float(points / _POINTS_PER_DAY)


def _weighted_mean(group: pd.DataFrame, metric: str) -> float:
    if metric == "accuracy" and {"correct_points", "total_points"}.issubset(group.columns):
        correct = pd.to_numeric(group["correct_points"], errors="coerce")
        total = pd.to_numeric(group["total_points"], errors="coerce")
        mask = correct.notna() & total.notna() & (total > 0)
        if mask.any():
            return float(correct[mask].sum() / total[mask].sum())
    vals = pd.to_numeric(group[metric], errors="coerce")
    pts = _event_points(group)
    mask = vals.notna() & pts.notna() & (pts > 0)
    if mask.any():
        return float((vals[mask] * pts[mask]).sum() / pts[mask].sum())
    return float(vals.mean()) if vals.notna().any() else np.nan


def _false_alarm_events(group: pd.DataFrame) -> float:
    if "false_alarm_events" in group.columns:
        counts = pd.to_numeric(group["false_alarm_events"], errors="coerce")
        valid = counts[counts >= 0]
        if not valid.empty:
            return float(valid.sum())
    if "false_alarms_per_turbine_day" not in group.columns:
        return np.nan
    rate = pd.to_numeric(group["false_alarms_per_turbine_day"], errors="coerce")
    days = _event_points(group) / _POINTS_PER_DAY
    mask = rate.notna() & days.notna() & (days > 0)
    if not mask.any():
        return np.nan
    return float((rate[mask] * days[mask]).sum())


def _false_alarm_rate(group: pd.DataFrame) -> float:
    count = _false_alarm_events(group)
    days = _total_days(group)
    if pd.notna(count) and pd.notna(days) and days > 0:
        return float(count / days)
    if "false_alarms_per_turbine_day" not in group.columns:
        return np.nan
    vals = pd.to_numeric(group["false_alarms_per_turbine_day"], errors="coerce")
    return float(vals.mean()) if vals.notna().any() else np.nan


def _mtbfa_points(group: pd.DataFrame) -> float:
    if "mtbfa" not in group.columns or "false_alarm_events" not in group.columns:
        return np.nan
    vals = pd.to_numeric(group["mtbfa"], errors="coerce")
    counts = pd.to_numeric(group["false_alarm_events"], errors="coerce")
    weights = (counts - 1).clip(lower=0)
    mask = vals.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float((vals[mask] * weights[mask]).sum() / weights[mask].sum())


def _mtbfa_days(group: pd.DataFrame) -> float:
    points = _mtbfa_points(group)
    return float(points / _POINTS_PER_DAY) if pd.notna(points) else np.nan


def _mtbfa_points_across_operating_points(group: pd.DataFrame) -> float:
    if group.empty:
        return np.nan
    if "typical_anomaly_ratio" not in group.columns:
        return _mtbfa_points(group)

    work = group.copy()
    work["_operating_point"] = pd.to_numeric(
        work["typical_anomaly_ratio"], errors="coerce"
    )
    values = [
        _mtbfa_points(op_group)
        for _, op_group in work.groupby("_operating_point", sort=False, dropna=False)
    ]
    valid = pd.Series(values, dtype=float).dropna()
    return float(valid.mean()) if not valid.empty else np.nan


def _mtbfa_days_across_operating_points(group: pd.DataFrame) -> float:
    points = _mtbfa_points_across_operating_points(group)
    return float(points / _POINTS_PER_DAY) if pd.notna(points) else np.nan


def _agg_metrics(
    group: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.Series:
    result = {}
    for m in metrics:
        if m in group.columns:
            if m == "accuracy":
                result[m] = _weighted_mean(group, m)
            elif m == "false_alarms_per_turbine_day":
                result[m] = _false_alarm_rate(group)
            elif m == "mtbfa":
                result[m] = _mtbfa_points(group)
            else:
                result[m] = pd.to_numeric(group[m], errors="coerce").mean()
    if _SERIES_COL in group.columns:
        result["n_series"] = int(group[_SERIES_COL].nunique())
    else:
        result["n_series"] = len(group)
    return pd.Series(result)


def _leaderboard_from_collapsed(
    collapsed: pd.DataFrame,
    metrics: Sequence[str],
    partition_cols: Sequence[str],
    rank_by: Optional[str] = None,
) -> pd.DataFrame:
    metrics = [m for m in metrics if m in collapsed.columns]
    if not metrics or collapsed.empty:
        return pd.DataFrame()

    group_cols = ["model_name"] + [
        c for c in partition_cols if c in collapsed.columns
    ]
    rows: List[dict] = []
    for keys, g in collapsed.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = _agg_metrics(g, metrics).to_dict()
        for col, val in zip(group_cols, keys):
            row[col] = val
        rows.append(row)

    board = pd.DataFrame(rows)
    if board.empty:
        return board
    rank_by = _pick_rank_metric(board, rank_by)
    if rank_by and rank_by in board.columns:
        sort_cols = list(partition_cols) + [rank_by]
        sort_cols = [c for c in sort_cols if c in board.columns]
        ascending = [True] * len(partition_cols) + [False]
        board.sort_values(sort_cols, ascending=ascending, inplace=True)
        if not partition_cols:
            board.insert(0, "rank", range(1, len(board) + 1))
    return board


def _operational_scope_values(group: pd.DataFrame, prefix: str) -> Dict[str, float]:
    points = _total_points(group)
    days = _total_days(group)
    return {
        f"{prefix}_points": points,
        f"{prefix}_days": days,
        f"accuracy_{prefix}": (
            _weighted_mean(group, "accuracy") if "accuracy" in group.columns else np.nan
        ),
        f"false_alarm_events_{prefix}": _false_alarm_events(group),
        f"false_alarms_per_turbine_day_{prefix}": _false_alarm_rate(group),
        f"mtbfa_{prefix}_days": _mtbfa_days(group),
    }


def _build_operational_summary(
    df: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    if not set(group_cols).issubset(df.columns):
        return pd.DataFrame()

    numeric_cols = [
        "accuracy",
        "false_alarms_per_turbine_day",
        "mtbfa",
        "test_lens",
        "correct_points",
        "total_points",
        "false_alarm_events",
        "turbine_days",
    ]
    df = _filter_operating_points(_safe_numeric(df, numeric_cols))
    rows: List[dict] = []
    for keys, raw_group in df.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group = collapse_ratios_per_series(raw_group, numeric_cols, group_cols)
        row = {col: val for col, val in zip(group_cols, keys)}
        normal = group[_is_normal_event(group)]
        raw_normal = raw_group[_is_normal_event(raw_group)]
        row["n_events"] = int(len(group))
        row["n_anomaly_events"] = int(len(group) - len(normal))
        row["n_normal_events"] = int(len(normal))
        row.update(_operational_scope_values(group, "all"))
        row.update(_operational_scope_values(normal, "normal"))
        row["mtbfa_all_days"] = _mtbfa_days_across_operating_points(raw_group)
        row["mtbfa_normal_days"] = _mtbfa_days_across_operating_points(raw_normal)
        rows.append(row)

    return pd.DataFrame(rows)


def _build_overall(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    rank_by: Optional[str] = None,
) -> pd.DataFrame:
    metrics = metrics or _default_metrics(df)
    df = _safe_numeric(df, metrics)
    collapsed = collapse_ratios_per_series(df, metrics, ())
    return _leaderboard_from_collapsed(collapsed, metrics, (), rank_by=rank_by)


def build_overall_leaderboard(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    rank_by: Optional[str] = None,
) -> pd.DataFrame:
    if "event_label" not in df.columns:
        return _build_overall(df, metrics=metrics, rank_by=rank_by)
    is_anomaly = df["event_label"].astype(str).str.lower().str.strip() != "normal"
    sub = df[is_anomaly]
    if sub.empty:
        return _build_overall(df, metrics=metrics, rank_by=rank_by)
    return _build_overall(sub, metrics=metrics, rank_by=rank_by)


def build_overall_leaderboard_with_normal(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    rank_by: Optional[str] = None,
) -> pd.DataFrame:
    return _build_overall(df, metrics=metrics, rank_by=rank_by)


def build_overall_leaderboard_anomaly_only(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    rank_by: Optional[str] = None,
) -> pd.DataFrame:
    return build_overall_leaderboard(df, metrics=metrics, rank_by=rank_by)


def build_by_farm_leaderboard(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    metrics = metrics or _default_metrics(df)
    df = _safe_numeric(df, metrics)
    collapsed = collapse_ratios_per_series(df, metrics, ["farm_id"])

    robustness = [
        m for m in ("accuracy", "false_alarms_per_turbine_day", "mtbfa") if m in metrics
    ]
    detection = [m for m in metrics if m not in robustness]
    board = _leaderboard_from_collapsed(collapsed, detection, ["farm_id"])
    if not robustness:
        return board

    if "event_label" in collapsed.columns:
        normal = collapsed[_is_normal_event(collapsed)]
        if normal.empty:
            normal = collapsed
    else:
        normal = collapsed
    board_op = _leaderboard_from_collapsed(normal, robustness, ["farm_id"])
    if board.empty:
        return board_op
    if board_op.empty:
        return board
    keys = [c for c in ("model_name", "farm_id") if c in board.columns and c in board_op.columns]
    op_cols = keys + [c for c in robustness if c in board_op.columns]
    return board.merge(board_op[op_cols], on=keys, how="left")


def build_operational_summary(df: pd.DataFrame) -> pd.DataFrame:
    from tsad_benchmark.report.care import is_current, main_tables
    if is_current(df):
        return main_tables(df)['operational_summary']
    return _build_operational_summary(df, ["model_name"])


def build_by_farm_operational_summary(df: pd.DataFrame) -> pd.DataFrame:
    from tsad_benchmark.report.care import is_current, main_tables
    if is_current(df):
        return main_tables(df)['by_farm_operational_summary']
    return _build_operational_summary(df, ["model_name", "farm_id"])


def build_by_event_label_leaderboard(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    metrics = metrics or _default_metrics(df)
    df = _safe_numeric(df, metrics)
    collapsed = collapse_ratios_per_series(df, metrics, ["event_label"])
    return _leaderboard_from_collapsed(collapsed, metrics, ["event_label"])


def build_by_event_length_leaderboard(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    bins: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    bins = bins or _ANOMALY_SPAN_BINS
    labels = labels or _ANOMALY_SPAN_LABELS
    metrics = metrics or _default_metrics(df)
    df = _safe_numeric(df, metrics + ["anomaly_span_len"])
    df = df.copy()
    if "anomaly_span_len" not in df.columns:
        df["anomaly_span_len"] = np.nan

    is_normal = df.get("event_label", pd.Series(dtype=str)).astype(str).str.lower().str.strip() == "normal"

    buckets = pd.Series(index=df.index, dtype="object")
    buckets[is_normal] = "normal"

    anomaly_mask = ~is_normal & df["anomaly_span_len"].notna()
    if anomaly_mask.any():
        buckets[anomaly_mask] = pd.cut(
            df.loc[anomaly_mask, "anomaly_span_len"],
            bins=bins,
            labels=labels,
            right=True,
        ).astype(str)

    buckets[buckets.isna()] = "unknown"
    df["length_bucket"] = buckets

    collapsed = collapse_ratios_per_series(df, metrics, ["length_bucket"])
    board = _leaderboard_from_collapsed(collapsed, metrics, ["length_bucket"])
    if board.empty:
        return board
    bucket_order = {
        "normal": 0,
        **{l: i + 1 for i, l in enumerate(labels)},
        "unknown": len(labels) + 1,
    }
    board["_sort"] = board["length_bucket"].map(bucket_order).fillna(99)
    rank_by = _pick_rank_metric(board)
    sort_cols = ["_sort"] + ([rank_by] if rank_by else [])
    sort_order = [True] + ([False] if rank_by else [])
    board.sort_values(sort_cols, ascending=sort_order, inplace=True)
    board.drop(columns=["_sort"], inplace=True)
    return board


def build_efficiency_report(
    df: pd.DataFrame,
    fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    fields = fields or EFFICIENCY_FIELDS
    present = [f for f in fields if f in df.columns]
    if not present:
        return pd.DataFrame()
    df = _safe_numeric(df, present)

    keys = _series_group_cols(df)
    collapsed = (
        df.groupby(keys, sort=False, dropna=False)[present].mean().reset_index()
        if _SERIES_COL in df.columns else df
    )
    agg = {}
    for f in present:
        agg[f + "_mean"] = (f, "mean")
        agg[f + "_std"] = (f, "std")
    return collapsed.groupby("model_name", sort=False).agg(**agg).reset_index()


def build_all_tables(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    event_length_bins: Optional[List[float]] = None,
    event_length_labels: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    from tsad_benchmark.report.care import is_current, main_tables
    if is_current(df):
        return main_tables(df)
    return {

        "overall_leaderboard": build_overall_leaderboard(df, metrics),

        "overall_leaderboard_with_normal": build_overall_leaderboard_with_normal(
            df, metrics
        ),
        "operational_summary": build_operational_summary(df),
        "by_farm_leaderboard": build_by_farm_leaderboard(df, metrics),
        "by_farm_operational_summary": build_by_farm_operational_summary(df),
        "by_event_label_leaderboard": build_by_event_label_leaderboard(df, metrics),
        "by_event_length_leaderboard": build_by_event_length_leaderboard(
            df, metrics, bins=event_length_bins, labels=event_length_labels,
        ),
        "efficiency_report": build_efficiency_report(df),
        "warning_support_by_farm": build_warning_support(df),
        "score_support_by_farm": build_score_support(df),
    }


def build_warning_support(df: pd.DataFrame) -> pd.DataFrame:
    fields = [name for name in (
        'eligible_event_count', 'early_detection_count', 'lead_support_count',
        'actionable_lead_support_count',
    ) if name in df]
    if not fields:
        return pd.DataFrame()
    collapsed = collapse_ratios_per_series(df, fields, ['farm_id'])
    keys = [name for name in ('model_name', 'farm_id') if name in collapsed]
    return collapsed.groupby(keys, dropna=False)[fields].sum(min_count=1).reset_index()


def build_score_support(df: pd.DataFrame) -> pd.DataFrame:
    fields = [name for name in SCORE_METRICS if name in df]
    if not fields:
        return pd.DataFrame()
    collapsed = collapse_ratios_per_series(df, fields, ['farm_id'])
    if 'event_label' in collapsed:
        collapsed = collapsed[~_is_normal_event(collapsed)]
    keys = [name for name in ('model_name', 'farm_id') if name in collapsed]
    grouped = collapsed.groupby(keys, dropna=False)
    counts = grouped[fields].count().rename(columns={name: name+'_event_support' for name in fields})
    counts['n_anomaly_series'] = grouped.size()
    return counts.reset_index()
