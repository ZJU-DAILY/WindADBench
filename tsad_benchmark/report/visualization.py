# -*- coding: utf-8 -*-

from __future__ import annotations

import base64
import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

_FIG_DPI = 150
_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

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
]
CORE_DETECTION_METRICS: List[str] = ["point_f1", "event_f1", "event_affiliation_f1", "affiliation_f1"]
EARLY_WARNING_METRICS: List[str] = ["early_detection_rate", "mean_detection_delay", "mean_lead_time"]
OPERATIONAL_METRICS: List[str] = ["false_alarms_per_turbine_day", "mtbfa"]
OPERATIONAL_SUMMARY_METRICS: List[str] = [
    "accuracy_all",
    "false_alarms_per_turbine_day_all",
    "mtbfa_all_days",
    "accuracy_normal",
    "false_alarms_per_turbine_day_normal",
    "mtbfa_normal_days",
]
SCORE_CURVE_METRICS: List[str] = [
    "auc_pr", "auc_roc", "range_auc_pr", "range_auc_roc", "vus_pr", "vus_roc",
]
ALL_REPORT_METRICS = (
    DETECTION_METRICS + EARLY_WARNING_METRICS + OPERATIONAL_METRICS + SCORE_CURVE_METRICS
)

_STEPS_TO_HOURS = 6.0  # 10-minute SCADA samples → hours
_HOURS_METRICS = ("mean_detection_delay", "mtbfa", "mean_lead_time")
_METRIC_DISPLAY = {
    **{m: f"{m} (h)" for m in _HOURS_METRICS},
    "accuracy": "accuracy (ratio)",
    "point_precision": "point_precision (ratio)",
    "point_recall": "point_recall (ratio)",
    "point_f1": "point_f1 (ratio)",
    "event_precision": "event_precision (ratio)",
    "event_recall": "event_recall (ratio)",
    "event_f1": "event_f1 (ratio)",
    "event_affiliation_f1": "event_affiliation_f1 (ratio)",
    "range_precision": "range_precision (ratio)",
    "range_recall": "range_recall (ratio)",
    "range_f1": "range_f1 (ratio)",
    "affiliation_precision": "affiliation_precision (ratio)",
    "affiliation_recall": "affiliation_recall (ratio)",
    "affiliation_f1": "affiliation_f1 (ratio)",
    "auc_pr": "auc_pr (ratio)",
    "auc_roc": "auc_roc (ratio)",
    "range_auc_pr": "range_auc_pr (ratio)",
    "range_auc_roc": "range_auc_roc (ratio)",
    "vus_pr": "vus_pr (ratio)",
    "vus_roc": "vus_roc (ratio)",
    "early_detection_rate": "early_detection_rate (ratio)",
    "false_alarms_per_turbine_day": "false_alarms_per_turbine_day (/turbine-day)",
    "accuracy_all": "accuracy_all (ratio)",
    "accuracy_normal": "accuracy_normal (ratio)",
    "false_alarms_per_turbine_day_all": "FA/day_all",
    "false_alarms_per_turbine_day_normal": "FA/day_normal",
    "mtbfa_all_days": "MTBFA_all (d)",
    "mtbfa_normal_days": "MTBFA_normal (d)",
    "anomaly_span_len": "anomaly_span_len (points)",
}


def _label(metric: str) -> str:
    return _METRIC_DISPLAY.get(metric, metric)


def _to_display_units(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in _HOURS_METRICS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") / _STEPS_TO_HOURS
    return out


def _ensure_mpl():
    if not HAS_MPL:
        raise ImportError("matplotlib is required.  pip install matplotlib")


def _save(fig, out_dir: str, name: str):
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart: %s", path)


def _filter_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    if "event_label" not in df.columns:
        return df
    return df[df["event_label"].astype(str).str.lower().str.strip() != "normal"]


# ========================================================================
# KPI summary card (for single-model or subset mode)
# ========================================================================

def _draw_kpi_row(cards: List, fig_axes_tuple=None):

    if not cards:
        return
    n = len(cards)
    if fig_axes_tuple is None:
        fig, axes = plt.subplots(1, n, figsize=(min(2.4 * n, 18), 2.4))
    else:
        fig, axes = fig_axes_tuple
    if n == 1:
        axes = [axes]
    for ax, (label, val, hb) in zip(axes, cards):
        color = "#2e7d32" if hb else "#c62828"
        fmt = f"{val:.4f}" if abs(val) < 10 else f"{val:.1f}"
        ax.text(0.5, 0.55, fmt, ha="center", va="center",
                fontsize=22, fontweight="bold", color=color,
                transform=ax.transAxes)
        ax.text(0.5, 0.15, label, ha="center", va="center",
                fontsize=9, color="#555", transform=ax.transAxes)
        arrow = "\u2191" if hb else "\u2193"
        ax.text(0.5, 0.88, arrow, ha="center", va="center",
                fontsize=10, color="#999", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.patch.set_facecolor("#f5f7fa")
        ax.patch.set_edgecolor("#ddd")
        ax.patch.set_linewidth(1)


def plot_kpi_summary(df: pd.DataFrame, out_dir: str):

    _ensure_mpl()
    anom = _filter_anomaly(df)
    model = df["model_name"].iloc[0] if "model_name" in df.columns else "Model"

    anomaly_specs = [
        ("event_f1",             "Event F1",        True),
        ("point_f1",             "Point F1",        True),
        ("event_affiliation_f1", "Event overlap F1", True),
        ("affiliation_f1",       "Affiliation F1",  True),
        ("early_detection_rate", "Early Det. Rate", True),
        ("mean_detection_delay", "Mean Delay (h)",  False),
    ]
    operational_specs = [
        ("false_alarms_per_turbine_day", "FA/Turb/Day", False),
        ("mtbfa",                        "MTBFA (h)",   True),
    ]

    def _collect(specs, source):
        cards = []
        for col, label, hb in specs:
            if col not in source.columns:
                continue
            v = source[col].mean()
            if pd.isna(v):
                continue
            cards.append((label, v, hb))
        return cards

    anom_cards = _collect(anomaly_specs, anom)
    ops_cards = _collect(operational_specs, df)

    n_anom = len(anom_cards)
    n_ops = len(ops_cards)
    if n_anom == 0 and n_ops == 0:
        return

    n_cols = max(n_anom, n_ops, 1)
    fig, all_axes = plt.subplots(2, n_cols, figsize=(min(2.4 * n_cols, 18), 5.2))

    # top row: anomaly KPIs
    for i in range(n_cols):
        ax = all_axes[0, i] if n_cols > 1 else all_axes[0]
        if i < n_anom:
            label, val, hb = anom_cards[i]
            _draw_kpi_row([(label, val, hb)], (fig, ax))
        else:
            ax.axis("off")

    # bottom row: operational KPIs
    for i in range(n_cols):
        ax = all_axes[1, i] if n_cols > 1 else all_axes[1]
        if i < n_ops:
            label, val, hb = ops_cards[i]
            _draw_kpi_row([(label, val, hb)], (fig, ax))
        else:
            ax.axis("off")

    # row labels
    fig.text(0.005, 0.75, "Anomaly\nEvents", fontsize=9, color="#4C72B0",
             fontweight="bold", va="center", ha="left")
    fig.text(0.005, 0.25, "All\nEvents", fontsize=9, color="#937860",
             fontweight="bold", va="center", ha="left")

    fig.suptitle(f"{model} — Key Metrics", fontsize=12, y=1.01)
    fig.tight_layout(rect=[0.04, 0, 1, 0.97])
    _save(fig, out_dir, "A0_kpi_summary")


# ========================================================================
# Multi-model comparison charts (full mode only, n_models >= 2)
# ========================================================================

def _grouped_bar(df, group_col, metrics, title, out_dir, filename, ylabel="Score"):
    _ensure_mpl()
    present = [m for m in metrics if m in df.columns]
    if not present or df.empty:
        return
    groups = df[group_col].unique().tolist()
    n_g, n_m = len(groups), len(present)
    x = np.arange(n_g)
    width = 0.8 / max(n_m, 1)
    fig, ax = plt.subplots(figsize=(max(7, n_g * 1.6), 5))
    for i, m in enumerate(present):
        vals = []
        for g in groups:
            cell = df.loc[df[group_col] == g, m]
            if cell.empty:
                vals.append(0.0)
                continue
            val = pd.to_numeric(cell.iloc[0], errors="coerce")
            vals.append(float(val) if pd.notna(val) and np.isfinite(val) else 0.0)
        ax.bar(x + i * width, vals, width, label=_label(m),
               color=_PALETTE[i % len(_PALETTE)])
    ax.set_xticks(x + width * (n_m - 1) / 2)
    ax.set_xticklabels(groups, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, filename)


def plot_detection_quality(df, out_dir):
    from tsad_benchmark.report.aggregation import aggregate_model_metrics

    anom = _filter_anomaly(df)
    present = [m for m in CORE_DETECTION_METRICS if m in anom.columns]
    if not present:
        return
    agg = aggregate_model_metrics(anom, present)
    _grouped_bar(agg, "model_name", present,
                 "Detection Quality (anomaly events)", out_dir, "A1_detection_quality")


def plot_early_warning(df, out_dir):
    _ensure_mpl()
    anom = _filter_anomaly(df)
    if anom.empty:
        return
    from tsad_benchmark.report.aggregation import aggregate_model_metrics

    present = [m for m in EARLY_WARNING_METRICS if m in anom.columns]
    if not present:
        return
    agg = aggregate_model_metrics(anom, present)
    models = agg["model_name"].tolist()
    if not models:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    col = "early_detection_rate"
    if col in agg.columns:
        vals = agg[col].fillna(0).tolist()
        axes[0].barh(models, vals, color=_PALETTE[0])
        axes[0].set_xlabel(_label(col))
        axes[0].set_title("Early Detection Rate (higher=better)")
        axes[0].set_xlim(0, max(max(vals) * 1.15, 0.01))
        axes[0].grid(axis="x", alpha=0.3)
    col = "mean_detection_delay"
    if col in agg.columns:
        vals = agg[col].fillna(0).tolist()
        axes[1].barh(models, vals, color=_PALETTE[3])
        axes[1].set_xlabel(_label(col))
        axes[1].set_title("Mean Detection Delay (lower=better)")
        axes[1].grid(axis="x", alpha=0.3)
    fig.suptitle("Early Warning Quality (anomaly events)", fontsize=13)
    fig.tight_layout()
    _save(fig, out_dir, "A2_early_warning")


def plot_operational_cost(df, out_dir):
    from tsad_benchmark.report.aggregation import build_operational_summary

    agg = build_operational_summary(df)
    chart_specs = [
        (
            ["accuracy_all", "accuracy_normal"],
            "Operational Accuracy (all vs normal events)",
            "A3_operational_accuracy",
            "Accuracy",
        ),
        (
            ["false_alarms_per_turbine_day_all", "false_alarms_per_turbine_day_normal"],
            "False-Alarm Rate (all vs normal events)",
            "A3_false_alarm_rate",
            "False alarms / turbine-day",
        ),
        (
            ["mtbfa_all_days", "mtbfa_normal_days"],
            "Mean Time Between False Alarms (all vs normal events)",
            "A3_mtbfa",
            "Days",
        ),
    ]
    for metrics, title, filename, ylabel in chart_specs:
        present = [m for m in metrics if m in agg.columns]
        if present:
            _grouped_bar(agg, "model_name", present, title, out_dir, filename, ylabel=ylabel)


def plot_farm_comparison(df, out_dir):
    _ensure_mpl()
    if "farm_id" not in df.columns:
        return
    farms = sorted(df["farm_id"].dropna().unique())
    if len(farms) < 2:
        return
    from tsad_benchmark.report.aggregation import (
        aggregate_model_metrics,
        build_by_farm_operational_summary,
    )

    dim_specs = [
        ("detection",     CORE_DETECTION_METRICS, _filter_anomaly(df), None),
        ("early_warning", EARLY_WARNING_METRICS,  _filter_anomaly(df), None),
        ("operational",   OPERATIONAL_SUMMARY_METRICS, df, build_by_farm_operational_summary),
    ]
    for tag, metrics, sub_df, builder in dim_specs:
        present = [m for m in metrics if m in sub_df.columns]
        if builder is not None:
            sub_df = builder(sub_df)
            present = [m for m in metrics if m in sub_df.columns]
        if not present or sub_df.empty:
            continue
        agg = (
            sub_df
            if builder is not None
            else aggregate_model_metrics(sub_df, present, ["farm_id"])
        )
        models = sorted(agg["model_name"].unique())
        n_f, n_m = len(farms), len(models)
        fig, axes_row = plt.subplots(1, len(present),
                                     figsize=(5 * len(present), 5), squeeze=False)
        for ax, metric in zip(axes_row[0], present):
            x = np.arange(n_f)
            w = 0.8 / max(n_m, 1)
            for i, mdl in enumerate(models):
                vals = []
                for f in farms:
                    cell = agg.loc[
                        (agg["model_name"] == mdl) & (agg["farm_id"] == f),
                        metric,
                    ]
                    if cell.empty:
                        vals.append(0.0)
                        continue
                    val = pd.to_numeric(cell.iloc[0], errors="coerce")
                    vals.append(float(val) if pd.notna(val) and np.isfinite(val) else 0.0)
                ax.bar(x + i * w, vals, w, label=mdl,
                       color=_PALETTE[i % len(_PALETTE)])
            ax.set_xticks(x + w * (n_m - 1) / 2)
            ax.set_xticklabels([f"Farm {f}" for f in farms])
            ax.set_title(_label(metric))
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=6)
        fig.suptitle(f"Farm Comparison — {tag.replace('_', ' ').title()}", fontsize=13)
        fig.tight_layout()
        _save(fig, out_dir, f"B_farm_{tag}")


def plot_cost_effectiveness(df, out_dir,
                            x_metric="false_alarms_per_turbine_day",
                            y_metric="event_f1",
                            size_metric="inference_time"):
    _ensure_mpl()
    needed = [x_metric, y_metric, size_metric]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return
    num_df = df.copy()
    for c in needed:
        num_df[c] = pd.to_numeric(num_df[c], errors="coerce")
    from tsad_benchmark.report.aggregation import aggregate_model_metrics

    agg = aggregate_model_metrics(num_df, needed).set_index("model_name").dropna()
    if agg.empty or len(agg) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = agg[size_metric]
    s_norm = (sizes - sizes.min()) / max(sizes.max() - sizes.min(), 1e-9)
    s_pts = 80 + s_norm * 400
    for i, (model, row) in enumerate(agg.iterrows()):
        ax.scatter(row[x_metric], row[y_metric], s=s_pts[model], alpha=0.75,
                   color=_PALETTE[i % len(_PALETTE)], edgecolors="black", linewidths=0.5)
        ax.annotate(model, (row[x_metric], row[y_metric]), fontsize=8, ha="left", va="bottom")
    ax.set_xlabel(_label(x_metric))
    ax.set_ylabel(_label(y_metric))
    ax.set_title(f"Cost-Effectiveness (bubble size = {_label(size_metric)})")
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "C_cost_effectiveness")


# ========================================================================
# D. Single-event time-series drill-down (enhanced annotations)
# ========================================================================

def _decode_column(encoded: str):
    try:
        return pickle.loads(base64.b64decode(encoded.encode("utf-8")))
    except Exception:
        return None


def _load_raw_features(meta_csv_path: str, file_name: str) -> Optional[pd.DataFrame]:
    try:
        meta = pd.read_csv(meta_csv_path)
        row = meta[meta["file_name"] == file_name]
        if row.empty:
            return None
        raw_path = row.iloc[0]["raw_path"]
        if not os.path.exists(raw_path):
            return None
        return pd.read_csv(raw_path, sep=";")
    except Exception:
        return None


def _is_score_track(row: pd.Series) -> bool:
    raw = row.get("strategy_args", "")
    if isinstance(raw, dict):
        cfg = raw
    elif isinstance(raw, str):
        try:
            cfg = json.loads(raw)
        except Exception:
            return "detect_score" in raw.lower() or raw.lower().endswith("_score")
    else:
        return False
    strategy = str(cfg.get("strategy_name", "")).lower()
    return "detect_score" in strategy or strategy.endswith("_score")


def _looks_binary(values: np.ndarray) -> bool:
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return False
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False
    return bool(np.all(np.isin(np.unique(arr), [0.0, 1.0])))


def _compute_hit_stats(gt: np.ndarray, pred: np.ndarray) -> Dict[str, str]:
    """Compute TP/FP/FN point counts and first-alarm info."""
    n = min(len(gt), len(pred))
    g, p = gt[:n].astype(int), pred[:n].astype(int)
    tp = int(np.sum((g == 1) & (p == 1)))
    fp = int(np.sum((g == 0) & (p == 1)))
    fn = int(np.sum((g == 1) & (p == 0)))
    gt_total = int(g.sum())
    pred_total = int(p.sum())

    first_alarm = None
    alarm_starts = np.where(np.diff(np.concatenate([[0], p])) == 1)[0]
    if len(alarm_starts) > 0:
        first_alarm = int(alarm_starts[0])

    gt_starts = np.where(np.diff(np.concatenate([[0], g])) == 1)[0]
    gt_start = int(gt_starts[0]) if len(gt_starts) > 0 else None

    status = "—"
    if gt_total == 0:
        status = f"Normal event, {fp} false-alarm points" if fp > 0 else "Normal event, clean"
    elif first_alarm is not None and gt_start is not None:
        if first_alarm < gt_start:
            status = f"Early hit (alarm {gt_start - first_alarm} steps before GT)"
        elif first_alarm <= gt_start + 0.1 * gt_total:
            status = f"Prompt hit (alarm at step {first_alarm})"
        else:
            status = f"Late hit (alarm {first_alarm - gt_start} steps into GT)"
    elif gt_total > 0 and pred_total == 0:
        status = "Missed (no alarm raised)"

    return {
        "TP": str(tp), "FP": str(fp), "FN": str(fn),
        "First alarm": str(first_alarm) if first_alarm is not None else "—",
        "Status": status,
    }


def plot_single_event_timeseries(
    result_df: pd.DataFrame,
    out_dir: str,
    file_name: str,
    meta_csv_path: Optional[str] = None,
):
    _ensure_mpl()
    row = result_df[result_df["file_name"] == file_name]
    if row.empty:
        return
    row = row.iloc[0]

    farm_id = row.get("farm_id", "?")
    event_id = row.get("event_id", "?")
    event_label = str(row.get("event_label", "?"))
    train_lens = int(row["train_lens"]) if pd.notna(row.get("train_lens")) else None
    test_lens = int(row["test_lens"]) if pd.notna(row.get("test_lens")) else None
    span_len = int(row["anomaly_span_len"]) if pd.notna(row.get("anomaly_span_len")) else None

    # --- decode per-timestep data ---
    gt_labels = None
    pred_labels = None
    anomaly_scores = None
    if "actual_data" in row.index and pd.notna(row["actual_data"]):
        decoded = _decode_column(str(row["actual_data"]))
        if decoded is not None:
            if isinstance(decoded, pd.DataFrame):
                gt_labels = decoded.values.reshape(-1)
            elif isinstance(decoded, np.ndarray):
                gt_labels = decoded.reshape(-1)
    is_score_track = _is_score_track(row)
    if "inference_data" in row.index and pd.notna(row["inference_data"]):
        decoded = _decode_column(str(row["inference_data"]))
        if decoded is not None:
            if isinstance(decoded, (list, tuple)) and len(decoded) >= 1:
                primary = np.asarray(decoded[0]).reshape(-1)
            elif isinstance(decoded, np.ndarray):
                primary = decoded.reshape(-1)
            else:
                primary = None
            if primary is not None:
                if is_score_track or not _looks_binary(primary):
                    anomaly_scores = primary.astype(float)
                else:
                    pred_labels = primary.astype(float)

    # --- load raw feature data for the test segment ---
    raw_features = None
    feature_col = None
    feature_reason = ""
    if meta_csv_path:
        raw_df = _load_raw_features(meta_csv_path, file_name)
        if raw_df is not None and train_lens is not None:
            test_segment = raw_df.iloc[train_lens:].reset_index(drop=True)
            numeric_cols = test_segment.select_dtypes(include=[np.number]).columns.tolist()
            skip = {"label", "train_test", "asset_id", "timestamp"}
            numeric_cols = [c for c in numeric_cols if c.lower() not in skip]
            if numeric_cols:
                variances = {c: test_segment[c].var() for c in numeric_cols}
                feature_col = max(variances, key=variances.get)
                raw_features = test_segment[feature_col].values
                feature_reason = "highest-variance feature"

    has_gt = gt_labels is not None and len(gt_labels) > 0
    has_pred = pred_labels is not None and len(pred_labels) > 0
    has_score = anomaly_scores is not None and len(anomaly_scores) > 0
    has_feat = raw_features is not None and len(raw_features) > 0

    if not (has_gt or has_pred or has_score or has_feat):
        return

    n_points = max(
        len(gt_labels) if has_gt else 0,
        len(pred_labels) if has_pred else 0,
        len(anomaly_scores) if has_score else 0,
        len(raw_features) if has_feat else 0,
    )
    t = np.arange(n_points)

    # --- compute hit stats ---
    hit_stats = {}
    if has_gt and has_pred:
        hit_stats = _compute_hit_stats(gt_labels, pred_labels)

    # --- build annotation text ---
    info_lines = []
    info_lines.append(f"Event: {file_name}  |  Farm {farm_id}  |  {event_label}")
    if test_lens is not None:
        info_lines.append(f"Test length: {test_lens} steps")
    if span_len is not None:
        info_lines.append(f"GT anomaly span: {span_len} steps")
    if hit_stats:
        info_lines.append(
            f"TP={hit_stats['TP']}  FP={hit_stats['FP']}  FN={hit_stats['FN']}  "
            f"| First alarm: step {hit_stats['First alarm']}  "
            f"| {hit_stats['Status']}"
        )
    elif has_score:
        info_lines.append("Score-track output: anomaly score shown; TP/FP/FN require a threshold.")

    n_axes = 1 + int(has_feat)
    fig, axes = plt.subplots(n_axes, 1, figsize=(14, 3.5 * n_axes + 0.8),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    def _shade_intervals(ax, binary, color, alpha, label):
        if binary is None or len(binary) == 0:
            return
        diff = np.diff(np.concatenate([[0], binary.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, alpha=alpha, color=color, label=label)
            label = None

    ax_idx = 0

    # --- top: feature curve ---
    if has_feat:
        ax = axes[ax_idx]
        ax.plot(t[:len(raw_features)], raw_features, linewidth=0.6,
                color="#333333", alpha=0.8)
        ax.set_ylabel(feature_col or "Feature")
        if has_gt:
            _shade_intervals(ax, gt_labels[:len(raw_features)],
                             "#FF4444", 0.20, "GT anomaly")
        if has_pred:
            _shade_intervals(ax, pred_labels[:len(raw_features)],
                             "#4C72B0", 0.25, "Predicted alarm")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)
        title = f"Highest-variance feature: {feature_col}" if feature_reason else f"Feature: {feature_col}"
        ax.set_title(title, fontsize=10)
        ax_idx += 1

    # --- bottom: GT vs predicted labels / anomaly scores ---
    ax = axes[ax_idx]
    if has_gt:
        ax.fill_between(t[:len(gt_labels)], 0, gt_labels[:n_points].astype(float),
                        step="mid", alpha=0.4, color="#FF4444", label="GT label")
    if has_pred:
        ax.fill_between(t[:len(pred_labels)], 0, pred_labels[:n_points].astype(float) * 0.8,
                        step="mid", alpha=0.5, color="#4C72B0", label="Predicted label")
    score_ax = None
    if has_score:
        score_ax = ax.twinx()
        score_ax.plot(t[:len(anomaly_scores)], anomaly_scores[:n_points],
                      linewidth=0.8, color="#4C72B0", label="Anomaly score")
        score_ax.set_ylabel("Score")
    ax.set_ylabel("Label")
    ax.set_xlabel("Timestep (test segment)")
    ax.set_ylim(-0.05, 1.15)
    handles, labels = ax.get_legend_handles_labels()
    if score_ax is not None:
        h2, l2 = score_ax.get_legend_handles_labels()
        handles += h2
        labels += l2
    if handles:
        ax.legend(handles, labels, fontsize=7, loc="upper right")
    ax.grid(alpha=0.2)

    # --- title + annotation box ---
    fig.suptitle(info_lines[0], fontsize=11, fontweight="bold")
    if len(info_lines) > 1:
        annotation = "\n".join(info_lines[1:])
        fig.text(0.01, -0.01, annotation, fontsize=8, color="#555",
                 va="top", family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#f5f7fa", ec="#ddd"))

    fig.tight_layout()
    safe_name = str(file_name).replace(".csv", "").replace("/", "_").replace("\\", "_")
    _save(fig, out_dir, f"D_timeseries_{safe_name}")


# ========================================================================
# Batch entry
# ========================================================================

def generate_all_charts(
    df: pd.DataFrame,
    out_dir: str,
    metrics: Optional[List[str]] = None,
    meta_csv_path: Optional[str] = None,
    drilldown_events: int = 3,
    is_subset: bool = False,
):

    os.makedirs(out_dir, exist_ok=True)

    for col in ALL_REPORT_METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = _to_display_units(df)

    n_models = df["model_name"].nunique() if "model_name" in df.columns else 0
    n_farms = df["farm_id"].nunique() if "farm_id" in df.columns else 0

    # -- Always: KPI summary when single model --
    if n_models <= 1:
        plot_kpi_summary(df, out_dir)

    if not is_subset:
        # Full report charts
        if n_models >= 2:
            plot_detection_quality(df, out_dir)
            plot_early_warning(df, out_dir)
            plot_operational_cost(df, out_dir)
            plot_cost_effectiveness(df, out_dir)

        if n_farms >= 2:
            plot_farm_comparison(df, out_dir)

    # -- Always: drill-down time series --
    if "file_name" in df.columns:
        anomaly_rows = _filter_anomaly(df)
        picks = anomaly_rows.drop_duplicates(subset=["file_name"]).head(drilldown_events)
        if len(picks) < drilldown_events:
            remaining = df[~df["file_name"].isin(picks["file_name"])]
            picks = pd.concat([picks, remaining.head(drilldown_events - len(picks))])
        for _, ev_row in picks.iterrows():
            plot_single_event_timeseries(
                df, out_dir,
                file_name=ev_row["file_name"],
                meta_csv_path=meta_csv_path,
            )
