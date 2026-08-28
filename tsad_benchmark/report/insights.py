# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NUMERIC_COLS = [
    "accuracy", "point_f1", "event_f1", "event_affiliation_f1", "affiliation_f1",
    "auc_pr", "auc_roc", "vus_pr", "vus_roc",
    "early_detection_rate", "mean_detection_delay", "mean_lead_time",
    "false_alarms_per_turbine_day", "mtbfa",
    "anomaly_span_len",
]


def _best(df: pd.DataFrame, metric: str, ascending: bool = False) -> Optional[str]:
    if metric not in df.columns or df[metric].dropna().empty:
        return None
    idx = df[metric].idxmin() if ascending else df[metric].idxmax()
    return str(df.loc[idx, "model_name"]) if "model_name" in df.columns else None


def _val(df: pd.DataFrame, model: str, metric: str) -> str:
    rows = df[df["model_name"] == model]
    if rows.empty or metric not in rows.columns:
        return "N/A"
    v = rows[metric].values[0]
    if pd.isna(v):
        return "N/A"
    return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"


def generate_insights(
    df: pd.DataFrame,
    series_limit: Optional[int] = None,
) -> List[str]:

    insights: List[str] = []
    df = df.copy()
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n_series = int(df["file_name"].nunique()) if "file_name" in df.columns else len(df)
    models = sorted(df["model_name"].unique()) if "model_name" in df.columns else []
    n_models = len(models)
    farms = sorted(df["farm_id"].dropna().unique()) if "farm_id" in df.columns else []
    n_farms = len(farms)

    is_subset = isinstance(series_limit, int) and series_limit > 0

    if not models:
        return ["No model results found."]

    # -- Subset warning (always first) --
    if is_subset:
        insights.append(
            f"NOTE: This is a SUBSET run (limit={series_limit}). "
            f"Results cover {n_series} of the full dataset and should "
            f"not be treated as final benchmark conclusions."
        )

    is_anomaly = (
        df.get("event_label", pd.Series(dtype=str))
        .astype(str).str.lower().str.strip() != "normal"
    )
    anom = df[is_anomaly]
    from tsad_benchmark.report.aggregation import aggregate_model_metrics

    det_cols = [c for c in ("event_f1", "early_detection_rate") if c in anom.columns]
    agg = (
        aggregate_model_metrics(anom, det_cols)
        if not anom.empty and det_cols
        else pd.DataFrame()
    )

    # -- Model performance (single model: state facts; multi: compare) --
    if not agg.empty:
        if n_models == 1:
            m = models[0]
            insights.append(
                f"{m} achieves event_f1 = {_val(agg, m, 'event_f1')} "
                f"on anomaly events."
            )
            edr = _val(agg, m, "early_detection_rate")
            if edr != "N/A":
                insights.append(
                    f"{m} early detection rate: {edr}."
                )
        else:
            best_det = _best(agg, "event_f1")
            if best_det:
                insights.append(
                    f"Best detection model: {best_det} "
                    f"(event_f1 = {_val(agg, best_det, 'event_f1')})."
                )
            best_ew = _best(agg, "early_detection_rate")
            if best_ew:
                insights.append(
                    f"Best early-warning model: {best_ew} "
                    f"(early_detection_rate = {_val(agg, best_ew, 'early_detection_rate')})."
                )

    # -- Cost-effectiveness (only meaningful with >=2 models) --
    if n_models >= 2:
        cost_cols = [
            c
            for c in ("false_alarms_per_turbine_day", "event_f1")
            if c in df.columns
        ]
        agg_all = aggregate_model_metrics(df, cost_cols) if cost_cols else pd.DataFrame()
        if "false_alarms_per_turbine_day" in agg_all.columns and "event_f1" in agg_all.columns:
            median_f1 = agg_all["event_f1"].median()
            qualified = agg_all[agg_all["event_f1"] >= median_f1]
            if not qualified.empty:
                best_cost = _best(qualified, "false_alarms_per_turbine_day", ascending=True)
                if best_cost:
                    insights.append(
                        f"Most cost-effective model: {best_cost} "
                        f"(false_alarms = {_val(qualified, best_cost, 'false_alarms_per_turbine_day')}/day, "
                        f"event_f1 = {_val(qualified, best_cost, 'event_f1')})."
                    )

    # -- Farm analysis (anomaly-only; only meaningful with >=2 farms) --
    if n_farms >= 2 and "event_f1" in anom.columns:
        from tsad_benchmark.report.aggregation import collapse_ratios_per_series

        per_series = collapse_ratios_per_series(anom, ["event_f1"], ["farm_id"])
        farm_agg = per_series.groupby("farm_id")["event_f1"].mean()
        if not farm_agg.empty:
            easiest = farm_agg.idxmax()
            hardest = farm_agg.idxmin()
            if easiest != hardest:
                insights.append(
                    f"Easiest anomaly farm: {easiest} "
                    f"(avg event_f1 = {farm_agg[easiest]:.4f}); "
                    f"hardest anomaly farm: {hardest} "
                    f"(avg event_f1 = {farm_agg[hardest]:.4f}).  "
                    f"[anomaly events only]"
                )

    # -- Event length insight (need both short and long events) --
    if "anomaly_span_len" in anom.columns and "event_f1" in anom.columns:
        from tsad_benchmark.report.aggregation import collapse_ratios_per_series

        per_ev = collapse_ratios_per_series(
            anom, ["event_f1", "anomaly_span_len"], ()
        )
        short_mask = per_ev["anomaly_span_len"] <= 500
        long_mask = per_ev["anomaly_span_len"] > 1500
        short_f1 = per_ev.loc[short_mask, "event_f1"].mean()
        long_f1 = per_ev.loc[long_mask, "event_f1"].mean()
        if not np.isnan(short_f1) and not np.isnan(long_f1):
            if long_f1 > short_f1:
                insights.append(
                    f"Long anomaly events are easier to detect "
                    f"(event_f1 {long_f1:.3f}) than short ones ({short_f1:.3f})."
                )
            else:
                insights.append(
                    f"Short anomaly events have higher detection "
                    f"(event_f1 {short_f1:.3f}) than long ones ({long_f1:.3f})."
                )

    # -- Normal event false-alarm summary --
    normal = df[~is_anomaly]
    if not normal.empty and "false_alarms_per_turbine_day" in normal.columns:
        fa_rate = normal["false_alarms_per_turbine_day"].mean()
        if not np.isnan(fa_rate):
            insights.append(
                f"Average false alarm rate on normal events: "
                f"{fa_rate:.3f} per turbine per day."
            )

    # -- Scope summary (always last) --
    scope_parts = [f"{n_models} model(s)", f"{n_series} events", f"{n_farms} farm(s)"]
    scope_str = ", ".join(scope_parts)
    if is_subset:
        scope_str += f"  [subset: limit={series_limit}]"
    insights.append(f"Scope: {scope_str}.")

    return insights
