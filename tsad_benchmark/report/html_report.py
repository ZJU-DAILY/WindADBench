# -*- coding: utf-8 -*-
"""
Self-contained HTML summary report for wind-turbine AD benchmark.

Two rendering paths:
    * **subset** — compact: config + insights + drill-downs + overall table
    * **full**  — complete: all charts, all tables, full insights
"""
from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_POINTS_PER_HOUR = 6.0  # 10-minute SCADA samples
_HOUR_METRICS = {"mean_lead_time", "mean_detection_delay", "mtbfa"}
_DISPLAY_UNITS = {
    "fit_time": "s",
    "inference_time": "s",
    "fit_peak_memory_mb": "MB",
    "inference_peak_memory_mb": "MB",
    "fit_gpu_peak_memory_mb": "MB",
    "inference_gpu_peak_memory_mb": "MB",
    "fit_cpu_usage_percent": "%",
    "inference_cpu_usage_percent": "%",
    "flops": "FLOPs",
    "n_params": "count",
    "model_size_mb": "MB",
    "accuracy": "ratio",
    "point_precision": "ratio",
    "point_recall": "ratio",
    "point_f1": "ratio",
    "event_precision": "ratio",
    "event_recall": "ratio",
    "event_f1": "ratio",
    "event_affiliation_f1": "ratio",
    "range_precision": "ratio",
    "range_recall": "ratio",
    "range_f1": "ratio",
    "affiliation_precision": "ratio",
    "affiliation_recall": "ratio",
    "affiliation_f1": "ratio",
    "auc_pr": "ratio",
    "auc_roc": "ratio",
    "range_auc_pr": "ratio",
    "range_auc_roc": "ratio",
    "vus_pr": "ratio",
    "vus_roc": "ratio",
    "early_detection_rate": "ratio",
    "mean_lead_time": "h",
    "mean_detection_delay": "h",
    "false_alarms_per_turbine_day": "/turbine-day",
    "mtbfa": "h",
    "correct_points": "count",
    "total_points": "count",
    "false_alarm_events": "count",
    "turbine_days": "d",
    "all_points": "points",
    "all_days": "d",
    "normal_points": "points",
    "normal_days": "d",
    "accuracy_all": "ratio",
    "accuracy_normal": "ratio",
    "false_alarm_events_all": "count",
    "false_alarm_events_normal": "count",
    "false_alarms_per_turbine_day_all": "/turbine-day",
    "false_alarms_per_turbine_day_normal": "/turbine-day",
    "mtbfa_all_days": "d",
    "mtbfa_normal_days": "d",
    "anomaly_span_len": "points",
}


def _img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _format_report_units(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for col in out.columns:
        base = col.removesuffix("_mean").removesuffix("_std")
        if base in _HOUR_METRICS:
            out[col] = pd.to_numeric(out[col], errors="coerce") / _POINTS_PER_HOUR
        if base in _DISPLAY_UNITS:
            rename[col] = f"{base}{col[len(base):]} ({_DISPLAY_UNITS[base]})"
    return out.rename(columns=rename)


def _table_html(df: pd.DataFrame, caption: str = "", footnote: str = "") -> str:
    if df.empty:
        return f"<p><em>{caption}: no data</em></p>"
    display_df = (
        _format_report_units(df)
        .replace({float("inf"): "no false alarms", float("-inf"): "no false alarms"})
        .fillna("\u2014")
    )
    html = display_df.to_html(
        index=False, border=0, classes="data-table",
        float_format=lambda x: f"{x:.4f}" if abs(x) < 100 else f"{x:.1f}",
    )
    if caption:
        html = f"<h3>{caption}</h3>\n" + html
    if footnote:
        html += f"\n<p style='font-size:11px;color:#888;margin-top:4px'>{footnote}</p>"
    return html


_CSS = """\
<style>
  :root { --bg: #f8f9fa; --card: #fff; --accent: #4C72B0; --text: #333; }
  * { box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 0;
         background: var(--bg); color: var(--text); }
  .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
  h1 { color: var(--accent); border-bottom: 3px solid var(--accent); padding-bottom: 8px; }
  h2 { color: var(--accent); margin-top: 36px; }
  h3 { margin-top: 24px; }
  .badge { font-size: 0.48em; padding: 2px 10px; border-radius: 4px;
           vertical-align: middle; color: #fff; }
  .badge-subset { background: #e8a735; }
  .badge-full { background: #2e7d32; }
  .card { background: var(--card); border-radius: 8px; padding: 20px;
          margin: 16px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .warning { color: #c44; font-weight: 600; }
  .insights-list { list-style: none; padding: 0; }
  .insights-list li { padding: 6px 0; border-bottom: 1px solid #eee; }
  .insights-list li:last-child { border-bottom: none; }
  .chart-grid { display: flex; flex-wrap: wrap; gap: 16px; }
  .chart-grid img { max-width: 100%; border-radius: 6px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .data-table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 13px; }
  .data-table th { background: var(--accent); color: #fff; padding: 8px 10px; text-align: left; }
  .data-table td { padding: 6px 10px; border-bottom: 1px solid #e0e0e0; }
  .data-table tr:hover td { background: #f0f4ff; }
  .footer { text-align: center; color: #999; font-size: 12px; margin-top: 40px; }
</style>
"""


def _render_header(parts, scope, source):
    is_subset = scope.get("is_subset", False)
    n_series = scope.get("n_series", "?")
    series_limit = scope.get("series_limit")

    if is_subset:
        parts.append(
            f"<h1>Wind Turbine AD — Sample Report "
            f"<span class='badge badge-subset'>SUBSET limit={series_limit}</span></h1>"
        )
        parts.append(
            f"<p class='warning'>This report covers only {n_series} events "
            f"(limit={series_limit}).  Do not cite as final benchmark results.</p>"
        )
    else:
        parts.append(
            "<h1>Wind Turbine AD — Benchmark Report "
            "<span class='badge badge-full'>FULL</span></h1>"
        )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    src = ""
    if source.get("result_csv"):
        src = f" &middot; Source: <code>{os.path.basename(source['result_csv'])}</code>"
        if source.get("result_rows"):
            src += f" ({source['result_rows']} rows)"
    parts.append(f"<p style='color:#888'>Generated: {ts}{src}</p>")


def _render_config(parts, experiment_summary):
    if not experiment_summary:
        return
    strat = experiment_summary.get("strategy", {})
    scope = experiment_summary.get("scope", {})
    rt = experiment_summary.get("runtime", {})
    parts.append("<div class='card'><h2>Experiment Configuration</h2>")
    parts.append("<table class='data-table'>")
    for k, v in [
        ("Strategy", strat.get("name", "—")),
        ("Series Evaluated", scope.get("n_series", "—")),
        ("Farms", scope.get("n_farms", "—")),
        ("Models", scope.get("n_models", "—")),
        ("Wall-Clock Time", f"{rt.get('elapsed_sec', '—')}s"),
    ]:
        parts.append(f"<tr><th>{k}</th><td>{v}</td></tr>")
    parts.append("</table></div>")


def _render_insights(parts, insights):
    if not insights:
        return
    parts.append("<div class='card'><h2>Key Findings</h2>")
    parts.append("<ul class='insights-list'>")
    for ins in insights:
        parts.append(f"  <li>{ins}</li>")
    parts.append("</ul></div>")


def _render_charts(parts, charts_dir, groups):
    if not os.path.isdir(charts_dir):
        return
    chart_files = sorted(f for f in os.listdir(charts_dir) if f.endswith(".png"))
    if not chart_files:
        return
    for group_title, prefixes in groups:
        files = [f for f in chart_files if f.startswith(prefixes)]
        if not files:
            continue
        parts.append(f"<div class='card'><h2>{group_title}</h2>")
        parts.append("<div class='chart-grid'>")
        for cf in files:
            b64 = _img_to_base64(os.path.join(charts_dir, cf))
            parts.append(f"<img src='data:image/png;base64,{b64}' alt='{cf}' title='{cf}'>")
        parts.append("</div></div>")


_TABLE_FOOTNOTES = {
    "overall_leaderboard": (
        "Headline ranking, restricted to series with <code>event_label != normal</code> "
        "(TSB-AD / TimeEval convention).  Score-track, event, and early-warning "
        "metrics are undefined on positive-free normal series, so excluding normal "
        "events gives the fairest detection-quality comparison. "
        "Lead time, detection delay, and MTBFA are displayed in hours."
    ),
    "overall_leaderboard_with_normal": (
        "Robustness / false-alarm sanity check: same aggregation but including normal "
        "series.  Normal rows keep only accuracy and false-alarm metrics; detection "
        "metrics are left as NaN and skipped by pandas means. "
        "<strong>Do not use as the primary headline ranking for detection quality.</strong> "
        "Lead time, detection delay, and MTBFA are displayed in hours."
    ),
    "by_event_label_leaderboard": (
        "\u2014 = not applicable.  Early-warning metrics are undefined for normal events "
        "(no ground-truth anomaly onset).  Lead time, detection delay, and MTBFA are "
        "displayed in hours."
    ),
    "by_farm_leaderboard": (
        "Lead time, detection delay, and MTBFA are displayed in hours."
    ),
    "operational_summary": (
        "All-events and normal-only robustness metrics.  accuracy is weighted by "
        "test points; false-alarm rate is total false-alarm events divided by "
        "total turbine-days; MTBFA is the mean adjacent false-alarm onset "
        "interval.  MTBFA is displayed in days in this table."
    ),
    "by_farm_operational_summary": (
        "Per-farm all-events and normal-only robustness metrics.  accuracy is "
        "weighted by test points; false-alarm rate is total false-alarm events "
        "divided by total turbine-days; MTBFA is displayed in days."
    ),
    "by_event_length_leaderboard": (
        "Event length is displayed in points; lead time, detection delay, and MTBFA "
        "are displayed in hours."
    ),
    "efficiency_report": (
        "Training cost (fit_*) and deployment cost (inference_*) are reported separately. "
        "Memory fields are process-level RSS / CUDA-allocator peaks for the current PID; "
        "GPU memory under the NVML fallback is also restricted to the benchmark process. "
        "flops is approximate inference-only FLOPs over the test segment; "
        "n_params and model_size_mb describe the deployed artifact."
    ),
}


def _render_tables(parts, tables, names):
    for key, display in names:
        tbl = tables.get(key)
        if tbl is not None and not tbl.empty:
            parts.append("<div class='card'>")
            footnote = _TABLE_FOOTNOTES.get(key, "")
            parts.append(_table_html(tbl, display, footnote=footnote))
            parts.append("</div>")


def generate_html_report(
    tables: Dict[str, pd.DataFrame],
    charts_dir: str,
    insights: List[str],
    experiment_summary: Optional[Dict[str, Any]] = None,
    out_path: Optional[str] = None,
) -> str:
    scope = (experiment_summary or {}).get("scope", {})
    source = (experiment_summary or {}).get("source", {})
    is_subset = scope.get("is_subset", False)

    parts: List[str] = []
    badge = "Subset" if is_subset else "Full"
    parts.append("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    parts.append(f"<title>Wind Turbine AD Report ({badge})</title>")
    parts.append(_CSS)
    parts.append("</head><body><div class='container'>")

    _render_header(parts, scope, source)

    if is_subset:
        # -- Subset template: config → insights → KPI card → drill-downs → overall table --
        _render_config(parts, experiment_summary)
        _render_insights(parts, insights)
        _render_charts(parts, charts_dir, [
            ("Model Summary", ("A0_",)),
            ("Event Drill-Down", ("D_",)),
        ])
        _render_tables(parts, tables, [
            ("overall_leaderboard", "Overall Leaderboard (Anomaly Series Only)"),
            ("operational_summary", "Operational Summary"),
            ("efficiency_report", "Efficiency"),
        ])
    else:
        # -- Full template: insights → config → all charts → all tables --
        _render_insights(parts, insights)
        _render_config(parts, experiment_summary)
        _render_charts(parts, charts_dir, [
            ("Model Summary", ("A0_",)),
            ("Detection & Warning Quality", ("A1_", "A2_")),
            ("Operational Cost", ("A3_",)),
            ("Farm Comparison", ("B_",)),
            ("Cost-Effectiveness", ("C_",)),
            ("Event Drill-Down", ("D_",)),
        ])
        _render_tables(parts, tables, [
            ("overall_leaderboard", "Overall Leaderboard (Anomaly Series Only)"),
            ("overall_leaderboard_with_normal", "Robustness Leaderboard (Includes Normal Series)"),
            ("operational_summary", "Operational Summary"),
            ("by_farm_operational_summary", "Per-Farm Operational Summary"),
            ("by_farm_leaderboard", "Per-Farm Leaderboard"),
            ("by_event_label_leaderboard", "Per-Event-Label Leaderboard"),
            ("by_event_length_leaderboard", "Per-Event-Length Leaderboard"),
            ("efficiency_report", "Efficiency Report"),
        ])

    parts.append(
        "<div class='footer'>Wind Turbine SCADA AD Benchmark &mdash; auto-generated report</div>"
    )
    parts.append("</div></body></html>")

    html = "\n".join(parts)
    if out_path is None:
        out_path = os.path.join(charts_dir, "..", "report.html")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("HTML report saved: %s", out_path)
    return out_path
