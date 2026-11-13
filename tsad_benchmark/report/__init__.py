from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def report(
    result_path: str,
    run_config: Optional[Dict[str, Any]] = None,
    out_dir: Optional[str] = None,
    *,
    export_tables: bool = True,
    export_figures: bool = True,
    save_summary: bool = True,
    export_html: bool = True,
    elapsed_sec: Optional[float] = None,
    metrics: Optional[List[str]] = None,
    meta_csv_path: Optional[str] = None,
    drilldown_events: int = 3,
    event_length_bins: Optional[List[float]] = None,
    event_length_labels: Optional[List[str]] = None,
) -> str:
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(result_path)), "report")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Loading result CSV: %s", result_path)
    df = pd.read_csv(result_path, float_precision='round_trip')
    logger.info("Loaded %d rows, %d columns.", len(df), len(df.columns))

    if meta_csv_path is None:
        root = ""
        if run_config:
            root = run_config.get("dataset_root") or ""
        candidate = os.path.join(root, "WIND_AD_META.csv")
        if os.path.exists(candidate):
            meta_csv_path = candidate
        else:
            try:
                from tsad_benchmark.common.paths import DEFAULT_META_CSV

                fallback = str(DEFAULT_META_CSV)
                if os.path.exists(fallback):
                    meta_csv_path = fallback
            except Exception:
                pass

    tables: Dict[str, pd.DataFrame] = {}
    experiment_summary: Optional[Dict[str, Any]] = None

    from tsad_benchmark.report.care import is_current
    if export_tables or is_current(df):
        from tsad_benchmark.report.aggregation import build_all_tables

        tables = build_all_tables(
            df, metrics,
            event_length_bins=event_length_bins,
            event_length_labels=event_length_labels,
        )
    if export_tables:
        tables_dir = os.path.join(out_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)
        from tsad_benchmark.report.html_report import _format_report_units

        for name, table_df in tables.items():
            if table_df is not None and not table_df.empty:
                path = os.path.join(tables_dir, f"{name}.csv")
                if name.startswith(('latest_', 'by_farm_ratio', 'macro_ratio')):
                    table_df.to_csv(path, index=False)
                else:
                    _format_report_units(table_df).to_csv(path, index=False)
                logger.info("Table saved: %s (%d rows)", path, len(table_df))

    series_limit = run_config.get("series_limit") if run_config else None
    is_subset = isinstance(series_limit, int) and series_limit > 0

    charts_dir = os.path.join(out_dir, "charts")
    if export_figures:
        try:
            from tsad_benchmark.report.visualization import generate_all_charts

            generate_all_charts(
                df, charts_dir, metrics,
                meta_csv_path=meta_csv_path,
                drilldown_events=drilldown_events,
                is_subset=is_subset,
                tables=tables,
            )
        except ImportError:
            logger.warning(
                "matplotlib not available; skipping chart generation. "
                "Install with: pip install matplotlib"
            )

    if save_summary and run_config is not None:
        from tsad_benchmark.report.experiment_log import (
            build_experiment_summary,
            save_experiment_summary,
        )

        experiment_summary = build_experiment_summary(
            df, run_config, elapsed_sec, result_path=result_path,
        )
        save_experiment_summary(experiment_summary, out_dir)

    insights: List[str] = []
    try:
        from tsad_benchmark.report.insights import generate_insights
        insights = generate_insights(df, series_limit=series_limit, tables=tables)
        for ins in insights:
            logger.info("Insight: %s", ins)
    except Exception as exc:
        logger.warning("Failed to generate insights: %s", exc)

    if export_html:
        try:
            from tsad_benchmark.report.html_report import generate_html_report

            if not tables:
                from tsad_benchmark.report.aggregation import build_all_tables
                tables = build_all_tables(
                    df, metrics,
                    event_length_bins=event_length_bins,
                    event_length_labels=event_length_labels,
                )

            generate_html_report(
                tables=tables,
                charts_dir=charts_dir,
                insights=insights,
                experiment_summary=experiment_summary,
                out_path=os.path.join(out_dir, "report.html"),
            )
        except Exception as exc:
            logger.warning("Failed to generate HTML report: %s", exc)

    logger.info("Report generation complete: %s", out_dir)
    return out_dir
