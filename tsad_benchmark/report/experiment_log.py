# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def build_experiment_summary(
    result_df: pd.DataFrame,
    run_config: Dict[str, Any],
    elapsed_sec: Optional[float] = None,
    result_path: Optional[str] = None,
) -> Dict[str, Any]:

    models = sorted(result_df["model_name"].unique().tolist()) if "model_name" in result_df.columns else []
    farms = sorted(result_df["farm_id"].dropna().unique().tolist()) if "farm_id" in result_df.columns else []
    events = sorted(result_df["event_id"].dropna().unique().tolist()) if "event_id" in result_df.columns else []
    event_labels = sorted(result_df["event_label"].dropna().unique().tolist()) if "event_label" in result_df.columns else []
    n_series = int(result_df["file_name"].nunique()) if "file_name" in result_df.columns else len(result_df)

    series_limit = run_config.get("series_limit")
    is_subset = isinstance(series_limit, int) and series_limit > 0

    strategy_args_raw = run_config.get("strategy_args") or {}

    strict_length = (
        strategy_args_raw.get("strict_length")
        if "strict_length" in strategy_args_raw
        else run_config.get("strict_length", True)
    )

    # Source traceability — hash the CSV so stale reports can be detected
    result_md5 = None
    if result_path and os.path.exists(result_path):
        h = hashlib.md5()
        with open(result_path, "rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 16), b""):
                h.update(chunk)
        result_md5 = h.hexdigest()

    summary: Dict[str, Any] = {
        "source": {
            "result_csv": os.path.abspath(result_path) if result_path else None,
            "result_rows": len(result_df),
            "result_md5": result_md5,
        },
        "dataset": {
            "name": "Wind Turbine SCADA AD Benchmark",
            "root": run_config.get("dataset_root", ""),
            "farms_used": farms,
            "n_events": len(events),
            "event_labels": event_labels,
        },
        "strategy": {
            "name": run_config.get("strategy_name", "unknown"),
            "strict_length": strict_length,
        },
        "models": {
            "names": models,
            "count": len(models),
            "config": run_config.get("model_config", {}),
        },
        "scope": {
            "n_series": n_series,
            "n_farms": len(farms),
            "n_models": len(models),
            "series_limit": series_limit,
            "is_subset": is_subset,
            "metrics": run_config.get("metrics", "all"),
        },
        "runtime": {
            "elapsed_sec": round(elapsed_sec, 2) if elapsed_sec is not None else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    return summary


def save_experiment_summary(
    summary: Dict[str, Any],
    out_dir: str,
    filename: str = "experiment_summary.json",
) -> str:
    """Write *summary* as pretty-printed JSON; return the file path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Experiment summary saved: %s", path)
    return path
