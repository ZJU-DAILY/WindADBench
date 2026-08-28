# -*- coding: utf-8 -*-
"""Recompute metric columns in result CSVs using current metric implementations.

Usage:
    python scripts/recompute_metrics.py results/non_learning/score/lof.csv
    python scripts/recompute_metrics.py results/non_learning/score/lof.csv --metrics vus_pr vus_roc --workers 8
    python scripts/recompute_metrics.py results/non_learning/score/lof.csv --metrics vus_pr vus_roc --workers 8 --report
    python scripts/recompute_metrics.py --all
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import logging
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tsad_benchmark.evaluation.metrics import (  # noqa: E402
    METRICS,
    clear_score_metric_cache,
    classification_metrics_score,
    classification_metrics_label,
)

logger = logging.getLogger("recompute_metrics")
SCORE_METRICS: List[str] = classification_metrics_score.__all__
LABEL_METRICS: List[str] = classification_metrics_label.__all__
NORMAL_EVENT_METRICS = {
    "accuracy", "false_alarms_per_turbine_day", "mtbfa",
    "correct_points", "total_points", "false_alarm_events", "turbine_days",
}
DROPPED_COLS = {"point_adjusted_f1"}
HEAD_COLS = ["model_name", "strategy_args", "model_params"]


def _decode(b64) -> object:
    if pd.isna(b64) or not isinstance(b64, str):
        return None
    try:
        return pickle.loads(base64.b64decode(b64.encode("utf-8")))
    except Exception:
        return None


def _detect_track(path: Path) -> str:
    s = str(path).lower().replace("\\", "/")
    if "/score/" in s:
        return "score"
    if "/label/" in s:
        return "label"
    raise ValueError(f"Cannot infer track from path (need /score/ or /label/): {path}")


def _to_1d(arr) -> np.ndarray:
    if isinstance(arr, pd.DataFrame):
        return arr.values.reshape(-1)
    if isinstance(arr, pd.Series):
        return arr.to_numpy().reshape(-1)
    return np.asarray(arr).reshape(-1)


def _normalise_requested_metrics(raw: List[str] | None) -> List[str] | None:
    if raw is None:
        return None
    names: List[str] = []
    for item in raw:
        for name in str(item).split(","):
            name = name.strip()
            if name:
                names.append(name)
    if len(names) == 1 and names[0].lower() == "all":
        return None
    return list(dict.fromkeys(names))


def _metric_names_for_track(track: str, requested: List[str] | None = None) -> List[str]:
    allowed = SCORE_METRICS if track == "score" else LABEL_METRICS
    if requested is None:
        return list(allowed)
    invalid = [m for m in requested if m not in allowed]
    if invalid:
        raise ValueError(f"Metrics not accepted by {track} track: {invalid}")
    return list(requested)


def _compute_row(track: str, metric_names: List[str], actual_b64, inference_b64) -> Dict[str, float]:
    clear_score_metric_cache()
    actual = _decode(actual_b64)
    inference = _decode(inference_b64)
    if actual is None or inference is None:
        return {}
    y_true = _to_1d(actual)
    pred = _to_1d(inference[0] if isinstance(inference, (list, tuple)) else inference)
    n = min(len(y_true), len(pred))
    if n == 0:
        return {}
    y_true, pred = y_true[:n], pred[:n]
    out: Dict[str, float] = {}
    for m in metric_names:
        try:
            out[m] = float(METRICS[m](y_true, pred))
        except Exception as exc:
            logger.warning("  %s failed on row: %s", m, exc)
            out[m] = float("nan")
    return out


def _compute_row_task(args) -> Tuple[int, Dict[str, float], str]:
    row_idx, track, metric_names, actual_b64, inference_b64 = args
    vals = _compute_row(track, metric_names, actual_b64, inference_b64)
    warning = "" if vals else "missing or empty actual_data/inference_data"
    return row_idx, vals, warning


def recompute_csv(
    csv_path: Path,
    metric_names: List[str] | None = None,
    workers: int = 1,
) -> Tuple[int, int]:
    track = _detect_track(csv_path)
    names = _metric_names_for_track(track, metric_names)
    df = pd.read_csv(csv_path)
    n = len(df)
    workers = max(1, int(workers))
    logger.info("[%s] %s rows  track=%s metrics=%s workers=%s", csv_path, n, track, names, workers)

    cols = {m: [] for m in names}
    ok = 0
    tasks = [
        (int(i), track, names, row.get("actual_data"), row.get("inference_data"))
        for i, row in df.iterrows()
    ]

    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            row_results = list(executor.map(_compute_row_task, tasks))
    else:
        row_results = [_compute_row_task(task) for task in tasks]
    row_results.sort(key=lambda x: x[0])

    for row_idx, vals, warning in row_results:
        if warning:
            logger.warning("  row %s skipped: %s", row_idx, warning)
        if vals:
            ok += 1
        for m in names:
            cols[m].append(vals.get(m, float("nan")))

    for m, v in cols.items():
        df[m] = v
    if "event_label" in df.columns:
        normal = df["event_label"].astype(str).str.lower().str.strip().eq("normal")
        for m in names:
            if m not in NORMAL_EVENT_METRICS and m in df.columns:
                df.loc[normal, m] = np.nan
    for c in DROPPED_COLS:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    head = [c for c in HEAD_COLS if c in df.columns]
    track_metric_order = SCORE_METRICS if track == "score" else LABEL_METRICS
    metric_block = [m for m in track_metric_order if m in df.columns]
    tail = [c for c in df.columns if c not in head and c not in metric_block]
    df = df[head + metric_block + tail]
    df.to_csv(csv_path, index=False)
    return ok, n


def _expand(paths: List[str], all_flag: bool, track: str = "all") -> List[Path]:
    targets: List[str] = list(paths)
    if all_flag:
        if track in ("all", "score"):
            targets += glob.glob("results/**/score/*.csv", recursive=True)
        if track in ("all", "label"):
            targets += glob.glob("results/**/label/*.csv", recursive=True)
    resolved = sorted({Path(t).resolve() for t in targets})
    return [p for p in resolved if p.exists()]


def _default_report_dir(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_report")


def _load_existing_summary(report_dir: Path) -> dict:
    path = report_dir / "experiment_summary.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _parse_strategy_args(csv_path: Path) -> dict:
    try:
        head = pd.read_csv(csv_path, nrows=1)
    except Exception:
        return {}
    if head.empty or "strategy_args" not in head.columns:
        return {}
    raw = head.iloc[0].get("strategy_args")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _report_run_config(csv_path: Path, report_dir: Path) -> dict:
    existing = _load_existing_summary(report_dir)
    strategy_args = _parse_strategy_args(csv_path)
    strategy_info = existing.get("strategy", {}) if isinstance(existing.get("strategy"), dict) else {}
    model_info = existing.get("models", {}) if isinstance(existing.get("models"), dict) else {}
    dataset_info = existing.get("dataset", {}) if isinstance(existing.get("dataset"), dict) else {}
    scope_info = existing.get("scope", {}) if isinstance(existing.get("scope"), dict) else {}

    if not strategy_args and strategy_info.get("name"):
        strategy_args = {"strategy_name": strategy_info.get("name")}
    return {
        "data_config": {},
        "model_config": model_info.get("config", {}),
        "evaluation_config": {
            "metrics": scope_info.get("metrics", "all"),
            "strategy_args": strategy_args,
        },
        "report_config": {},
        "dataset_root": dataset_info.get("root", str(_REPO_ROOT)),
        "series_limit": scope_info.get("series_limit"),
        "strategy_name": strategy_args.get("strategy_name") or strategy_info.get("name", "unknown"),
        "strategy_args": strategy_args,
        "metrics": scope_info.get("metrics", "all"),
    }


def generate_report_for_csv(
    csv_path: Path,
    out_dir: Path | None = None,
    export_figures: bool = True,
    elapsed_sec: float | None = None,
) -> str:
    from tsad_benchmark.report import report

    report_dir = out_dir or _default_report_dir(csv_path)
    if elapsed_sec is None:
        runtime = _load_existing_summary(report_dir).get("runtime", {})
        if isinstance(runtime, dict):
            try:
                raw_elapsed = runtime.get("elapsed_sec")
                elapsed_sec = None if raw_elapsed is None else float(raw_elapsed)
            except (TypeError, ValueError):
                elapsed_sec = None
    return report(
        result_path=str(csv_path),
        run_config=_report_run_config(csv_path, report_dir),
        out_dir=str(report_dir),
        export_figures=export_figures,
        elapsed_sec=elapsed_sec,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--all", action="store_true",
                        help="Recompute every results/**/(score|label)/*.csv.")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Metric names to recompute, comma or space separated. Defaults to all.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel row workers; 1 keeps serial execution.")
    parser.add_argument("--track", choices=["all", "score", "label"], default="all",
                        help="When --all is used, restrict discovered result CSVs by track.")
    parser.add_argument("--report", action="store_true",
                        help="Regenerate the benchmark report after recomputing each CSV.")
    parser.add_argument("--report-dir", default=None,
                        help="Report output directory for a single CSV; ignored for --all/multiple paths.")
    parser.add_argument("--no-figures", action="store_true",
                        help="When --report is used, skip chart/image generation.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    targets = _expand(args.paths, args.all, args.track)
    if not targets:
        parser.error("no existing CSV paths given (use --all or pass paths)")

    requested_metrics = _normalise_requested_metrics(args.metrics)
    report_dir_arg = Path(args.report_dir).resolve() if args.report_dir else None
    if args.report_dir and len(targets) != 1:
        logger.warning("--report-dir is only used for a single CSV; using per-file *_report dirs.")
        report_dir_arg = None

    for p in targets:
        if args.dry_run:
            logger.info("would recompute: %s", p)
            continue
        try:
            started = time.perf_counter()
            ok, n = recompute_csv(p, metric_names=requested_metrics, workers=args.workers)
            elapsed = time.perf_counter() - started
            logger.info("[%s] recomputed %s/%s rows in %.3fs", p, ok, n, elapsed)
            if args.report:
                report_dir = generate_report_for_csv(
                    p,
                    out_dir=report_dir_arg,
                    export_figures=not args.no_figures,
                )
                logger.info("[%s] report regenerated: %s", p, report_dir)
        except Exception:
            logger.exception("failed: %s", p)


if __name__ == "__main__":
    main()
