"""Compute deferred VUS-PR/VUS-ROC from saved cross-domain scores."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.cross_domain.aggregation import write_result_tables
from experiments.cross_domain.artifacts import (
    artifact_inventory,
    atomic_json,
    sha256_file,
)
from tsad_benchmark.evaluation.metrics import METRICS, clear_score_metric_cache


VUS_METRICS = ("vus_pr", "vus_roc")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inside_run(run_dir: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("prediction_path must be a non-empty relative path.")
    path = (run_dir / relative).resolve()
    try:
        path.relative_to(run_dir)
    except ValueError as error:
        raise ValueError(f"Prediction path escapes run directory: {relative}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _valid_mask(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).to_numpy(dtype=bool)
    normalized = values.astype("string").str.strip().str.lower()
    valid_tokens = {"true", "false", "1", "0"}
    observed = set(normalized.dropna())
    if not observed.issubset(valid_tokens):
        raise ValueError(f"score_valid contains unexpected values: {sorted(observed)}")
    return normalized.isin({"true", "1"}).to_numpy(dtype=bool)


def _compute_task(task: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(task["prediction_path"]))
    expected_sha256 = str(task.get("prediction_sha256") or "")
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise ValueError(f"Prediction digest mismatch: {path}")

    frame = pd.read_csv(path)
    required = {"label", "score", "score_valid"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    valid = _valid_mask(frame["score_valid"])
    if not valid.any():
        raise ValueError(f"{path}: no valid scores")
    labels = pd.to_numeric(frame.loc[valid, "label"], errors="raise").to_numpy()
    scores = pd.to_numeric(frame.loc[valid, "score"], errors="raise").to_numpy()
    if not np.isfinite(scores).all():
        raise ValueError(f"{path}: valid scores contain non-finite values")
    if not np.isin(labels, (0, 1)).all() or not np.any(labels == 1):
        raise ValueError(f"{path}: anomaly score row has invalid labels")

    clear_score_metric_cache()
    try:
        values = {name: float(METRICS[name](labels, scores)) for name in VUS_METRICS}
    finally:
        clear_score_metric_cache()
    return {
        "row_index": int(task["row_index"]),
        **values,
    }


def _refresh_manifest(run_dir: Path, summary: Mapping[str, Any]) -> None:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    postprocessing = dict(manifest.get("postprocessing") or {})
    postprocessing["score_vus"] = dict(summary)
    manifest["postprocessing"] = postprocessing
    manifest["artifacts"] = artifact_inventory(run_dir, exclude="run_manifest.json")
    atomic_json(manifest, path)


def recompute_score_vus(
    run_dir: Path,
    *,
    workers: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    results_dir = run_dir / "results"
    per_event_path = results_dir / "per_event.csv"
    if not per_event_path.is_file():
        raise FileNotFoundError(per_event_path)
    workers = max(1, int(workers))
    started_utc = _timestamp()
    started = time.perf_counter()

    try:
        state_path = run_dir / "state.json"
        if state_path.exists():
            with state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
            if state.get("status") != "completed":
                raise RuntimeError("VUS postprocessing requires a completed run.")

        per_event = pd.read_csv(per_event_path)
        required = {
            "track",
            "target_event_label",
            "prediction_path",
            "prediction_sha256",
            *VUS_METRICS,
        }
        missing = sorted(required - set(per_event.columns))
        if missing:
            raise ValueError(f"per_event.csv is missing columns: {missing}")
        if "score_metric_status" not in per_event.columns:
            per_event["score_metric_status"] = np.where(
                per_event["target_event_label"].astype(str).str.lower().eq("normal")
                & per_event["track"].eq("score"),
                "not_applicable",
                "",
            )

        eligible = per_event["track"].eq("score") & per_event[
            "target_event_label"
        ].astype(str).str.strip().str.lower().eq("anomaly")
        tasks: list[dict[str, Any]] = []
        skipped_rows = 0
        for row_index in per_event.index[eligible]:
            row = per_event.loc[row_index]
            already_complete = str(row.get("score_metric_status", "")) == "complete"
            if not already_complete:
                already_complete = all(pd.notna(row[name]) for name in VUS_METRICS)
            if already_complete and not force:
                per_event.at[row_index, "score_metric_status"] = "complete"
                skipped_rows += 1
                continue
            path = _inside_run(run_dir, row["prediction_path"])
            tasks.append(
                {
                    "row_index": int(row_index),
                    "prediction_path": str(path),
                    "prediction_sha256": row["prediction_sha256"],
                }
            )

        if workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                computed = list(executor.map(_compute_task, tasks))
        else:
            computed = [_compute_task(task) for task in tasks]

        for result in computed:
            row_index = int(result["row_index"])
            for name in VUS_METRICS:
                per_event.at[row_index, name] = result[name]
            per_event.at[row_index, "score_metric_status"] = "complete"

        per_asset, per_direction = write_result_tables(run_dir, per_event)
        for legacy_name in ("vus_recompute.csv", "vus_status.json"):
            (results_dir / legacy_name).unlink(missing_ok=True)
        summary = {
            "status": "completed",
            "started_utc": started_utc,
            "completed_utc": _timestamp(),
            "workers": workers,
            "force": bool(force),
            "eligible_rows": int(eligible.sum()),
            "computed_rows": len(computed),
            "skipped_rows": skipped_rows,
            "turbine_rows": len(per_asset),
            "direction_rows": len(per_direction),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        _refresh_manifest(run_dir, summary)
        return summary
    except BaseException as error:
        failure = {
            "status": "failed",
            "started_utc": started_utc,
            "failed_utc": _timestamp(),
            "workers": workers,
            "force": bool(force),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _refresh_manifest(run_dir, failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1)
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = recompute_score_vus(
        args.run_dir,
        workers=args.workers,
        force=args.force,
    )
    print(
        f"VUS completed: {summary['computed_rows']} computed, "
        f"{summary['skipped_rows']} skipped. Results: {args.run_dir.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
