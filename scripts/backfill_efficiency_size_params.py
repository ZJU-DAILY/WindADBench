# -*- coding: utf-8 -*-
"""Backfill model-size and parameter-count fields in existing result CSVs.

This script is for the narrow efficiency-table gap where a model run already
finished, but ``model_size_mb`` / ``n_params`` were not recorded because the
model adapter lacked the statistic hook at run time.

Default behavior:
  - updates only explicitly listed score CSVs under results/
  - fills model_size_mb for eif/kmeans/torsk/dada/units
  - fills n_params for DADA/UniTS and Merlion deep wrappers
  - writes .bak backups before modifying CSVs
  - flushes after every updated row, so interrupted runs can resume
  - regenerates export/benchmark_* CSV tables

It does not rerun scoring or metrics.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tsad_benchmark.common.paths import ROOT_DIR, resolve_config_path  # noqa: E402
from tsad_benchmark.common.random_utils import DEFAULT_SEED, fix_random_seed  # noqa: E402
from tsad_benchmark.data.series_registry import DataPool  # noqa: E402
from tsad_benchmark.data.suites.wind_series_backend import DatasetPoolImpl  # noqa: E402
from tsad_benchmark.data.wind_frame_ops import feature_columns_only  # noqa: E402
from tsad_benchmark.data.wind_sources import LocalWindFarmDetectDataSource  # noqa: E402


LOGGER = logging.getLogger(__name__)

TARGETS: dict[str, dict[str, Any]] = {
    "ae": {
        "csv": Path("deep_learning/score/ae.csv"),
        "class": "tsad_benchmark.baselines.deep_learning.ae.AutoEncoderModel",
        "fill_n_params": True,
        "fill_model_size": False,
        "needs_fit": True,
    },
    "lstmed": {
        "csv": Path("deep_learning/score/lstmed.csv"),
        "class": "tsad_benchmark.baselines.deep_learning.lstmed.LSTMEDModel",
        "fill_n_params": True,
        "fill_model_size": False,
        "needs_fit": True,
    },
    "eif": {
        "csv": Path("machine_learning/score/eif.csv"),
        "class": "tsad_benchmark.baselines.machine_learning.eif.EIFModel",
        "fill_n_params": False,
        "fill_model_size": True,
        "needs_fit": True,
    },
    "kmeans": {
        "csv": Path("machine_learning/score/kmeans.csv"),
        "class": "tsad_benchmark.baselines.machine_learning.kmeans.KMeansModel",
        "fill_n_params": False,
        "fill_model_size": True,
        "needs_fit": True,
    },
    "torsk": {
        "csv": Path("machine_learning/score/torsk.csv"),
        "class": "tsad_benchmark.baselines.machine_learning.torsk.TorskModel",
        "fill_n_params": False,
        "fill_model_size": True,
        "needs_fit": True,
    },
    "dagmm": {
        "csv": Path("machine_learning/score/dagmm.csv"),
        "class": "tsad_benchmark.baselines.machine_learning.dagmm.DAGMMModel",
        "fill_n_params": True,
        "fill_model_size": False,
        "needs_fit": True,
    },
    "deeppoint": {
        "csv": Path("machine_learning/score/deeppoint.csv"),
        "class": "tsad_benchmark.baselines.machine_learning.deeppoint.DeepPointModel",
        "fill_n_params": True,
        "fill_model_size": False,
        "needs_fit": True,
    },
    "dada": {
        "csv": Path("ts_pretrained/score/dada.csv"),
        "class": "tsad_benchmark.baselines.ts_pretrained.dada.DADAModel",
        "fill_n_params": True,
        "fill_model_size": True,
        "needs_fit": False,
    },
    "units": {
        "csv": Path("ts_pretrained/score/units.csv"),
        "class": "tsad_benchmark.baselines.ts_pretrained.units.UniTSModel",
        "fill_n_params": True,
        "fill_model_size": True,
        "needs_fit": False,
    },
}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_config_data_section(config_path: str, dataset_root: str | None) -> dict:
    resolved = resolve_config_path(config_path)
    with open(resolved, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    data_cfg = dict(cfg.get("data_config", {}))
    if dataset_root:
        data_cfg["dataset_root"] = dataset_root
    return data_cfg


def _bootstrap_data_pool(data_cfg: dict) -> None:
    raw_root = data_cfg.get("dataset_root")
    dataset_root = Path(raw_root).resolve() if raw_root else ROOT_DIR
    LOGGER.info("Initialising data source at: %s", dataset_root)
    data_source = LocalWindFarmDetectDataSource(
        str(dataset_root),
        domain_preprocessing=bool(data_cfg.get("domain_preprocessing", True)),
    )
    DataPool().register_backend(DatasetPoolImpl(data_source))


def _import_class(path: str):
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _parse_params(value: Any) -> dict:
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    if isinstance(value, dict):
        return dict(value)
    return {}


def _json_key(value: dict) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value)


def _split_train_data(series_name: str, result_row: pd.Series) -> pd.DataFrame:
    backend = DataPool().backend()
    data = backend.series_frame(str(series_name))
    if data is None or data.empty:
        raise RuntimeError(f"Series is empty or missing: {series_name}")

    split_idx = result_row.get("train_lens")
    if _is_missing(split_idx):
        meta = backend.metadata_for_series(str(series_name))
        split_idx = meta.get("train_lens") if meta is not None else None
    if _is_missing(split_idx):
        if "train_test" not in data.columns:
            raise RuntimeError(f"Cannot infer train split for {series_name}")
        split_idx = int(data["train_test"].astype(str).str.strip().eq("train").sum())
    split_idx = max(0, min(int(float(split_idx)), len(data)))
    return feature_columns_only(data.reset_index(drop=True).iloc[:split_idx].copy())


def _cleanup_torch() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _model_stats(
    model: Any,
    *,
    want_n_params: bool = True,
    want_model_size: bool = True,
) -> dict[str, float]:
    out = {"n_params": math.nan, "model_size_mb": math.nan}
    methods = []
    if want_n_params:
        methods.append(("n_params", "estimate_n_params"))
    if want_model_size:
        methods.append(("model_size_mb", "estimate_model_size_mb"))
    for field, method in methods:
        fn = getattr(model, method, None)
        if callable(fn):
            try:
                out[field] = float(fn())
            except Exception:
                out[field] = math.nan
    return out


def _pretrained_stats(
    model_name: str,
    cls: Any,
    params: dict,
    n_features: int,
    cache: dict[tuple[Any, ...], dict[str, float]],
    *,
    want_n_params: bool = True,
    want_model_size: bool = True,
) -> dict[str, float]:
    key = (
        model_name,
        _json_key(params),
        int(n_features),
        bool(want_n_params),
        bool(want_model_size),
    )
    if key in cache:
        return cache[key]
    model = cls(**params)
    if model_name == "units":
        model._model = model._load_model(int(n_features))
    elif model_name == "dada":
        model._model = model._load_model()
    else:
        raise ValueError(f"Unsupported no-fit target: {model_name}")
    stats = _model_stats(
        model,
        want_n_params=want_n_params,
        want_model_size=want_model_size,
    )
    cache[key] = stats
    del model
    _cleanup_torch()
    return stats


def _fit_stats(
    cls: Any,
    params: dict,
    train_data: pd.DataFrame,
    *,
    want_n_params: bool = True,
    want_model_size: bool = True,
) -> dict[str, float]:
    seed = params.get("seed", params.get("random_state", DEFAULT_SEED))
    fix_random_seed(None if seed is None else int(seed))
    model = cls(**params)
    model.fit(train_data)
    stats = _model_stats(
        model,
        want_n_params=want_n_params,
        want_model_size=want_model_size,
    )
    del model
    _cleanup_torch()
    return stats


def _backup_path(path: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.name}.{stamp}.bak")


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _append_report(record: dict[str, Any], report_path: Path | None) -> None:
    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([record])
    row.to_csv(
        report_path,
        mode="a",
        header=not report_path.exists(),
        index=False,
    )


def _backfill_one(
    model_name: str,
    spec: dict[str, Any],
    csv_path: Path,
    *,
    overwrite: bool,
    dry_run: bool,
    backup: bool,
    device: str | None,
    pretrained_cache: dict[tuple[Any, ...], dict[str, float]],
    flush_every: int,
    report_path: Path | None,
) -> list[dict[str, Any]]:
    if not csv_path.exists():
        LOGGER.warning("[%s] CSV missing, skipped: %s", model_name, csv_path)
        return [
            {
                "model": model_name,
                "csv": str(csv_path),
                "file_name": "",
                "status": "csv_missing",
                "n_params": math.nan,
                "model_size_mb": math.nan,
            }
        ]

    cls = _import_class(str(spec["class"]))
    df = pd.read_csv(csv_path)
    for col in ("n_params", "model_size_mb"):
        if col not in df.columns:
            df[col] = math.nan

    changed = False
    backed_up = False
    pending_writes = 0
    flush_every = max(int(flush_every), 1)
    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        fill_size = bool(spec.get("fill_model_size", True)) and (
            overwrite or _is_missing(row.get("model_size_mb"))
        )
        fill_params = bool(spec["fill_n_params"]) and (
            overwrite or _is_missing(row.get("n_params"))
        )
        if not fill_size and not fill_params:
            continue

        file_name = str(row.get("file_name", ""))
        params = _parse_params(row.get("model_params"))
        if device and model_name in {"dada", "units"}:
            params["device"] = device
        try:
            train_data = _split_train_data(file_name, row)
            if spec["needs_fit"]:
                stats = _fit_stats(
                    cls,
                    params,
                    train_data,
                    want_n_params=fill_params,
                    want_model_size=fill_size,
                )
            else:
                stats = _pretrained_stats(
                    model_name,
                    cls,
                    params,
                    train_data.shape[1],
                    pretrained_cache,
                    want_n_params=fill_params,
                    want_model_size=fill_size,
                )
            if fill_size:
                df.at[idx, "model_size_mb"] = stats["model_size_mb"]
            if fill_params:
                df.at[idx, "n_params"] = stats["n_params"]
            changed = True
            pending_writes += 1
            status = "updated"
            if not dry_run and pending_writes >= flush_every:
                if backup and not backed_up:
                    dst = _backup_path(csv_path)
                    shutil.copy2(csv_path, dst)
                    backed_up = True
                    LOGGER.info("[%s] backup written: %s", model_name, dst)
                _write_csv_atomic(df, csv_path)
                pending_writes = 0
                LOGGER.info("[%s] flushed progress through %s", model_name, file_name)
        except Exception as exc:
            stats = {"n_params": math.nan, "model_size_mb": math.nan}
            status = f"error: {type(exc).__name__}: {exc}"
            LOGGER.exception("[%s] failed on %s", model_name, file_name)

        record = {
            "model": model_name,
            "csv": str(csv_path),
            "file_name": file_name,
            "status": status,
            "n_params": stats["n_params"],
            "model_size_mb": stats["model_size_mb"],
        }
        records.append(record)
        if not dry_run:
            _append_report(record, report_path)

    if changed and not dry_run and pending_writes > 0:
        if backup and not backed_up:
            dst = _backup_path(csv_path)
            shutil.copy2(csv_path, dst)
            backed_up = True
            LOGGER.info("[%s] backup written: %s", model_name, dst)
        _write_csv_atomic(df, csv_path)
        LOGGER.info("[%s] updated CSV: %s", model_name, csv_path)
    elif not changed:
        LOGGER.info("[%s] no missing target cells found.", model_name)
    return records


def _regenerate_exports(results_dir: Path, export_dir: Path) -> None:
    from scripts.export_results_tables import export_tables

    export_tables(
        results_dir,
        performance_out=export_dir / "benchmark_results_by_farm.csv",
        efficiency_out=export_dir / "benchmark_efficiency_by_farm.csv",
        performance_wide_dir=export_dir / "benchmark_results_by_farm_wide",
        efficiency_wide_dir=export_dir / "benchmark_efficiency_by_farm_wide",
        track="score",
    )


def _validate_backfill_outputs(
    results_dir: Path,
    models: list[str],
    records: list[dict[str, Any]],
) -> list[str]:
    issues: list[str] = []
    for record in records:
        status = str(record.get("status", ""))
        if status.startswith("error") or status == "csv_missing":
            issues.append(
                f"{record.get('model')} {record.get('file_name')}: {status}"
            )

    for model_name in models:
        spec = TARGETS[model_name]
        csv_path = results_dir / spec["csv"]
        if not csv_path.exists():
            issues.append(f"{model_name}: CSV not found after backfill: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if spec.get("fill_model_size", True):
            if "model_size_mb" not in df.columns:
                issues.append(f"{model_name}: model_size_mb column is missing.")
            else:
                size = pd.to_numeric(df["model_size_mb"], errors="coerce")
                bad = size.isna() | (size <= 0)
                if bad.any():
                    issues.append(
                        f"{model_name}: {int(bad.sum())} rows still have invalid model_size_mb."
                    )

        if spec["fill_n_params"]:
            if "n_params" not in df.columns:
                issues.append(f"{model_name}: n_params column is missing.")
            else:
                n_params = pd.to_numeric(df["n_params"], errors="coerce")
                bad = n_params.isna() | (n_params <= 0)
                if bad.any():
                    issues.append(
                        f"{model_name}: {int(bad.sum())} rows still have invalid n_params."
                    )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default="unfixed_score.json")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--results-dir", type=Path, default=ROOT_DIR / "results")
    parser.add_argument("--export-dir", type=Path, default=ROOT_DIR / "export")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(TARGETS),
        choices=sorted(TARGETS),
        help="Subset of target models to backfill.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Override DADA/UniTS device while loading models.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Flush updated CSV progress every N updated rows (default: 1).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Do not fail if some target cells remain missing after backfill.",
    )
    args = parser.parse_args()

    _setup_logging()
    data_cfg = _load_config_data_section(args.config_path, args.dataset_root)
    _bootstrap_data_pool(data_cfg)

    results_dir = args.results_dir.resolve()
    export_dir = args.export_dir.resolve()
    report_path = None
    if not args.dry_run:
        export_dir.mkdir(parents=True, exist_ok=True)
        report_path = export_dir / "efficiency_size_params_backfill_report.csv"
    pretrained_cache: dict[tuple[Any, ...], dict[str, float]] = {}
    all_records: list[dict[str, Any]] = []

    for model_name in args.models:
        spec = TARGETS[model_name]
        csv_path = results_dir / spec["csv"]
        all_records.extend(
            _backfill_one(
                model_name,
                spec,
                csv_path,
                overwrite=bool(args.overwrite),
                dry_run=bool(args.dry_run),
                backup=not bool(args.no_backup),
                device=args.device,
                pretrained_cache=pretrained_cache,
                flush_every=max(int(args.flush_every), 1),
                report_path=report_path,
            )
        )

    if not args.dry_run:
        LOGGER.info("Backfill report written: %s", report_path)
        issues = _validate_backfill_outputs(results_dir, list(args.models), all_records)
        if issues:
            for issue in issues:
                LOGGER.error("Backfill validation failed: %s", issue)
            if not args.allow_partial:
                raise SystemExit(1)
        if not args.skip_export:
            _regenerate_exports(results_dir, export_dir)


if __name__ == "__main__":
    main()
