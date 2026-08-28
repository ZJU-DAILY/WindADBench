# -*- coding: utf-8 -*-

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tsad_benchmark.common.paths import RESULTS_DIR, ROOT_DIR, ensure_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bootstrap_data_pool(data_config: Dict[str, Any]):

    from tsad_benchmark.data.series_registry import DataPool
    from tsad_benchmark.data.wind_sources import LocalWindFarmDetectDataSource
    from tsad_benchmark.data.suites.wind_series_backend import DatasetPoolImpl

    raw_root = data_config.get("dataset_root")
    dataset_root = Path(raw_root).resolve() if raw_root else ROOT_DIR
    meta_path = dataset_root / "WIND_AD_META.csv"

    if data_config.get("rebuild_meta") and meta_path.exists():
        meta_path.unlink()
        logger.info("Removed stale meta index; rebuilding: %s", meta_path)

    logger.info("Initialising data source at: %s", dataset_root)
    data_source = LocalWindFarmDetectDataSource(
        str(dataset_root),
        domain_preprocessing=bool(data_config.get("domain_preprocessing", True)),
    )
    n = (
        len(data_source.dataset.metadata)
        if data_source.dataset.metadata is not None
        else 0
    )
    logger.info("Metadata loaded: %d series.", n)

    DataPool().register_backend(DatasetPoolImpl(data_source))
    return data_source


def _resolve_series_list(data_source, data_config: Dict[str, Any]) -> List[str]:

    meta = data_source.dataset.metadata
    if meta is None or meta.empty:
        raise RuntimeError(
            "Wind-farm metadata is empty; check the dataset directory layout."
        )

    series = list(meta["file_name"])
    limit = data_config.get("series_limit")
    if isinstance(limit, int) and limit > 0:
        series = series[:limit]
        logger.info("Series list truncated to first %d entries.", limit)
    else:
        logger.info("Evaluating all %d series.", len(series))
    return series


def _resolve_save_path(save_path: Optional[str]) -> Path:

    if save_path:
        p = Path(save_path)
        return p if p.is_absolute() else (ROOT_DIR / p).resolve()
    ts = time.strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"result_{ts}.csv"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def execute_run(
    data_config: Dict[str, Any],
    model_config: Dict[str, Any],
    evaluation_config: Dict[str, Any],
    save_path: Optional[str] = None,
) -> str:

    from tsad_benchmark.evaluation.evaluate_model import eval_model, save_result
    from tsad_benchmark.models.loader import build_model_factories

    started = time.perf_counter()

    data_source = _bootstrap_data_pool(data_config)
    series_list = _resolve_series_list(data_source, data_config)

    evaluation_config = dict(evaluation_config)
    strategy_args = dict(evaluation_config.get("strategy_args", {}))
    seed = model_config.get("recommend_hyper_params", {}).get("seed")
    if seed is not None:
        strategy_args.setdefault("seed", seed)
    evaluation_config["strategy_args"] = strategy_args

    strategy_name = evaluation_config["strategy_args"]["strategy_name"]

    enriched_model_cfg = dict(model_config)
    enriched_model_cfg["strategy_name"] = strategy_name
    factories = build_model_factories(enriched_model_cfg)
    if not factories:
        raise RuntimeError(
            "No models loaded — check `models` in model_config / CLI overrides."
        )

    out_csv = _resolve_save_path(save_path)
    ensure_dir(out_csv.parent)

    first = True
    for factory in factories:
        logger.info(
            "Evaluating: model=%s strategy=%s n_series=%d",
            factory.model_name, strategy_name, len(series_list),
        )
        result = eval_model(factory, series_list, evaluation_config)
        save_result(result, str(out_csv), append=not first)
        first = False

    elapsed = time.perf_counter() - started
    logger.info("Run finished in %.1fs. Output: %s", elapsed, out_csv)
    return str(out_csv)
