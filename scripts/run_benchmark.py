from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tsad_benchmark.cli.arg_parser import build_arg_parser  # noqa: E402
from tsad_benchmark.cli.overrides import (  # noqa: E402
    apply_data_overrides,
    apply_evaluation_overrides,
    apply_model_overrides,
    apply_report_overrides,
)
from tsad_benchmark.common.paths import resolve_config_path  # noqa: E402
from tsad_benchmark.runner import execute_run  # noqa: E402

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_template(config_path: str) -> dict:
    resolved = resolve_config_path(config_path)
    logger.info("Loading config template: %s", resolved)
    with open(resolved, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    for required in ("data_config", "model_config", "evaluation_config", "report_config"):
        if required not in cfg:
            raise ValueError(
                f"Config template {resolved} is missing required section: {required!r}"
            )
    return cfg


def main() -> None:
    _setup_logging()

    args = build_arg_parser().parse_args()
    template = _load_template(args.config_path)

    data_cfg = apply_data_overrides(args, template["data_config"])
    model_cfg = apply_model_overrides(args, template["model_config"])
    evaluation_cfg = apply_evaluation_overrides(args, template["evaluation_config"])
    report_cfg = apply_report_overrides(args, template["report_config"])
    from tsad_benchmark.evaluation.care_protocol import require_current
    require_current(evaluation_cfg.get('strategy_args', {}).get('evaluation_protocol'))
    if evaluation_cfg['strategy_args']['strategy_name'] != 'unfixed_detect_score':
        raise ValueError('The formal CARE launcher requires unfixed_detect_score.')

    seed = model_cfg.get("recommend_hyper_params", {}).get("seed")
    if seed is not None:
        evaluation_cfg.setdefault("strategy_args", {}).setdefault("seed", seed)

    save_path = args.save_path
    started = time.perf_counter()
    result_csv = execute_run(
        data_config=data_cfg,
        model_config=model_cfg,
        evaluation_config=evaluation_cfg,
        save_path=save_path,
    )
    elapsed_sec = time.perf_counter() - started

    if not report_cfg.get("generate_report", True):
        logger.info("Report generation skipped (--no-report).")
        return

    from tsad_benchmark.report import report

    report_dir = report_cfg.get("report_dir") or args.report_dir
    if report_dir is None:
        report_dir = os.path.join(os.path.dirname(os.path.abspath(result_csv)), "report")

    dataset_root = data_cfg.get("dataset_root") or "<repo_root>"
    full_run_config = {
        "data_config": data_cfg,
        "model_config": model_cfg,
        "evaluation_config": evaluation_cfg,
        "report_config": report_cfg,
        "dataset_root": dataset_root,
        "series_limit": data_cfg.get("series_limit"),
        "strategy_name": evaluation_cfg.get("strategy_args", {}).get("strategy_name"),
        "strategy_args": evaluation_cfg.get("strategy_args", {}),
        "metrics": evaluation_cfg.get("metrics", "all"),
    }

    report(
        result_path=result_csv,
        run_config=full_run_config,
        out_dir=report_dir,
        export_figures=bool(report_cfg.get("export_figures", True)),
        elapsed_sec=elapsed_sec,
    )


if __name__ == "__main__":
    main()
