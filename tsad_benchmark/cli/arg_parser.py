# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the benchmark CLI parser."""
    parser = argparse.ArgumentParser(
        prog="run_benchmark",
        description="Wind-turbine anomaly-detection benchmark — unified launcher.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ----- 1. Config template ------------------------------------------------
    g_cfg = parser.add_argument_group("config")
    g_cfg.add_argument(
        "--config-path",
        required=True,
        help=(
            "JSON template under config/ (e.g. unfixed_label.json) or an "
            "explicit relative/absolute path."
        ),
    )

    # ----- 2. Data overrides -------------------------------------------------
    g_data = parser.add_argument_group("data")
    g_data.add_argument(
        "--dataset-root",
        default=None,
        help="Wind-farm dataset root; defaults to the repo root.",
    )
    g_data.add_argument(
        "--limit",
        type=int,
        default=None,
        dest="series_limit",
        help="Evaluate only the first N series (smoke-test shortcut).",
    )
    g_data.add_argument(
        "--rebuild-meta",
        action="store_true",
        default=False,
        help="Force rebuild of WIND_AD_META.csv before running.",
    )

    # ----- 3. Model injection (one entry per model) --------------------------
    g_model = parser.add_argument_group("model")
    g_model.add_argument(
        "--model-name",
        nargs="+",
        required=True,
        help="Display name(s) of the model(s) to evaluate (one per model).",
    )
    g_model.add_argument(
        "--model-path",
        nargs="+",
        required=True,
        help=(
            "Fully-qualified import path of the model class or factory, "
            "e.g. tsad_benchmark.baselines.non_learning.lof.LOFModel."
        ),
    )
    g_model.add_argument(
        "--model-hyper-params",
        nargs="+",
        default=None,
        help=(
            "JSON string per model with explicit hyper-parameters, "
            "e.g. '{\"n_neighbors\": 20}'. Use 'None' for defaults."
        ),
    )
    g_model.add_argument(
        "--adapter",
        nargs="+",
        default=None,
        help=(
            "Adapter name per model: sklearn / pytorch / rule, or 'None'. "
            "When omitted, no adapter is applied."
        ),
    )
    g_model.add_argument(
        "--expected-output",
        nargs="+",
        default=None,
        help="Expected output per model: score / label.",
    )

    # ----- 4. Evaluation overrides ------------------------------------------
    g_eval = parser.add_argument_group("evaluation")
    g_eval.add_argument(
        "--strategy-args",
        default=None,
        help=(
            "JSON dict whose keys deep-merge into evaluation_config."
            "strategy_args (e.g. '{\"train_test_split\": 0.5}')."
        ),
    )
    g_eval.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metric names to compute, or 'all'. Overrides the JSON template.",
    )
    g_eval.add_argument(
        "--defer-score-vus",
        action="store_true",
        default=False,
        help=(
            "For score-track runs, compute fast score metrics now and leave "
            "vus_pr/vus_roc as NaN for later recomputation."
        ),
    )

    # ----- 5. Output / report -----------------------------------------------
    g_out = parser.add_argument_group("output")
    g_out.add_argument(
        "--save-path",
        default=None,
        help="Result CSV path (relative to repo root or absolute).",
    )
    g_out.add_argument(
        "--report-dir",
        default=None,
        help="Report output directory; defaults to <save-path-dir>/report.",
    )
    g_out.add_argument(
        "--no-report",
        action="store_true",
        default=False,
        help="Skip report generation after the run finishes.",
    )

    return parser
