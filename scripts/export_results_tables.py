# -*- coding: utf-8 -*-
"""Export benchmark performance + efficiency tables from ``results/``.

Reads all categories under ``results/`` (non_learning, machine_learning,
deep_learning, …) and writes two long tables plus per-farm wide CSVs aligned
with the user's Excel templates.

Usage:
    python scripts/export_results_tables.py
    python scripts/export_results_tables.py --results-dir results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

FARMS = ["A", "B", "C"]

CATEGORY_LABELS = {
    "non_learning": "non-learning",
    "machine_learning": "machine-learning",
    "deep_learning": "deep-learning",
    "llm_based": "llm-based",
    "ts_pretrained": "ts pre-trained",
    "finetune_llm": "domain llm",
}

# Category / model order aligned with the user's Excel template (top → bottom).
TEMPLATE_CATEGORY_DIRS = [
    "non_learning",
    "machine_learning",
    "deep_learning",
    "llm_based",
    "ts_pretrained",
    "finetune_llm",
]

# When Excel category differs from results/ subdir.
RESULTS_SUBDIR_OVERRIDES: dict[tuple[str, str], str] = {}

TEMPLATE_MODEL_ORDER: dict[str, list[str]] = {
    "non_learning": ["cblof", "hbos", "lof"],
    "machine_learning": [
        "eif",
        "iforest",
        "kmeans",
        "knn",
        "loda",
        "pca",
        "torsk",
        "ocsvm",
        "dagmm",
        "deeppoint",
    ],
    # Order aligned with the user's Excel deep-learning block.
    "deep_learning": [
        "ae",
        "anomaly_transformer",
        "catch",
        "dcdetector",
        "duet",
        "lstmed",
        "timesnet",
        "tranad",
        "d3r",
        "gdn",
        "mscred",
        "mtad_gat",
        "omnianomaly",
        "usad",
        "mtgflow",
        "sarad",
    ],
    "llm_based": ["gpt4ts", "unitime"],
    "ts_pretrained": ["chronos", "dada", "moment", "units"],
    "finetune_llm": ["rpcl_tcne_mts_llm"],
}

# Row labels in the spreadsheet (when different from result CSV stems).
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "iforest": "if",
    "tranad": "tranad",
    "rpcl_tcne_mts_llm": "pcl_tcne_mts_llm",
}

LABEL_DIR_BY_CATEGORY = {v: k for k, v in CATEGORY_LABELS.items()}

# Excel template: acc → point_* → event_* → point_affiliation → range_* →
# affiliation_* → score AUC/VUS → operational KPIs.
PERFORMANCE_METRIC_COLUMNS = [
    "acc",
    "point_precision",
    "point_recall",
    "point_f1",
    "event_precision",
    "event_recall",
    "event_f1",
    "point_affiliation",
    "range_precision",
    "range_recall",
    "range_f1",
    "affiliation_precision",
    "affiliation_recall",
    "affiliation_f1",
    "auc_pr",
    "auc_roc",
    "range_auc_pr",
    "range_auc_roc",
    "vus_pr",
    "vus_roc",
    "mean_lead_time",
    "mean_detection_delay",
    "early_detection_rate",
    "false_alarms_per_turbine_day",
    "mtbfa",
]

PERFORMANCE_COLUMNS = [
    "farm_id",
    "category",
    "model",
    *PERFORMANCE_METRIC_COLUMNS,
    "n_series_label",
    "n_series_score",
]

# Export column -> label-report column in ``by_farm_leaderboard.csv``.
# Detection / early-warning metrics stay on the mixed farm board (normal
# rows are already NaN for those metrics).  Robustness metrics are taken
# from ``by_farm_operational_summary`` normal-only columns instead.
LABEL_REPORT_MAP = {
    "point_precision": "point_precision",
    "point_recall": "point_recall",
    "point_f1": "point_f1",
    "event_precision": "event_precision",
    "event_recall": "event_recall",
    "event_f1": "event_f1",
    "point_affiliation": "event_affiliation_f1",
    "range_precision": "range_precision",
    "range_recall": "range_recall",
    "range_f1": "range_f1",
    "affiliation_precision": "affiliation_precision",
    "affiliation_recall": "affiliation_recall",
    "affiliation_f1": "affiliation_f1",
    "mean_lead_time": "mean_lead_time",
    "mean_detection_delay": "mean_detection_delay",
    "early_detection_rate": "early_detection_rate",
}

# Robustness metrics: normal-event-only from operational farm summary.
# ``mtbfa_normal_days`` is converted to hours to match the existing export unit.
OPERATIONAL_NORMAL_MAP = {
    "acc": "accuracy_normal",
    "false_alarms_per_turbine_day": "false_alarms_per_turbine_day_normal",
    "mtbfa": "mtbfa_normal_days",
}
_MTBFA_DAYS_TO_HOURS = 24.0

SCORE_METRICS = [
    "auc_pr",
    "auc_roc",
    "range_auc_pr",
    "range_auc_roc",
    "vus_pr",
    "vus_roc",
]

EFFICIENCY_COLUMNS = [
    "farm_id",
    "category",
    "model",
    "fit_time_s",
    "fit_gpu_mem_mb",
    "fit_ram_mb",
    "fit_cpu_use_per",
    "infer_time_s",
    "infer_gpu_mem_mb",
    "infer_ram_mb",
    "infer_cpu_use_per",
    "flops",
    "n_params",
    "model_size_mb",
    "n_series",
]

EFFICIENCY_SOURCE_COLS = [
    "fit_time",
    "fit_peak_memory_mb",
    "fit_gpu_peak_memory_mb",
    "fit_cpu_usage_percent",
    "inference_time",
    "inference_peak_memory_mb",
    "inference_gpu_peak_memory_mb",
    "inference_cpu_usage_percent",
    "flops",
    "n_params",
    "model_size_mb",
]


def _strip_suffix(col: str) -> str:
    return str(col).split(" (", 1)[0].strip()


def _read_farm_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=_strip_suffix)
    if "farm_id" in df.columns:
        df["farm_id"] = df["farm_id"].astype(str).str.strip()
    return df


def _display_model(stem: str) -> str:
    return MODEL_DISPLAY_NAMES.get(stem.lower(), stem.lower())


def _model_sort_key(category_label: str, model_stem: str) -> tuple[int, int, str]:
    cat_dir = LABEL_DIR_BY_CATEGORY.get(category_label, category_label)
    cat_idx = (
        TEMPLATE_CATEGORY_DIRS.index(cat_dir)
        if cat_dir in TEMPLATE_CATEGORY_DIRS
        else len(TEMPLATE_CATEGORY_DIRS)
    )
    order = TEMPLATE_MODEL_ORDER.get(cat_dir, [])
    model_idx = order.index(model_stem.lower()) if model_stem.lower() in order else len(order)
    return cat_idx, model_idx, model_stem.lower()


def _apply_template_order(table: pd.DataFrame, id_col: str = "model_stem") -> pd.DataFrame:
    if table.empty:
        return table
    out = table.copy()
    out["_sort"] = out.apply(
        lambda r: _model_sort_key(r["category"], r[id_col]), axis=1,
    )
    out["farm_id"] = pd.Categorical(out["farm_id"], categories=FARMS, ordered=True)
    out = out.sort_values(["farm_id", "_sort"]).drop(columns="_sort")
    out["model"] = out[id_col].map(_display_model)
    return out.reset_index(drop=True)


def _results_subdir(category_label: str, model_stem: str, default_cat_dir: str) -> str:
    return RESULTS_SUBDIR_OVERRIDES.get(
        (category_label, model_stem.lower()), default_cat_dir
    )


def _template_model_entries(include_missing: bool) -> list[tuple[str, str, str]]:
    """Return (category_dir, category_label, model_stem) in template order."""
    entries: list[tuple[str, str, str]] = []
    for cat_dir in TEMPLATE_CATEGORY_DIRS:
        category_label = CATEGORY_LABELS[cat_dir]
        for model in TEMPLATE_MODEL_ORDER.get(cat_dir, []):
            entries.append((cat_dir, category_label, model))
    return entries


def _category_root(results_dir: Path, cat_dir: str, category_label: str, model: str) -> Path:
    subdir = _results_subdir(category_label, model, cat_dir)
    return results_dir / subdir


def _empty_performance_row(farm: str, category_label: str, model_stem: str) -> dict:
    row = {col: float("nan") for col in PERFORMANCE_COLUMNS}
    row.update(
        {
            "farm_id": farm,
            "category": category_label,
            "model": _display_model(model_stem),
            "model_stem": model_stem,
        }
    )
    return row


def _empty_efficiency_row(farm: str, category_label: str, model_stem: str) -> dict:
    row = {col: float("nan") for col in EFFICIENCY_COLUMNS}
    row.update(
        {
            "farm_id": farm,
            "category": category_label,
            "model": _display_model(model_stem),
            "model_stem": model_stem,
            "n_series": float("nan"),
        }
    )
    return row


def load_performance_rows(
    category_root: Path,
    category_label: str,
    model: str,
) -> pd.DataFrame:
    label_path = (
        category_root / "label" / f"{model}_report" / "tables" / "by_farm_leaderboard.csv"
    )
    score_path = (
        category_root / "score" / f"{model}_report" / "tables" / "by_farm_leaderboard.csv"
    )
    op_path = (
        category_root
        / "label"
        / f"{model}_report"
        / "tables"
        / "by_farm_operational_summary.csv"
    )
    if not label_path.exists() and not score_path.exists():
        raise FileNotFoundError(f"missing reports for {category_label}/{model}")

    label_df = _read_farm_table(label_path) if label_path.exists() else pd.DataFrame()
    score_df = _read_farm_table(score_path) if score_path.exists() else pd.DataFrame()
    op_df = _read_farm_table(op_path) if op_path.exists() else pd.DataFrame()

    rows = []
    for farm in FARMS:
        lsub = label_df[label_df["farm_id"] == farm] if not label_df.empty else pd.DataFrame()
        ssub = score_df[score_df["farm_id"] == farm] if not score_df.empty else pd.DataFrame()
        osub = op_df[op_df["farm_id"] == farm] if not op_df.empty else pd.DataFrame()
        if lsub.empty and ssub.empty:
            continue
        row = {
            "farm_id": farm,
            "category": category_label,
            "model": _display_model(model),
            "model_stem": model.lower(),
        }
        if not lsub.empty:
            lr = lsub.iloc[0]
            for export_col, source_col in LABEL_REPORT_MAP.items():
                row[export_col] = lr.get(source_col, float("nan"))
            row["n_series_label"] = lr.get("n_series", float("nan"))
        # Prefer normal-only robustness metrics when the operational table exists.
        if not osub.empty:
            orow = osub.iloc[0]
            for export_col, source_col in OPERATIONAL_NORMAL_MAP.items():
                val = orow.get(source_col, float("nan"))
                if export_col == "mtbfa" and pd.notna(val):
                    val = float(val) * _MTBFA_DAYS_TO_HOURS
                row[export_col] = val
        elif not lsub.empty:
            # Fallback for older reports without operational farm summary.
            lr = lsub.iloc[0]
            row["acc"] = lr.get("accuracy", float("nan"))
            row["false_alarms_per_turbine_day"] = lr.get(
                "false_alarms_per_turbine_day", float("nan")
            )
            row["mtbfa"] = lr.get("mtbfa", float("nan"))
        if not ssub.empty:
            sr = ssub.iloc[0]
            for metric in SCORE_METRICS:
                row[metric] = sr.get(metric, float("nan"))
            row["n_series_score"] = sr.get("n_series", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def load_efficiency_rows(
    category_root: Path,
    category_label: str,
    model: str,
    track: str = "score",
) -> pd.DataFrame:
    csv_path = category_root / track / f"{model}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    usecols = {"farm_id", *EFFICIENCY_SOURCE_COLS}
    df = pd.read_csv(csv_path, usecols=lambda c: c in usecols)
    if "farm_id" not in df.columns:
        raise ValueError(f"Missing farm_id in {csv_path}")
    df["farm_id"] = df["farm_id"].astype(str).str.strip()

    rows = []
    for farm in FARMS:
        sub = df[df["farm_id"] == farm]
        if sub.empty:
            continue
        row = {
            "farm_id": farm,
            "category": category_label,
            "model": _display_model(model),
            "model_stem": model.lower(),
            "n_series": len(sub),
        }
        for col in EFFICIENCY_SOURCE_COLS:
            if col in sub.columns:
                row[col] = pd.to_numeric(sub[col], errors="coerce").mean()
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["fit_time_s"] = out.get("fit_time")
    # GPU memory strictly from the GPU counter; CPU-only models stay NaN.
    out["fit_gpu_mem_mb"] = pd.to_numeric(
        out.get("fit_gpu_peak_memory_mb"), errors="coerce"
    )
    out["fit_ram_mb"] = pd.to_numeric(
        out.get("fit_peak_memory_mb"), errors="coerce"
    )
    out["fit_cpu_use_per"] = out.get("fit_cpu_usage_percent")
    out["infer_time_s"] = out.get("inference_time")
    out["infer_gpu_mem_mb"] = pd.to_numeric(
        out.get("inference_gpu_peak_memory_mb"), errors="coerce"
    )
    out["infer_ram_mb"] = pd.to_numeric(
        out.get("inference_peak_memory_mb"), errors="coerce"
    )
    out["infer_cpu_use_per"] = out.get("inference_cpu_usage_percent")
    for col in EFFICIENCY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[EFFICIENCY_COLUMNS]


def build_performance_table(
    results_dir: Path,
    *,
    include_missing: bool = True,
) -> pd.DataFrame:
    loaded: dict[tuple[str, str], pd.DataFrame] = {}
    skipped: list[str] = []

    for cat_dir, category_label, model in _template_model_entries(include_missing=True):
        category_root = _category_root(results_dir, cat_dir, category_label, model)
        try:
            loaded[(category_label, model)] = load_performance_rows(
                category_root, category_label, model
            )
        except FileNotFoundError:
            skipped.append(f"{category_label}/{model}")

    if skipped and not include_missing:
        print("Skipped performance (missing report):", ", ".join(skipped))

    rows: list[dict] = []
    for farm in FARMS:
        for cat_dir, category_label, model in _template_model_entries(include_missing):
            key = (category_label, model)
            part = loaded.get(key)
            if part is not None:
                sub = part[part["farm_id"] == farm]
                if not sub.empty:
                    row = sub.iloc[0].to_dict()
                    row["model"] = _display_model(model)
                    row["model_stem"] = model
                    rows.append(row)
                    continue
            if include_missing:
                rows.append(_empty_performance_row(farm, category_label, model))

    if not rows:
        return pd.DataFrame(columns=PERFORMANCE_COLUMNS)

    out = pd.DataFrame(rows)
    out = _apply_template_order(out)
    for col in PERFORMANCE_COLUMNS:
        if col not in out.columns:
            out[col] = float("nan")
    return out[PERFORMANCE_COLUMNS]


def build_efficiency_table(
    results_dir: Path,
    track: str = "score",
    *,
    include_missing: bool = True,
) -> pd.DataFrame:
    loaded: dict[tuple[str, str], pd.DataFrame] = {}
    skipped: list[str] = []

    for cat_dir, category_label, model in _template_model_entries(include_missing=True):
        category_root = _category_root(results_dir, cat_dir, category_label, model)
        try:
            loaded[(category_label, model)] = load_efficiency_rows(
                category_root, category_label, model, track=track
            )
        except FileNotFoundError:
            skipped.append(f"{category_label}/{model}")

    if skipped and not include_missing:
        print("Skipped efficiency (missing CSV):", ", ".join(skipped))

    rows: list[dict] = []
    for farm in FARMS:
        for cat_dir, category_label, model in _template_model_entries(include_missing):
            key = (category_label, model)
            part = loaded.get(key)
            if part is not None:
                sub = part[part["farm_id"] == farm]
                if not sub.empty:
                    row = sub.iloc[0].to_dict()
                    row["model"] = _display_model(model)
                    row["model_stem"] = model
                    rows.append(row)
                    continue
            if include_missing:
                rows.append(_empty_efficiency_row(farm, category_label, model))

    if not rows:
        return pd.DataFrame(columns=EFFICIENCY_COLUMNS)

    out = pd.DataFrame(rows)
    out = _apply_template_order(out)
    for col in EFFICIENCY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[EFFICIENCY_COLUMNS]


def _write_wide(table: pd.DataFrame, metric_cols: list[str], wide_dir: Path, prefix: str) -> None:
    wide_dir.mkdir(parents=True, exist_ok=True)
    for farm in FARMS:
        block = table[table["farm_id"] == farm].copy()
        if block.empty:
            continue
        block = block[["model", *metric_cols]]
        path = wide_dir / f"farm_{farm}.csv"
        block.to_csv(path, index=False, float_format="%.6f")
        print(f"  wide {prefix} farm {farm}: {len(block)} rows -> {path.name}")


def export_tables(
    results_dir: Path,
    performance_out: Path | None = None,
    efficiency_out: Path | None = None,
    performance_wide_dir: Path | None = None,
    efficiency_wide_dir: Path | None = None,
    track: str = "score",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_dir = results_dir.resolve()
    performance_out = performance_out or (results_dir / "benchmark_results_by_farm.csv")
    efficiency_out = efficiency_out or (results_dir / "benchmark_efficiency_by_farm.csv")
    performance_wide_dir = performance_wide_dir or (
        results_dir / "benchmark_results_by_farm_wide"
    )
    efficiency_wide_dir = efficiency_wide_dir or (
        results_dir / "benchmark_efficiency_by_farm_wide"
    )

    perf = build_performance_table(results_dir, include_missing=True)
    eff = build_efficiency_table(results_dir, track=track, include_missing=True)

    performance_out.parent.mkdir(parents=True, exist_ok=True)
    perf.to_csv(performance_out, index=False, float_format="%.6f")
    print(f"Wrote performance: {len(perf)} rows -> {performance_out}")

    eff.to_csv(efficiency_out, index=False, float_format="%.6f")
    print(f"Wrote efficiency: {len(eff)} rows -> {efficiency_out}")

    perf_metric_cols = list(PERFORMANCE_METRIC_COLUMNS)
    eff_metric_cols = [
        c for c in EFFICIENCY_COLUMNS if c not in ("farm_id", "category", "model", "n_series")
    ]
    _write_wide(perf, perf_metric_cols, performance_wide_dir, "performance")
    _write_wide(eff, eff_metric_cols, efficiency_wide_dir, "efficiency")

    return perf, eff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_REPO / "results",
        help="Root directory containing category subdirs (default: <repo>/results).",
    )
    parser.add_argument(
        "--track",
        choices=["score", "label"],
        default="score",
        help="Which track CSV to use for efficiency aggregation (default: score).",
    )
    args = parser.parse_args()
    export_tables(args.results_dir, track=args.track)


if __name__ == "__main__":
    main()
