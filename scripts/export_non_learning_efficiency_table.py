# -*- coding: utf-8 -*-
"""Export non-learning efficiency metrics by wind farm (A/B/C).

Aggregates per-series resource columns from result CSVs (score track by default).

Usage:
    python scripts/export_non_learning_efficiency_table.py
    python scripts/export_non_learning_efficiency_table.py --results-root results/non_learning
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

MODELS = ["hbos", "lof", "cblof"]
FARMS = ["A", "B", "C"]

# Headers aligned with the user's efficiency spreadsheet template.
OUTPUT_COLUMNS = [
    "farm_id",
    "category",
    "model",
    "fit_time_s",
    "fit_gpu_mem_mb",
    "fit_gpu_use_per",
    "infer_time_s",
    "infer_gpu_mem_mb",
    "infer_cpu_use_per",
    "flops",
    "n_params",
    "model_size_mb",
    "n_series",
]

# Columns in result CSV -> export name (after aggregation).
SERIES_COLS = {
    "fit_time": "fit_time_s",
    "fit_peak_memory_mb": "_fit_mem",
    "fit_gpu_peak_memory_mb": "_fit_gpu_mem",
    "fit_cpu_usage_percent": "_fit_cpu",
    "inference_time": "infer_time_s",
    "inference_peak_memory_mb": "_infer_mem",
    "inference_gpu_peak_memory_mb": "_infer_gpu_mem",
    "inference_cpu_usage_percent": "infer_cpu_use_per",
    "flops": "flops",
    "n_params": "n_params",
    "model_size_mb": "model_size_mb",
}


def _coalesce_gpu_or_mem(gpu: pd.Series, mem: pd.Series) -> pd.Series:
    """Prefer GPU peak MB when recorded; else RAM peak MB (CPU-only baselines)."""
    gpu_n = pd.to_numeric(gpu, errors="coerce")
    mem_n = pd.to_numeric(mem, errors="coerce")
    return gpu_n.where(gpu_n.notna(), mem_n)


def _coalesce_gpu_or_cpu(gpu: pd.Series, cpu: pd.Series) -> pd.Series:
    gpu_n = pd.to_numeric(gpu, errors="coerce")
    cpu_n = pd.to_numeric(cpu, errors="coerce")
    return gpu_n.where(gpu_n.notna(), cpu_n)


def _load_efficiency_by_farm(csv_path: Path, model: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    usecols = ["farm_id"] + [c for c in SERIES_COLS if c != "farm_id"]
    df = pd.read_csv(csv_path, usecols=lambda c: c in usecols)
    if "farm_id" not in df.columns:
        raise ValueError(f"Missing farm_id in {csv_path}")
    df["farm_id"] = df["farm_id"].astype(str).str.strip()

    rows = []
    for farm in FARMS:
        sub = df[df["farm_id"] == farm]
        if sub.empty:
            continue
        row = {"farm_id": farm, "category": "non-learning", "model": model.lower(), "n_series": len(sub)}
        for src, _ in SERIES_COLS.items():
            if src in sub.columns:
                row[src] = pd.to_numeric(sub[src], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["fit_gpu_mem_mb"] = _coalesce_gpu_or_mem(
        out.get("fit_gpu_peak_memory_mb"),
        out.get("fit_peak_memory_mb"),
    )
    out["fit_gpu_use_per"] = _coalesce_gpu_or_cpu(
        out.get("fit_gpu_peak_memory_mb"),
        out.get("fit_cpu_usage_percent"),
    )
    out["infer_gpu_mem_mb"] = _coalesce_gpu_or_mem(
        out.get("inference_gpu_peak_memory_mb"),
        out.get("inference_peak_memory_mb"),
    )
    out["fit_time_s"] = out.get("fit_time")
    out["infer_time_s"] = out.get("inference_time")
    out["infer_cpu_use_per"] = out.get("inference_cpu_usage_percent")
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[OUTPUT_COLUMNS]


def build_table(results_root: Path, models: list[str], track: str = "score") -> pd.DataFrame:
    parts = []
    for model in models:
        csv_path = results_root / track / f"{model}.csv"
        parts.append(_load_efficiency_by_farm(csv_path, model))
    out = pd.concat(parts, ignore_index=True)
    out = _finalize(out)
    out["farm_id"] = pd.Categorical(out["farm_id"], categories=FARMS, ordered=True)
    out["model"] = pd.Categorical(
        out["model"], categories=[m.lower() for m in models], ordered=True
    )
    return out.sort_values(["farm_id", "model"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_REPO / "non_learning",
    )
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--track", choices=["score", "label"], default="score")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--wide-dir", type=Path, default=None)
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    output = args.output or (results_root / "non_learning_efficiency_by_farm.csv")
    wide_dir = args.wide_dir or (results_root / "non_learning_efficiency_by_farm_wide")

    table = build_table(results_root, [m.lower() for m in args.models], track=args.track)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, float_format="%.6f")
    print(f"Wrote {len(table)} rows -> {output}")

    metric_cols = [c for c in OUTPUT_COLUMNS if c not in ("farm_id", "category", "model", "n_series")]
    wide_dir.mkdir(parents=True, exist_ok=True)
    for farm in FARMS:
        block = table[table["farm_id"] == farm].set_index("model")[metric_cols]
        block.to_csv(wide_dir / f"farm_{farm}.csv", float_format="%.6f")
        print(f"  wide farm {farm}: {len(block)} models")


if __name__ == "__main__":
    main()
