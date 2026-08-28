# -*- coding: utf-8 -*-
"""Audit missing cells in benchmark efficiency exports."""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.export_results_tables import (  # noqa: E402
    CATEGORY_LABELS,
    EFFICIENCY_COLUMNS,
    EFFICIENCY_SOURCE_COLS,
    FARMS,
    LABEL_DIR_BY_CATEGORY,
    TEMPLATE_CATEGORY_DIRS,
    TEMPLATE_MODEL_ORDER,
    _display_model,
)

FIELD_SOURCE = {
    "fit_time_s": "fit_time",
    "fit_gpu_mem_mb": "fit_gpu_peak_memory_mb",
    "fit_ram_mb": "fit_peak_memory_mb",
    "fit_cpu_use_per": "fit_cpu_usage_percent",
    "infer_time_s": "inference_time",
    "infer_gpu_mem_mb": "inference_gpu_peak_memory_mb",
    "infer_ram_mb": "inference_peak_memory_mb",
    "infer_cpu_use_per": "inference_cpu_usage_percent",
    "flops": "flops",
    "n_params": "n_params",
    "model_size_mb": "model_size_mb",
}

MODEL_MODULE = {
    "cblof": "tsad_benchmark/baselines/non_learning/cblof.py",
    "hbos": "tsad_benchmark/baselines/non_learning/hbos.py",
    "lof": "tsad_benchmark/baselines/non_learning/lof.py",
    "eif": "tsad_benchmark/baselines/machine_learning/eif.py",
    "iforest": "tsad_benchmark/baselines/machine_learning/iforest.py",
    "kmeans": "tsad_benchmark/baselines/machine_learning/kmeans.py",
    "knn": "tsad_benchmark/baselines/machine_learning/knn.py",
    "loda": "tsad_benchmark/baselines/machine_learning/loda.py",
    "pca": "tsad_benchmark/baselines/machine_learning/pca.py",
    "torsk": "tsad_benchmark/baselines/machine_learning/torsk.py",
    "ocsvm": "tsad_benchmark/baselines/machine_learning/ocsvm.py",
    "dagmm": "tsad_benchmark/baselines/machine_learning/dagmm.py",
    "deeppoint": "tsad_benchmark/baselines/machine_learning/deeppoint.py",
    "ae": "tsad_benchmark/baselines/deep_learning/ae.py",
    "anomaly_transformer": "tsad_benchmark/baselines/deep_learning/anomaly_transformer.py",
    "catch": "tsad_benchmark/baselines/deep_learning/catch.py",
    "dcdetector": "tsad_benchmark/baselines/deep_learning/dcdetector.py",
    "duet": "tsad_benchmark/baselines/deep_learning/duet.py",
    "lstmed": "tsad_benchmark/baselines/deep_learning/lstmed.py",
    "timesnet": "tsad_benchmark/baselines/deep_learning/timesnet.py",
    "tranad": "tsad_benchmark/baselines/deep_learning/tranad.py",
    "d3r": "tsad_benchmark/baselines/deep_learning/d3r.py",
    "gdn": "tsad_benchmark/baselines/deep_learning/gdn.py",
    "mscred": "tsad_benchmark/baselines/deep_learning/mscred.py",
    "mtad_gat": "tsad_benchmark/baselines/deep_learning/mtad_gat.py",
    "omnianomaly": "tsad_benchmark/baselines/deep_learning/omnianomaly.py",
    "usad": "tsad_benchmark/baselines/deep_learning/usad.py",
    "mtgflow": "tsad_benchmark/baselines/deep_learning/mtgflow.py",
    "sarad": "tsad_benchmark/baselines/deep_learning/sarad.py",
    "gpt4ts": "tsad_benchmark/baselines/llm_based/gpt4ts.py",
    "unitime": "tsad_benchmark/baselines/llm_based/unitime.py",
    "chronos": "tsad_benchmark/baselines/ts_pretrained/chronos.py",
    "dada": "tsad_benchmark/baselines/ts_pretrained/dada.py",
    "moment": "tsad_benchmark/baselines/ts_pretrained/moment.py",
    "units": "tsad_benchmark/baselines/ts_pretrained/units.py",
    "rpcl_tcne_mts_llm": "tsad_benchmark/baselines/finetune_llm/rpcl_tcne_mts_llm.py",
}

BASE_REASON = {
    "fit_gpu_mem_mb": "source GPU peak column is empty; CPU-only or no PyTorch/NVML GPU allocation was recorded. RAM column is the valid memory value.",
    "infer_gpu_mem_mb": "source GPU peak column is empty; CPU-only or no PyTorch/NVML GPU allocation was recorded. RAM column is the valid memory value.",
    "n_params": "model did not provide estimate_n_params, or the value is not applicable for this non-neural estimator.",
    "model_size_mb": "model did not provide estimate_model_size_mb, or serialization-based size estimation returned NaN.",
    "flops": "model did not provide a FLOPs estimate, or the estimator explicitly returned NaN.",
}


def template_entries() -> list[tuple[str, str, str]]:
    rows = []
    for cat_dir in TEMPLATE_CATEGORY_DIRS:
        label = CATEGORY_LABELS[cat_dir]
        for model in TEMPLATE_MODEL_ORDER[cat_dir]:
            rows.append((cat_dir, label, model))
    return rows


def read_log(repo: Path, cat_dir: str, model: str, track: str) -> tuple[str, str]:
    candidates = [
        repo / "logs" / cat_dir / f"{model}_{track}.log",
        repo / f"{model}_{track}.log",
    ]
    parts = []
    used = []
    for path in candidates:
        if path.exists():
            used.append(str(path.relative_to(repo)))
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts), ";".join(used)


def classify_log(text: str) -> tuple[str, str]:
    if not text:
        return "no_log", ""
    tail = "\n".join(text.splitlines()[-8:])
    if re.search(r"\bKilled\b", text):
        return "killed", tail
    if "Run finished" in text and "Results saved" in text:
        return "finished", tail
    if re.search(r"Traceback|RuntimeError|Exception|CUDA out of memory|MemoryError|No such file", text):
        return "error", tail
    return "not_finished", tail


def safe_mean_status(df: pd.DataFrame, source_col: str) -> tuple[int, int]:
    if source_col not in df.columns:
        return 0, 0
    vals = pd.to_numeric(df[source_col], errors="coerce")
    return int(vals.notna().sum()), int(len(vals))


def source_reason(repo: Path, model: str, field: str, source_col: str) -> str:
    reason = BASE_REASON.get(field, f"source column {source_col} is empty.")
    module = MODEL_MODULE.get(model)
    if not module:
        return reason
    path = repo / module
    if not path.exists():
        return reason
    text = path.read_text(encoding="utf-8", errors="replace")
    if field == "flops" and re.search(r"def estimate_flops[\s\S]{0,180}return math\.nan", text):
        return "code explicitly returns math.nan for estimate_flops; FLOPs are intentionally not estimated."
    if field == "flops" and "DLBaseModel" in text:
        return "inherits DLBaseModel FLOPs estimator; fvcore tracing returned NaN for this model/runtime, so this is estimator coverage rather than a missing CSV."
    if field == "n_params" and "def estimate_n_params" not in text:
        return "no model-specific estimate_n_params implementation; base method returns NaN/not applicable."
    if field == "model_size_mb" and "def estimate_model_size_mb" not in text:
        return "no model-specific estimate_model_size_mb implementation; base method returns NaN/unknown."
    return reason


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=_REPO / "results")
    parser.add_argument("--export-dir", type=Path, default=_REPO / "export")
    parser.add_argument("--track", default="score", choices=["score", "label"])
    args = parser.parse_args()

    repo = _REPO
    results_dir = args.results_dir.resolve()
    export_dir = args.export_dir.resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    model_rows = []
    field_rows = []
    eff_path = export_dir / "benchmark_efficiency_by_farm.csv"
    eff = pd.read_csv(eff_path) if eff_path.exists() else pd.DataFrame(columns=EFFICIENCY_COLUMNS)

    for cat_dir, category_label, model in template_entries():
        display = _display_model(model)
        score_csv = results_dir / cat_dir / args.track / f"{model}.csv"
        label_csv = results_dir / cat_dir / "label" / f"{model}.csv"
        log_text, log_paths = read_log(repo, cat_dir, model, args.track)
        log_status, log_tail = classify_log(log_text)
        csv_exists = score_csv.exists()
        row_status = "ok"
        n_rows = 0
        farm_counts = ""
        df = pd.DataFrame()

        if csv_exists:
            usecols = {"farm_id", *EFFICIENCY_SOURCE_COLS}
            df = pd.read_csv(score_csv, usecols=lambda c: c in usecols)
            n_rows = len(df)
            if "farm_id" in df.columns:
                farm_counts = ";".join(
                    f"{farm}:{int((df['farm_id'].astype(str).str.strip() == farm).sum())}"
                    for farm in FARMS
                )
            if n_rows == 0:
                row_status = "empty_score_csv"
            elif n_rows < 95:
                row_status = "partial_score_csv"
        else:
            row_status = "missing_score_csv"

        if row_status == "missing_score_csv":
            if log_status == "killed":
                row_reason = "score CSV is absent because the run was killed before completion."
            elif log_status == "not_finished":
                row_reason = "score CSV is absent because the run has no completion marker; likely interrupted or still unfinished."
            elif label_csv.exists():
                row_reason = "score CSV is absent; only label-track output exists, while efficiency export reads score track by default."
            else:
                row_reason = "score CSV is absent and no completed score run was found."
        elif row_status == "partial_score_csv":
            row_reason = "score CSV exists but has fewer than the expected 95 series; export aggregates available rows only."
        else:
            row_reason = "score CSV exists and efficiency rows are aggregated from it."

        model_rows.append(
            {
                "category": category_label,
                "model": display,
                "model_stem": model,
                "score_csv": str(score_csv.relative_to(repo)),
                "score_csv_exists": csv_exists,
                "score_rows": n_rows,
                "farm_counts": farm_counts,
                "label_csv_exists": label_csv.exists(),
                "log_status": log_status,
                "log_paths": log_paths,
                "row_status": row_status,
                "row_reason": row_reason,
                "log_tail": log_tail,
            }
        )

        block = eff[(eff.get("category") == category_label) & (eff.get("model") == display)]
        for field, source_col in FIELD_SOURCE.items():
            if field not in block.columns:
                missing_farms = ",".join(FARMS)
            else:
                missing_farms = ",".join(
                    str(row["farm_id"])
                    for _, row in block.iterrows()
                    if pd.isna(row.get(field))
                )
            if not missing_farms:
                continue

            present_count, total_count = safe_mean_status(df, source_col) if csv_exists else (0, 0)
            if not csv_exists:
                reason = row_reason
            elif present_count == 0:
                reason = source_reason(repo, model, field, source_col)
            elif present_count < total_count:
                reason = f"source column {source_col} is partially NaN; export farm means remain blank where all rows in a farm are NaN."
            else:
                reason = f"field is blank in export although source column {source_col} has values; check export mapping."

            field_rows.append(
                {
                    "category": category_label,
                    "model": display,
                    "model_stem": model,
                    "field": field,
                    "source_column": source_col,
                    "missing_farms": missing_farms,
                    "source_non_null_rows": present_count,
                    "source_total_rows": total_count,
                    "reason": reason,
                    "score_csv_exists": csv_exists,
                    "log_status": log_status,
                    "module": MODEL_MODULE.get(model, ""),
                }
            )

    model_audit = pd.DataFrame(model_rows)
    field_audit = pd.DataFrame(field_rows)
    model_audit.to_csv(export_dir / "efficiency_model_status_audit.csv", index=False)
    field_audit.to_csv(export_dir / "efficiency_missing_field_audit.csv", index=False)
    print(f"Wrote {len(model_audit)} model rows -> {export_dir / 'efficiency_model_status_audit.csv'}")
    print(f"Wrote {len(field_audit)} field rows -> {export_dir / 'efficiency_missing_field_audit.csv'}")


if __name__ == "__main__":
    main()
