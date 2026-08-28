# -*- coding: utf-8 -*-
"""Export non-learning results into a per-farm (A/B/C) spreadsheet-style table.

Reads merged metrics from ``{model}_report/tables/by_farm_leaderboard.csv``
(label + score tracks) produced by ``run_benchmark.py --report``.

Usage:
    python scripts/export_non_learning_farm_table.py
    python scripts/export_non_learning_farm_table.py --results-root results/non_learning
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

MODELS = ["hbos", "lof", "cblof"]
FARMS = ["A", "B", "C"]

# Column order aligned with the user's Excel template (left → right).
OUTPUT_COLUMNS = [
    "farm_id",
    "category",
    "model",
    "point_precision",
    "point_recall",
    "point_f1",
    "event_precision",
    "event_recall",
    "event_f1",
    "event_affiliation_f1",
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
    "n_series_label",
    "n_series_score",
]

LABEL_METRICS = [
    "point_precision",
    "point_recall",
    "point_f1",
    "event_precision",
    "event_recall",
    "event_f1",
    "event_affiliation_f1",
    "range_precision",
    "range_recall",
    "range_f1",
    "affiliation_precision",
    "affiliation_recall",
    "affiliation_f1",
    "early_detection_rate",
    "mean_lead_time",
    "mean_detection_delay",
    "false_alarms_per_turbine_day",
    "mtbfa",
]

SCORE_METRICS = [
    "auc_pr",
    "auc_roc",
    "range_auc_pr",
    "range_auc_roc",
    "vus_pr",
    "vus_roc",
]


def _strip_suffix(col: str) -> str:
    return str(col).split(" (", 1)[0].strip()


def _read_farm_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df.rename(columns=_strip_suffix)
    if "farm_id" in df.columns:
        df["farm_id"] = df["farm_id"].astype(str).str.strip()
    return df


def _load_model_farm_rows(results_root: Path, model: str) -> pd.DataFrame:
    label_path = results_root / "label" / f"{model}_report" / "tables" / "by_farm_leaderboard.csv"
    score_path = results_root / "score" / f"{model}_report" / "tables" / "by_farm_leaderboard.csv"
    label_df = _read_farm_table(label_path)
    score_df = _read_farm_table(score_path)

    rows = []
    for farm in FARMS:
        lsub = label_df[label_df["farm_id"] == farm]
        ssub = score_df[score_df["farm_id"] == farm]
        if lsub.empty and ssub.empty:
            continue
        row = {
            "farm_id": farm,
            "category": "non-learning",
            "model": model.lower(),
        }
        if not lsub.empty:
            lr = lsub.iloc[0]
            for m in LABEL_METRICS:
                row[m] = lr.get(m, float("nan"))
            row["n_series_label"] = lr.get("n_series", float("nan"))
        if not ssub.empty:
            sr = ssub.iloc[0]
            for m in SCORE_METRICS:
                row[m] = sr.get(m, float("nan"))
            row["n_series_score"] = sr.get("n_series", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def build_table(results_root: Path, models: list[str]) -> pd.DataFrame:
    parts = [_load_model_farm_rows(results_root, m) for m in models]
    out = pd.concat(parts, ignore_index=True)
    out["farm_id"] = pd.Categorical(out["farm_id"], categories=FARMS, ordered=True)
    out["model"] = pd.Categorical(
        out["model"], categories=[m.lower() for m in models], ordered=True
    )
    out = out.sort_values(["farm_id", "model"]).reset_index(drop=True)
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = float("nan")
    return out[OUTPUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_REPO / "non_learning",
        help="Root with score/ and label/ subdirs (default: <repo>/non_learning).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Model stems matching CSV/report names (default: hbos lof cblof).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <results-root>/non_learning_results_by_farm.csv).",
    )
    parser.add_argument(
        "--wide-dir",
        type=Path,
        default=None,
        help="If set, also write one wide CSV per farm (rows=models, cols=metrics) for Excel paste.",
    )
    args = parser.parse_args()
    results_root = args.results_root.resolve()
    output = args.output or (results_root / "non_learning_results_by_farm.csv")
    wide_dir = args.wide_dir or (results_root / "non_learning_results_by_farm_wide")

    table = build_table(results_root, [m.lower() for m in args.models])
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, float_format="%.6f")
    print(f"Wrote {len(table)} rows -> {output}")

    metric_cols = [c for c in OUTPUT_COLUMNS if c not in ("farm_id", "category", "model", "n_series_label", "n_series_score")]
    wide_dir.mkdir(parents=True, exist_ok=True)
    for farm in FARMS:
        block = table[table["farm_id"] == farm].set_index("model")[metric_cols]
        wide_path = wide_dir / f"farm_{farm}.csv"
        block.to_csv(wide_path, float_format="%.6f")
        print(f"  wide farm {farm}: {wide_path.name} ({len(block)} models)")


if __name__ == "__main__":
    main()
