# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
perf = pd.read_csv(ROOT / "export" / "benchmark_results_by_farm.csv")
rt = pd.read_csv(ROOT / "export" / "experiment_runtime_by_method.csv")

score_m = ["auc_pr", "auc_roc", "range_auc_pr", "range_auc_roc", "vus_pr", "vus_roc"]
label_m = [
    "point_f1",
    "event_f1",
    "affiliation_f1",
    "event_affiliation_f1",
    "range_f1",
    "early_detection_rate",
    "false_alarms_per_turbine_day",
    "mean_lead_time",
]
present_score = [c for c in score_m if c in perf.columns]
present_label = [c for c in label_m if c in perf.columns]

agg = perf.groupby(["category", "model"], as_index=False)[
    present_score + present_label
].mean(numeric_only=True)

rt = rt.copy()
rt["model"] = rt["model"].str.lower()
agg["model_l"] = agg["model"].str.lower()
name_map = {"if": "iforest"}
agg["model_key"] = agg["model_l"].map(lambda x: name_map.get(x, x))
rt["model_key"] = rt["model"].map(lambda x: name_map.get(x, x))

m = agg.merge(rt, on="model_key", how="left", suffixes=("", "_rt"))

keys = [c for c in ["auc_pr", "event_f1", "affiliation_f1", "range_auc_pr"] if c in m.columns]
m["perf_score"] = m[keys].mean(axis=1, skipna=True)
m["perf_rank"] = m["perf_score"].rank(ascending=False, method="min")

fast = m[m["total_h_both"].fillna(999) <= 12].copy()
fast = fast.sort_values(["perf_rank", "total_h_both"])

cols = [
    "category",
    "model",
    "total_h_both",
    "score_h",
    "label_h",
    "perf_score",
    "perf_rank",
    *keys,
]
for c in ["point_f1", "false_alarms_per_turbine_day", "early_detection_rate"]:
    if c in m.columns:
        cols.append(c)
cols = [c for c in cols if c in fast.columns]

print("=== ALL top 15 by composite (auc_pr+event_f1+aff_f1+range_auc_pr) ===")
print(m.sort_values("perf_rank")[cols].head(15).to_string(index=False))

print("\n=== FAST (<=12h score+label) ranked by performance ===")
print(fast[cols].to_string(index=False))

print("\n=== among fast: top half by perf ===")
if len(fast):
    med = fast["perf_score"].median()
    good = fast[fast["perf_score"] >= med].sort_values("total_h_both")
    print(f"(perf_score >= median {med:.4f})")
    print(good[cols].to_string(index=False))

print("\n=== global medians ===")
print(m[keys + ["total_h_both"]].median(numeric_only=True))

out = ROOT / "export" / "fast_and_good_methods.csv"
fast[cols].to_csv(out, index=False)
print(f"\nWrote {out}")
