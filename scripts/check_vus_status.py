# -*- coding: utf-8 -*-
"""Audit vus_pr / vus_roc in score CSVs and reports."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
RESULTS = _REPO / "results"

# Server screenshot + template (for reference when files not synced locally)
SERVER_DL = [
    "ae", "anomaly_transformer", "d3r", "dcdetector", "duet", "gdn",
    "lstmed", "omnianomaly", "timesnet", "tranad", "usad",
]
PENDING_DL = ["catch", "mscred", "mtad_gat"]


def audit_csv(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path, usecols=lambda c: c in ["event_label", "vus_pr", "vus_roc"])
    anom = df[df["event_label"] == "anomaly"]
    n = len(anom)
    vp = pd.to_numeric(anom["vus_pr"], errors="coerce")
    vr = pd.to_numeric(anom["vus_roc"], errors="coerce")
    np_, nr_ = int(vp.notna().sum()), int(vr.notna().sum())
    if np_ == 0:
        status = "ALL_MISSING"
    elif np_ < n:
        status = "PR_PARTIAL"
    elif nr_ < n:
        status = "ROC_ONLY_PARTIAL"  # usually B_53
    else:
        status = "OK"
    return {
        "anomaly_n": n,
        "vus_pr": f"{np_}/{n}",
        "vus_roc": f"{nr_}/{n}",
        "status": status,
    }


def audit_report(report_path: Path) -> str:
    if not report_path.exists():
        return "NO_REPORT"
    df = pd.read_csv(report_path)
    df.columns = [str(c).split(" (", 1)[0].strip() for c in df.columns]
    cols = [c for c in df.columns if c in ("vus_pr", "vus_roc")]
    if not cols:
        return "NO_VUS_COLS"
    sub = df[df["farm_id"].astype(str).str.strip() == "A"]
    if sub.empty:
        return "NO_FARM_A"
    row = sub.iloc[0]
    if any(pd.isna(row.get(c)) for c in cols):
        return "REPORT_NAN"
    return "REPORT_OK"


def main() -> None:
    results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    rows = []
    for cat_dir in sorted(results_root.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith("benchmark"):
            continue
        score_dir = cat_dir / "score"
        if not score_dir.is_dir():
            continue
        for csv in sorted(score_dir.glob("*.csv")):
            if "_report" in csv.stem:
                continue
            info = audit_csv(csv)
            rep = audit_report(
                score_dir / f"{csv.stem}_report" / "tables" / "by_farm_leaderboard.csv"
            )
            rows.append(
                {
                    "category": cat_dir.name,
                    "model": csv.stem,
                    "csv_status": info["status"],
                    "vus_pr": info["vus_pr"],
                    "vus_roc": info["vus_roc"],
                    "report": rep,
                    "need_recompute": info["status"] == "ALL_MISSING"
                    or rep in ("NO_REPORT", "REPORT_NAN", "NO_VUS_COLS"),
                }
            )

    df = pd.DataFrame(rows)
    need = df[df["need_recompute"]]
    print("=== Need VUS recompute (CSV empty or report missing/NaN) ===")
    if need.empty:
        print("(none in local results/)")
    else:
        print(need[["category", "model", "csv_status", "vus_pr", "vus_roc", "report"]].to_string(index=False))

    print("\n=== CSV ALL_MISSING only ===")
    miss = df[df["csv_status"] == "ALL_MISSING"]
    print(miss[["category", "model"]].to_string(index=False) if len(miss) else "(none)")

    print("\n=== ROC_ONLY_PARTIAL (vus_pr OK, vus_roc 43/44 — B_53, no recompute needed) ===")
    print(f"count: {(df['csv_status'] == 'ROC_ONLY_PARTIAL').sum()}")

    print("\n=== Local DL vs server screenshot ===")
    local_dl = set(df[df["category"] == "deep_learning"]["model"])
    for m in SERVER_DL:
        flag = "local OK" if m in local_dl else "NOT_IN_LOCAL_SYNC"
        if m in local_dl:
            r = df[(df["category"] == "deep_learning") & (df["model"] == m)].iloc[0]
            print(f"  {m}: csv={r['csv_status']} report={r['report']} {flag}")
        else:
            print(f"  {m}: (no local csv) — CHECK ON SERVER {flag}")
    for m in PENDING_DL:
        print(f"  {m}: not in screenshot / pending run")

    print("\n=== Summary ===")
    print(f"total models: {len(df)}")
    print(f"need_recompute flag: {need['need_recompute'].sum()}")
    print(f"ALL_MISSING csv: {(df['csv_status'] == 'ALL_MISSING').sum()}")


if __name__ == "__main__":
    main()
