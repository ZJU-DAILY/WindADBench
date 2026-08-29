"""Generate website/data.js from the WindADBench-DA knowledge base.

Run after refreshing the knowledge base::

    python refresh_from_xlsx.py --workbook .../benchmark.xlsx --analysis ../analysis
    python ../analysis/agent/build_kb.py
    python ../analysis/agent/rebuild_results.py
    python build_data.py

Exports the full metric matrix (25 detection/operational metrics plus cost
columns) so the site can rank any model x workload x metric combination, the
four-dimension percentiles the decision policies consume, and the M2B active
evaluation plan with its conformal stopping radius.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DEFAULT_ANALYSIS = HERE.parent / "analysis"

METRIC_COLS = [
    "acc", "point_precision", "point_recall", "point_f1",
    "event_precision", "event_recall", "event_f1", "event_affiliation_f1",
    "range_precision", "range_recall", "range_f1",
    "affiliation_precision", "affiliation_recall", "affiliation_f1",
    "auc_pr", "auc_roc", "range_auc_pr", "range_auc_roc", "vus_pr", "vus_roc",
    "mean_lead_time", "mean_detection_delay", "early_detection_rate",
    "false_alarms_per_turbine_day", "mtbfa",
    "fit_time", "infer_time", "infer_gpu_mem", "model_size",
]

FARM_META = {
    "A": {"turbines": 5, "features": 86, "type": "onshore", "anomaly_events": 11},
    "B": {"turbines": 9, "features": 257, "type": "offshore", "anomaly_events": 6},
    "C": {"turbines": 22, "features": 957, "type": "offshore", "anomaly_events": 27},
}


def r4(x) -> Optional[float]:
    return None if pd.isna(x) else round(float(x), 4)


def diagnostic_value(kb, agent_mod, models: Sequence[str]) -> Dict[str, float]:
    """Single-workload CV error reduction, the first step of the M2B policy.

    A workload is diagnostic to the degree that revealing it alone lowers the
    nested reference-model completion error, so this is the same quantity the
    active planner maximises (before the cost and coverage terms).
    """
    workloads, values = agent_mod._m2b_arrays(kb, list(models))
    references = list(range(len(models)))
    base = agent_mod._cv_completion_error(values, references, [])
    gains = {w: base - agent_mod._cv_completion_error(values, references, [i])
             for i, w in enumerate(workloads)}
    lo, hi = min(gains.values()), max(gains.values())
    return {w: (g - lo) / (hi - lo + 1e-12) for w, g in gains.items()}


def radius_curve(kb, agent_mod, models: Sequence[str],
                 order: Sequence[str]) -> List[Dict[str, float]]:
    """90% reference-residual radius after each prefix of the evaluation plan."""
    out = []
    for k in range(1, len(order) + 1):
        radius = agent_mod.conformal_radius(kb, list(models), list(order[:k]))
        out.append({"K": k, "radius": round(float(radius), 4)})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS,
                        help=f"analysis directory (default: {DEFAULT_ANALYSIS})")
    parser.add_argument("--out", type=Path, default=HERE / "data.js")
    args = parser.parse_args(argv)

    agent_dir = args.analysis / "agent"
    if not (agent_dir / "agent_core.py").is_file():
        logger.error("agent_core.py not found under %s", agent_dir)
        return 1
    sys.path.insert(0, str(agent_dir))
    import agent_core  # noqa: E402
    from agent_core import KB, active_workload_order, failure_flags  # noqa: E402

    kb = KB.load()
    models = sorted(kb.dim_pct["model"].unique())

    registry = [{
        "id": r["model"], "family": r["family"],
        "needs_gpu": bool(r["needs_gpu"]),
        "model_size": r4(r["model_size"]), "infer_time": r4(r["infer_time"]),
        "fit_time": r4(r["fit_time"]),
        "flags": failure_flags(kb, r["model"]),
    } for _, r in kb.registry.iterrows()]

    # The plan a brand-new detector should follow: every published model acts as
    # a reference, so the order is fixed before the new detector is measured.
    order = active_workload_order(kb, models)
    order_pos = {w: i + 1 for i, w in enumerate(order)}
    disc = diagnostic_value(kb, agent_core, models)

    workloads = [{
        "workload": r["workload"], "track": r["track"], "source": r["source"],
        "target": r["target"], "n_events": int(r["n_events"]),
        "cost_proxy": r4(r["cost_proxy"]),
        "disc": r4(disc[r["workload"]]),
        "order_pos": order_pos[r["workload"]],
    } for _, r in kb.meta.iterrows()]

    dims = [{"m": r["model"], "w": r["workload"],
             "a": r4(r["accuracy"]), "e": r4(r["earliness"]),
             "r": r4(r["reliability"]), "c": r4(r["cost"])}
            for r in kb.dim_pct.to_dict("records")]

    metrics = [{"m": r["model"], "w": r["workload"],
                **{c: r4(r[c]) for c in METRIC_COLS}}
               for r in kb.wide.to_dict("records")]

    cards = {}
    for m in models:
        g = kb.dim_pct[kb.dim_pct["model"] == m]
        cards[m] = {
            "accuracy": r4(g["accuracy"].mean()),
            "earliness": r4(g["earliness"].mean()),
            "reliability": r4(g["reliability"].mean()),
            "generalization": r4(
                g[g["track"].isin(["cross-turbine", "cross-farm"])]["accuracy"].mean()),
            "cost": r4(g["cost"].mean()),
        }

    results = agent_dir / "results"
    summary = json.loads((results / "summary.json").read_text())
    data = {
        "registry": registry, "workloads": workloads, "dims": dims,
        "metrics": metrics, "cards": cards, "farms": FARM_META,
        "modeB": {
            "curve": pd.read_csv(results / "modeB_curve.csv").round(4).to_dict("records"),
            "random": pd.read_csv(results / "modeB_random.csv").round(4).to_dict("records"),
            "order": order,
            "radius": radius_curve(kb, agent_core, models, order),
            "summary": summary["mode_b"],
        },
        "modeA": summary["mode_a"],
        "policy": {
            "risk_aversion": 0.5, "min_support": 4.0, "abstain_gap": 0.02,
            "top_k": 3, "knn_k": 3, "max_active": 6,
            "replay_budgets": sorted(int(k) for k in summary["mode_b"]["mae_at_K"]),
        },
    }
    args.out.write_text("window.WINDAD = " + json.dumps(data) + ";", encoding="utf-8")
    logger.info("wrote %s (%.0f KB)", args.out, args.out.stat().st_size / 1024)
    logger.info("evaluation plan: %s", " > ".join(order[:6]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
