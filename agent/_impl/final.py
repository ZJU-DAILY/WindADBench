from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evidence as kb

kb.DIM_METRICS["reliability"] = [("fa_day", False, "error"), ("mtbfa_h", True, "omit"), ("normal_fpr", False, "error")]
import d2m as d2m
import ppca as m3
import planning as a4

assert [c for d, c, h, r in m3.METRICS if d == 2] == ["fa_day", "mtbfa_h", "normal_fpr"]


DEFAULT_EVIDENCE = HERE.parent / "evidence_cells.csv"


def load_evidence(root: Path, evidence_csv: Path = DEFAULT_EVIDENCE) -> kb.Evidence:
    cells = pd.read_csv(Path(root) / evidence_csv, float_precision="round_trip", converters={"fit_key": str})
    required = {
        "model", "family", "workload", "event_f1", "vus_pr", "edr", "delay_h", "fa_day", "mtbfa_h",
        "n_events", "fit_s", "infer_s", "gpu_mb", "eval_fit_s", "eval_infer_s", "fit_key", "n_sequences",
        "event_recall", "point_recall", "acc", "track", "source", "target", "cpu_only", "normal_fpr",
    }
    missing = required.difference(cells.columns)
    if missing:
        raise ValueError(f"Evidence snapshot is missing columns: {', '.join(sorted(missing))}")
    if (len(cells) != 432 or cells.model.nunique() != 36
            or set(cells.workload) != set(kb.workload_table().workload)
            or cells.duplicated(["model", "workload"]).any() or cells.normal_fpr.isna().any()):
        raise ValueError("Evidence snapshot must contain the complete model/workload grid and normal FPR")
    return kb.Evidence(cells=cells, meta=kb.workload_meta(cells), models=sorted(cells.model.unique()))


def recommend(fh: d2m.FayHerriotD2M, models: Sequence[str], omega: np.ndarray,
              excluded: Optional[Dict[str, List[str]]] = None, k: int = 3) -> dict:
    excluded = excluded or {}
    feas = [i for i, m in enumerate(models) if m not in excluded]
    if not feas:
        return {"decision": "no_feasible_detector", "primary": None, "top_k": [], "margin": None, "excluded": excluded}
    u, lcb = fh.scores(omega, a4.LAMBDA)
    order = sorted(feas, key=lambda i: (-lcb[i], -u[i], i))
    if len(order) == 1:
        return {"decision": "recommend_single_feasible", "primary": models[order[0]], "top_k": [models[order[0]]],
                "margin": float("inf"), "excluded": excluded}
    margin = float(lcb[order[0]] - lcb[order[1]])
    return {"decision": "recommend" if margin >= a4.EPSILON else "abstain", "primary": models[order[0]],
            "top_k": [models[i] for i in order[:k]], "margin": margin, "excluded": excluded}


def exclusion_reasons(cells: pd.DataFrame, models: Sequence[str], evidence: Sequence[str], no_gpu: bool = False,
                      max_fa_day: Optional[float] = None, max_normal_fpr: Optional[float] = None) -> Dict[str, List[str]]:
    ev = cells[cells.workload.isin(list(evidence))]
    fa = ev.groupby("model").fa_day.mean()
    cf = ev[ev.track == "cross-farm"].groupby("model").normal_fpr.mean()
    cpu = cells.groupby("model").cpu_only.first()
    out: Dict[str, List[str]] = {}
    for m in models:
        r = []
        if no_gpu and not cpu[m]:
            r.append("requires a GPU")
        if max_fa_day is not None and fa[m] > max_fa_day:
            r.append(f"mean FA/day {fa[m]:.2f} > {max_fa_day}")
        if max_normal_fpr is not None and cf[m] > max_normal_fpr:
            r.append(f"cross-farm normal FPR {cf[m]:.3f} > {max_normal_fpr}")
        if r:
            out[m] = r
    return out


def topsis(profile: np.ndarray, omega: np.ndarray) -> np.ndarray:
    r = profile / np.sqrt((profile ** 2).sum(axis=0, keepdims=True))
    v = r * omega[None]
    best, worst = v.max(axis=0), v.min(axis=0)
    dp, dm = np.sqrt(((v - best) ** 2).sum(axis=1)), np.sqrt(((v - worst) ** 2).sum(axis=1))
    return dm / np.maximum(dp + dm, 1e-12)
