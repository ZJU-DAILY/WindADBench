from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

FARMS = ["A", "B", "C"]
DIMS = ["accuracy", "earliness", "reliability", "cost"]


DIM_METRICS: Dict[str, List[tuple]] = {
    "accuracy": [("event_f1", True, "error"), ("vus_pr", True, "error")],
    "earliness": [("edr", True, "error"), ("delay_h", False, "worst")],
    "reliability": [("fa_day", False, "error"), ("mtbfa_h", True, "omit")],
    "cost": [("fit_s", False, "error"), ("infer_s", False, "error"), ("gpu_mb", False, "best")],
}


def workload_table() -> pd.DataFrame:
    rows = []
    for x in FARMS:
        rows.append(dict(workload=f"IF-{x}", track="in-farm", source=x, target=x))
    for x in FARMS:
        rows.append(dict(workload=f"CT-{x}", track="cross-turbine", source=x, target=x))
    for x in FARMS:
        for y in FARMS:
            if x != y:
                rows.append(dict(workload=f"CF-{x}>{y}", track="cross-farm", source=x, target=y))
    return pd.DataFrame(rows)


def _percentile(values: np.ndarray, higher: bool, rule: str, population: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    p = np.asarray(population, dtype=float)
    sign = 1.0 if higher else -1.0

    def key(a):
        k = sign * a
        if rule == "worst":
            k = np.where(np.isnan(a), -np.inf, k)
        elif rule == "best":
            k = np.where(np.isnan(a), np.inf, k)
        return k

    kv, kp = key(v), key(p)
    if rule == "omit":
        kp = kp[~np.isnan(kp)]
    elif rule == "error":
        assert not np.isnan(kv).any() and not np.isnan(kp).any()
    n = len(kp)
    out = np.full(len(v), np.nan)
    for i, x in enumerate(kv):
        if np.isnan(x) or n == 0:
            continue
        worse = np.sum(kp < x)
        ties = np.sum(kp == x)
        out[i] = (worse + 0.5 * (ties + 1)) / n if ties else (worse + 1.0) / (n + 1)
    return out


def dim_scores(cells: pd.DataFrame, population: Sequence[str],
               models: Optional[Sequence[str]] = None) -> pd.DataFrame:
    population = list(population)
    models = list(models) if models is not None else population
    rows = []
    for wl, g in cells.groupby("workload", sort=True):
        g = g.set_index("model")
        pop = g.loc[population]
        tgt = g.loc[models]
        out = pd.DataFrame(index=tgt.index)
        for dim, spec in DIM_METRICS.items():
            parts = []
            for col, higher, rule in spec:
                pv = pop[col].to_numpy(float)
                tv = tgt[col].to_numpy(float)
                pct = np.empty(len(tv))
                in_pop = np.isin(tgt.index, population)

                if in_pop.any():
                    pct[in_pop] = _percentile(tv[in_pop], higher, rule, pv)

                if (~in_pop).any():
                    pct[~in_pop] = _insert_percentile(tv[~in_pop], higher, rule, pv)
                parts.append(pct)
            with np.errstate(all="ignore"):
                out[dim] = np.nanmean(np.vstack(parts), axis=0)
        out["workload"] = wl
        rows.append(out.reset_index())
    d = pd.concat(rows, ignore_index=True)
    assert np.isfinite(d[DIMS].to_numpy()).all()
    return d.merge(workload_table(), on="workload")


def _insert_percentile(values: np.ndarray, higher: bool, rule: str, population: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    p = np.asarray(population, dtype=float)
    sign = 1.0 if higher else -1.0
    kv, kp = sign * v, sign * p
    if rule == "worst":
        kv, kp = np.where(np.isnan(v), -np.inf, kv), np.where(np.isnan(p), -np.inf, kp)
    elif rule == "best":
        kv, kp = np.where(np.isnan(v), np.inf, kv), np.where(np.isnan(p), np.inf, kp)
    elif rule == "omit":
        kp = kp[~np.isnan(kp)]
    out = np.full(len(v), np.nan)
    n = len(kp)
    for i, x in enumerate(kv):
        if np.isnan(x):
            continue
        out[i] = (1.0 + np.sum(kp < x) + 0.5 * np.sum(kp == x)) / (n + 1)
    return out


def support_weights(workloads: Sequence[str], meta: pd.DataFrame) -> pd.Series:
    n = meta.set_index("workload").loc[list(workloads), "n_events"].astype(float)
    q = np.log1p(n.clip(lower=1.0))
    return q / q.sum()


def workload_meta(cells: pd.DataFrame) -> pd.DataFrame:
    meta = cells.groupby("workload").agg(n_events=("n_events", "first"),
                                         n_sequences=("n_sequences", "first")).reset_index()
    assert cells.groupby("workload").n_events.nunique().max() == 1
    return meta.merge(workload_table(), on="workload")


@dataclass
class Evidence:
    cells: pd.DataFrame
    meta: pd.DataFrame
    models: List[str]

    def array(self, scores: pd.DataFrame, models: Sequence[str], workloads: Sequence[str]) -> np.ndarray:
        return np.stack([scores.pivot(index="model", columns="workload", values=d)
                         .reindex(index=list(models), columns=list(workloads)).to_numpy(float)
                         for d in DIMS], axis=2)
