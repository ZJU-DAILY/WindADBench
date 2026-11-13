from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

import evidence as kb
import d2m as d2m
import ppca as m3

LAMBDA, EPSILON = d2m.LAMBDA, d2m.EPSILON
RHO, SMOOTH, MAX_ACTIVE = 0.02, 0.5, 6
QUANTILE, STOP_RADIUS = 0.90, 0.08


def recommend(fh: d2m.FayHerriotD2M, models: Sequence[str], omega: np.ndarray, feasible: Optional[Dict[str, List[str]]] = None,
              k: int = 3, draws: int = 4000, seed: int = 0) -> dict:
    excluded = feasible or {}
    keep = np.array([m not in excluded for m in models])
    u, lcb = fh.scores(omega, LAMBDA)
    u, lcb = np.where(keep, u, -np.inf), np.where(keep, lcb, -np.inf)
    order = d2m.order_by(lcb, u)
    order = [i for i in order if keep[i]]
    margin = float(lcb[order[0]] - lcb[order[1]])
    rng = np.random.default_rng(seed)
    th = fh.theta[keep][None] + rng.standard_normal((draws,) + fh.theta[keep].shape) * np.sqrt(fh.V[keep])[None]
    best = np.argmax(th @ omega, axis=1)
    idx = {i: j for j, i in enumerate(np.flatnonzero(keep))}
    pcs = float(np.mean(best == idx[order[0]]))
    theta = fh.theta[keep]
    front = [models[i] for j, i in enumerate(np.flatnonzero(keep))
             if not np.any(np.all(theta >= theta[j], axis=1) & np.any(theta > theta[j], axis=1))]
    return {"decision": "recommend" if margin >= EPSILON else "abstain", "primary": models[order[0]],
            "top_k": [models[i] for i in order[:k]], "margin": margin, "confidence_pcs": pcs,
            "pareto_front": front,
            "low_cost_fallback": models[max(np.flatnonzero(keep), key=lambda i: fh.theta[i, 3])],
            "high_sensitivity_fallback": models[max(np.flatnonzero(keep), key=lambda i: fh.theta[i, 1])],
            "excluded": excluded}


class InnerReplay:
    def __init__(self, raw: m3.RawTensor, refs: Sequence[int], A: np.ndarray):
        self.A = A
        self.items = []
        for j in refs:
            pop = [x for x in refs if x != j]
            self.items.append((m3.PPCA(raw.scores(pop, pop).reshape(len(pop), -1)), raw.scores(pop, [j]).reshape(-1)))
        self._cache = {}

    def residuals(self, chosen: Sequence[int]) -> np.ndarray:
        key = tuple(sorted(chosen))
        if key not in self._cache:
            obs = m3.cells_of(list(key))
            self._cache[key] = np.array([np.abs(self.A @ (mdl.complete(obs, y[obs]) - y)) for mdl, y in self.items])
        return self._cache[key]

    def ecv(self, chosen: Sequence[int]) -> float:
        return float(self.residuals(chosen).mean())

    def radius(self, chosen: Sequence[int]) -> float:
        r = np.sort(self.residuals(chosen).ravel())
        k = int(np.ceil(QUANTILE * (len(r) + 1)))
        return float(r[k - 1]) if k <= len(r) else float("inf")


def plan_order(inner: InnerReplay, tracks: np.ndarray, marginal, standalone: np.ndarray,
               rho: float = RHO, smooth: float = SMOOTH, max_active: int = MAX_ACTIVE) -> List[int]:
    W = len(tracks)
    scale = float(np.median(standalone))
    chosen: List[int] = []
    while len(chosen) < min(W, max_active):
        base = inner.ecv(chosen)
        seen = set(tracks[chosen]) if chosen else set()
        cands = []
        for i in range(W):
            if i in chosen:
                continue
            gain = base - inner.ecv(chosen + [i])
            c = marginal(chosen, i)
            if gain > 0:
                cands.append((gain / (c / scale + smooth) + (rho if tracks[i] not in seen else 0.0), -c, -i))
        if not cands:
            break
        chosen.append(-max(cands)[2])
    while len(chosen) < W:
        rest = [i for i in range(W) if i not in chosen]
        chosen.append(min(rest, key=lambda i: (marginal(chosen, i), i)))
    return chosen
