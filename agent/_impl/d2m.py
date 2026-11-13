from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from evidence import DIMS, FARMS, support_weights

LAMBDA = 0.5
EPSILON = 0.02
FIXED_POOL = ("anomaly_transformer", "d3r", "lof")


def preference_grid() -> List[np.ndarray]:
    return [np.array(w, dtype=float) for w in itertools.product(np.arange(0, 1.01, 0.25), repeat=4)
            if abs(sum(w) - 1.0) < 1e-9]


def lofo_sets(target: str) -> Dict[str, List[str]]:
    a, b = [f for f in FARMS if f != target]
    return {"evidence": [f"IF-{a}", f"IF-{b}", f"CT-{a}", f"CT-{b}", f"CF-{a}>{b}", f"CF-{b}>{a}"],
            "truth": [f"CF-{a}>{target}", f"CF-{b}>{target}"]}


def track_mean(R: np.ndarray, workloads: Sequence[str], members: Sequence[str], meta: pd.DataFrame) -> np.ndarray:
    idx = [list(workloads).index(w) for w in members]
    q = support_weights(members, meta).to_numpy()
    return np.tensordot(R[:, idx, :], q, axes=([1], [0]))


class FayHerriotD2M:
    def __init__(self, R: np.ndarray, workloads: Sequence[str], meta: pd.DataFrame, evidence: Sequence[str]):
        workloads = list(workloads)
        ev_if = [w for w in evidence if w.startswith("IF-")]
        ev_ct = [w for w in evidence if w.startswith("CT-")]
        ev_cf = [w for w in evidence if w.startswith("CF-")]
        assert len(ev_cf) == 2 and len(ev_if) == 2 and len(ev_ct) == 2
        x_if = track_mean(R, workloads, ev_if, meta)
        x_ct = track_mean(R, workloads, ev_ct, meta)
        q = support_weights(ev_cf, meta).to_numpy()
        y1 = R[:, workloads.index(ev_cf[0]), :]
        y2 = R[:, workloads.index(ev_cf[1]), :]
        ybar = q[0] * y1 + q[1] * y2
        n, p = R.shape[0], 3
        self.theta = np.empty_like(ybar)
        self.V = np.empty_like(ybar)
        self.reg = np.empty_like(ybar)
        self.ybar = ybar
        self.x_if = x_if
        self.diag = {}
        for d, dim in enumerate(DIMS):
            X = np.column_stack([np.ones(n), x_if[:, d], x_ct[:, d]])
            beta, *_ = np.linalg.lstsq(X, ybar[:, d], rcond=None)
            fitted = X @ beta
            resid = ybar[:, d] - fitted
            sigma2 = float(np.mean((y1[:, d] - y2[:, d]) ** 2) / 2.0)
            psi = float((q[0] ** 2 + q[1] ** 2) * sigma2)
            tau2 = max(0.0, float(resid @ resid / (n - p)) - psi)
            gamma = tau2 / (tau2 + psi) if tau2 + psi > 0 else 0.0
            h = np.einsum("ij,jk,ik->i", X, np.linalg.pinv(X.T @ X), X)
            self.theta[:, d] = fitted + gamma * resid
            self.V[:, d] = gamma * psi + (1.0 - gamma) ** 2 * h * (tau2 + psi)
            self.reg[:, d] = fitted
            self.diag[dim] = {"gamma": gamma, "tau2": tau2, "psi": psi, "sigma2_direction": sigma2,
                              "beta_intercept": float(beta[0]), "beta_in_farm": float(beta[1]),
                              "beta_cross_turbine": float(beta[2])}

    def scores(self, omega: np.ndarray, lam: float = LAMBDA):
        u = self.theta @ omega
        se = np.sqrt(self.V @ (omega ** 2))
        return u, u - lam * se


def order_by(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    return np.lexsort((-secondary, -primary))


def truth_utility(R: np.ndarray, workloads: Sequence[str], meta: pd.DataFrame,
                  truth: Sequence[str], omega: np.ndarray) -> np.ndarray:
    return track_mean(R, workloads, truth, meta) @ omega


class OriginalD2M:
    def __init__(self, core, scores: pd.DataFrame, cells: pd.DataFrame, meta: pd.DataFrame):
        wide = cells.rename(columns={"fa_day": "false_alarms_per_turbine_day", "edr": "early_detection_rate"})
        registry = cells.groupby(["model", "family"]).agg(
            needs_gpu=("cpu_only", lambda s: not bool(s.iloc[0])), infer_time=("infer_s", "median")).reset_index()
        registry["model_size"] = np.nan
        dim_pct = scores[["model", "workload", "track", "source", "target"] + DIMS].copy()
        self.core = core
        self.kb = core.KB(dim_pct=dim_pct, coords=dim_pct[["model", "workload"]].copy(), registry=registry,
                          meta=meta.assign(cost_proxy=np.nan), wide=wide)

    def recommend(self, omega: np.ndarray, evidence: Sequence[str]):
        prof = self.core.Profile(dict(zip(DIMS, omega)), risk_aversion=LAMBDA, abstain_gap=0.0)
        rec = self.core.recommend(self.kb, prof, list(evidence), k=5, allow_abstain=False)
        return rec

    def utility(self, omega: np.ndarray, workloads: Sequence[str], models=None) -> pd.Series:
        return self.core.utility(self.kb, dict(zip(DIMS, omega)), list(workloads), models)
