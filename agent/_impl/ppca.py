from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from evidence import DIM_METRICS, DIMS

METRICS = [(DIMS.index(d), col, higher, rule) for d, spec in DIM_METRICS.items() for col, higher, rule in spec]
R_MAX = 15
ND = len(DIMS)


class RawTensor:
    def __init__(self, cells: pd.DataFrame, models: Sequence[str], workloads: Sequence[str]):
        self.models, self.workloads = list(models), list(workloads)
        c = cells.set_index(["model", "workload"])
        self.X = np.array([[[float(c.at[(m, w), col]) for _, col, _, _ in METRICS] for w in self.workloads]
                           for m in self.models])
        self.dim_of = np.array([d for d, _, _, _ in METRICS])

    def scores(self, pop: Sequence[int], tgt: Sequence[int]) -> np.ndarray:
        pop, tgt = np.asarray(pop), np.asarray(tgt)
        member = np.isin(tgt, pop)
        T, W, K = len(tgt), self.X.shape[1], self.X.shape[2]
        P = np.full((T, W, K), np.nan)
        for k, (_, _, higher, rule) in enumerate(METRICS):
            s = 1.0 if higher else -1.0
            for w in range(W):
                pv, tv = self.X[pop, w, k], self.X[tgt, w, k]
                kp, kt = s * pv, s * tv
                if rule == "worst":
                    kp, kt = np.where(np.isnan(pv), -np.inf, kp), np.where(np.isnan(tv), -np.inf, kt)
                elif rule == "best":
                    kp, kt = np.where(np.isnan(pv), np.inf, kp), np.where(np.isnan(tv), np.inf, kt)
                elif rule == "omit":
                    kp = kp[~np.isnan(kp)]
                srt = np.sort(kp)
                n = len(srt)
                with np.errstate(invalid="ignore"):
                    lo = np.searchsorted(srt, kt, "left")
                    ties = np.searchsorted(srt, kt, "right") - lo
                pct = np.where(member, (lo + (ties + 1) / 2) / n, (1 + lo + 0.5 * ties) / (n + 1))
                P[:, w, k] = np.where(np.isnan(kt), np.nan, pct)
        with np.errstate(all="ignore"):
            return np.stack([np.nanmean(P[:, :, self.dim_of == d], axis=2) for d in range(ND)], axis=2)


def _eig(Y: np.ndarray):
    mu = Y.mean(axis=0)
    X = Y - mu
    ev, U = np.linalg.eigh(X.T @ X / len(Y))
    o = np.argsort(ev)[::-1]
    return mu, np.clip(ev[o], 0.0, None), U[:, o]


def _sigma2(ev: np.ndarray, r: int) -> float:
    return max(float(ev[r:].mean()), 1e-10)


def select_rank(Y: np.ndarray, r_max: int = R_MAX) -> int:
    n, D = Y.shape
    r_max = min(r_max, n - 2, D - 1)
    ll = np.zeros(r_max)
    for j in range(n):
        mu, ev, U = _eig(np.delete(Y, j, axis=0))
        z2 = (U.T @ (Y[j] - mu)) ** 2
        for r in range(1, r_max + 1):
            s2 = _sigma2(ev, r)
            lam = np.maximum(ev[:r], s2)
            ll[r - 1] += -0.5 * (np.log(lam).sum() + (D - r) * np.log(s2) + (z2[:r] / lam).sum() + z2[r:].sum() / s2)
    return int(np.argmax(ll) + 1)


class PPCA:
    def __init__(self, Y: np.ndarray, rank: Optional[int] = None):
        self.rank = select_rank(Y) if rank is None else rank
        self.mu, ev, U = _eig(Y)
        self.sigma2 = _sigma2(ev, self.rank)
        self.W = U[:, :self.rank] * np.sqrt(np.clip(ev[:self.rank] - self.sigma2, 0.0, None))
        self.D = Y.shape[1]

    def _post(self, obs):
        Wo = self.W[obs]
        return np.linalg.inv(np.eye(self.rank) + Wo.T @ Wo / self.sigma2)

    def complete(self, obs: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
        if len(obs) == 0:
            return self.mu.copy()
        z = self._post(obs) @ self.W[obs].T @ (y_obs - self.mu[obs]) / self.sigma2
        y = self.mu + self.W @ z
        y[obs] = y_obs
        return y

    def card_cov(self, obs: np.ndarray, A: np.ndarray) -> np.ndarray:
        miss = np.setdiff1d(np.arange(self.D), obs)
        C = self._post(obs) if len(obs) else np.eye(self.rank)
        Am = A[:, miss]
        B = Am @ self.W[miss]
        return B @ C @ B.T + self.sigma2 * (Am @ Am.T)


def card_matrix(q: np.ndarray) -> np.ndarray:
    W = len(q)
    A = np.zeros((ND, W * ND))
    for w in range(W):
        for d in range(ND):
            A[d, w * ND + d] = q[w]
    return A


def cells_of(idx: Sequence[int]) -> np.ndarray:
    return np.array([w * ND + d for w in idx for d in range(ND)], dtype=int)


def plan_order(model: PPCA, A: np.ndarray, omega2: np.ndarray, marginal) -> List[int]:
    W = A.shape[1] // ND
    chosen: List[int] = []

    def spread(sel):
        return float(np.sqrt(max(np.trace(model.card_cov(cells_of(sel), A) @ omega2), 0.0)))

    cur = spread(chosen)
    while len(chosen) < W:
        best = None
        for i in range(W):
            if i in chosen:
                continue
            s = spread(chosen + [i])
            c = marginal(chosen, i)
            key = ((cur - s) / c, -c, -i)
            if best is None or key > best[0]:
                best = (key, i, s)
        chosen.append(best[1])
        cur = best[2]
    return chosen


def knn_complete(Y_ref: np.ndarray, y: np.ndarray, obs: np.ndarray, k: int = 3) -> np.ndarray:
    if len(obs) == 0:
        return Y_ref.mean(axis=0)
    d = np.abs(Y_ref[:, obs] - y[obs]).mean(axis=1)
    nb = np.argsort(d)[:k]
    out = Y_ref[nb].mean(axis=0)
    out[obs] = y[obs]
    return out


def pareto_member(cards: np.ndarray, idx: int, cols=(3, 1)) -> bool:
    v = cards[:, list(cols)]
    return not bool(np.any(np.all(v >= v[idx], axis=1) & np.any(v > v[idx], axis=1)))


def rank_of(cards: np.ndarray, idx: int) -> float:
    return float(pd.Series(cards.mean(axis=1)).rank(ascending=False, method="average").iloc[idx])
