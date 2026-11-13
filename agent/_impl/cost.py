from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from evidence import DIMS


def preference_second_moment(grid: Sequence[np.ndarray]) -> np.ndarray:
    G = np.vstack(grid)
    return G.T @ G / len(G)


class CostModel:
    def __init__(self, cells: pd.DataFrame, workloads: Sequence[str]):
        self.workloads = list(workloads)
        c = cells.set_index(["model", "workload"])
        self.fit = c["eval_fit_s"]
        self.infer = c["eval_infer_s"]
        self.key = cells.drop_duplicates("workload").set_index("workload").loc[self.workloads, "fit_key"].to_numpy()

    def components(self, models: Sequence[str], how: str = "median"):
        fit = np.array([np.median([self.fit[(m, w)] for m in models]) for w in self.workloads])
        inf = np.array([np.median([self.infer[(m, w)] for m in models]) for w in self.workloads])
        return fit, inf

    def total(self, fit: np.ndarray, inf: np.ndarray, chosen: Sequence[int]) -> float:
        paid, cost = set(), 0.0
        for i in chosen:
            cost += inf[i]
            k = self.key[i]
            if k and k not in paid:
                cost += fit[i]
                paid.add(k)
        return float(cost)

    def marginal(self, fit: np.ndarray, inf: np.ndarray, chosen: Sequence[int], i: int) -> float:
        k = self.key[i]
        paid = {self.key[j] for j in chosen if self.key[j]}
        return float(inf[i] + (fit[i] if k and k not in paid else 0.0))


def cheapest_order(cost: CostModel, est_fit: np.ndarray, est_inf: np.ndarray, n_workloads: int) -> List[int]:
    chosen: List[int] = []
    while len(chosen) < n_workloads:
        rest = [i for i in range(n_workloads) if i not in chosen]
        chosen.append(min(rest, key=lambda i: (cost.marginal(est_fit, est_inf, chosen, i), i)))
    return chosen



