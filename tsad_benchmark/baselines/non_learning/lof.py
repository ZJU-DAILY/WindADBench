# -*- coding: utf-8 -*-


from __future__ import annotations

from typing import Optional

from pyod.models.lof import LOF as _PyodLOF
from sklearn.neighbors import LocalOutlierFactor

from tsad_benchmark.baselines._pyod_base import PyODBaseModel


class LOFModel(PyODBaseModel):

    model_name = "LOF"
    _scale_inputs = True

    def __init__(
        self,
        n_neighbors: int = 20,
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: int = 2,
        contamination: float = 0.05,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(contamination=contamination)
        self.n_neighbors = int(n_neighbors)
        self.algorithm = algorithm
        self.leaf_size = int(leaf_size)
        self.metric = metric
        self.p = int(p)
        self.n_jobs = int(n_jobs)
        self.model_hyper_params.update(
            {
                "n_neighbors": self.n_neighbors,
                "algorithm": self.algorithm,
                "leaf_size": self.leaf_size,
                "metric": self.metric,
                "p": self.p,
            }
        )

    def _build_pyod_model(self):
        return _PyodLOF(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            contamination=self.contamination,
            n_jobs=self.n_jobs,
            novelty=True,
        )

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:
        n_test = self._nrows(test_data)
        d = self._ncols(test_data, fallback=self._n_features or 1)
        k = max(self.n_neighbors, 1)
        flops = n_test * k * (d + 1) if n_test > 0 else 0.0
        return float(flops) if flops > 0 else float("nan")


# ---------------------------------------------------------------------------
# Legacy factory — kept so configs still using adapter="sklearn" keep working.
# ---------------------------------------------------------------------------


def make_lof(
    n_neighbors: int = 20,
    algorithm: str = "auto",
    leaf_size: int = 30,
    metric: str = "minkowski",
    p: int = 2,
    contamination: float = 0.05,
    n_jobs: int = 1,
    novelty: bool = True,
    **kwargs,
) -> LocalOutlierFactor:
    """Build a sklearn ``LocalOutlierFactor`` (inductive by default)."""
    return LocalOutlierFactor(
        n_neighbors=int(n_neighbors),
        algorithm=algorithm,
        leaf_size=int(leaf_size),
        metric=metric,
        p=int(p),
        contamination=float(contamination),
        n_jobs=n_jobs,
        novelty=bool(novelty),
        **kwargs,
    )
