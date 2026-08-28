# -*- coding: utf-8 -*-


from __future__ import annotations

from pyod.models.knn import KNN as _PyodKNN

from tsad_benchmark.baselines._pyod_base import PyODBaseModel


class KNNModel(PyODBaseModel):


    model_name = "KNN"
    _scale_inputs = True

    def __init__(
        self,
        n_neighbors: int = 5,
        method: str = "largest",
        radius: float = 1.0,
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: int = 2,
        contamination: float = 0.05,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(contamination=contamination)
        self.n_neighbors = int(n_neighbors)
        self.method = method
        self.radius = float(radius)
        self.algorithm = algorithm
        self.leaf_size = int(leaf_size)
        self.metric = metric
        self.p = int(p)
        self.n_jobs = int(n_jobs)
        self.model_hyper_params.update(
            {
                "n_neighbors": self.n_neighbors,
                "method": self.method,
                "metric": self.metric,
                "p": self.p,
            }
        )

    def _build_pyod_model(self):
        return _PyodKNN(
            contamination=self.contamination,
            n_neighbors=self.n_neighbors,
            method=self.method,
            radius=self.radius,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            n_jobs=self.n_jobs,
        )

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:
        n_test = self._nrows(test_data)
        d = self._ncols(test_data, fallback=self._n_features or 1)
        k = max(self.n_neighbors, 1)
        flops = n_test * k * d if n_test > 0 else 0.0
        return float(flops) if flops > 0 else float("nan")
