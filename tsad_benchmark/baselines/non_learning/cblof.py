# -*- coding: utf-8 -*-


from __future__ import annotations

from typing import Optional

from pyod.models.cblof import CBLOF as _PyodCBLOF

from tsad_benchmark.baselines._pyod_base import PyODBaseModel
from tsad_benchmark.common.random_utils import DEFAULT_SEED


class CBLOFModel(PyODBaseModel):


    model_name = "CBLOF"
    _scale_inputs = True

    def __init__(
        self,
        n_clusters: int = 8,
        alpha: float = 0.9,
        beta: float = 5.0,
        use_weights: bool = False,
        contamination: float = 0.05,
        random_state: int = DEFAULT_SEED,
    ) -> None:
        super().__init__(contamination=contamination)
        self.n_clusters = int(n_clusters)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.use_weights = bool(use_weights)
        self.random_state = int(random_state)
        self.model_hyper_params.update(
            {
                "n_clusters": self.n_clusters,
                "alpha": self.alpha,
                "beta": self.beta,
                "use_weights": self.use_weights,
                "random_state": self.random_state,
            }
        )

    def _build_pyod_model(self):
        return _PyodCBLOF(
            n_clusters=self.n_clusters,
            contamination=self.contamination,
            alpha=self.alpha,
            beta=self.beta,
            use_weights=self.use_weights,
            random_state=self.random_state,
        )

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:
        n_test = self._nrows(test_data)
        d = self._ncols(test_data, fallback=self._n_features or 1)
        k = max(self.n_clusters, 1)
        flops = n_test * k * d
        return float(flops) if flops > 0 else float("nan")
