# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Optional

from pyod.models.iforest import IForest as _PyodIForest

from tsad_benchmark.baselines._pyod_base import PyODBaseModel
from tsad_benchmark.common.random_utils import DEFAULT_SEED


class IForestModel(PyODBaseModel):
    """PyOD-backed Isolation Forest."""

    model_name = "IForest"

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: str | int | float = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        contamination: float = 0.05,
        random_state: int = DEFAULT_SEED,
        n_jobs: int = -1,
    ) -> None:
        super().__init__(contamination=contamination)
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.max_features = float(max_features)
        self.bootstrap = bool(bootstrap)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.model_hyper_params.update(
            {
                "n_estimators": self.n_estimators,
                "max_samples": self.max_samples,
                "max_features": self.max_features,
                "bootstrap": self.bootstrap,
                "random_state": self.random_state,
            }
        )

    def _build_pyod_model(self):
        return _PyodIForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:
        n_train = self._nrows(train_data)
        n_test = self._nrows(test_data)
        if isinstance(self.max_samples, (int, float)) and self.max_samples != "auto":
            psi = float(self.max_samples)
            if 0.0 < psi <= 1.0:
                psi = max(psi * max(n_train, 1), 1.0)
        else:
            psi = min(256.0, max(n_train, 1.0))
        log_psi = math.log2(max(psi, 2.0))
        flops = self.n_estimators * n_test * log_psi
        return float(flops) if flops > 0 else float("nan")


# ---------------------------------------------------------------------------
# Legacy factory (sklearn adapter)
# ---------------------------------------------------------------------------


def make_isolation_forest(
    n_estimators: int = 200,
    max_samples: str | int | float = "auto",
    contamination: float = 0.05,
    random_state: int = DEFAULT_SEED,
    n_jobs: Optional[int] = -1,
    **kwargs,
):
 
    from sklearn.ensemble import IsolationForest

    return IsolationForest(
        n_estimators=int(n_estimators),
        max_samples=max_samples,
        contamination=float(contamination),
        random_state=int(random_state),
        n_jobs=n_jobs,
        **kwargs,
    )
