# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Union

from pyod.models.ocsvm import OCSVM as _PyodOCSVM

from tsad_benchmark.baselines._pyod_base import PyODBaseModel


class OCSVMModel(PyODBaseModel):

    model_name = "OCSVM"
    _scale_inputs = True

    def __init__(
        self,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: Union[str, float] = "auto",
        coef0: float = 0.0,
        tol: float = 1e-3,
        nu: float = 0.5,
        shrinking: bool = True,
        cache_size: int = 200,
        max_iter: int = -1,
        contamination: float = 0.05,
    ) -> None:
        super().__init__(contamination=contamination)
        self.kernel = kernel
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.nu = float(nu)
        self.shrinking = bool(shrinking)
        self.cache_size = int(cache_size)
        self.max_iter = int(max_iter)
        self.model_hyper_params.update(
            {
                "kernel": self.kernel,
                "gamma": self.gamma,
                "nu": self.nu,
                "degree": self.degree,
            }
        )

    def _build_pyod_model(self):
        return _PyodOCSVM(
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            tol=self.tol,
            nu=self.nu,
            shrinking=self.shrinking,
            cache_size=self.cache_size,
            max_iter=self.max_iter,
            contamination=self.contamination,
        )

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:
        n_train = self._nrows(train_data)
        n_test = self._nrows(test_data)
        d = self._ncols(test_data, fallback=self._n_features or 1)
        nu = max(min(self.nu, 1.0), 1e-3)
        n_sv = max(int(n_train * nu), 1)
        flops = n_test * n_sv * d if n_test > 0 else 0.0
        return float(flops) if flops > 0 else float("nan")
