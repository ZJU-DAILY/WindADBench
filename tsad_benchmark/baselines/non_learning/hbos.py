# -*- coding: utf-8 -*-

from __future__ import annotations

from pyod.models.hbos import HBOS as _PyodHBOS

from tsad_benchmark.baselines._pyod_base import PyODBaseModel


class HBOSModel(PyODBaseModel):
    """PyOD-backed HBOS detector."""

    model_name = "HBOS"

    def __init__(
        self,
        n_bins: int = 10,
        alpha: float = 0.1,
        tol: float = 0.5,
        contamination: float = 0.05,
    ) -> None:
        super().__init__(contamination=contamination)
        self.n_bins = int(n_bins)
        self.alpha = float(alpha)
        self.tol = float(tol)
        self.model_hyper_params.update(
            {"n_bins": self.n_bins, "alpha": self.alpha, "tol": self.tol}
        )

    def _build_pyod_model(self):
        return _PyodHBOS(
            n_bins=self.n_bins,
            alpha=self.alpha,
            tol=self.tol,
            contamination=self.contamination,
        )

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:
        n_test = self._nrows(test_data)
        d = self._ncols(test_data, fallback=self._n_features or 1)
        flops = n_test * d * 3
        return float(flops) if flops > 0 else float("nan")
