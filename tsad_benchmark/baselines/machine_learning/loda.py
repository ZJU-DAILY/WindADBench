# -*- coding: utf-8 -*-

from __future__ import annotations

from pyod.models.loda import LODA as _PyodLODA

from tsad_benchmark.baselines._pyod_base import PyODBaseModel


class LODAModel(PyODBaseModel):
    """PyOD-backed LODA detector."""

    model_name = "LODA"

    def __init__(
        self,
        n_bins: int = 10,
        n_random_cuts: int = 100,
        contamination: float = 0.05,
    ) -> None:
        super().__init__(contamination=contamination)
        self.n_bins = int(n_bins)
        self.n_random_cuts = int(n_random_cuts)
        self.model_hyper_params.update(
            {
                "n_bins": self.n_bins,
                "n_random_cuts": self.n_random_cuts,
            }
        )

    def _build_pyod_model(self):
        return _PyodLODA(
            contamination=self.contamination,
            n_bins=self.n_bins,
            n_random_cuts=self.n_random_cuts,
        )

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:
        n_test = self._nrows(test_data)
        d = self._ncols(test_data, fallback=self._n_features or 1)
        cuts = max(self.n_random_cuts, 1)
        flops = n_test * (d * cuts + cuts)
        return float(flops) if flops > 0 else float("nan")
