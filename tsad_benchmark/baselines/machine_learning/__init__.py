# -*- coding: utf-8 -*-
"""
Classical / shallow machine-learning baselines.

Mirrors ``scripts/baselines/machine_learning/``: IForest, EIF, KMeans,
KNN, OCSVM, PCA, LODA, Torsk, DAGMM, DeepPoint.

Heavy / optional dependencies (``eif``, ``salesforce-merlion``, ``torch``)
are imported lazily so a minimal install can still load the lightweight
PyOD-only baselines.
"""

from tsad_benchmark.baselines.machine_learning.iforest import (
    IForestModel,
    make_isolation_forest,
)
from tsad_benchmark.baselines.machine_learning.kmeans import KMeansModel
from tsad_benchmark.baselines.machine_learning.knn import KNNModel
from tsad_benchmark.baselines.machine_learning.loda import LODAModel
from tsad_benchmark.baselines.machine_learning.ocsvm import OCSVMModel
from tsad_benchmark.baselines.machine_learning.pca import PCAModel

try:
    from tsad_benchmark.baselines.machine_learning.eif import EIFModel
except Exception:  # pragma: no cover
    EIFModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.machine_learning.torsk import TorskModel
except Exception:  # pragma: no cover
    TorskModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.machine_learning.dagmm import DAGMMModel
except Exception:  # pragma: no cover
    DAGMMModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.machine_learning.deeppoint import DeepPointModel
except Exception:  # pragma: no cover
    DeepPointModel = None  # type: ignore[assignment]


__all__ = [
    "DAGMMModel",
    "DeepPointModel",
    "EIFModel",
    "IForestModel",
    "KMeansModel",
    "KNNModel",
    "LODAModel",
    "OCSVMModel",
    "PCAModel",
    "TorskModel",
    "make_isolation_forest",
]
