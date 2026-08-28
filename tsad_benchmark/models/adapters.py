# -*- coding: utf-8 -*-
"""
Third-party model adapters.

Background
----------
External models have varying interfaces:

- Different method names (fit / train / learn / ...).
- Different input formats (DataFrame / ndarray / tensor).
- Different output formats (list / ndarray / dict).
- Some only support windowed input.

Adapters absorb these differences inside the model layer so that the
evaluator always calls the unified ``fit / detect_score / detect_label``
interface.

Adapters are NOT responsible for:

- Metric computation
- Data loading
- Training orchestration
- Result export

Usage
-----
Specify ``adapter`` in model config::

    {"model_path": "...", "adapter": "sklearn"}

Or wrap manually::

    from tsad_benchmark.models.adapters import SklearnAdapter
    wrapped = SklearnAdapter(my_sklearn_model)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Sklearn-style adapter
# ---------------------------------------------------------------------------


class SklearnAdapter(AnomalyModelBase):
    """
    Adapt sklearn-style anomaly detectors.

    Conventions:

    - Training: ``model.fit(X)`` where X is an ndarray.
    - Inference: ``model.decision_function(X)`` or ``model.score_samples(X)``
      (lower = more anomalous; negated to match "higher = more anomalous").
    - If the model has ``predict``, it returns -1 (anomaly) / 1 (normal).

    Typical models: IsolationForest, OneClassSVM, LocalOutlierFactor.
    """

    model_name = "SklearnModel"
    capability = ModelCapability.score_and_label()

    def __init__(self, raw_model: Any) -> None:
        self._raw = raw_model

    def fit(
        self,
        train_data: pd.DataFrame,
        train_label: Optional[pd.DataFrame] = None,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> None:
        X = train_data.values if isinstance(train_data, pd.DataFrame) else train_data
        self._raw.fit(X)

    def detect_score(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        X = test_data.values if isinstance(test_data, pd.DataFrame) else test_data
        if hasattr(self._raw, "decision_function"):
            scores = self._raw.decision_function(X)
        elif hasattr(self._raw, "score_samples"):
            scores = self._raw.score_samples(X)
        else:
            raise AttributeError(
                f"{type(self._raw).__name__} has neither "
                "decision_function nor score_samples."
            )
        # sklearn: lower = more anomalous; negate to match benchmark convention
        return -np.asarray(scores, dtype=float)

    def detect_label(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        X = test_data.values if isinstance(test_data, pd.DataFrame) else test_data
        if not hasattr(self._raw, "predict"):
            raise AttributeError(
                f"{type(self._raw).__name__} has no predict method."
            )
        raw_labels = np.asarray(self._raw.predict(X))
        # sklearn convention: -1 = anomaly, 1 = normal -> convert to 0/1
        return np.where(raw_labels == -1, 1, 0).astype(int)


# ---------------------------------------------------------------------------
# 2. PyTorch-style adapter
# ---------------------------------------------------------------------------


class PytorchAdapter(AnomalyModelBase):
    """
    Adapt PyTorch deep-learning anomaly detectors.

    Conventions:

    - Model has its network structure already initialised.
    - Training: ``model.fit(data)`` or ``model.train_model(data)``.
    - Inference: ``model.anomaly_score(data)`` returns an ndarray.

    This adapter handles:

    - DataFrame -> ndarray conversion (if needed).
    - Method name unification.
    - Output format normalisation (list / tensor -> ndarray).
    """

    model_name = "PytorchModel"
    capability = ModelCapability.score_only()

    def __init__(self, raw_model: Any) -> None:
        self._raw = raw_model

    def fit(
        self,
        train_data: pd.DataFrame,
        train_label: Optional[pd.DataFrame] = None,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> None:
        data = train_data.values if isinstance(train_data, pd.DataFrame) else train_data
        if hasattr(self._raw, "fit"):
            self._raw.fit(data, **kwargs)
        elif hasattr(self._raw, "train_model"):
            self._raw.train_model(data, **kwargs)
        else:
            raise AttributeError(
                f"{type(self._raw).__name__} has neither fit nor train_model."
            )

    def detect_score(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        data = test_data.values if isinstance(test_data, pd.DataFrame) else test_data
        for method_name in ("anomaly_score", "detect_score", "score", "predict_score"):
            if hasattr(self._raw, method_name):
                result = getattr(self._raw, method_name)(data, **kwargs)
                return _as_float_vector(result)
        raise AttributeError(
            f"{type(self._raw).__name__} has no recognised inference method. "
            "Supported: anomaly_score / detect_score / score / predict_score"
        )


# ---------------------------------------------------------------------------
# 3. Rule-based adapter
# ---------------------------------------------------------------------------


class RuleAdapter(AnomalyModelBase):
    """
    Adapt rule-based / threshold / statistical models.

    These models typically:

    - Do not need training (``fit`` is a no-op).
    - Directly output 0/1 decisions per row.

    Convention: ``model.detect(data)`` returns a 0/1 array.
    """

    model_name = "RuleModel"
    capability = ModelCapability.label_only(supports_fit=False)

    def __init__(self, raw_model: Any) -> None:
        self._raw = raw_model

    def fit(
        self,
        train_data: pd.DataFrame,
        train_label: Optional[pd.DataFrame] = None,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> None:
        if hasattr(self._raw, "fit"):
            data = train_data.values if isinstance(train_data, pd.DataFrame) else train_data
            self._raw.fit(data)

    def detect_label(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        data = test_data.values if isinstance(test_data, pd.DataFrame) else test_data
        for method_name in ("detect", "detect_label", "predict", "classify"):
            if hasattr(self._raw, method_name):
                result = getattr(self._raw, method_name)(data, **kwargs)
                return _as_binary_vector(result)
        raise AttributeError(
            f"{type(self._raw).__name__} has no recognised inference method. "
            "Supported: detect / detect_label / predict / classify"
        )


# ---------------------------------------------------------------------------
# Factory wrapper (used by loader)
# ---------------------------------------------------------------------------


def _adapter_factory_for(
    adapter_cls: type,
    raw_factory: Callable,
) -> Callable:
    """
    Wrap a raw model factory so it returns an adapted instance.

    The loader uses this to replace ``ModelInfo.model_factory``.
    """
    def wrapped_factory(**kwargs) -> AnomalyModelBase:
        raw_model = raw_factory(**kwargs)
        return adapter_cls(raw_model)

    wrapped_factory.__name__ = f"{adapter_cls.__name__}({raw_factory.__name__})"
    wrapped_factory.__wrapped__ = raw_factory
    return wrapped_factory


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

ADAPTER: Dict[str, Callable] = {
    "sklearn": lambda factory: _adapter_factory_for(SklearnAdapter, factory),
    "pytorch": lambda factory: _adapter_factory_for(PytorchAdapter, factory),
    "rule": lambda factory: _adapter_factory_for(RuleAdapter, factory),
}

ADAPTER_CAPABILITY: Dict[str, ModelCapability] = {
    "sklearn": SklearnAdapter.capability,
    "pytorch": PytorchAdapter.capability,
    "rule": RuleAdapter.capability,
}


def register_model_adapter(
    name: str,
    wrapper_fn: Callable,
    capability: Optional[ModelCapability] = None,
) -> None:
    """
    Register a custom adapter.

    :param name:       Adapter name (value of ``adapter`` in config).
    :param wrapper_fn: Function that takes a raw factory and returns a
                       wrapped factory.
    """
    if name in ADAPTER:
        logger.warning("Adapter %r already exists; it will be overwritten.", name)
    ADAPTER[name] = wrapper_fn
    if capability is not None:
        ADAPTER_CAPABILITY[name] = capability


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _as_float_vector(x: Any) -> np.ndarray:
    """Convert various types to a 1-D float ndarray."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1)


def _as_binary_vector(x: Any) -> np.ndarray:
    """Convert various types to a 1-D int ndarray (0/1)."""
    arr = _as_float_vector(x)
    return (arr >= 0.5).astype(int)
