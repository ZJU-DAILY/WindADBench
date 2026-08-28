# -*- coding: utf-8 -*-
"""
Unified interface for anomaly detection models.

All models plugged into the benchmark must inherit from
:class:`AnomalyModelBase`.

Minimum contract
----------------
- ``fit`` must be implemented (unsupervised models may ignore *train_label*;
  rule-based models may leave the body empty).
- At least one of ``detect_score`` / ``detect_label`` must be implemented.

Capability declaration
----------------------
Subclasses declare their supported output types and other traits via the
class-level ``capability`` attribute.  The benchmark checks compatibility
at *load* time so mismatches surface early rather than mid-evaluation.
"""

from __future__ import annotations

import abc
import math
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Capability constants
# ---------------------------------------------------------------------------

SUPPORT_SCORE = "score"
SUPPORT_LABEL = "label"


class ModelCapability:
    """
    Describes the capability boundary of a model.

    Every field has a sensible default; subclasses only need to override
    fields that differ from the default.

    Attributes
    ----------
    supported_outputs :
        Output types the model can produce.  One of ``["score"]``,
        ``["label"]``, or ``["score", "label"]``.
    training_paradigm :
        ``"unsupervised"`` / ``"semi_supervised"`` / ``"supervised"``.
    supports_fit :
        Whether the model needs (or supports) a training phase.
        Rule-based / statistical models may set this to ``False``.
    supports_covariates :
        Whether the model accepts covariate inputs.
    input_granularity :
        Expected input granularity: ``"point"`` or ``"window"``.
    supports_flops :
        Whether ``estimate_flops`` is implemented.
    """

    def __init__(
        self,
        supported_outputs: Optional[List[str]] = None,
        training_paradigm: str = "unsupervised",
        supports_fit: bool = True,
        supports_covariates: bool = False,
        input_granularity: str = "point",
        supports_flops: bool = False,
    ) -> None:
        self.supported_outputs: List[str] = (
            [SUPPORT_SCORE] if supported_outputs is None else supported_outputs
        )
        self.training_paradigm = training_paradigm
        self.supports_fit = supports_fit
        self.supports_covariates = supports_covariates
        self.input_granularity = input_granularity
        self.supports_flops = supports_flops

    # Convenience constructors ------------------------------------------------

    @classmethod
    def score_only(cls, **kwargs) -> "ModelCapability":
        """Model that only outputs anomaly scores."""
        return cls(supported_outputs=[SUPPORT_SCORE], **kwargs)

    @classmethod
    def label_only(cls, **kwargs) -> "ModelCapability":
        """Model that directly outputs binary labels (rule / classifier)."""
        return cls(supported_outputs=[SUPPORT_LABEL], **kwargs)

    @classmethod
    def score_and_label(cls, **kwargs) -> "ModelCapability":
        """Model that supports both score and label outputs."""
        return cls(supported_outputs=[SUPPORT_SCORE, SUPPORT_LABEL], **kwargs)

    # Query helpers ------------------------------------------------------------

    def has_score_output(self) -> bool:
        return SUPPORT_SCORE in self.supported_outputs

    def has_label_output(self) -> bool:
        return SUPPORT_LABEL in self.supported_outputs

    def __repr__(self) -> str:
        return (
            f"ModelCapability(outputs={self.supported_outputs}, "
            f"paradigm={self.training_paradigm}, "
            f"fit={self.supports_fit})"
        )


# ---------------------------------------------------------------------------
# Default capability (score-only, requires training, unsupervised)
# ---------------------------------------------------------------------------

_DEFAULT_CAPABILITY = ModelCapability.score_only()


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class AnomalyModelBase(metaclass=abc.ABCMeta):
    """
    Unified model interface for the wind-turbine anomaly-detection benchmark.

    Subclasses must implement at minimum:

    - ``fit``
    - one of ``detect_score`` or ``detect_label``

    Quick-start example::

        class MyModel(AnomalyModelBase):
            model_name = "MyModel"
            model_hyper_params = {"window": 50}
            capability = ModelCapability.score_only()

            def fit(self, train_data, train_label=None, **kwargs):
                ...

            def detect_score(self, test_data, **kwargs):
                return np.zeros(len(test_data))
    """

    # --- Class-level attributes (override in subclass) -----------------------

    model_name: str = "AnomalyModel"
    model_hyper_params: dict = {}
    capability: ModelCapability = _DEFAULT_CAPABILITY

    # --- Required interface ---------------------------------------------------

    @abc.abstractmethod
    def fit(
        self,
        train_data: pd.DataFrame,
        train_label: Optional[pd.DataFrame] = None,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Train the model.

        Parameters
        ----------
        train_data :
            Training features (metadata columns already stripped).
        train_label :
            Optional binary label column (column name ``label``).
            Purely unsupervised models may ignore this argument.
        covariates :
            Optional external covariate dict.
        """

    # --- Inference interface (implement at least one) -------------------------

    def detect_score(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Return per-row anomaly scores (higher = more anomalous).

        Returns
        -------
        np.ndarray
            Float array of shape ``(len(test_data),)``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement detect_score. "
            "If the model only outputs labels, declare label_only in capability."
        )

    def detect_label(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Return per-row binary anomaly labels (0 = normal, 1 = anomaly).

        The default implementation raises ``NotImplementedError``.
        Subclasses may override this to output labels directly.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(len(test_data),)`` with values 0/1.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement detect_label. "
            "Models that only output scores should be evaluated under the "
            "score-track strategies (fixed_detect_score / unfixed_detect_score / "
            "all_detect_score)."
        )

    # --- Optional extensions --------------------------------------------------

    def estimate_flops(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        **kwargs,
    ) -> float:
        """Estimate FLOPs. Returns ``nan`` by default (not supported)."""
        return math.nan

    def estimate_n_params(self) -> float:
        """Total number of trainable parameters; ``nan`` if not applicable."""
        return math.nan

    def estimate_model_size_mb(self) -> float:
        """On-disk footprint of the fitted model, in MB; ``nan`` if unknown."""
        return math.nan

    # --- Internal validation --------------------------------------------------

    def _validate_capability(self) -> None:
        """Verify that the subclass actually implements what it declares."""
        cap = self.capability
        has_score = _is_overridden(self, "detect_score")
        has_label = _is_overridden(self, "detect_label")

        if SUPPORT_SCORE in cap.supported_outputs and not has_score:
            raise RuntimeError(
                f"{self.__class__.__name__} declares score output "
                "but does not implement detect_score."
            )
        if SUPPORT_LABEL in cap.supported_outputs and not has_label:
            raise RuntimeError(
                f"{self.__class__.__name__} declares label output "
                "but does not implement detect_label."
            )
        if not has_score and not has_label:
            raise RuntimeError(
                f"{self.__class__.__name__} implements neither "
                "detect_score nor detect_label."
            )

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}" for k, v in self.model_hyper_params.items()
        )
        return f"{self.model_name}({params})"


def _is_overridden(instance: AnomalyModelBase, method_name: str) -> bool:
    """Check whether *method_name* is truly overridden by the subclass."""
    base_method = getattr(AnomalyModelBase, method_name)
    subclass_method = getattr(type(instance), method_name)
    return subclass_method is not base_method
