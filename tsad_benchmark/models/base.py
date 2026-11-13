from __future__ import annotations

import abc
import math
from typing import List, Optional

import numpy as np
import pandas as pd


SUPPORT_SCORE = "score"
SUPPORT_LABEL = "label"


class ModelCapability:
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

    @classmethod
    def score_only(cls, **kwargs) -> "ModelCapability":
        return cls(supported_outputs=[SUPPORT_SCORE], **kwargs)

    @classmethod
    def label_only(cls, **kwargs) -> "ModelCapability":
        return cls(supported_outputs=[SUPPORT_LABEL], **kwargs)

    @classmethod
    def score_and_label(cls, **kwargs) -> "ModelCapability":
        return cls(supported_outputs=[SUPPORT_SCORE, SUPPORT_LABEL], **kwargs)

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


_DEFAULT_CAPABILITY = ModelCapability.score_only()


class AnomalyModelBase(metaclass=abc.ABCMeta):
    model_name: str = "AnomalyModel"
    model_hyper_params: dict = {}
    capability: ModelCapability = _DEFAULT_CAPABILITY

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

    def detect_score(
        self,
        test_data: pd.DataFrame,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement detect_label. "
            "Models that only output scores should be evaluated under the "
            "score-track strategies (fixed_detect_score / unfixed_detect_score / "
            "all_detect_score)."
        )

    def estimate_flops(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        **kwargs,
    ) -> float:
        return math.nan

    def estimate_n_params(self) -> float:
        return math.nan

    def estimate_model_size_mb(self) -> float:
        return math.nan

    def _validate_capability(self) -> None:
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
    base_method = getattr(AnomalyModelBase, method_name)
    subclass_method = getattr(type(instance), method_name)
    return subclass_method is not base_method
