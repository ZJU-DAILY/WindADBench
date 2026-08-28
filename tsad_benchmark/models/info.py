# -*- coding: utf-8 -*-
"""
Model information descriptor.

:class:`ModelInfo` carries all metadata the benchmark needs to identify,
configure and instantiate a model.  It separates *model definition* from
*model instantiation* so that the loader, evaluator, and result recorder
all depend on a single standardised object.

Standard hyper-parameters
-------------------------
A model may declare which parameters should be supplied by the benchmark
via ``required_hyper_params``.  The mapping format is::

    {model_param_name: benchmark_standard_param_name}

Currently supported standard parameter names:

- window_size        : input window length
- step_size          : sliding step size
- train_ratio_in_tv  : training ratio within train/validation split
- threshold          : fixed decision threshold
- contamination      : expected anomaly ratio in training set
- batch_size         : mini-batch size
- num_epochs         : number of training epochs
- learning_rate      : optimiser learning rate
- seed               : random seed
- device             : compute device ("cpu" / "cuda")
- points_per_day     : sampling points per day (144 for 10-min interval)
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability

# ---------------------------------------------------------------------------
# Supported standard hyper-parameter names
# ---------------------------------------------------------------------------

STANDARD_HYPER_PARAMS = {
    "window_size",
    "step_size",
    "train_ratio_in_tv",
    "threshold",
    "contamination",
    "batch_size",
    "num_epochs",
    "learning_rate",
    "seed",
    "device",
    "points_per_day",
}


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """
    Full metadata required for the benchmark to manage a model.

    Attributes
    ----------
    model_name :
        Display name used in result tables.
    model_factory :
        A callable that accepts hyper-parameter kwargs and returns an
        :class:`AnomalyModelBase` instance.
    model_hyper_params :
        Default (or current) hyper-parameters for the model.
    required_hyper_params :
        Mapping from model parameter names to benchmark standard parameter
        names, e.g. ``{"contamination": "contamination"}``.
    capability :
        Capability declaration (output types, training paradigm, etc.).
    adapter :
        Adapter name if wrapping a third-party model (key in
        ``adapters.ADAPTER``).
    description :
        Optional free-text description.
    """

    model_name: str
    model_factory: Callable[..., AnomalyModelBase]
    model_hyper_params: Dict[str, Any] = field(default_factory=dict)
    required_hyper_params: Dict[str, str] = field(default_factory=dict)
    capability: ModelCapability = field(
        default_factory=ModelCapability.score_only
    )
    adapter: Optional[str] = None
    description: Optional[str] = None

    def validate_spec(self) -> None:
        """Check internal consistency (does not involve strategy compat)."""
        if not callable(self.model_factory):
            raise ValueError(
                f"[{self.model_name}] model_factory must be callable."
            )
        unknown = set(self.required_hyper_params.values()) - STANDARD_HYPER_PARAMS
        if unknown:
            raise ValueError(
                f"[{self.model_name}] required_hyper_params contains unknown "
                f"standard params: {unknown}. "
                f"Supported: {sorted(STANDARD_HYPER_PARAMS)}"
            )

    def build(self, **override_params) -> AnomalyModelBase:
        """
        Create a model instance and validate capability consistency.

        Parameters
        ----------
        override_params :
            Temporary overrides with highest priority.
        """
        params = {**self.model_hyper_params, **override_params}
        model = self.model_factory(**params)
        if hasattr(model, "_validate_capability"):
            model._validate_capability()
        return model

    def __repr__(self) -> str:
        return (
            f"ModelInfo(name={self.model_name!r}, "
            f"capability={self.capability}, "
            f"adapter={self.adapter!r})"
        )


# ---------------------------------------------------------------------------
# Helper: dynamic import from dotted path
# ---------------------------------------------------------------------------


def load_model_spec(attr_path: str) -> Any:
    """
    Dynamically import a model definition by its fully-qualified path.

    :param attr_path: e.g. ``"tsad_benchmark.baselines.machine_learning.iforest.IForestModel"``
    :return: A :class:`ModelInfo`, a callable model class, or a dict.
    """
    package_name, name = attr_path.rsplit(".", 1)
    try:
        module = importlib.import_module(package_name)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module {package_name!r}. "
            f"Original error: {e}"
        ) from e
    obj = getattr(module, name, None)
    if obj is None:
        raise AttributeError(
            f"Module {package_name!r} has no attribute {name!r}."
        )
    return obj
