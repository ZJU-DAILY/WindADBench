# -*- coding: utf-8 -*-
"""
tsad_benchmark.models - Unified model-layer interface.

Quick import example::

    from tsad_benchmark.models import (
        AnomalyModelBase,
        ModelCapability,
        ModelInfo,
        ModelFactory,
        build_model_factories,
        SklearnAdapter,
        PytorchAdapter,
        RuleAdapter,
        register_model_adapter,
    )
"""

from tsad_benchmark.models.base import (
    SUPPORT_LABEL,
    SUPPORT_SCORE,
    AnomalyModelBase,
    ModelCapability,
)
from tsad_benchmark.models.info import (
    STANDARD_HYPER_PARAMS,
    ModelInfo,
    load_model_spec,
)
from tsad_benchmark.models.loader import (
    ModelFactory,
    build_model_factories,
    validate_strategy_compatibility,
)
from tsad_benchmark.models.adapters import (
    ADAPTER_CAPABILITY,
    PytorchAdapter,
    RuleAdapter,
    SklearnAdapter,
    register_model_adapter,
)

__all__ = [
    # base
    "AnomalyModelBase",
    "ModelCapability",
    "SUPPORT_SCORE",
    "SUPPORT_LABEL",
    # info
    "ModelInfo",
    "load_model_spec",
    "STANDARD_HYPER_PARAMS",
    # loader
    "ModelFactory",
    "build_model_factories",
    "validate_strategy_compatibility",
    # adapters
    "SklearnAdapter",
    "PytorchAdapter",
    "RuleAdapter",
    "ADAPTER_CAPABILITY",
    "register_model_adapter",
]
