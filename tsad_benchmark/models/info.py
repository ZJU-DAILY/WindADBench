from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability


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


@dataclass
class ModelInfo:
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


def load_model_spec(attr_path: str) -> Any:
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
