from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

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

logger = logging.getLogger(__name__)


class ModelFactory:
    def __init__(self, model_info: ModelInfo) -> None:
        self.model_info = model_info

    @property
    def model_name(self) -> str:
        return self.model_info.model_name

    @property
    def model_hyper_params(self) -> dict:
        return self.model_info.model_hyper_params

    @property
    def capability(self) -> ModelCapability:
        return self.model_info.capability

    def __call__(self) -> AnomalyModelBase:
        return self.model_info.build()

    def __repr__(self) -> str:
        return f"ModelFactory({self.model_info!r})"


_STRATEGY_OUTPUT_REQUIREMENT: Dict[str, str] = {
    "fixed_detect_score": SUPPORT_SCORE,
    "unfixed_detect_score": SUPPORT_SCORE,
    "all_detect_score": SUPPORT_SCORE,
    "fixed_detect_label": SUPPORT_LABEL,
    "unfixed_detect_label": SUPPORT_LABEL,
    "all_detect_label": SUPPORT_LABEL,
}


def validate_strategy_compatibility(
    model_info: ModelInfo,
    strategy_name: Optional[str] = None,
    expected_output: Optional[str] = None,
) -> None:
    cap = model_info.capability

    required_from_strategy: Optional[str] = None
    if strategy_name is not None:
        required_from_strategy = _STRATEGY_OUTPUT_REQUIREMENT.get(strategy_name)

    required = required_from_strategy or expected_output
    if required is None:
        return

    required = required.lower().strip()

    if required == SUPPORT_SCORE and not cap.has_score_output():
        raise RuntimeError(
            f"[{model_info.model_name}] Strategy requires score output, "
            f"but model only supports: {cap.supported_outputs}. "
            "Use a score-capable model; unified label calibration also needs scores."
        )

    if required == SUPPORT_LABEL and not cap.has_label_output():
        raise RuntimeError(
            f"[{model_info.model_name}] Strategy requires label output, "
            f"but the model only supports: {cap.supported_outputs}. "
            "Either implement detect_label on the model, or evaluate it "
            "with a *DetectScore strategy."
        )


def _compose_hyper_params(
    recommend: Dict[str, Any],
    required_mapping: Dict[str, str],
    model_explicit: Dict[str, Any],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}

    model_explicit = dict(model_explicit or {})

    for model_param, std_param in required_mapping.items():
        if model_param in model_explicit:
            continue
        if std_param in recommend:
            merged[model_param] = recommend[std_param]

    merged.update(model_explicit)
    missing = sorted(set(required_mapping) - set(merged))
    if missing:
        details = {
            model_param: required_mapping[model_param]
            for model_param in missing
        }
        raise ValueError(
            "Missing required hyper-parameters. Provide them in "
            "`recommend_hyper_params` using the mapped standard names, "
            f"or in the model's explicit `model_hyper_params`: {details}"
        )
    return merged


def _factory_has_parameter(factory: Callable[..., Any], name: str) -> bool:
    factory = inspect.unwrap(factory)
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters


def _add_seed_if_supported(
    recommend: Dict[str, Any],
    factory: Callable[..., Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    if "seed" not in recommend:
        return params
    merged = dict(params)
    for name in ("seed", "random_state"):
        if name not in merged and _factory_has_parameter(factory, name):
            merged[name] = recommend["seed"]
    return merged


def _resolve_model_spec(
    model_config: Dict[str, Any],
    recommend: Dict[str, Any],
) -> ModelInfo:
    model_path: str = model_config.get("model_path", "")
    if not model_path:
        raise ValueError("Model config is missing the 'model_path' field.")

    raw = load_model_spec(model_path)

    if isinstance(raw, ModelInfo):
        model_info = raw
    elif callable(raw):
        capability: ModelCapability = getattr(
            raw, "capability", ModelCapability.score_only()
        )
        raw_required = getattr(raw, "required_hyper_params", {})
        if callable(raw_required):
            raw_required = raw_required()
        required_hyper_params = {
            **dict(raw_required or {}),
            **model_config.get("required_hyper_params", {}),
        }
        fallback_name = model_path.rsplit(".", 1)[-1]
        model_info = ModelInfo(
            model_name=model_config.get("model_name", fallback_name),
            model_factory=raw,
            model_hyper_params=model_config.get("model_hyper_params", {}),
            required_hyper_params=required_hyper_params,
            capability=capability,
            adapter=model_config.get("adapter"),
        )
    elif isinstance(raw, dict):
        factory = raw.get("model_factory")
        if factory is None:
            raise ValueError(
                f"Model info dict is missing 'model_factory': {model_path}"
            )
        fallback_name = model_path.rsplit(".", 1)[-1]
        model_info = ModelInfo(
            model_name=raw.get(
                "model_name",
                model_config.get("model_name", fallback_name),
            ),
            model_factory=factory,
            model_hyper_params=raw.get("model_hyper_params", {}),
            required_hyper_params=raw.get("required_hyper_params", {}),
            capability=raw.get("capability", ModelCapability.score_only()),
            adapter=raw.get("adapter", model_config.get("adapter")),
            description=raw.get("description"),
        )
    else:
        raise ValueError(
            f"Unsupported model info type: {type(raw).__name__}. "
            "Expected ModelInfo, callable, or dict."
        )

    if model_info.adapter is not None:
        _attach_model_adapter(model_info)

    merged = _compose_hyper_params(
        recommend,
        model_info.required_hyper_params,
        model_config.get("model_hyper_params", model_info.model_hyper_params),
    )
    merged = _add_seed_if_supported(recommend, model_info.model_factory, merged)
    model_info.model_hyper_params = merged

    if "model_name" in model_config:
        model_info.model_name = model_config["model_name"]

    model_info.validate_spec()
    return model_info


def _attach_model_adapter(model_info: ModelInfo) -> None:
    from tsad_benchmark.models.adapters import ADAPTER, ADAPTER_CAPABILITY

    adapter_name = model_info.adapter
    if adapter_name not in ADAPTER:
        raise ValueError(
            f"Unknown adapter: {adapter_name!r}. "
            f"Registered: {sorted(ADAPTER)}"
        )
    wrapper_fn: Callable = ADAPTER[adapter_name]
    original_factory = model_info.model_factory
    model_info.model_factory = wrapper_fn(original_factory)
    if adapter_name in ADAPTER_CAPABILITY:
        model_info.capability = ADAPTER_CAPABILITY[adapter_name]
    logger.info(
        "Applied adapter %r to %s", adapter_name, model_info.model_name
    )


def build_model_factories(all_model_config: Dict[str, Any]) -> List[ModelFactory]:
    recommend: Dict[str, Any] = all_model_config.get(
        "recommend_hyper_params", {}
    )
    strategy_name: Optional[str] = all_model_config.get("strategy_name")
    factories: List[ModelFactory] = []

    for model_config in all_model_config.get("models", []):
        name_hint = model_config.get(
            "model_name", model_config.get("model_path", "?")
        )
        try:
            logger.info("Loading model: %s", name_hint)
            model_info = _resolve_model_spec(model_config, recommend)

            validate_strategy_compatibility(
                model_info,
                strategy_name=strategy_name,
                expected_output=model_config.get("expected_output"),
            )

            factories.append(ModelFactory(model_info))
            logger.info(
                "Model %s loaded, capability: %s",
                model_info.model_name,
                model_info.capability,
            )
        except Exception as exc:
            logger.error("Failed to load model %s: %s", name_hint, exc)
            raise

    return factories
