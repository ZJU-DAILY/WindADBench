"""Plan or execute the fixed-holdout Core-3 cross-domain experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import logging
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.cross_domain.artifacts import (
    artifact_inventory,
    atomic_csv,
    atomic_json,
    sha256_file,
    write_failure,
)
from experiments.cross_domain.engine import (
    CrossDomainEngine,
    ExperimentModelSpec,
    safe_name,
)
from experiments.cross_domain.protocol import (
    build_dataset_split,
    build_protocol_plan,
    direction_summary,
    load_manifest,
    validate_protocol,
)
from experiments.cross_domain.transfer_adapters import (
    available_transfer_adapters,
    build_transfer_adapter,
)


RUNNER_VERSION = "cross-domain-v7"


def _load_json(path: Path, label: str) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return document


def _resolve_dataset_root(config: Mapping[str, object]) -> Path:
    raw = config.get("dataset_root")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Config dataset_root must be a non-empty string.")
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _single_model_document(args: argparse.Namespace) -> Dict[str, object]:
    try:
        params = json.loads(args.model_params)
    except json.JSONDecodeError as error:
        raise ValueError(f"--model-params is not valid JSON: {error}") from error
    if not isinstance(params, dict):
        raise ValueError("--model-params must decode to a JSON object.")
    try:
        adapter_params = json.loads(args.transfer_adapter_params)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"--transfer-adapter-params is not valid JSON: {error}"
        ) from error
    if not isinstance(adapter_params, dict):
        raise ValueError("--transfer-adapter-params must decode to a JSON object.")
    return {
        "models": [
            {
                "model_name": args.model_name or args.model_path.rsplit(".", 1)[-1],
                "model_path": args.model_path,
                "model_hyper_params": params,
                "adapter": args.model_adapter,
                "expected_output": "score",
                "transfer_adapter": args.transfer_adapter,
                "transfer_adapter_params": adapter_params,
            }
        ]
    }


def _load_model_document(args: argparse.Namespace) -> tuple[Optional[Path], Dict[str, object]]:
    if args.models_config is not None:
        path = args.models_config.resolve()
        return path, _load_json(path, "Models config")
    if args.model_path:
        return None, _single_model_document(args)
    return None, {"models": []}


def _validate_model_instance(adapter: Any, model: Any) -> None:
    validator = getattr(adapter, "_validate_model", None)
    if callable(validator):
        validator(model)


def _effective_model_params(
    model: Any, factory_params: Mapping[str, object]
) -> Dict[str, object]:
    effective: Dict[str, object] = dict(factory_params)
    declared = getattr(model, "model_hyper_params", None)
    if isinstance(declared, Mapping):
        effective.update(declared)
    try:
        parameters = inspect.signature(type(model).__init__).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    for parameter in parameters:
        if parameter.name == "self" or parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if hasattr(model, parameter.name):
            value = getattr(model, parameter.name)
            if not callable(value):
                effective[parameter.name] = value
    return json.loads(json.dumps(effective, default=str))


def _build_model_specs(
    document: Mapping[str, object], seed: int
) -> tuple[List[ExperimentModelSpec], Dict[str, object]]:
    from tsad_benchmark.models.loader import build_model_factories

    raw_models = document.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("Models config must contain a models list.")
    if not raw_models:
        return [], {"recommend_hyper_params": {"seed": seed}, "models": []}

    loader_entries: List[Dict[str, object]] = []
    source_entries: List[Dict[str, object]] = []
    transfer_names: List[str] = []
    transfer_params: List[Dict[str, object]] = []
    for index, raw in enumerate(raw_models):
        if not isinstance(raw, Mapping):
            raise ValueError(f"models[{index}] must be an object.")
        entry = dict(raw)
        transfer = entry.pop("transfer_adapter", None)
        raw_transfer_params = entry.pop("transfer_adapter_params", {}) or {}
        if not isinstance(raw_transfer_params, Mapping):
            raise ValueError(
                f"models[{index}].transfer_adapter_params must be an object."
            )
        if not isinstance(transfer, str) or not transfer.strip():
            raise ValueError(f"models[{index}].transfer_adapter is required.")
        transfer = transfer.strip().lower()
        if transfer not in available_transfer_adapters():
            raise ValueError(
                f"models[{index}] has unknown transfer_adapter={transfer!r}; "
                f"choose one of {available_transfer_adapters()}."
            )
        expected = entry.setdefault("expected_output", "score")
        if expected != "score":
            raise ValueError("Cross-domain models must use expected_output='score'.")
        if not isinstance(entry.get("model_path"), str):
            raise ValueError(f"models[{index}].model_path is required.")
        if entry.get("adapter") is None:
            entry.pop("adapter", None)
        source_entries.append(dict(raw))
        loader_entries.append(entry)
        transfer_names.append(transfer)
        transfer_params.append(dict(raw_transfer_params))

    raw_recommend = document.get("recommend_hyper_params", {}) or {}
    if not isinstance(raw_recommend, Mapping):
        raise ValueError("recommend_hyper_params must be an object.")
    recommend = dict(raw_recommend)
    recommend["seed"] = seed
    factories = build_model_factories(
        {"recommend_hyper_params": recommend, "models": loader_entries}
    )
    if len(factories) != len(source_entries):
        raise RuntimeError("Model loader returned an unexpected factory count.")

    specs: List[ExperimentModelSpec] = []
    names = set()
    path_names = set()
    for factory, source, transfer, requested_adapter_params in zip(
        factories, source_entries, transfer_names, transfer_params
    ):
        if not factory.capability.has_score_output():
            raise ValueError(f"{factory.model_name} has no continuous score output.")
        if factory.model_name in names:
            raise ValueError(f"Duplicate model_name={factory.model_name!r}.")
        names.add(factory.model_name)
        model_path_name = safe_name(factory.model_name)
        folded_path_name = model_path_name.casefold()
        if len(model_path_name) > 64:
            raise ValueError(f"Model path name is too long: {model_path_name!r}.")
        if model_path_name.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES:
            raise ValueError(f"Model name maps to reserved path {model_path_name!r}.")
        if folded_path_name in path_names:
            raise ValueError(
                f"Model names collide after path sanitization: {model_path_name!r}."
            )
        path_names.add(folded_path_name)
        for seed_name in ("seed", "random_seed", "random_state"):
            effective_seed = factory.model_hyper_params.get(seed_name)
            if effective_seed is not None and int(effective_seed) != seed:
                raise ValueError(
                    f"{factory.model_name} overrides {seed_name}={effective_seed}; "
                    f"expected {seed}."
                )
        adapter = build_transfer_adapter(transfer, **requested_adapter_params)
        model = factory()
        try:
            _validate_model_instance(adapter, model)
            effective_params = _effective_model_params(
                model, factory.model_hyper_params
            )
            adapter_params = dict(adapter.resolved_params())
        finally:
            del model
            gc.collect()
        specs.append(
            ExperimentModelSpec(
                factory=factory,
                transfer_adapter=transfer,
                transfer_adapter_params=adapter_params,
                model_path=str(source["model_path"]),
                benchmark_adapter=(
                    str(source["adapter"]) if source.get("adapter") else None
                ),
                effective_model_params=effective_params,
                config_entry=source,
            )
        )

    resolved = {
        "recommend_hyper_params": recommend,
        "models": [
            {
                **source,
                "model_name": spec.model_name,
                "effective_model_hyper_params": spec.effective_model_params,
                "effective_transfer_adapter_params": (
                    spec.transfer_adapter_params
                ),
            }
            for source, spec in zip(source_entries, specs)
        ],
    }
    return specs, resolved


def _write_plan(
    output_dir: Path,
    manifest: pd.DataFrame,
    config: Mapping[str, object],
    source_farms: Sequence[str] = ("A", "B", "C"),
) -> pd.DataFrame:
    plan = build_protocol_plan(manifest, config, source_farms=source_farms)
    atomic_csv(build_dataset_split(manifest, config), output_dir / "dataset_split.csv")
    atomic_csv(plan, output_dir / "protocol_plan.csv")
    summary = direction_summary(plan)
    atomic_csv(summary, output_dir / "direction_plan.csv")
    return summary


def _environment() -> Dict[str, object]:
    versions: Dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import torch

        versions.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
            }
        )
    except Exception as error:
        versions["torch"] = f"unavailable: {error}"
    return versions


_WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def _run_id(requested: Optional[str], specs: Sequence[ExperimentModelSpec]) -> str:
    if requested:
        if safe_name(requested) != requested or len(requested) > 80:
            raise ValueError(
                "--run-id must be at most 80 characters and contain only letters, "
                "digits, underscore, dot, and hyphen."
            )
        if requested.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES:
            raise ValueError("--run-id is a reserved Windows path name.")
        return requested
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    identity = json.dumps(
        [
            [
                spec.model_name,
                spec.transfer_adapter,
                spec.transfer_adapter_params,
                spec.effective_model_params,
            ]
            for spec in specs
        ],
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    first_model = safe_name(specs[0].model_name)[:24]
    return f"{stamp}_{first_model}_{digest}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=_THIS_DIR / "configs" / "fixed_holdout.json"
    )
    parser.add_argument("--output-dir", type=Path, default=_THIS_DIR / "outputs")
    parser.add_argument("--run-id")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing run directory with the same run id.",
    )
    run_mode.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a verified failed run and infer only unfinished target events.",
    )
    parser.add_argument(
        "--source-farm",
        choices=("A", "B", "C"),
        help="Fit and evaluate exactly one source farm.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate models and write plans only."
    )
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--models-config", type=Path)
    models.add_argument("--model-path")
    parser.add_argument("--model-name")
    parser.add_argument("--model-params", default="{}")
    parser.add_argument("--model-adapter")
    parser.add_argument(
        "--transfer-adapter", choices=available_transfer_adapters()
    )
    parser.add_argument("--transfer-adapter-params", default="{}")
    return parser


def _validate_resume_run(
    run_dir: Path,
    *,
    config: Mapping[str, object],
    config_path: Path,
    resolved_models: Mapping[str, object],
    model_config_path: Optional[Path],
    source_farm: str,
) -> Dict[str, object]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
    state = _load_json(run_dir / "state.json", "Resume state")
    if state.get("status") not in {"failed", "running"}:
        raise ValueError("--resume requires a failed or interrupted run.")
    experiment = _load_json(
        run_dir / "resolved_experiment_config.json", "Resume experiment config"
    )
    if experiment.get("config") != config:
        raise ValueError("Resume experiment config differs from the saved run.")
    if experiment.get("source_sha256") != sha256_file(config_path):
        raise ValueError("Resume experiment config checksum differs from the saved run.")
    if experiment.get("source_farms") != [source_farm]:
        raise ValueError("Resume source farm differs from the saved run.")

    models = _load_json(run_dir / "resolved_models_config.json", "Resume models config")
    if models.get("recommend_hyper_params") != resolved_models.get(
        "recommend_hyper_params"
    ) or models.get("models") != resolved_models.get("models"):
        raise ValueError("Resume model configuration differs from the saved run.")
    expected_model_sha = (
        sha256_file(model_config_path) if model_config_path is not None else None
    )
    if models.get("source_sha256") != expected_model_sha:
        raise ValueError("Resume models config checksum differs from the saved run.")
    return state


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parser().parse_args()
    if args.model_path and not args.transfer_adapter:
        raise ValueError("--model-path requires --transfer-adapter.")
    if not args.model_path and (
        any(
            value is not None
            for value in (args.model_name, args.model_adapter, args.transfer_adapter)
        )
        or args.model_params != "{}"
        or args.transfer_adapter_params != "{}"
    ):
        raise ValueError(
            "Single-model options require --model-path."
        )

    config_path = args.config.resolve()
    config = _load_json(config_path, "Experiment config")
    dataset_root = _resolve_dataset_root(config)
    manifest = load_manifest(dataset_root)
    validate_protocol(manifest, config)
    protocol = config["protocol"]
    assert isinstance(protocol, Mapping)
    seed = int(protocol.get("seed", 2026))
    model_config_path, model_document = _load_model_document(args)
    specs, resolved_models = _build_model_specs(model_document, seed)
    if specs and args.source_farm is None:
        raise ValueError("Model runs require --source-farm A, B, or C.")
    source_farms = (args.source_farm,) if args.source_farm else ("A", "B", "C")

    output_dir = args.output_dir.resolve()
    if args.dry_run or not specs:
        summary = _write_plan(
            output_dir, manifest, config, source_farms=source_farms
        )
        print(summary.to_string(index=False), flush=True)
        task_count = int(summary["target_event_count"].sum())
        protocol_count = len(protocol["evaluation_protocols"])
        track_count = len(protocol["evaluation_tracks"])
        print(
            f"Protocol valid: {task_count} source-to-target-event tasks in "
            f"{len(summary)} directions, "
            f"{task_count * protocol_count * track_count} rows/model after "
            f"protocol/track expansion; plans written to {output_dir}.",
            flush=True,
        )
        if not specs:
            print("Plan-only mode: no model was fitted.", flush=True)
        else:
            print(f"Validated {len(specs)} model(s); dry-run skipped fitting.", flush=True)
        return

    run_id = _run_id(args.run_id, specs)
    run_dir = output_dir / run_id
    replaced_existing = run_dir.exists() or run_dir.is_symlink()
    previous_state: Dict[str, object] = {}
    if args.resume:
        previous_state = _validate_resume_run(
            run_dir,
            config=config,
            config_path=config_path,
            resolved_models=resolved_models,
            model_config_path=model_config_path,
            source_farm=str(args.source_farm),
        )
        replaced_existing = False
    elif replaced_existing:
        if not args.overwrite:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}; pass --overwrite to replace it."
            )
        if run_dir.is_symlink():
            run_dir.unlink()
        else:
            shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    logging.info(
        "[%s] run_id=%s source=%s overwrite=%s replaced_existing=%s",
        RUNNER_VERSION,
        run_id,
        args.source_farm,
        bool(args.overwrite),
        replaced_existing,
    )
    if args.resume:
        logging.info("[%s] resume=true; verified artifacts will be reused", RUNNER_VERSION)
    logging.info("[%s] results=%s", RUNNER_VERSION, run_dir)
    resumed = datetime.now(timezone.utc).isoformat() if args.resume else None
    started = str(
        previous_state.get("started_utc")
        or resumed
        or datetime.now(timezone.utc).isoformat()
    )
    summary: Dict[str, int]
    try:
        atomic_json(
            {
                "status": "running",
                "run_id": run_id,
                "runner_version": RUNNER_VERSION,
                "overwrite": bool(args.overwrite),
                "resume": bool(args.resume),
                "replaced_existing": replaced_existing,
                "started_utc": started,
                "resumed_utc": resumed,
            },
            run_dir / "state.json",
        )
        if not args.resume:
            atomic_json(
                {
                    "source_path": str(config_path),
                    "source_sha256": sha256_file(config_path),
                    "dataset_root_resolved": str(dataset_root),
                    "source_farms": list(source_farms),
                    "config": config,
                },
                run_dir / "resolved_experiment_config.json",
            )
            atomic_json(
                {
                    "source_path": str(model_config_path) if model_config_path else None,
                    "source_sha256": (
                        sha256_file(model_config_path) if model_config_path else None
                    ),
                    **resolved_models,
                },
                run_dir / "resolved_models_config.json",
            )
        engine = CrossDomainEngine(
            manifest=manifest,
            config=config,
            dataset_root=dataset_root,
            run_dir=run_dir,
            run_id=run_id,
            source_farms=source_farms,
            resume=bool(args.resume),
        )
        summary = engine.run(specs)
        completed = datetime.now(timezone.utc).isoformat()
        atomic_json(
            {
                "status": "completed",
                "run_id": run_id,
                "runner_version": RUNNER_VERSION,
                "overwrite": bool(args.overwrite),
                "resume": bool(args.resume),
                "replaced_existing": replaced_existing,
                "started_utc": started,
                "resumed_utc": resumed,
                "completed_utc": completed,
                **summary,
            },
            run_dir / "state.json",
        )
        (run_dir / "failure.json").unlink(missing_ok=True)
        inventory = artifact_inventory(run_dir, exclude="run_manifest.json")
        atomic_json(
            {
                "status": "completed",
                "run_id": run_id,
                "runner_version": RUNNER_VERSION,
                "overwrite": bool(args.overwrite),
                "resume": bool(args.resume),
                "replaced_existing": replaced_existing,
                "started_utc": started,
                "resumed_utc": resumed,
                "completed_utc": completed,
                "command": sys.argv,
                "working_directory": str(Path.cwd()),
                "repository_root": str(_REPO_ROOT),
                "dataset_root": str(dataset_root),
                "source_farms": list(source_farms),
                "dataset_manifest_path": str(dataset_root / "manifest.csv"),
                "dataset_manifest_sha256": sha256_file(
                    dataset_root / "manifest.csv"
                ),
                "environment": _environment(),
                "experiment_code": artifact_inventory(
                    _THIS_DIR,
                    exclude=lambda relative: (
                        relative.parts[0] == "outputs"
                        or "__pycache__" in relative.parts
                    ),
                ),
                "summary": summary,
                "artifacts": inventory,
            },
            run_dir / "run_manifest.json",
        )
    except BaseException as error:
        write_failure(
            run_dir / "failure.json",
            error,
            phase="cross_domain_run",
            context={"run_id": run_id, "models": [spec.model_name for spec in specs]},
        )
        atomic_json(
            {
                "status": "failed",
                "run_id": run_id,
                "runner_version": RUNNER_VERSION,
                "overwrite": bool(args.overwrite),
                "resume": bool(args.resume),
                "replaced_existing": replaced_existing,
                "started_utc": started,
                "resumed_utc": resumed,
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "error": str(error),
            },
            run_dir / "state.json",
        )
        raise

    print(
        f"Completed run {run_id}: {summary['source_fits']} source fits, "
        f"{summary['event_results']} event results. Artifacts: {run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
