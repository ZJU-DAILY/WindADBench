from __future__ import annotations

import gc
import gzip
import hashlib
import io
import json
import math
import os
import tempfile
import traceback
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional


PathLike = str | os.PathLike[str]
CheckpointValidator = Callable[[Any], bool]
CHECKPOINT_FORMAT = "torch-cloudpickle-v1"
EIF_RECIPE_FORMAT = "eif-deterministic-recipe-v1"
_EIF_RECIPE_KEY = "cross_domain_eif_recipe"
_CHECKPOINT_CACHE: dict[str, Any] = {}


def _json_value(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _json_value(value.item())
        if isinstance(value, np.ndarray):
            return [_json_value(item) for item in value.tolist()]
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=str)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="xb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    return temporary


def _atomic_write(path: PathLike, writer: Callable[[Any], None]) -> Path:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_json(value: Any, path: PathLike) -> Path:
    payload = json.dumps(
        _json_value(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return _atomic_write(path, lambda handle: handle.write(payload))


def atomic_csv(frame: Any, path: PathLike, **kwargs: Any) -> Path:
    destination = Path(path)
    options = {"index": False, **kwargs}
    encoding = str(options.pop("encoding", "utf-8"))
    options.pop("compression", None)

    def write(handle: Any) -> None:
        if destination.suffix.lower() == ".gz":
            zipped = gzip.GzipFile(
                filename="", fileobj=handle, mode="wb", mtime=0
            )
            text = io.TextIOWrapper(zipped, encoding=encoding, newline="")
            try:
                frame.to_csv(text, **options)
                text.flush()
                text.detach()
                zipped.close()
            except BaseException:
                try:
                    text.close()
                finally:
                    zipped.close()
                raise
        else:
            text = io.TextIOWrapper(handle, encoding=encoding, newline="")
            try:
                frame.to_csv(text, **options)
                text.flush()
            finally:
                text.detach()

    return _atomic_write(destination, write)


def atomic_npz(
    arrays: Mapping[str, Any], path: PathLike, *, compressed: bool = True
) -> Path:
    import numpy as np

    save = np.savez_compressed if compressed else np.savez
    return _atomic_write(path, lambda handle: save(handle, **dict(arrays)))


def sha256_file(path: PathLike, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_map_location(torch: Any) -> Optional[str]:
    return None if torch.cuda.is_available() else "cpu"


def _loaded_device(map_location: Any, torch: Any) -> Any:
    if isinstance(map_location, (str, torch.device)):
        return torch.device(map_location)
    return None


def _move_loaded_model(model: Any, device: Any, torch: Any) -> Any:
    if device is None:
        return model
    if isinstance(model, torch.nn.Module):
        model.to(device)
    for name in ("_network", "_model", "_raw"):
        module = getattr(model, name, None)
        if isinstance(module, torch.nn.Module):
            module.to(device)
    pipeline = getattr(model, "_pipeline", None)
    module = getattr(pipeline, "model", None)
    if isinstance(module, torch.nn.Module):
        module.to(device)
    if hasattr(model, "_device"):
        model._device = device
    if device.type == "cpu" and isinstance(getattr(model, "device", None), str):
        if model.device.lower().startswith("cuda"):
            model.device = "cpu"
    return model


def _checkpoint_key(path: PathLike) -> str:
    return str(Path(path).resolve())


def _eif_recipe(model: Any) -> Optional[dict[str, Any]]:
    if not (
        type(model).__module__ == "tsad_benchmark.baselines.machine_learning.eif"
        and type(model).__name__ == "EIFModel"
    ):
        return None
    data = getattr(model, "_cross_domain_checkpoint_data", None)
    columns = getattr(model, "_cross_domain_checkpoint_columns", None)
    if data is None or columns is None:
        raise RuntimeError(
            "EIF checkpoint data was not retained by its transfer adapter."
        )
    import numpy as np

    matrix = np.asarray(data, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != len(columns):
        raise ValueError("EIF checkpoint training matrix is invalid.")
    if not np.isfinite(matrix).all():
        raise ValueError("EIF checkpoint training matrix contains non-finite values.")
    return {
        _EIF_RECIPE_KEY: 1,
        "constructor_params": {
            "n_estimators": int(model.n_estimators),
            "sample_size": int(model.sample_size),
            "extension_level": model.extension_level,
            "anomaly_ratio": list(model.anomaly_ratio),
            "seed": int(model.seed),
        },
        "feature_columns": list(columns),
        "training_features": np.ascontiguousarray(matrix),
    }


def _restore_eif_recipe(payload: Mapping[str, Any]) -> Any:
    import numpy as np
    import pandas as pd

    from tsad_benchmark.baselines.machine_learning.eif import EIFModel

    if payload.get(_EIF_RECIPE_KEY) != 1:
        raise ValueError("Unsupported EIF checkpoint recipe version.")
    params = payload.get("constructor_params")
    columns = payload.get("feature_columns")
    data = payload.get("training_features")
    if not isinstance(params, Mapping) or not isinstance(columns, list):
        raise ValueError("EIF checkpoint recipe metadata is invalid.")
    matrix = np.asarray(data, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != len(columns):
        raise ValueError("EIF checkpoint recipe training matrix is invalid.")
    if not np.isfinite(matrix).all():
        raise ValueError("EIF checkpoint recipe contains non-finite values.")
    model = EIFModel(**dict(params))
    model.fit(
        pd.DataFrame(matrix, columns=[str(column) for column in columns]),
        None,
    )
    return model


def load_checkpoint(path: PathLike, map_location: Any = None) -> Any:
    import torch

    key = _checkpoint_key(path)
    cached = _CHECKPOINT_CACHE.get(key)
    if cached is not None:
        return cached
    location = _default_map_location(torch) if map_location is None else map_location
    payload = torch.load(
        Path(path), map_location=location, weights_only=False
    )
    model = (
        _restore_eif_recipe(payload)
        if isinstance(payload, Mapping) and _EIF_RECIPE_KEY in payload
        else payload
    )
    return _move_loaded_model(model, _loaded_device(location, torch), torch)


def _release_checkpoint() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def save_checkpoint(
    model: Any,
    path: PathLike,
    validator: Optional[CheckpointValidator] = None,
) -> dict[str, Any]:
    import torch
    from joblib.externals import cloudpickle

    destination = Path(path)
    _CHECKPOINT_CACHE.pop(_checkpoint_key(destination), None)
    recipe = _eif_recipe(model)
    payload = model if recipe is None else recipe
    checkpoint_format = CHECKPOINT_FORMAT if recipe is None else EIF_RECIPE_FORMAT
    temporary = _temporary_path(destination)
    cached_model = None
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle, pickle_module=cloudpickle)
            handle.flush()
            os.fsync(handle.fileno())

        for attempt in (1, 2):
            restored = load_checkpoint(temporary)
            accepted = False
            try:
                if validator is not None and not bool(validator(restored)):
                    raise RuntimeError(
                        f"Checkpoint validator rejected reload {attempt}."
                    )
                accepted = True
            finally:
                if recipe is not None and attempt == 2 and accepted:
                    cached_model = restored
                else:
                    del restored
                    _release_checkpoint()

        digest = sha256_file(temporary)
        size = temporary.stat().st_size
        os.replace(temporary, destination)
        if cached_model is not None:
            _CHECKPOINT_CACHE[_checkpoint_key(destination)] = cached_model
    except BaseException:
        temporary.unlink(missing_ok=True)
        if cached_model is not None:
            del cached_model
            _release_checkpoint()
        raise
    return {"format": checkpoint_format, "sha256": digest, "bytes": size}


def _is_excluded(
    relative: Path,
    exclude: Optional[PathLike | Iterable[PathLike] | Callable[[Path], bool]],
) -> bool:
    if exclude is None:
        return False
    if callable(exclude):
        return bool(exclude(relative))
    name = relative.as_posix()
    items = (exclude,) if isinstance(exclude, (str, os.PathLike)) else exclude
    for item in items:
        pattern = str(item).replace("\\", "/")
        if name == pattern or fnmatchcase(name, pattern):
            return True
    return False


def artifact_inventory(
    root: PathLike,
    exclude: Optional[PathLike | Iterable[PathLike] | Callable[[Path], bool]] = None,
) -> list[dict[str, Any]]:
    base = Path(root)
    if not base.is_dir():
        raise NotADirectoryError(base)
    inventory = []
    for path in sorted((item for item in base.rglob("*") if item.is_file())):
        relative = path.relative_to(base)
        if _is_excluded(relative, exclude):
            continue
        inventory.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def write_failure(
    path: PathLike,
    error: BaseException,
    *,
    phase: str = "unknown",
    attempt: int = 1,
    context: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        "status": "failed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "attempt": int(attempt),
        "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
        "context": dict(context or {}),
    }
    atomic_json(payload, path)
    return payload


__all__ = [
    "CHECKPOINT_FORMAT",
    "artifact_inventory",
    "atomic_csv",
    "atomic_json",
    "atomic_npz",
    "load_checkpoint",
    "save_checkpoint",
    "sha256_file",
    "write_failure",
]
