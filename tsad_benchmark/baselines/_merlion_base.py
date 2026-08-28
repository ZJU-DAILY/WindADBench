# -*- coding: utf-8 -*-

from __future__ import annotations

import contextlib
import logging
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from tsad_benchmark.models.base import AnomalyModelBase, ModelCapability

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CUDA helpers (best-effort; fail open to CPU)
# ---------------------------------------------------------------------------


def _cuda_available() -> bool:

    if os.environ.get("TSAD_DISABLE_CUDA", "").strip() in ("1", "true", "True"):
        return False
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@contextlib.contextmanager
def _torch_default_cuda():

    if not _cuda_available():
        yield
        return
    try:
        import torch
    except Exception:
        yield
        return
    if not hasattr(torch, "set_default_device"):
        yield
        return
    prev = "cpu"
    try:
        prev = str(torch.empty(0).device.type)
    except Exception:
        pass
    torch.set_default_device("cuda")
    try:
        yield
    finally:
        try:
            torch.set_default_device(prev)
        except Exception:
            torch.set_default_device("cpu")


def _move_to_cuda(obj, *, max_depth: int = 6) -> int:

    if not _cuda_available():
        return 0
    try:
        import torch
    except Exception:
        return 0

    moved = 0
    seen: set[int] = set()

    def _visit(o, depth: int) -> None:
        nonlocal moved
        if depth > max_depth or id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, torch.nn.Module):
            try:
                o.cuda()
                moved += 1
            except Exception:
                pass
            return  # nn.Module.cuda() recurses through submodules
        if isinstance(o, (list, tuple, set)):
            for v in o:
                _visit(v, depth + 1)
            return
        if isinstance(o, dict):
            for v in o.values():
                _visit(v, depth + 1)
            return
        if hasattr(o, "__dict__"):
            for v in vars(o).values():
                _visit(v, depth + 1)

    _visit(obj, 0)
    return moved


def _describe_torch_device() -> str:
    if os.environ.get("TSAD_DISABLE_CUDA", "").strip() in ("1", "true", "True"):
        return "cpu (TSAD_DISABLE_CUDA=1)"
    try:
        import torch
    except Exception as exc:
        return f"cpu (torch unavailable: {exc})"
    if not torch.cuda.is_available():
        return f"cpu (torch={getattr(torch, '__version__', '?')}, cuda.is_available=False)"
    try:
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        return f"cuda:{idx} ({name}, torch={torch.__version__})"
    except Exception:
        return f"cuda (torch={torch.__version__})"


def _describe_merlion_device(model) -> str:
    if model is None:
        return "unknown"
    if hasattr(model, "device"):
        return str(getattr(model, "device"))
    if hasattr(model, "use_cuda"):
        return "cuda" if bool(getattr(model, "use_cuda")) else "cpu"
    return "unknown"


def _module_devices(obj, *, max_depth: int = 6) -> list[str]:
    try:
        import torch
    except Exception:
        return []

    found: list[str] = []
    seen: set[int] = set()

    def _visit(o, depth: int) -> None:
        if o is None or depth > max_depth or id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, torch.nn.Module):
            try:
                for param in o.parameters(recurse=True):
                    found.append(str(param.device))
                    break
            except Exception:
                pass
            return
        if isinstance(o, (list, tuple, set)):
            for value in o:
                _visit(value, depth + 1)
            return
        if isinstance(o, dict):
            for value in o.values():
                _visit(value, depth + 1)
            return
        if hasattr(o, "__dict__"):
            for value in vars(o).values():
                _visit(value, depth + 1)

    _visit(obj, 0)
    # Preserve order while uniquifying.
    return list(dict.fromkeys(found))


def _count_torch_parameters(obj, *, max_depth: int = 8) -> float:
    try:
        import torch
    except Exception:
        return float("nan")

    seen_obj: set[int] = set()
    seen_param: set[int] = set()
    total = 0

    def _visit(o, depth: int) -> None:
        nonlocal total
        if o is None or depth > max_depth:
            return
        oid = id(o)
        if oid in seen_obj:
            return
        seen_obj.add(oid)

        if isinstance(o, torch.nn.Module):
            try:
                for param in o.parameters(recurse=True):
                    pid = id(param)
                    if pid not in seen_param:
                        seen_param.add(pid)
                        total += int(param.numel())
            except Exception:
                pass
            return

        if isinstance(o, (str, bytes, int, float, bool)):
            return
        if isinstance(o, (list, tuple, set)):
            for value in o:
                _visit(value, depth + 1)
            return
        if isinstance(o, dict):
            for value in o.values():
                _visit(value, depth + 1)
            return
        if hasattr(o, "__dict__"):
            for value in vars(o).values():
                _visit(value, depth + 1)

    _visit(obj, 0)
    return float(total) if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# DataFrame ↔ TimeSeries
# ---------------------------------------------------------------------------


def df_to_merlion_ts(df: pd.DataFrame):

    from merlion.utils import TimeSeries

    if df is None or len(df) == 0:
        return TimeSeries.from_pd(pd.DataFrame({"_": []}))

    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range("2020-01-01", periods=len(df), freq="10s")
    # Numeric only — Merlion will choke on object dtype columns
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] == 0:
        return TimeSeries.from_pd(pd.DataFrame({"_": [0.0] * len(df)}, index=df.index))
    if not np.isfinite(df.to_numpy()).all():
        col_mean = df.replace([np.inf, -np.inf], np.nan).mean(numeric_only=True)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(col_mean).fillna(0.0)
    return TimeSeries.from_pd(df)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class MerlionBaseModel(AnomalyModelBase):

    capability = ModelCapability.score_and_label()

    def __init__(self) -> None:
        self.model_hyper_params = {}
        self._model = None

    def _build_merlion_model(self):
        raise NotImplementedError("Subclasses must build a Merlion model")

    # ------------------------------------------------------------------
    # AnomalyModelBase interface
    # ------------------------------------------------------------------

    def fit(self, train_data, train_label=None, covariates=None, **kwargs) -> None:
        if train_data is None or len(train_data) == 0:
            self._model = None
            return
        ts = df_to_merlion_ts(train_data)
        if len(ts) == 0:
            self._model = None
            return
        # Build the model (subclasses may already pass device='cuda' to
        # the merlion config when our helpers see CUDA).
        self._model = self._build_merlion_model()
        torch_device = _describe_torch_device()
        merlion_device = _describe_merlion_device(self._model)
        logger.info(
            "[%s] device preflight: torch=%s | merlion=%s",
            self.model_name,
            torch_device,
            merlion_device,
        )
        if merlion_device == "cpu" or str(torch_device).startswith("cpu"):
            logger.warning(
                "[%s] running on CPU — check that this env has a CUDA "
                "torch build and that TSAD_DISABLE_CUDA is unset.",
                self.model_name,
            )
        # Pre-move parameters onto CUDA so that the training loop sees
        # GPU tensors from the very first batch.
        moved_pre = _move_to_cuda(self._model)
        with warnings.catch_warnings(), _torch_default_cuda():
            warnings.simplefilter("ignore")
            self._model.train(ts)
        # Some Merlion versions rebuild internal sub-modules during
        # train(); re-walk and re-move so inference also runs on GPU.
        moved_post = _move_to_cuda(self._model)
        param_devices = _module_devices(self._model)
        logger.info(
            "[%s] device after fit: torch=%s | merlion=%s | "
            "nn.Module devices=%s | moved_modules(pre=%d, post=%d)",
            self.model_name,
            _describe_torch_device(),
            _describe_merlion_device(self._model),
            param_devices or ["(none exposed; Merlion may build MLP ephemerally)"],
            moved_pre,
            moved_post,
        )

    def detect_score(self, test_data, covariates=None, **kwargs) -> np.ndarray:
        n = len(test_data) if test_data is not None else 0
        if self._model is None or n == 0:
            return np.zeros(n, dtype=float)
        ts = df_to_merlion_ts(test_data)
        if len(ts) == 0:
            return np.zeros(n, dtype=float)
        with warnings.catch_warnings(), _torch_default_cuda():
            warnings.simplefilter("ignore")
            scores_ts = self._model.get_anomaly_score(ts)
        scores = np.asarray(scores_ts.to_pd().values, dtype=float).ravel()
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return self._align_length(scores, n)

    def detect_label(self, test_data, covariates=None, **kwargs) -> np.ndarray:
        n = len(test_data) if test_data is not None else 0
        if self._model is None or n == 0:
            return np.zeros(n, dtype=np.int32)

        ts = df_to_merlion_ts(test_data)
        if len(ts) == 0:
            return np.zeros(n, dtype=np.int32)
        with warnings.catch_warnings(), _torch_default_cuda():
            warnings.simplefilter("ignore")
            labels_ts = self._model.get_anomaly_label(ts)
        raw = np.asarray(labels_ts.to_pd().values, dtype=float).ravel()
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        labels = (raw != 0.0).astype(np.int32)
        return self._align_length(labels, n).astype(np.int32)

    def estimate_flops(self, train_data, test_data, **kwargs) -> float:

        return float("nan")

    def estimate_model_size_mb(self) -> float:
        if self._model is None:
            return float("nan")
        try:
            import pickle
            return float(len(pickle.dumps(self._model)) / (1024.0 * 1024.0))
        except Exception:
            return float("nan")

    def estimate_n_params(self) -> float:
        if self._model is None:
            return float("nan")
        return _count_torch_parameters(self._model)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _align_length(scores: np.ndarray, target_len: int) -> np.ndarray:

        if scores.size == target_len:
            return scores
        out = np.empty(target_len, dtype=float)
        if scores.size == 0:
            out.fill(0.0)
            return out
        median = float(np.median(scores))
        if scores.size < target_len:
            pad = target_len - scores.size
            out[:pad] = median
            out[pad:] = scores
        else:
            out[:] = scores[:target_len]
        return out
