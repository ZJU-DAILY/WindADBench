
from __future__ import annotations

import contextlib
import inspect
import logging
import time
import types
from abc import ABC, abstractmethod
from typing import Dict, Iterable, NamedTuple, Optional, Sequence, Type

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


LOGGER = logging.getLogger(__name__)


class _ProgressLoader:
    def __init__(self, loader, *, label: str, mode: str, passes: int) -> None:
        self._loader = loader
        self._label = label
        self._mode = mode
        self._passes = max(int(passes), 1)
        self._pass_index = 0
        self._started: Optional[float] = None

    def __len__(self) -> int:
        return len(self._loader)

    def __getattr__(self, name: str):
        return getattr(self._loader, name)

    def __iter__(self):
        self._pass_index += 1
        pass_index = self._pass_index
        batches = len(self)
        if self._started is None:
            self._started = time.perf_counter()
        pass_started = time.perf_counter()
        LOGGER.info(
            "%s | %s epoch %d/%d | started | batches=%d",
            self._label,
            self._mode,
            pass_index,
            self._passes,
            batches,
        )
        interval = max(1, (batches + 9) // 10)
        for batch_index, batch in enumerate(self._loader, start=1):
            yield batch
            if self._mode == "train" and (
                batch_index == batches or batch_index % interval == 0
            ):
                completed = min(
                    (pass_index - 1) * batches + batch_index,
                    self._passes * batches,
                )
                planned = self._passes * batches
                elapsed = time.perf_counter() - self._started
                eta = elapsed * (planned - completed) / completed if completed else 0.0
                LOGGER.info(
                    "%s | train epoch %d/%d | batch %d/%d | "
                    "epoch_progress=%.1f%% overall_progress=%.1f%% "
                    "elapsed=%.1fs eta=%.1fs",
                    self._label,
                    pass_index,
                    self._passes,
                    batch_index,
                    batches,
                    100.0 * batch_index / batches,
                    100.0 * completed / planned,
                    elapsed,
                    eta,
                )
        LOGGER.info(
            "%s | %s epoch %d/%d | completed | elapsed=%.1fs",
            self._label,
            self._mode,
            pass_index,
            self._passes,
            time.perf_counter() - pass_started,
        )


class ScoreOutput(NamedTuple):
    scores: np.ndarray
    valid_mask: np.ndarray


def _numeric_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    columns: Optional[Sequence[str]] = None,
    allow_empty: bool = False,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if frame.columns.has_duplicates or frame.shape[1] == 0:
        raise ValueError(f"{name} must have unique, non-empty feature columns.")
    if columns is not None and list(frame.columns) != list(columns):
        raise ValueError(
            f"{name} feature columns differ: expected {list(columns)}, "
            f"got {list(frame.columns)}."
        )
    if frame.empty and not allow_empty:
        raise ValueError(f"{name} must not be empty.")
    try:
        numeric = frame.apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains non-numeric feature values.") from exc
    if not np.isfinite(numeric.to_numpy(dtype=float, copy=False)).all():
        raise ValueError(f"{name} contains non-finite feature values.")
    return numeric.reset_index(drop=True)


def _source_segments(segments: Iterable[pd.DataFrame]) -> list[pd.DataFrame]:
    materialized = list(segments)
    if not materialized:
        raise ValueError("Source training needs at least one segment.")
    first = _numeric_frame(materialized[0], name="source segment 0")
    columns = list(first.columns)
    result = [first]
    for index, segment in enumerate(materialized[1:], start=1):
        result.append(
            _numeric_frame(
                segment,
                name=f"source segment {index}",
                columns=columns,
            )
        )
    return result


def _score_vector(raw, expected: int, *, context: str) -> np.ndarray:
    if isinstance(raw, (dict, tuple)):
        raise TypeError(
            f"{context}: detect_score must return one score vector, got "
            f"{type(raw).__name__}."
        )
    scores = np.asarray(raw, dtype=float).reshape(-1)
    if scores.size != expected:
        raise ValueError(
            f"{context}: model returned {scores.size} scores for {expected} rows."
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"{context}: model returned non-finite scores.")
    return scores


def _valid_vector(raw, expected: int, *, context: str) -> np.ndarray:
    values = np.asarray(raw).reshape(-1)
    if values.size != expected:
        raise ValueError(
            f"{context}: valid_mask has {values.size} rows; expected {expected}."
        )
    if values.dtype == np.bool_:
        return values.copy()
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: valid_mask must be boolean or binary.") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{context}: valid_mask must contain only 0/1 values.")
    return numeric.astype(bool)


def _event_inputs(
    frame: pd.DataFrame,
    context_frame: Optional[pd.DataFrame],
    *,
    context_rows: int,
    pad_left: bool = False,
) -> tuple[pd.DataFrame, int]:
    target = _numeric_frame(frame, name="event frame", allow_empty=True)
    if target.empty or context_rows <= 0:
        return target, 0
    context = None
    if context_frame is not None:
        context = _numeric_frame(
            context_frame,
            name="event context",
            columns=list(target.columns),
            allow_empty=True,
        )
    if context is not None and not context.empty:
        tail = context.iloc[-context_rows:].reset_index(drop=True)
        if pad_left and len(tail) < context_rows:
            missing = context_rows - len(tail)
            prefix = pd.DataFrame(
                np.repeat(tail.iloc[[0]].to_numpy(), missing, axis=0),
                columns=target.columns,
            )
            tail = pd.concat([prefix, tail], ignore_index=True)
    elif pad_left:
        tail = pd.DataFrame(
            np.repeat(target.iloc[[0]].to_numpy(), context_rows, axis=0),
            columns=target.columns,
        )
    else:
        return target, 0
    return pd.concat([tail, target], ignore_index=True), len(tail)


def _slice_segment_lengths(
    lengths: Sequence[int], start: int, stop: int
) -> list[int]:
    result: list[int] = []
    cursor = 0
    for length in lengths:
        left = max(cursor, start)
        right = min(cursor + length, stop)
        if right > left:
            result.append(right - left)
        cursor += length
    if sum(result) != max(stop - start, 0):
        raise AssertionError("Segment partition does not cover the requested slice.")
    return result


def _safe_window_starts(
    lengths: Sequence[int],
    window: int,
    *,
    step: int = 1,
    sample_rate: float = 1.0,
) -> np.ndarray:
    step = max(int(step), 1)
    starts: list[np.ndarray] = []
    offset = 0
    for length in lengths:
        count = max(int(length) - int(window) + 1, 0)
        if count:
            starts.append(offset + np.arange(0, count, step, dtype=np.int64))
        offset += int(length)
    result = np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)
    rate = float(np.clip(sample_rate, 0.0, 1.0))
    if 0.0 < rate < 1.0 and result.size:
        keep = max(1, int(np.ceil(result.size * rate)))
        indices = np.linspace(0, result.size - 1, keep, dtype=np.int64)
        result = result[np.unique(indices)]
    return result


def _array_partition(
    base: np.ndarray,
    lengths: Sequence[int],
    array: np.ndarray,
) -> list[int]:
    if array is None:
        return []
    if len(array) == len(base) and not np.shares_memory(array, base):
        return list(lengths)
    if not np.shares_memory(array, base):
        raise RuntimeError("Model copied a temporal slice before window construction.")
    if base.ndim == 0 or array.ndim == 0 or base.strides[0] <= 0:
        raise RuntimeError("Cannot resolve the temporal slice boundaries.")
    byte_offset = int(array.__array_interface__["data"][0]) - int(
        base.__array_interface__["data"][0]
    )
    if byte_offset < 0 or byte_offset % int(base.strides[0]):
        raise RuntimeError("Temporal slice is not aligned to source rows.")
    start = byte_offset // int(base.strides[0])
    stop = start + len(array)
    if stop > len(base):
        raise RuntimeError("Temporal slice exceeds the fitted source array.")
    return _slice_segment_lengths(lengths, start, stop)


@contextlib.contextmanager
def _instance_method(model, name: str, replacement):
    had_value = name in model.__dict__
    previous = model.__dict__.get(name)
    setattr(model, name, types.MethodType(replacement, model))
    try:
        yield
    finally:
        if had_value:
            setattr(model, name, previous)
        else:
            delattr(model, name)


class TransferAdapter(ABC):
    name: str

    def resolved_params(self) -> dict[str, object]:
        return {}

    def fit_metadata(self, model) -> dict[str, object]:
        return {}

    def minimum_probe_rows(self, model) -> int:
        return 1

    @abstractmethod
    def fit_source(self, model, segments: Iterable[pd.DataFrame]) -> None:
        raise NotImplementedError

    @abstractmethod
    def score_event(
        self,
        model,
        frame: pd.DataFrame,
        context_frame: Optional[pd.DataFrame] = None,
    ) -> ScoreOutput:
        raise NotImplementedError


class PointConcatAdapter(TransferAdapter):
    name = "point_concat"

    _UNSAFE_MODULE_MARKERS = (
        ".deep_learning.",
        ".ts_pretrained.",
        ".llm_based.",
        ".finetune_llm.",
        ".machine_learning.kmeans",
        ".machine_learning.torsk",
        "merlion",
    )

    @classmethod
    def _validate_model(cls, model) -> None:
        module = type(model).__module__.lower()
        from tsad_benchmark.baselines._merlion_base import MerlionBaseModel

        if isinstance(model, MerlionBaseModel):
            raise TypeError(
                f"{type(model).__name__} is Merlion-backed and requires an "
                "explicit audited transfer adapter."
            )
        if any(marker in module for marker in cls._UNSAFE_MODULE_MARKERS):
            raise TypeError(
                f"{type(model).__name__} is temporal and cannot use point_concat."
            )
        if hasattr(model, "_make_windows") or hasattr(model, "_run_reservoir"):
            raise TypeError(
                f"{type(model).__name__} constructs temporal state or windows; "
                "select an explicit segmented adapter."
            )
        capability = getattr(model, "capability", None)
        granularity = getattr(capability, "input_granularity", None)
        if granularity not in (None, "point"):
            raise TypeError(
                f"{type(model).__name__} declares input_granularity={granularity!r}."
            )

    def fit_metadata(self, model) -> dict[str, object]:
        stats = getattr(model, "_transfer_fit_audit_stats", None)
        return dict(stats) if isinstance(stats, dict) else {}

    def fit_source(self, model, segments: Iterable[pd.DataFrame]) -> None:
        self._validate_model(model)
        source = _source_segments(segments)
        combined = pd.concat(source, ignore_index=True)
        model.fit(combined, None)
        if (
            type(model).__module__
            == "tsad_benchmark.baselines.machine_learning.eif"
            and type(model).__name__ == "EIFModel"
        ):
            model._cross_domain_checkpoint_data = np.ascontiguousarray(
                combined.to_numpy(dtype=np.float64, copy=True)
            )
            model._cross_domain_checkpoint_columns = [
                str(column) for column in combined.columns
            ]
        lengths = [len(segment) for segment in source]
        model._transfer_fit_audit_stats = {
            "source_event_count": len(source),
            "source_rows": int(sum(lengths)),
            "source_rows_per_event": lengths,
            "boundary_policy": "pointwise_concat",
        }

    def score_event(
        self,
        model,
        frame: pd.DataFrame,
        context_frame: Optional[pd.DataFrame] = None,
    ) -> ScoreOutput:
        self._validate_model(model)
        target = _numeric_frame(frame, name="event frame", allow_empty=True)
        if target.empty:
            return ScoreOutput(np.zeros(0, dtype=float), np.zeros(0, dtype=bool))
        scores = _score_vector(
            model.detect_score(target), len(target), context=type(model).__name__
        )
        return ScoreOutput(scores, np.ones(len(target), dtype=bool))


class DLBaseSegmentedAdapter(TransferAdapter):
    name = "dlbase_segmented"

    def fit_metadata(self, model) -> dict[str, object]:
        stats = getattr(model, "_transfer_fit_audit_stats", None)
        if isinstance(stats, dict):
            return dict(stats)
        return {"native_label_train_score_cache_saved": False}

    def minimum_probe_rows(self, model) -> int:
        return self._validate_model(model)

    @staticmethod
    def _validate_model(model):
        from tsad_benchmark.baselines._dl_base import DLBaseModel

        if type(model).__name__ == "D3RModel" or type(model).__module__.endswith(
            ".d3r"
        ):
            raise TypeError(
                "D3R performs rolling preprocessing before window loading and "
                "needs a dedicated segmented adapter."
            )
        if not isinstance(model, DLBaseModel):
            raise TypeError(
                f"{type(model).__name__} is not a DLBaseModel instance."
            )
        if type(model)._make_loader is not DLBaseModel._make_loader:
            raise TypeError(
                f"{type(model).__name__} overrides _make_loader and needs a "
                "model-specific segmented adapter."
            )
        fit_impl = type(model).fit
        is_known_override = (
            type(model).__module__.endswith(".catch")
            and type(model).__name__ == "CATCHAnomalyModel"
        )
        if fit_impl is not DLBaseModel.fit and not is_known_override:
            raise TypeError(
                f"{type(model).__name__} overrides fit and has not been audited "
                "for DLBase segmented training."
            )
        win_size = int(getattr(model, "win_size", 0))
        if win_size <= 0:
            raise ValueError("DLBase win_size must be positive.")
        return win_size

    @staticmethod
    def _segmented_loader(model, arr, mode, shuffle, lengths):
        from torch.utils.data import ConcatDataset, DataLoader
        from tsad_benchmark.baselines._dl_base import _build_torch_dataset_cls

        if len(arr) != sum(lengths):
            raise RuntimeError(
                f"Unexpected {mode} array length {len(arr)}; expected {sum(lengths)}."
            )
        dataset_cls = _build_torch_dataset_cls()
        datasets = []
        cursor = 0
        for length in lengths:
            part = arr[cursor : cursor + length]
            cursor += length
            dataset = dataset_cls(part, model.win_size, mode=mode)
            if len(dataset):
                datasets.append(dataset)
        if not datasets:
            return None
        dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        return DataLoader(
            dataset,
            batch_size=model.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    def fit_source(self, model, segments: Iterable[pd.DataFrame]) -> None:
        win_size = self._validate_model(model)
        source = _source_segments(segments)
        lengths = [len(segment) for segment in source]
        total = sum(lengths)
        val_n = int(total * float(model.val_ratio))
        has_val = val_n >= win_size + 1 and total - val_n >= win_size
        split = total - val_n if has_val else total
        train_lengths = _slice_segment_lengths(lengths, 0, split)
        val_lengths = _slice_segment_lengths(lengths, split, total) if has_val else []
        if not any(length >= win_size for length in train_lengths):
            raise ValueError(
                "No source segment is long enough to form a training window "
                "after the train/validation split."
            )
        train_windows = sum(max(length - win_size + 1, 0) for length in train_lengths)
        validation_windows = sum(
            max(length - win_size + 1, 0) for length in val_lengths
        )
        batch_size = max(int(model.batch_size), 1)
        epochs = max(int(model.num_epochs), 1)
        train_batches = (train_windows + batch_size - 1) // batch_size
        validation_batches = (
            (validation_windows + batch_size - 1) // batch_size
            if validation_windows
            else 0
        )
        progress_label = f"[{model.model_name} source training]"
        LOGGER.info(
            "%s plan | events=%d rows=%d train_windows=%d "
            "validation_windows=%d batch_size=%d epochs=%d "
            "train_batches_per_epoch=%d validation_batches_per_epoch=%d",
            progress_label,
            len(source),
            total,
            train_windows,
            validation_windows,
            batch_size,
            epochs,
            train_batches,
            validation_batches,
        )

        had_instance_method = "_make_loader" in model.__dict__
        previous_instance_method = model.__dict__.get("_make_loader")
        had_inference_method = "_inference_scores" in model.__dict__
        previous_inference_method = model.__dict__.get("_inference_scores")
        original_inference = model._inference_scores

        def segmented_loader(this, arr, mode, shuffle):
            if mode == "val":
                if not has_val:
                    raise RuntimeError("Model requested an unexpected validation loader.")
                selected = val_lengths
            elif mode == "train":
                selected = train_lengths
            else:
                raise RuntimeError(
                    f"Unexpected loader mode {mode!r} during source fitting."
                )
            loader = self._segmented_loader(this, arr, mode, shuffle, selected)
            if loader is None:
                return None
            return _ProgressLoader(
                loader,
                label=progress_label,
                mode=mode,
                passes=epochs,
            )

        def inference_without_label_cache(this, arr, criterion, mode="thre"):
            if mode == "train":
                return np.zeros(0, dtype=float)
            return original_inference(arr, criterion, mode=mode)

        model._make_loader = types.MethodType(segmented_loader, model)
        model._inference_scores = types.MethodType(
            inference_without_label_cache, model
        )
        try:
            model.fit(pd.concat(source, ignore_index=True), None)
        finally:
            if had_instance_method:
                model._make_loader = previous_instance_method
            else:
                delattr(model, "_make_loader")
            if had_inference_method:
                model._inference_scores = previous_inference_method
            else:
                delattr(model, "_inference_scores")
            model._train_scores = None
        model._transfer_fit_audit_stats = {
            "source_event_count": len(source),
            "source_rows": int(total),
            "source_rows_per_event": lengths,
            "train_rows_per_segment": train_lengths,
            "validation_rows_per_segment": val_lengths,
            "training_windows": int(train_windows),
            "validation_windows": int(validation_windows),
            "boundary_policy": "independent_source_segments",
            "native_label_train_score_cache_saved": False,
        }

    def score_event(
        self,
        model,
        frame: pd.DataFrame,
        context_frame: Optional[pd.DataFrame] = None,
    ) -> ScoreOutput:
        win_size = self._validate_model(model)
        combined, context_len = _event_inputs(
            frame,
            context_frame,
            context_rows=win_size - 1,
            pad_left=True,
        )
        target_len = len(frame)
        if target_len == 0:
            return ScoreOutput(np.zeros(0, dtype=float), np.zeros(0, dtype=bool))
        all_scores = _score_vector(
            model.detect_score(combined),
            len(combined),
            context=type(model).__name__,
        )
        scores = all_scores[context_len:].copy()
        valid = np.full(target_len, len(combined) >= win_size, dtype=bool)
        return ScoreOutput(scores, valid)


class KMeansSegmentedAdapter(TransferAdapter):
    name = "kmeans_segmented"

    def __init__(self) -> None:
        self.fit_audit_stats_: Optional[dict[str, object]] = None

    def resolved_params(self) -> dict[str, object]:
        return {}

    def fit_metadata(self, model) -> dict[str, object]:
        stats = self.fit_audit_stats_ or getattr(
            model, "_transfer_fit_audit_stats", None
        )
        if not stats:
            return {}
        return {
            "source_event_count": int(stats["source_event_count"]),
            "source_rows": int(stats["source_rows"]),
            "candidate_fit_windows": int(stats["candidate_windows"]),
            "fit_windows": int(stats["fit_windows"]),
            "candidate_windows_per_event": list(
                stats["candidate_windows_per_event"]
            ),
            "fit_windows_per_event": list(stats["fit_windows_per_event"]),
            "native_label_train_score_cache_saved": False,
        }

    def minimum_probe_rows(self, model) -> int:
        window, _ = self._validate_model(model)
        return window

    @staticmethod
    def _validate_model(model) -> tuple[int, int]:
        from tsad_benchmark.baselines.machine_learning.kmeans import KMeansModel

        if type(model) is not KMeansModel:
            raise TypeError(
                "kmeans_segmented only supports the audited KMeansModel class."
            )
        window = int(model.window_size)
        stride = max(int(model.stride), 1)
        if window <= 0:
            raise ValueError("KMeans window_size must be positive.")
        if stride > window:
            raise ValueError(
                "KMeans stride cannot exceed window_size because it leaves "
                "unscored gaps inside an event."
            )
        return window, stride

    def fit_source(self, model, segments: Iterable[pd.DataFrame]) -> None:
        window, stride = self._validate_model(model)
        source = _source_segments(segments)
        lengths = [len(segment) for segment in source]
        if not any(length >= window for length in lengths):
            raise ValueError("No source segment is long enough for a KMeans window.")

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        arrays = [segment.to_numpy(dtype=float, copy=False) for segment in source]
        scaler = StandardScaler()
        for values in arrays:
            scaler.partial_fit(values)

        fit_window_parts = []
        candidate_counts = []
        for values in arrays:
            scaled = scaler.transform(values)
            count = max((len(scaled) - window) // stride + 1, 0)
            candidate_counts.append(count)
            if count == 0:
                continue
            starts = np.arange(count, dtype=np.int64) * stride
            views = sliding_window_view(scaled, window_shape=window, axis=0)
            fit_window_parts.append(views[starts].reshape(count, -1))

        fit_windows = np.concatenate(fit_window_parts, axis=0)
        k = max(min(model.n_clusters, len(fit_windows)), 1)
        estimator = KMeans(
            n_clusters=k,
            random_state=model.random_state,
            n_init=model.n_init,
        )
        estimator.fit(fit_windows)

        model._n_features = arrays[0].shape[1]
        model._scaler = scaler
        model._kmeans = estimator
        model._train_scores = None
        stats: dict[str, object] = {
            "source_event_count": len(source),
            "source_rows": int(sum(lengths)),
            "candidate_windows": int(sum(candidate_counts)),
            "fit_windows": int(sum(candidate_counts)),
            "candidate_windows_per_event": candidate_counts,
            "fit_windows_per_event": candidate_counts,
        }
        self.fit_audit_stats_ = stats
        model._transfer_fit_audit_stats = dict(stats)

    @staticmethod
    def _coverage(length: int, window: int, stride: int) -> np.ndarray:
        covered = np.zeros(length, dtype=bool)
        if length < window:
            return covered
        starts = np.arange(0, length - window + 1, stride, dtype=np.int64)
        delta = np.zeros(length + 1, dtype=np.int64)
        np.add.at(delta, starts, 1)
        np.add.at(delta, starts + window, -1)
        return np.cumsum(delta[:-1]) > 0

    def score_event(
        self,
        model,
        frame: pd.DataFrame,
        context_frame: Optional[pd.DataFrame] = None,
    ) -> ScoreOutput:
        window, stride = self._validate_model(model)
        combined, context_len = _event_inputs(
            frame,
            context_frame,
            context_rows=window - 1,
            pad_left=True,
        )
        target_len = len(frame)
        if target_len == 0:
            return ScoreOutput(np.zeros(0, dtype=float), np.zeros(0, dtype=bool))
        all_scores = _score_vector(
            model.detect_score(combined),
            len(combined),
            context=type(model).__name__,
        )
        valid = self._coverage(len(combined), window, stride)[context_len:]
        return ScoreOutput(all_scores[context_len:].copy(), valid.copy())


class ModelSegmentedAdapter(TransferAdapter):
    name = "model_segmented"

    _MODELS = {
        "tsad_benchmark.baselines.deep_learning.ae.AutoEncoderModel": "merlion",
        "tsad_benchmark.baselines.deep_learning.lstmed.LSTMEDModel": "merlion",
        "tsad_benchmark.baselines.machine_learning.dagmm.DAGMMModel": "merlion",
        "tsad_benchmark.baselines.machine_learning.deeppoint.DeepPointModel": "merlion",
        "tsad_benchmark.baselines.machine_learning.torsk.TorskModel": "torsk",
        "tsad_benchmark.baselines.deep_learning.d3r.D3RModel": "d3r",
        "tsad_benchmark.baselines.deep_learning.gdn.GDNModel": "gdn",
        "tsad_benchmark.baselines.deep_learning.mscred.MSCREDModel": "mscred",
        "tsad_benchmark.baselines.deep_learning.mtad_gat.MTADGATModel": "make_xy",
        "tsad_benchmark.baselines.deep_learning.mtgflow.MTGFlowModel": "raw_loader",
        "tsad_benchmark.baselines.deep_learning.omnianomaly.OmniAnomalyModel": "make_windows",
        "tsad_benchmark.baselines.deep_learning.sarad.SARADModel": "raw_loader",
        "tsad_benchmark.baselines.deep_learning.usad.USADModel": "make_windows",
        "tsad_benchmark.baselines.llm_based.gpt4ts.GPT4TSModel": "gpt4ts",
        "tsad_benchmark.baselines.llm_based.unitime.UniTimeModel": "unitime",
        "tsad_benchmark.baselines.ts_pretrained.chronos.ChronosModel": "pretrained",
        "tsad_benchmark.baselines.ts_pretrained.dada.DADAModel": "pretrained",
        "tsad_benchmark.baselines.ts_pretrained.moment.MOMENTModel": "moment",
        "tsad_benchmark.baselines.ts_pretrained.units.UniTSModel": "units",
        "tsad_benchmark.baselines.finetune_llm.rpcl_tcne_mts_llm.RPCLTCNEMTSLLMModel": "rpcl",
    }

    def fit_metadata(self, model) -> dict[str, object]:
        stats = getattr(model, "_transfer_fit_audit_stats", None)
        return dict(stats) if isinstance(stats, dict) else {}

    def minimum_probe_rows(self, model) -> int:
        self._validate_model(model)
        _, minimum, _ = self._score_contract(model)
        return minimum

    @classmethod
    def _validate_model(cls, model) -> str:
        identity = f"{type(model).__module__}.{type(model).__name__}"
        strategy = cls._MODELS.get(identity)
        if strategy is None:
            raise TypeError(
                f"model_segmented has not audited {identity}; refusing unsafe fallback."
            )
        if identity.endswith("AutoEncoderModel") and int(model.sequence_len) != 1:
            raise ValueError("Cross-domain AutoEncoder is audited only for sequence_len=1.")
        if identity.endswith("DAGMMModel") and int(model.sequence_len) != 1:
            raise ValueError("Cross-domain DAGMM is audited only for sequence_len=1.")
        if identity.endswith("GPT4TSModel") and bool(model.channel_independent):
            raise ValueError(
                "The main GPT4TS configuration is channel-mixed; "
                "channel_independent requires a separate boundary audit."
            )
        return strategy

    @staticmethod
    def _audit(model, source: Sequence[pd.DataFrame], strategy: str) -> None:
        lengths = [len(segment) for segment in source]
        model._transfer_fit_audit_stats = {
            "source_event_count": len(source),
            "source_rows": int(sum(lengths)),
            "source_rows_per_event": lengths,
            "boundary_policy": "independent_source_segments",
            "fit_strategy": strategy,
            "native_label_train_score_cache_saved": False,
        }

    @staticmethod
    def _concat(source: Sequence[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(source, ignore_index=True)

    @staticmethod
    def _concat_arrays(parts: Sequence[np.ndarray], empty_shape) -> np.ndarray:
        nonempty = [part for part in parts if len(part)]
        return np.concatenate(nonempty, axis=0) if nonempty else np.zeros(empty_shape)

    def _fit_make_windows(self, model, source, method_name: str) -> None:
        lengths = [len(segment) for segment in source]
        original = getattr(model, method_name)

        def segmented(this, array):
            parts = []
            cursor = 0
            for length in lengths:
                parts.append(original(array[cursor : cursor + length]))
                cursor += length
            if parts:
                shape = (0, *parts[0].shape[1:])
                return self._concat_arrays(parts, shape)
            return original(array)

        with _instance_method(model, method_name, segmented):
            model.fit(self._concat(source), None)
        model._train_scores = None

    def _fit_make_xy(self, model, source) -> None:
        lengths = [len(segment) for segment in source]
        original = model._make_xy

        def segmented(this, array):
            x_parts, y_parts = [], []
            cursor = 0
            for length in lengths:
                x, y = original(array[cursor : cursor + length])
                cursor += length
                if len(x):
                    x_parts.append(x)
                    y_parts.append(y)
            if not x_parts:
                features = max(int(getattr(this, "_n_features", 0)), 1)
                return (
                    np.zeros((0, this.win_size, features), dtype=np.float32),
                    np.zeros((0, features), dtype=np.float32),
                )
            return np.concatenate(x_parts), np.concatenate(y_parts)

        with _instance_method(model, "_make_xy", segmented):
            model.fit(self._concat(source), None)
        model._train_scores = None

    def _fit_raw_loader(self, model, source) -> None:
        from torch.utils.data import ConcatDataset, DataLoader

        lengths = [len(segment) for segment in source]
        original_scale = model._scale_fit
        original_loader = model._make_loader
        state: dict[str, np.ndarray] = {}

        def capture_scale(this, array):
            scaled = original_scale(array)
            state["base"] = scaled
            return scaled

        def segmented_loader(this, array, shuffle, stride=1):
            if array is None:
                return None
            parts = _array_partition(state["base"], lengths, array)
            datasets = []
            cursor = 0
            template = None
            for length in parts:
                loader = original_loader(
                    array[cursor : cursor + length], shuffle=False, stride=stride
                )
                cursor += length
                if loader is not None and len(loader.dataset):
                    template = loader
                    datasets.append(loader.dataset)
            if not datasets:
                return None
            dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
            return DataLoader(
                dataset,
                batch_size=template.batch_size,
                shuffle=shuffle,
                num_workers=0,
                drop_last=False,
            )

        with _instance_method(model, "_scale_fit", capture_scale), _instance_method(
            model, "_make_loader", segmented_loader
        ):
            model.fit(self._concat(source), None)
        model._train_scores = None

    def _fit_d3r(self, model, source) -> None:
        from torch.utils.data import ConcatDataset, DataLoader

        lengths = [len(segment) for segment in source]
        original_stable = model._stable_data_and_target
        original_loader = model._make_loader
        original_score = model._score_array
        state: dict[str, object] = {}

        def segmented_stable(this, array, time_array):
            data_parts, time_parts, stable_parts = [], [], []
            processed_lengths = []
            cursor = 0
            for length in lengths:
                local_time = this._make_time_embedding(
                    None, length, start_offset=0
                )
                data, times, stable = original_stable(
                    array[cursor : cursor + length],
                    local_time,
                )
                cursor += length
                data_parts.append(data)
                time_parts.append(times)
                stable_parts.append(stable)
                processed_lengths.append(len(data))
            merged = np.concatenate(data_parts, axis=0)
            state["base"] = merged
            state["lengths"] = processed_lengths
            return (
                merged,
                np.concatenate(time_parts, axis=0),
                np.concatenate(stable_parts, axis=0),
            )

        def segmented_loader(this, array, time_array, stable_array, shuffle):
            parts = _array_partition(state["base"], state["lengths"], array)
            datasets = []
            template = None
            cursor = 0
            for length in parts:
                loader = original_loader(
                    array[cursor : cursor + length],
                    time_array[cursor : cursor + length],
                    stable_array[cursor : cursor + length],
                    False,
                )
                cursor += length
                if loader is not None and len(loader.dataset):
                    template = loader
                    datasets.append(loader.dataset)
            if not datasets:
                return None
            dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
            return DataLoader(
                dataset,
                batch_size=template.batch_size,
                shuffle=shuffle,
                num_workers=0,
                drop_last=False,
            )

        def skip_train_score(this, array, time_array):
            return np.zeros(0, dtype=float)

        with _instance_method(
            model, "_stable_data_and_target", segmented_stable
        ), _instance_method(model, "_make_loader", segmented_loader), _instance_method(
            model, "_score_array", skip_train_score
        ):
            model.fit(self._concat(source), None)
        model._train_scores = None
        if original_score is None:
            raise AssertionError

    def _fit_gdn(self, model, source) -> None:
        import tsad_benchmark.baselines.deep_learning.gdn as module
        from torch.utils.data import ConcatDataset

        lengths = [len(segment) for segment in source]
        original = module._GDNWindowDataset

        class SegmentedGDNWindowDataset:
            def __init__(self, array, win_size, stride, train):
                datasets = []
                cursor = 0
                for length in lengths:
                    dataset = original(
                        array[cursor : cursor + length], win_size, stride, train
                    ).dataset
                    cursor += length
                    if len(dataset):
                        datasets.append(dataset)
                if not datasets:
                    self.dataset = []
                elif len(datasets) == 1:
                    self.dataset = datasets[0]
                else:
                    self.dataset = ConcatDataset(datasets)

        module._GDNWindowDataset = SegmentedGDNWindowDataset
        try:
            model.fit(self._concat(source), None)
        finally:
            module._GDNWindowDataset = original
        model._train_scores = None
        model._normal_scores = None

    def _fit_mscred(self, model, source) -> None:
        lengths = [len(segment) for segment in source]
        original = model._make_target_ends

        def segmented(this, array, target_stride=1):
            ends = []
            cursor = 0
            for length in lengths:
                local = original(
                    array[cursor : cursor + length], target_stride=target_stride
                )
                if local.size:
                    ends.append(local + cursor)
                cursor += length
            return np.concatenate(ends) if ends else np.zeros(0, dtype=np.int64)

        with _instance_method(model, "_make_target_ends", segmented):
            model.fit(self._concat(source), None)
        model._train_scores = None

    def _fit_gpt4ts(self, model, source) -> None:
        import tsad_benchmark.baselines.llm_based.gpt4ts as module

        lengths = [len(segment) for segment in source]
        original_scale = model._scale_fit
        original_windows = module.rolling_windows
        state: dict[str, np.ndarray] = {}

        def capture_scale(this, array):
            scaled = original_scale(array)
            state["base"] = scaled
            return scaled

        def segmented_windows(array, window, step=1):
            parts = _array_partition(state["base"], lengths, array)
            windows = []
            cursor = 0
            for length in parts:
                local = original_windows(
                    array[cursor : cursor + length], window, step=step
                )
                cursor += length
                if len(local):
                    windows.append(local)
            shape = (0, int(window), array.shape[1])
            return np.concatenate(windows) if windows else np.zeros(shape, dtype=np.float32)

        module.rolling_windows = segmented_windows
        try:
            with _instance_method(model, "_scale_fit", capture_scale):
                model.fit(self._concat(source), None)
        finally:
            module.rolling_windows = original_windows
        model._train_scores = None

    @staticmethod
    def _fit_split_lengths(model, lengths: Sequence[int]) -> tuple[list[int], list[int]]:
        total = sum(lengths)
        val_n = int(total * float(model.val_ratio))
        window = int(model.win_size)
        if val_n >= window + 1 and total - val_n >= window:
            split = total - val_n
            return (
                _slice_segment_lengths(lengths, 0, split),
                _slice_segment_lengths(lengths, split, total),
            )
        return list(lengths), []

    def _fit_unitime(self, model, source) -> None:
        lengths = [len(segment) for segment in source]
        train_lengths, val_lengths = self._fit_split_lengths(model, lengths)
        train_n, val_n = sum(train_lengths), sum(val_lengths)
        calls = 0

        def starts(this, size, step=1):
            nonlocal calls
            calls += 1
            if val_lengths and int(size) == val_n and not (
                train_n == val_n and calls != 2
            ):
                selected = val_lengths
            elif int(size) == train_n:
                selected = train_lengths
            elif int(size) == sum(lengths):
                selected = lengths
            else:
                raise RuntimeError(f"Unexpected UniTime fit slice length {size}.")
            return _safe_window_starts(selected, model.win_size, step=step)

        with _instance_method(model, "_window_starts", starts):
            model.fit(self._concat(source), None)
        model._train_scores = None
        model._train_score_windows = None

    def _fit_moment(self, model, source) -> None:
        lengths = [len(segment) for segment in source]
        total = sum(lengths)
        if model.fine_tune_epochs > 0 and model.fine_tune_val_ratio > 0.0:
            split = int(total * (1.0 - model.fine_tune_val_ratio))
            split = min(max(split, model.win_size), total)
            train_lengths = _slice_segment_lengths(lengths, 0, split)
            val_lengths = _slice_segment_lengths(lengths, split, total)
        else:
            train_lengths, val_lengths = list(lengths), []
        fit_calls = 0

        def starts(this, size, step, sample_rate=1.0):
            nonlocal fit_calls
            caller = inspect.currentframe().f_back.f_code.co_name
            if caller == "_validation_loss":
                selected = val_lengths
            elif caller == "_fine_tune":
                selected = train_lengths
            elif caller == "fit":
                fit_calls += 1
                if fit_calls == 1:
                    selected = train_lengths
                elif val_lengths and fit_calls == 2:
                    selected = val_lengths
                else:
                    selected = lengths
            else:
                raise RuntimeError(f"Unexpected MOMENT window caller {caller!r}.")
            if int(size) != sum(selected):
                raise RuntimeError(
                    f"MOMENT fit slice has {size} rows; expected {sum(selected)}."
                )
            return _safe_window_starts(
                selected,
                model.win_size,
                step=step,
                sample_rate=sample_rate,
            )

        with _instance_method(model, "_window_starts", starts):
            model.fit(self._concat(source), None)
        model._train_scores = None
        model._train_arr = None

    def _fit_units(self, model, source) -> None:
        def skip_scores(this, array, step=1):
            return np.zeros((0, this.win_size), dtype=float)

        with _instance_method(model, "_window_scores", skip_scores):
            model.fit(self._concat(source), None)
        model._train_scores = None

    def _fit_rpcl(self, model, source) -> None:
        import tsad_benchmark.baselines.finetune_llm.rpcl_tcne_mts_llm as module

        lengths = [len(segment) for segment in source]
        original = module._window_array

        def segmented(array, window, stride=1, starts=None):
            parts = []
            cursor = 0
            effective_stride = max(1, int(getattr(model, "train_stride", stride)))
            extra_starts = np.zeros(0, dtype=np.int64)
            if starts is not None:
                starts_array = np.asarray(starts, dtype=np.int64).reshape(-1)
                expected = module._window_starts(array.shape[0], window, effective_stride)
                extra_starts = np.setdiff1d(starts_array, expected, assume_unique=False)
            for length in lengths:
                local_starts = None
                if starts is not None:
                    local_starts = module._window_starts(length, window, effective_stride)
                    if extra_starts.size:
                        mask = (extra_starts >= cursor) & (
                            extra_starts + int(window) <= cursor + length
                        )
                        if np.any(mask):
                            local_starts = np.unique(
                                np.concatenate([local_starts, extra_starts[mask] - cursor])
                            )
                windows = original(
                    array[cursor : cursor + length],
                    window,
                    stride=effective_stride,
                    starts=local_starts,
                )
                cursor += length
                if len(windows):
                    parts.append(windows)
            shape = (0, int(window), array.shape[1])
            return np.concatenate(parts) if parts else np.zeros(shape, dtype=np.float32)

        module._window_array = segmented
        try:
            model.fit(self._concat(source), None)
        finally:
            module._window_array = original
        model._train_scores = None
        model._train_array = None

    def _fit_torsk(self, model, source) -> None:
        arrays = [segment.to_numpy(dtype=float, copy=False) for segment in source]
        merged = np.concatenate(arrays, axis=0)
        model._n_features = merged.shape[1]
        model._mean = merged.mean(axis=0, keepdims=True)
        model._std = np.maximum(merged.std(axis=0, keepdims=True), 1e-8)
        model._build_reservoir(model._n_features)
        ztz = None
        zty = None
        used = 0
        for array in arrays:
            normalized = (array - model._mean) / model._std
            if len(normalized) < 2:
                continue
            states = model._run_reservoir(normalized)
            pairs = len(normalized) - 1
            start = max(0, min(model.transient, pairs - 1))
            count = pairs - start
            if count <= 0:
                start, count = 0, pairs
            design = np.concatenate(
                [
                    states[start : start + count],
                    normalized[start : start + count],
                    np.ones((count, 1)),
                ],
                axis=1,
            )
            target = normalized[start + 1 : start + 1 + count]
            local_ztz = design.T @ design
            local_zty = design.T @ target
            ztz = local_ztz if ztz is None else ztz + local_ztz
            zty = local_zty if zty is None else zty + local_zty
            used += count
        if not used:
            raise ValueError("No source event has enough rows for Torsk training.")
        regularizer = np.eye(ztz.shape[0]) * model.ridge_lambda
        try:
            model._w_out = np.linalg.solve(ztz + regularizer, zty)
        except np.linalg.LinAlgError:
            model._w_out = np.linalg.pinv(ztz + regularizer) @ zty
        model._train_scores = None

    def _fit_merlion(self, model, source) -> None:
        if type(model).__name__ != "LSTMEDModel":
            model.fit(self._concat(source), None)
            return

        import merlion.models.utils.rolling_window_dataset as rolling_module

        lengths = [len(segment) for segment in source]
        total = sum(lengths)
        original = rolling_module.RollingWindowDataset
        original_builder = model._build_merlion_model
        state: dict[str, object] = {}

        class SegmentedRollingWindowDataset:
            def __init__(self, data, *args, **kwargs):
                if len(data) != total:
                    self.single = original(data, *args, **kwargs)
                    self.datasets = None
                    return
                self.single = None
                self.datasets = []
                cursor = 0
                for length in lengths:
                    part = data.iloc[cursor : cursor + length]
                    cursor += length
                    dataset = original(part, *args, **kwargs)
                    if dataset.n_points > 0:
                        self.datasets.append(dataset)

            def __len__(self):
                if self.single is not None:
                    return len(self.single)
                return sum(len(dataset) for dataset in self.datasets)

            def __iter__(self):
                if self.single is not None:
                    yield from self.single
                    return
                for dataset in self.datasets:
                    yield from dataset

        def build_segmented(this):
            inner = original_builder()
            original_score = inner._get_anomaly_score
            state["inner"] = inner
            state["had_score"] = "_get_anomaly_score" in inner.__dict__
            state["previous_score"] = inner.__dict__.get("_get_anomaly_score")

            def segmented_train_score(inner_self, time_series, time_series_prev=None):
                if time_series_prev is not None or len(time_series) != total:
                    return original_score(time_series, time_series_prev)
                outputs = []
                cursor = 0
                for length in lengths:
                    part = time_series.iloc[cursor : cursor + length]
                    cursor += length
                    outputs.append(original_score(part, None))
                return pd.concat(outputs, axis=0)

            inner._get_anomaly_score = types.MethodType(
                segmented_train_score, inner
            )
            return inner

        rolling_module.RollingWindowDataset = SegmentedRollingWindowDataset
        try:
            with _instance_method(model, "_build_merlion_model", build_segmented):
                model.fit(self._concat(source), None)
        finally:
            rolling_module.RollingWindowDataset = original
            inner = state.get("inner")
            if inner is not None:
                if state["had_score"]:
                    inner._get_anomaly_score = state["previous_score"]
                else:
                    delattr(inner, "_get_anomaly_score")

    def fit_source(self, model, segments: Iterable[pd.DataFrame]) -> None:
        strategy = self._validate_model(model)
        source = _source_segments(segments)
        if strategy == "merlion":
            self._fit_merlion(model, source)
        elif strategy == "torsk":
            self._fit_torsk(model, source)
        elif strategy == "make_windows":
            self._fit_make_windows(model, source, "_make_windows")
        elif strategy == "make_xy":
            self._fit_make_xy(model, source)
        elif strategy == "raw_loader":
            self._fit_raw_loader(model, source)
        elif strategy == "d3r":
            self._fit_d3r(model, source)
        elif strategy == "gdn":
            self._fit_gdn(model, source)
        elif strategy == "mscred":
            self._fit_mscred(model, source)
        elif strategy == "gpt4ts":
            self._fit_gpt4ts(model, source)
        elif strategy == "unitime":
            self._fit_unitime(model, source)
        elif strategy == "moment":
            self._fit_moment(model, source)
        elif strategy == "units":
            self._fit_units(model, source)
        elif strategy == "rpcl":
            self._fit_rpcl(model, source)
        elif strategy == "pretrained":
            model.fit(self._concat(source), None)
            if hasattr(model, "_train_arr"):
                model._train_arr = None
        else:
            raise AssertionError(strategy)
        self._audit(model, source, strategy)

    @staticmethod
    def _score_contract(model) -> tuple[int, int, bool]:
        name = type(model).__name__
        if name == "TorskModel":
            return 1, 2, False
        if name == "LSTMEDModel":
            window = int(model.sequence_len)
            return window - 1, window, False
        if name in {"AutoEncoderModel", "DAGMMModel", "DeepPointModel"}:
            return 0, 1, False
        if name == "ChronosModel":
            minimum = int(model.context_length) + 1
            return int(model.context_length), minimum, False
        if name == "GDNModel" or name == "MTADGATModel":
            window = int(model.win_size)
            return window, window + 1, True
        if name in {"D3RModel", "MTGFlowModel", "USADModel", "RPCLTCNEMTSLLMModel"}:
            window = int(model.win_size)
            return window - 1, window, True
        if name == "SARADModel":
            window = int(model.win_size)
            return window - 1, 1, False
        if name == "MSCREDModel":
            minimum = int(model._first_target_end()) + 1
            return minimum - 1, minimum, False
        window = int(getattr(model, "win_size", 1))
        return max(window - 1, 0), window, False

    def score_event(
        self,
        model,
        frame: pd.DataFrame,
        context_frame: Optional[pd.DataFrame] = None,
    ) -> ScoreOutput:
        self._validate_model(model)
        context_rows, minimum, endpoint = self._score_contract(model)
        combined, context_len = _event_inputs(
            frame,
            context_frame,
            context_rows=context_rows,
            pad_left=True,
        )
        target_len = len(frame)
        if target_len == 0:
            return ScoreOutput(np.zeros(0, dtype=float), np.zeros(0, dtype=bool))
        scores = _score_vector(
            model.detect_score(combined), len(combined), context=type(model).__name__
        )[context_len:].copy()
        if endpoint:
            valid = np.arange(context_len, len(combined)) >= (minimum - 1)
        else:
            valid = np.full(target_len, len(combined) >= minimum, dtype=bool)
        return ScoreOutput(scores, valid)


class NativeSegmentedAdapter(TransferAdapter):
    name = "native_segmented"

    def fit_source(self, model, segments: Iterable[pd.DataFrame]) -> None:
        source = _source_segments(segments)
        fit_segments = getattr(model, "fit_segments", None)
        if not callable(fit_segments):
            raise TypeError(
                f"{type(model).__name__} does not implement fit_segments(segments)."
            )
        fit_segments(source)

    def score_event(
        self,
        model,
        frame: pd.DataFrame,
        context_frame: Optional[pd.DataFrame] = None,
    ) -> ScoreOutput:
        target = _numeric_frame(frame, name="event frame", allow_empty=True)
        if target.empty:
            return ScoreOutput(np.zeros(0, dtype=float), np.zeros(0, dtype=bool))
        context = None
        if context_frame is not None:
            context = _numeric_frame(
                context_frame,
                name="event context",
                columns=list(target.columns),
                allow_empty=True,
            )
        score_segment = getattr(model, "detect_score_segment", None)
        if not callable(score_segment):
            raise TypeError(
                f"{type(model).__name__} does not implement "
                "detect_score_segment(frame, context_frame=...)."
            )
        raw = score_segment(target, context_frame=context)
        if not isinstance(raw, tuple) or len(raw) != 2:
            raise TypeError(
                "detect_score_segment must return (scores, valid_mask)."
            )
        scores = _score_vector(raw[0], len(target), context=type(model).__name__)
        valid = _valid_vector(raw[1], len(target), context=type(model).__name__)
        return ScoreOutput(scores, valid)


_REGISTRY: Dict[str, Type[TransferAdapter]] = {}


def register_transfer_adapter(
    name: str, adapter_type: Type[TransferAdapter], *, replace: bool = False
) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Adapter name must be non-empty.")
    if not isinstance(adapter_type, type) or not issubclass(
        adapter_type, TransferAdapter
    ):
        raise TypeError("adapter_type must be a TransferAdapter subclass.")
    if key in _REGISTRY and not replace:
        raise KeyError(f"Transfer adapter {key!r} is already registered.")
    _REGISTRY[key] = adapter_type


def build_transfer_adapter(name: str, **kwargs) -> TransferAdapter:
    key = str(name).strip().lower()
    adapter_type = _REGISTRY.get(key)
    if adapter_type is None:
        raise KeyError(
            f"Unknown transfer adapter {name!r}; choose one of "
            f"{sorted(_REGISTRY)}."
        )
    return adapter_type(**kwargs)


def available_transfer_adapters() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


for _adapter in (
    PointConcatAdapter,
    DLBaseSegmentedAdapter,
    KMeansSegmentedAdapter,
    ModelSegmentedAdapter,
    NativeSegmentedAdapter,
):
    register_transfer_adapter(_adapter.name, _adapter)


__all__ = [
    "ScoreOutput",
    "TransferAdapter",
    "PointConcatAdapter",
    "DLBaseSegmentedAdapter",
    "KMeansSegmentedAdapter",
    "ModelSegmentedAdapter",
    "NativeSegmentedAdapter",
    "register_transfer_adapter",
    "build_transfer_adapter",
    "available_transfer_adapters",
]
