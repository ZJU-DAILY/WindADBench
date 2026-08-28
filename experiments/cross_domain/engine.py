
from __future__ import annotations

import gc
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from tsad_benchmark.common.random_utils import fix_random_seed
from tsad_benchmark.evaluation.strategy.anomaly_detect import AnomalyDetect

from .aggregation import write_result_tables
from .artifacts import (
    atomic_csv,
    atomic_json,
    atomic_npz,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
    write_failure,
)
from .core3_contract import FEATURE_COLUMNS
from .metrics import compute_label_metrics, compute_score_metrics
from .protocol import (
    MedianImputerState,
    PreparedEvent,
    apply_threshold,
    build_dataset_split,
    build_protocol_plan,
    calibrate_at_fpr,
    direction_summary,
    fit_median_imputer,
    prepare_event,
    source_records,
    target_records,
    validate_protocol,
)
from .transfer_adapters import TransferAdapter, build_transfer_adapter


LOGGER = logging.getLogger(__name__)


@contextmanager
def _progress_heartbeat(label: str, interval_seconds: float = 60.0):
    started = time.perf_counter()
    stopped = threading.Event()

    def report() -> None:
        while not stopped.wait(interval_seconds):
            LOGGER.info("%s | still running | elapsed=%.1fs", label, time.perf_counter() - started)

    LOGGER.info("%s | started", label)
    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()
        LOGGER.info("%s | completed | elapsed=%.1fs", label, time.perf_counter() - started)


@dataclass(frozen=True)
class ExperimentModelSpec:
    factory: Any
    transfer_adapter: str
    transfer_adapter_params: Mapping[str, object]
    model_path: str
    benchmark_adapter: Optional[str]
    effective_model_params: Mapping[str, object]
    config_entry: Mapping[str, object]

    @property
    def model_name(self) -> str:
        return str(self.factory.model_name)


def safe_name(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("._") or "item"


_WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def _path_name(value: object, *, maximum: int = 80) -> str:
    name = safe_name(value)
    if len(name) > maximum:
        raise ValueError(f"Path component is too long after sanitization: {name!r}.")
    if name.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"Reserved Windows path component: {name!r}.")
    return name


def _json_params(value: object) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True, default=str)


def _scalar_stat(model: Any, name: str) -> float:
    try:
        method = getattr(model, name, None)
        return float(method()) if callable(method) else float("nan")
    except Exception:
        return float("nan")


def _release() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _measure(callable_) -> tuple[Any, Dict[str, float]]:
    monitor = AnomalyDetect._ResourceMonitor()
    AnomalyDetect._synchronize_torch_cuda()
    started = time.perf_counter()
    monitor.start()
    try:
        result = callable_()
        AnomalyDetect._synchronize_torch_cuda()
    finally:
        monitor.stop()
    return result, {
        "seconds": float(time.perf_counter() - started),
        "peak_memory_mb": float(monitor.peak_memory_mb()),
        "peak_gpu_memory_mb": float(monitor.peak_gpu_memory_mb()),
        "cpu_usage_percent": float(monitor.cpu_usage_percent()),
    }


class CrossDomainEngine:
    def __init__(
        self,
        *,
        manifest: pd.DataFrame,
        config: Mapping[str, object],
        dataset_root: Path,
        run_dir: Path,
        run_id: str,
        source_farms: Sequence[str] = ("A", "B", "C"),
        resume: bool = False,
    ) -> None:
        self.manifest = manifest
        self.config = config
        self.dataset_root = dataset_root
        self.run_dir = run_dir
        self.run_id = run_id
        self.resume = bool(resume)
        self.source_farms = tuple(str(farm).upper() for farm in source_farms)
        if (
            not self.source_farms
            or len(set(self.source_farms)) != len(self.source_farms)
            or not set(self.source_farms).issubset({"A", "B", "C"})
        ):
            raise ValueError("source_farms must contain unique values from A, B, C.")
        self.holdouts = validate_protocol(manifest, config)
        protocol = config["protocol"]
        if not isinstance(protocol, Mapping):
            raise ValueError("protocol must be a mapping.")
        self.protocols = tuple(
            str(value) for value in protocol["evaluation_protocols"]
        )
        self.tracks = tuple(str(value) for value in protocol["evaluation_tracks"])
        self.label_policy = str(protocol["label_policy"])
        self.budget = float(protocol["false_positive_budget"])
        self.defer_score_vus = bool(protocol.get("defer_score_vus", False))
        self.points_per_day = int(protocol["points_per_day"])
        self.lead_delta_points = int(protocol["lead_delta_points"])
        self.seed = int(protocol.get("seed", 2026))
        self.event_rows: List[Dict[str, object]] = []
        self.threshold_rows: List[Dict[str, object]] = []
        self.training_rows: List[Dict[str, object]] = []
        self.resource_rows: List[Dict[str, object]] = []
        self.imputer_rows: List[Dict[str, object]] = []
        self._target_cache: OrderedDict[str, PreparedEvent] = OrderedDict()
        self._target_imputers: Dict[str, MedianImputerState] = {}
        self._verified_events: set[str] = set()
        if self.resume:
            self._load_resume_ledgers()
        for farm in ("A", "B", "C"):
            seen = set()
            for file_name in target_records(manifest, farm, self.holdouts)[
                "file_name"
            ].astype(str):
                key = _path_name(Path(file_name).stem).casefold()
                if key in seen:
                    raise ValueError(
                        f"Target event paths collide after sanitization in farm {farm}: "
                        f"{file_name!r}."
                    )
                seen.add(key)

    def _load_resume_ledgers(self) -> None:
        ledgers = {
            "event_rows": "per_event.csv",
            "threshold_rows": "thresholds.csv",
            "training_rows": "training.csv",
            "resource_rows": "resources.csv",
        }
        for attribute, name in ledgers.items():
            path = self.run_dir / "results" / name
            if path.is_file():
                setattr(self, attribute, pd.read_csv(path).to_dict("records"))
        imputer_ledger = self.run_dir / "preprocessing" / "imputers.csv"
        if imputer_ledger.is_file():
            self.imputer_rows = pd.read_csv(imputer_ledger).to_dict("records")

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.run_dir).as_posix()

    def _write_ledger(self, rows: List[Dict[str, object]], name: str) -> None:
        if rows:
            atomic_csv(pd.DataFrame(rows), self.run_dir / "results" / name)

    def write_plans(self) -> None:
        plan = build_protocol_plan(
            self.manifest, self.config, source_farms=self.source_farms
        )
        atomic_csv(
            build_dataset_split(self.manifest, self.config),
            self.run_dir / "plans" / "dataset_split.csv",
        )
        atomic_csv(plan, self.run_dir / "plans" / "protocol_plan.csv")
        atomic_csv(
            direction_summary(plan), self.run_dir / "plans" / "direction_plan.csv"
        )

    def _record_imputer(
        self,
        imputer: MedianImputerState,
        path: Path,
        *,
        scope: str,
        farm_id: str,
        event: Optional[PreparedEvent] = None,
    ) -> None:
        atomic_json(imputer.to_dict(), path)
        row: Dict[str, object] = {
            "track": "shared",
            "operation": "median_imputation",
            "normalization_applied": False,
            "scope": scope,
            "farm_id": farm_id,
            "target_asset_id": "" if event is None else event.record["asset_id"],
            "target_event_id": "" if event is None else event.record["event_id"],
            "target_file_name": "" if event is None else event.record["file_name"],
            "fitted_rows": imputer.fitted_rows,
            "path": self._relative(path),
            "sha256": sha256_file(path),
        }
        self.imputer_rows.append(row)
        atomic_csv(
            pd.DataFrame(self.imputer_rows),
            self.run_dir / "preprocessing" / "imputers.csv",
        )

    def _source_data(
        self, source_farm: str
    ) -> tuple[pd.DataFrame, List[pd.DataFrame], MedianImputerState]:
        records = source_records(self.manifest, source_farm, self.holdouts)
        prepared = []
        total = len(records)
        for position, (_, record) in enumerate(records.iterrows(), start=1):
            LOGGER.info(
                "[source %s] loading train event %d/%d: %s",
                source_farm,
                position,
                total,
                record["file_name"],
            )
            prepared.append(self._prepare_event(record.to_dict()))
        imputer = fit_median_imputer(event.train_raw for event in prepared)
        imputer_path = (
            self.run_dir / "preprocessing" / "source_imputers" / f"farm_{source_farm}.json"
        )
        self._record_imputer(
            imputer, imputer_path, scope="source_farm_train", farm_id=source_farm
        )
        segments = [event.train_features(imputer) for event in prepared]
        del prepared
        gc.collect()
        return records, segments, imputer

    def _resume_source_data(
        self, source_farm: str
    ) -> tuple[pd.DataFrame, List[pd.DataFrame], MedianImputerState]:
        records = source_records(self.manifest, source_farm, self.holdouts)
        path = (
            self.run_dir
            / "preprocessing"
            / "source_imputers"
            / f"farm_{source_farm}.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Resume source imputer is missing: {path}")
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        imputer = MedianImputerState.from_dict(document)
        expected_rows = int(records["train_rows"].sum())
        if imputer.fitted_rows != expected_rows:
            raise ValueError(
                f"Resume source imputer fitted_rows={imputer.fitted_rows}, "
                f"expected {expected_rows}."
            )
        matches = [
            row
            for row in self.imputer_rows
            if str(row.get("scope")) == "source_farm_train"
            and str(row.get("farm_id")) == source_farm
        ]
        if len(matches) != 1 or str(matches[0].get("sha256")) != sha256_file(path):
            raise ValueError("Resume source imputer ledger or checksum is invalid.")
        return records, [], imputer

    def _prepare_event(self, record: Mapping[str, object]) -> PreparedEvent:
        key = str(record["file_name"])
        verify = key not in self._verified_events
        event = prepare_event(
            self.dataset_root, record, verify_digest=verify
        )
        if verify:
            self._verified_events.add(key)
        return event

    def _target_event(self, record: Mapping[str, object]) -> PreparedEvent:
        key = str(record["file_name"])
        event = self._target_cache.get(key)
        if event is None:
            event = self._prepare_event(record)
            self._target_cache[key] = event
            if len(self._target_cache) > 2:
                self._target_cache.popitem(last=False)
        else:
            self._target_cache.move_to_end(key)
        return event

    def _target_imputer(self, event: PreparedEvent) -> MedianImputerState:
        key = str(event.record["file_name"])
        imputer = self._target_imputers.get(key)
        if imputer is not None:
            return imputer
        imputer = fit_median_imputer([event.train_raw])
        farm = str(event.record["farm_id"])
        path = (
            self.run_dir
            / "preprocessing"
            / "target_imputers"
            / f"farm_{farm}"
            / f"{safe_name(Path(key).stem)}.json"
        )
        self._record_imputer(
            imputer, path, scope="target_event_train", farm_id=farm, event=event
        )
        self._target_imputers[key] = imputer
        return imputer

    @staticmethod
    def _validate_adapter_model(adapter: TransferAdapter, model: Any) -> None:
        validator = getattr(adapter, "_validate_model", None)
        if callable(validator):
            validator(model)

    @staticmethod
    def _probe_segment(
        model: Any,
        adapter: TransferAdapter,
        segments: Sequence[pd.DataFrame],
    ) -> pd.DataFrame:
        longest = max(segments, key=len)
        minimum = max(int(adapter.minimum_probe_rows(model)), 1)
        if len(longest) < minimum:
            raise ValueError(
                f"Checkpoint probe needs {minimum} rows, but the longest source "
                f"event contains {len(longest)}."
            )
        rows = min(len(longest), max(256, 2 * minimum))
        return longest.iloc[:rows].reset_index(drop=True)

    def _save_model(
        self,
        model: Any,
        adapter: TransferAdapter,
        segments: Sequence[pd.DataFrame],
        path: Path,
    ) -> Dict[str, object]:
        probe = self._probe_segment(model, adapter, segments)
        reference: Dict[str, np.ndarray] = {}
        reloads: List[Dict[str, object]] = []
        rtol = 1e-5
        atol = 1e-7

        def validator(restored: Any) -> bool:
            fix_random_seed(self.seed)
            output = adapter.score_event(restored, probe, context_frame=None)
            if not output.valid_mask.any():
                raise ValueError("Checkpoint probe produced no valid scores.")
            attempt = len(reloads) + 1
            valid = output.valid_mask.copy()
            scores = output.scores[valid].copy()
            if not reference:
                reference["valid"] = valid
                reference["scores"] = scores
                reloads.append(
                    {
                        "attempt": attempt,
                        "valid_rows": int(valid.sum()),
                        "max_abs_error": 0.0,
                        "max_relative_error": 0.0,
                        "passed": True,
                    }
                )
                return True
            masks_equal = np.array_equal(reference["valid"], valid)
            if not masks_equal or reference["scores"].shape != scores.shape:
                raise RuntimeError(
                    "Checkpoint reload changed the valid-score mask or score shape."
                )
            absolute = np.abs(reference["scores"] - scores)
            denominator = np.maximum(
                np.maximum(np.abs(reference["scores"]), np.abs(scores)), 1e-12
            )
            max_abs = float(absolute.max(initial=0.0))
            max_relative = float((absolute / denominator).max(initial=0.0))
            passed = bool(
                np.allclose(reference["scores"], scores, rtol=rtol, atol=atol)
            )
            reloads.append(
                {
                    "attempt": attempt,
                    "valid_rows": int(valid.sum()),
                    "max_abs_error": max_abs,
                    "max_relative_error": max_relative,
                    "passed": passed,
                }
            )
            LOGGER.info(
                "checkpoint reload validation %d | rows=%d max_abs=%.3e "
                "max_rel=%.3e passed=%s",
                attempt,
                int(valid.sum()),
                max_abs,
                max_relative,
                passed,
            )
            if not passed:
                raise RuntimeError(
                    "Checkpoint reload scores differ beyond tolerance: "
                    f"max_abs={max_abs:.6e}, max_relative={max_relative:.6e}, "
                    f"rtol={rtol:.1e}, atol={atol:.1e}."
                )
            return True

        metadata = save_checkpoint(model, path, validator=validator)
        validation_path = path.with_name("checkpoint_validation.json")
        atomic_json(
            {
                "status": "passed",
                "policy": "two_independent_same_device_reloads",
                "seed": self.seed,
                "probe_rows": len(probe),
                "rtol": rtol,
                "atol": atol,
                "reloads": reloads,
            },
            validation_path,
        )
        metadata["validation_path"] = str(validation_path)
        metadata["validation_sha256"] = sha256_file(validation_path)
        return metadata

    @staticmethod
    def _score_loaded(
        checkpoint: Path,
        adapter: TransferAdapter,
        frame: pd.DataFrame,
        context: Optional[pd.DataFrame],
    ) -> tuple[Any, Dict[str, float]]:
        load_started = time.perf_counter()
        model = load_checkpoint(checkpoint)
        load_seconds = time.perf_counter() - load_started
        try:
            output, resource = _measure(
                lambda: adapter.score_event(model, frame, context_frame=context)
            )
            resource["checkpoint_load_seconds"] = float(load_seconds)
            return output, resource
        finally:
            del model
            _release()

    def _source_threshold(
        self,
        *,
        spec: ExperimentModelSpec,
        source_farm: str,
        checkpoint: Path,
        checkpoint_meta: Mapping[str, object],
        adapter: TransferAdapter,
        segments: Sequence[pd.DataFrame],
        records: pd.DataFrame,
    ) -> tuple[float, float]:
        scores: List[np.ndarray] = []
        valid: List[np.ndarray] = []
        offsets = [0]
        calibration_seconds = 0.0
        load_seconds = 0.0
        peak_memory_mb = 0.0
        peak_gpu_memory_mb = 0.0
        peak_cpu_usage_percent = 0.0
        total_segments = len(segments)
        for position, segment in enumerate(segments, start=1):
            threshold_label = (
                f"[{spec.model_name} source {source_farm}] "
                f"threshold scores {position}/{total_segments}"
            )
            with _progress_heartbeat(threshold_label):
                output, resource = self._score_loaded(
                    checkpoint, adapter, segment, context=None
                )
            scores.append(output.scores)
            valid.append(output.valid_mask)
            offsets.append(offsets[-1] + len(segment))
            calibration_seconds += resource["seconds"]
            load_seconds += resource["checkpoint_load_seconds"]
            peak_memory_mb = max(peak_memory_mb, resource["peak_memory_mb"])
            peak_gpu_memory_mb = max(
                peak_gpu_memory_mb, resource["peak_gpu_memory_mb"]
            )
            peak_cpu_usage_percent = max(
                peak_cpu_usage_percent, resource["cpu_usage_percent"]
            )
        all_scores = np.concatenate(scores)
        all_valid = np.concatenate(valid)
        if not all_valid.any():
            raise ValueError("Source calibration produced no valid scores.")
        threshold, _, realized = calibrate_at_fpr(
            all_scores[all_valid], self.budget
        )
        model_key = safe_name(spec.model_name)
        artifact = (
            self.run_dir
            / "calibration"
            / model_key
            / "label"
            / f"source_{source_farm}"
            / "source_train_scores.npz"
        )
        atomic_npz(
            {
                "scores": all_scores,
                "valid_mask": all_valid,
                "event_offsets": np.asarray(offsets, dtype=np.int64),
                "event_file_names": records["file_name"].astype(str).to_numpy(),
            },
            artifact,
        )
        self.threshold_rows.append(
            {
                "run_id": self.run_id,
                "track": "label",
                "track_variant": "fpr_label",
                "label_policy": self.label_policy,
                "model_name": spec.model_name,
                "model_params": _json_params(spec.effective_model_params),
                "transfer_adapter": spec.transfer_adapter,
                "transfer_adapter_params": _json_params(
                    spec.transfer_adapter_params
                ),
                "protocol": "strict_zero_shot",
                "source_farm": source_farm,
                "target_farm": "",
                "target_asset_id": "",
                "target_event_id": "",
                "target_file_name": "",
                "calibration_scope": "source_farm_train",
                "imputation_scope": "source_farm_train",
                "normalization_scope": "model_native_source_fit_only",
                "target_train_used": False,
                "target_prediction_used": False,
                "target_labels_used": False,
                "calibration_rows": int(len(all_scores)),
                "calibration_valid_rows": int(all_valid.sum()),
                "fpr_budget": self.budget,
                "threshold": threshold,
                "calibration_fpr_realized": realized,
                "calibration_seconds": calibration_seconds,
                "checkpoint_load_seconds": load_seconds,
                "calibration_peak_memory_mb": peak_memory_mb,
                "calibration_peak_gpu_memory_mb": peak_gpu_memory_mb,
                "calibration_peak_cpu_usage_percent": peak_cpu_usage_percent,
                "checkpoint_sha256": checkpoint_meta["sha256"],
                "scores_path": self._relative(artifact),
                "scores_sha256": sha256_file(artifact),
            }
        )
        self._write_ledger(self.threshold_rows, "thresholds.csv")
        return threshold, realized

    def _target_threshold(
        self,
        *,
        spec: ExperimentModelSpec,
        source_farm: str,
        target_farm: str,
        event: PreparedEvent,
        checkpoint: Path,
        checkpoint_meta: Mapping[str, object],
        adapter: TransferAdapter,
        train_frame: pd.DataFrame,
    ) -> tuple[float, float, Dict[str, float]]:
        output, resource = self._score_loaded(
            checkpoint, adapter, train_frame, context=None
        )
        if not output.valid_mask.any():
            raise ValueError("Target-normal calibration produced no valid scores.")
        threshold, _, realized = calibrate_at_fpr(
            output.scores[output.valid_mask], self.budget
        )
        file_stem = safe_name(Path(str(event.record["file_name"])).stem)
        artifact = (
            self.run_dir
            / "calibration"
            / safe_name(spec.model_name)
            / "label"
            / f"source_{source_farm}"
            / "target_normal_calibrated"
            / f"{source_farm}_to_{target_farm}"
            / f"{file_stem}.npz"
        )
        atomic_npz(
            {
                "scores": output.scores,
                "valid_mask": output.valid_mask,
                "row_id": pd.to_numeric(
                    event.train_metadata["id"], errors="raise"
                ).to_numpy(dtype=np.int64),
            },
            artifact,
        )
        self.threshold_rows.append(
            {
                "run_id": self.run_id,
                "track": "label",
                "track_variant": "fpr_label",
                "label_policy": self.label_policy,
                "model_name": spec.model_name,
                "model_params": _json_params(spec.effective_model_params),
                "transfer_adapter": spec.transfer_adapter,
                "transfer_adapter_params": _json_params(
                    spec.transfer_adapter_params
                ),
                "protocol": "target_normal_calibrated",
                "source_farm": source_farm,
                "target_farm": target_farm,
                "target_asset_id": event.record["asset_id"],
                "target_event_id": event.record["event_id"],
                "target_file_name": event.record["file_name"],
                "calibration_scope": "target_event_train",
                "imputation_scope": "target_event_train",
                "normalization_scope": "model_native_source_fit_only",
                "target_train_used": True,
                "target_prediction_used": False,
                "target_labels_used": False,
                "calibration_rows": int(len(output.scores)),
                "calibration_valid_rows": int(output.valid_mask.sum()),
                "fpr_budget": self.budget,
                "threshold": threshold,
                "calibration_fpr_realized": realized,
                "calibration_seconds": resource["seconds"],
                "checkpoint_load_seconds": resource["checkpoint_load_seconds"],
                "calibration_peak_memory_mb": resource["peak_memory_mb"],
                "calibration_peak_gpu_memory_mb": resource[
                    "peak_gpu_memory_mb"
                ],
                "calibration_peak_cpu_usage_percent": resource[
                    "cpu_usage_percent"
                ],
                "checkpoint_sha256": checkpoint_meta["sha256"],
                "scores_path": self._relative(artifact),
                "scores_sha256": sha256_file(artifact),
            }
        )
        self._write_ledger(self.threshold_rows, "thresholds.csv")
        return threshold, realized, resource

    def _prediction_path(
        self,
        spec: ExperimentModelSpec,
        source_farm: str,
        target_farm: str,
        track: str,
        protocol: str,
        file_name: str,
    ) -> Path:
        if track not in (*self.tracks, "shared"):
            raise ValueError(f"Unknown track {track!r}.")
        return (
            self.run_dir
            / "predictions"
            / safe_name(spec.model_name)
            / f"source_{source_farm}"
            / track
            / protocol
            / f"{source_farm}_to_{target_farm}"
            / f"{safe_name(Path(file_name).stem)}.csv.gz"
        )

    def _evaluate_event(
        self,
        *,
        spec: ExperimentModelSpec,
        source_farm: str,
        target_farm: str,
        source_records_frame: pd.DataFrame,
        source_imputer: MedianImputerState,
        source_threshold: float,
        source_realized: float,
        training_row: Mapping[str, object],
        checkpoint: Path,
        checkpoint_meta: Mapping[str, object],
        adapter: TransferAdapter,
        event: PreparedEvent,
        protocol: str,
    ) -> List[Dict[str, object]]:
        if protocol == "strict_zero_shot":
            imputer = source_imputer
            threshold = source_threshold
            realized = source_realized
            calibration_resource = {
                "seconds": 0.0,
                "checkpoint_load_seconds": 0.0,
                "peak_memory_mb": 0.0,
                "peak_gpu_memory_mb": 0.0,
                "cpu_usage_percent": 0.0,
            }
            context = None
            imputation_scope = "source_farm_train"
            threshold_scope = "source_farm_train"
            target_train_used = False
            temporal_start_policy = "repeat_first_prediction_point_if_required"
        elif protocol == "target_normal_calibrated":
            imputer = self._target_imputer(event)
            train_frame = event.train_features(imputer)
            threshold, realized, calibration_resource = self._target_threshold(
                spec=spec,
                source_farm=source_farm,
                target_farm=target_farm,
                event=event,
                checkpoint=checkpoint,
                checkpoint_meta=checkpoint_meta,
                adapter=adapter,
                train_frame=train_frame,
            )
            context = train_frame
            imputation_scope = "target_event_train"
            threshold_scope = "target_event_train"
            target_train_used = True
            temporal_start_policy = "target_train_tail_then_repeat_first_if_short"
        else:
            raise ValueError(f"Unknown protocol {protocol!r}.")

        test_frame = event.test_features(imputer)
        output, resource = self._score_loaded(
            checkpoint, adapter, test_frame, context=context
        )
        valid = output.valid_mask
        labels = event.test_labels
        if not valid.any():
            raise ValueError("Target event produced no valid test scores.")
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size > 1 and np.any(np.diff(valid_indices) != 1):
            raise ValueError(
                "Non-contiguous valid scores would distort event and range metrics."
            )
        if str(event.record["event_label"]) == "anomaly" and np.any(
            (labels == 1) & ~valid
        ):
            raise ValueError("Temporal warm-up masks part of the anomaly interval.")
        valid_predictions = apply_threshold(output.scores[valid], threshold)
        event_label = str(event.record["event_label"])
        score_metrics = compute_score_metrics(
            labels[valid],
            output.scores[valid],
            event_label=event_label,
            defer_vus=self.defer_score_vus,
        )
        label_metrics = compute_label_metrics(
            labels[valid],
            valid_predictions,
            event_label=event_label,
            points_per_day=self.points_per_day,
            lead_delta_points=self.lead_delta_points,
        )

        predictions = pd.array([pd.NA] * len(test_frame), dtype="Int8")
        predictions[valid] = valid_predictions
        prediction_frame = event.test_metadata.copy()
        prediction_frame["label"] = labels
        prediction_frame["score"] = np.where(valid, output.scores, np.nan)
        prediction_frame["prediction"] = predictions
        prediction_frame["score_valid"] = valid
        prediction_frame["track"] = "shared"
        prediction_frame["tracks"] = _json_params(self.tracks)
        prediction_frame["track_variants"] = _json_params(
            {"score": "continuous_score", "label": "fpr_label"}
        )
        prediction_frame["protocol"] = protocol
        prediction_frame["threshold"] = threshold
        prediction_frame["label_policy"] = self.label_policy
        prediction_frame["imputation_scope"] = imputation_scope
        prediction_frame["normalization_scope"] = "model_native_source_fit_only"
        prediction_frame["temporal_start_policy"] = temporal_start_policy
        prediction_path = self._prediction_path(
            spec,
            source_farm,
            target_farm,
            "shared",
            protocol,
            str(event.record["file_name"]),
        )
        atomic_csv(prediction_frame, prediction_path)
        prediction_sha256 = sha256_file(prediction_path)
        evaluation_wall_seconds = (
            calibration_resource["seconds"]
            + calibration_resource["checkpoint_load_seconds"]
            + resource["seconds"]
            + resource["checkpoint_load_seconds"]
        )
        self.resource_rows.append(
            {
                "run_id": self.run_id,
                "track": "shared",
                "tracks": _json_params(self.tracks),
                "model_name": spec.model_name,
                "protocol": protocol,
                "source_farm": source_farm,
                "target_farm": target_farm,
                "target_asset_id": event.record["asset_id"],
                "target_event_id": event.record["event_id"],
                "target_file_name": event.record["file_name"],
                "resource_accounting": "shared_once",
                "calibration_seconds_event": calibration_resource["seconds"],
                "calibration_checkpoint_load_seconds_event": (
                    calibration_resource["checkpoint_load_seconds"]
                ),
                "calibration_peak_memory_mb": calibration_resource[
                    "peak_memory_mb"
                ],
                "calibration_peak_gpu_memory_mb": calibration_resource[
                    "peak_gpu_memory_mb"
                ],
                "calibration_cpu_usage_percent": calibration_resource[
                    "cpu_usage_percent"
                ],
                "inference_seconds_event": resource["seconds"],
                "checkpoint_load_seconds_event": resource[
                    "checkpoint_load_seconds"
                ],
                "inference_points_event": int(len(test_frame)),
                "inference_peak_memory_mb": resource["peak_memory_mb"],
                "inference_peak_gpu_memory_mb": resource["peak_gpu_memory_mb"],
                "inference_cpu_usage_percent": resource["cpu_usage_percent"],
                "evaluation_wall_seconds_event": evaluation_wall_seconds,
                "prediction_path": self._relative(prediction_path),
                "prediction_sha256": prediction_sha256,
            }
        )
        self._write_ledger(self.resource_rows, "resources.csv")

        common: Dict[str, object] = {
            "run_id": self.run_id,
            "seed": self.seed,
            "model_name": spec.model_name,
            "model_path": spec.model_path,
            "model_params": _json_params(spec.effective_model_params),
            "benchmark_adapter": spec.benchmark_adapter or "",
            "transfer_adapter": spec.transfer_adapter,
            "transfer_adapter_params": _json_params(
                spec.transfer_adapter_params
            ),
            "protocol": protocol,
            "source_farm": source_farm,
            "target_farm": target_farm,
            "transfer_type": (
                "cross_turbine" if source_farm == target_farm else "cross_farm"
            ),
            "source_asset_count": int(source_records_frame["asset_id"].nunique()),
            "source_event_count": int(len(source_records_frame)),
            "source_train_rows": int(training_row["source_train_rows"]),
            "target_asset_id": event.record["asset_id"],
            "target_event_id": event.record["event_id"],
            "target_file_name": event.record["file_name"],
            "target_event_label": event.record["event_label"],
            "target_train_rows": int(len(event.train_raw)),
            "target_test_rows": int(len(event.test_raw)),
            "target_test_valid_rows": int(valid.sum()),
            "imputation_scope": imputation_scope,
            "normalization_scope": "model_native_source_fit_only",
            "temporal_start_policy": temporal_start_policy,
            "target_train_used": target_train_used,
            "checkpoint_path": self._relative(checkpoint),
            "checkpoint_sha256": checkpoint_meta["sha256"],
            "resource_accounting": "shared_once",
            "fit_seconds_source": training_row["fit_seconds_source"],
            "calibration_seconds_event": calibration_resource["seconds"],
            "calibration_checkpoint_load_seconds_event": (
                calibration_resource["checkpoint_load_seconds"]
            ),
            "calibration_peak_memory_mb": calibration_resource[
                "peak_memory_mb"
            ],
            "calibration_peak_gpu_memory_mb": calibration_resource[
                "peak_gpu_memory_mb"
            ],
            "calibration_cpu_usage_percent": calibration_resource[
                "cpu_usage_percent"
            ],
            "inference_seconds_event": resource["seconds"],
            "checkpoint_load_seconds_event": resource[
                "checkpoint_load_seconds"
            ],
            "inference_points_event": int(len(test_frame)),
            "inference_peak_memory_mb": resource["peak_memory_mb"],
            "inference_peak_gpu_memory_mb": resource["peak_gpu_memory_mb"],
            "inference_cpu_usage_percent": resource["cpu_usage_percent"],
            "evaluation_wall_seconds_event": evaluation_wall_seconds,
        }
        score_result = {
            **common,
            "track": "score",
            "track_variant": "continuous_score",
            "score_rule": "continuous_anomaly_score",
            "score_metric_status": (
                "not_applicable"
                if event_label == "normal"
                else "partial_deferred_vus"
                if self.defer_score_vus
                else "complete"
            ),
            "label_policy": "not_applicable",
            "prediction_path": self._relative(prediction_path),
            "prediction_sha256": prediction_sha256,
            **score_metrics,
        }
        label_result = {
            **common,
            "track": "label",
            "track_variant": "fpr_label",
            "label_policy": self.label_policy,
            "threshold_scope": threshold_scope,
            "fpr_budget": self.budget,
            "threshold": threshold,
            "calibration_fpr_realized": realized,
            "prediction_path": self._relative(prediction_path),
            "prediction_sha256": prediction_sha256,
            **label_metrics,
        }
        return [score_result, label_result]

    def _resume_source_state(
        self,
        *,
        spec: ExperimentModelSpec,
        source_farm: str,
    ) -> tuple[Dict[str, object], Path, Dict[str, object], float, float]:
        training = [
            row
            for row in self.training_rows
            if str(row.get("model_name")) == spec.model_name
            and str(row.get("source_farm")) == source_farm
        ]
        if len(training) != 1:
            raise ValueError("Resume requires exactly one matching training record.")
        training_row = training[0]
        expected = {
            "run_id": self.run_id,
            "model_path": spec.model_path,
            "model_params": _json_params(spec.effective_model_params),
            "transfer_adapter": spec.transfer_adapter,
            "transfer_adapter_params": _json_params(spec.transfer_adapter_params),
        }
        for field, value in expected.items():
            if str(training_row.get(field)) != str(value):
                raise ValueError(f"Resume training identity mismatch for {field}.")

        checkpoint = (self.run_dir / str(training_row["checkpoint_path"])).resolve()
        try:
            checkpoint.relative_to(self.run_dir.resolve())
        except ValueError as error:
            raise ValueError("Resume checkpoint path escapes the run directory.") from error
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint is missing: {checkpoint}")
        checkpoint_sha256 = str(training_row["checkpoint_sha256"])
        if sha256_file(checkpoint) != checkpoint_sha256:
            raise ValueError("Resume checkpoint checksum does not match training.csv.")
        if int(checkpoint.stat().st_size) != int(training_row["checkpoint_bytes"]):
            raise ValueError("Resume checkpoint size does not match training.csv.")

        validation = (
            self.run_dir / str(training_row["checkpoint_validation_path"])
        ).resolve()
        try:
            validation.relative_to(self.run_dir.resolve())
        except ValueError as error:
            raise ValueError(
                "Resume checkpoint validation path escapes the run directory."
            ) from error
        if (
            not validation.is_file()
            or sha256_file(validation)
            != str(training_row["checkpoint_validation_sha256"])
        ):
            raise ValueError("Resume checkpoint validation artifact is invalid.")

        thresholds = [
            row
            for row in self.threshold_rows
            if str(row.get("model_name")) == spec.model_name
            and str(row.get("source_farm")) == source_farm
            and str(row.get("protocol")) == "strict_zero_shot"
            and str(row.get("calibration_scope")) == "source_farm_train"
        ]
        if len(thresholds) != 1:
            raise ValueError("Resume requires exactly one source threshold record.")
        threshold_row = thresholds[0]
        if str(threshold_row.get("checkpoint_sha256")) != checkpoint_sha256:
            raise ValueError("Resume threshold was not produced by this checkpoint.")
        if not np.isclose(
            float(threshold_row["fpr_budget"]), self.budget, rtol=0.0, atol=1e-12
        ):
            raise ValueError("Resume threshold FPR budget does not match the protocol.")
        scores_path = (self.run_dir / str(threshold_row["scores_path"])).resolve()
        try:
            scores_path.relative_to(self.run_dir.resolve())
        except ValueError as error:
            raise ValueError("Resume score path escapes the run directory.") from error
        if (
            not scores_path.is_file()
            or sha256_file(scores_path) != str(threshold_row["scores_sha256"])
        ):
            raise ValueError("Resume source calibration scores are invalid.")

        checkpoint_meta = {
            "format": str(training_row["checkpoint_format"]),
            "sha256": checkpoint_sha256,
            "bytes": int(training_row["checkpoint_bytes"]),
            "validation_path": str(validation),
            "validation_sha256": str(training_row["checkpoint_validation_sha256"]),
        }
        return (
            training_row,
            checkpoint,
            checkpoint_meta,
            float(threshold_row["threshold"]),
            float(threshold_row["calibration_fpr_realized"]),
        )

    def _completed_event(
        self,
        *,
        spec: ExperimentModelSpec,
        source_farm: str,
        target_farm: str,
        file_name: str,
        protocol: str,
        checkpoint_sha256: str,
    ) -> bool:
        def matches(row: Mapping[str, object]) -> bool:
            return (
                str(row.get("model_name")) == spec.model_name
                and str(row.get("source_farm")) == source_farm
                and str(row.get("target_farm")) == target_farm
                and str(row.get("target_file_name")) == file_name
                and str(row.get("protocol")) == protocol
            )

        events = [row for row in self.event_rows if matches(row)]
        resources = [row for row in self.resource_rows if matches(row)]
        if not events and not resources:
            return False
        event_tracks = {str(row.get("track")) for row in events}
        if len(events) != len(self.tracks) or event_tracks != set(self.tracks):
            raise ValueError(f"Resume found incomplete event ledger rows for {file_name}.")
        if len(resources) != 1:
            raise ValueError(f"Resume found invalid resource ledger rows for {file_name}.")
        rows = [*events, resources[0]]
        paths = {str(row.get("prediction_path")) for row in rows}
        digests = {str(row.get("prediction_sha256")) for row in rows}
        checkpoint_digests = {
            str(row.get("checkpoint_sha256")) for row in events
        }
        if len(paths) != 1 or len(digests) != 1:
            raise ValueError(f"Resume prediction ledger disagrees for {file_name}.")
        if checkpoint_digests != {checkpoint_sha256}:
            raise ValueError(f"Resume checkpoint identity disagrees for {file_name}.")
        prediction = (self.run_dir / next(iter(paths))).resolve()
        try:
            prediction.relative_to(self.run_dir.resolve())
        except ValueError as error:
            raise ValueError("Resume prediction path escapes the run directory.") from error
        if not prediction.is_file() or sha256_file(prediction) != next(iter(digests)):
            raise ValueError(f"Resume prediction artifact is invalid for {file_name}.")
        return True

    def _fit_source_model(
        self,
        *,
        spec: ExperimentModelSpec,
        source_farm: str,
        records: pd.DataFrame,
        segments: Sequence[pd.DataFrame],
        imputer: MedianImputerState,
    ) -> None:
        adapter = build_transfer_adapter(
            spec.transfer_adapter, **dict(spec.transfer_adapter_params)
        )
        model_key = safe_name(spec.model_name)
        source_dir = self.run_dir / "models" / model_key / f"source_{source_farm}"
        source_status = source_dir / "status.json"
        atomic_json(
            {
                "status": "running",
                "track": "shared",
                "tracks": list(self.tracks),
                "model_name": spec.model_name,
                "source_farm": source_farm,
            },
            source_status,
        )
        checkpoint = source_dir / "checkpoint.pt"
        model: Any = None
        try:
            if self.resume:
                (
                    training_row,
                    checkpoint,
                    checkpoint_meta,
                    source_threshold,
                    source_realized,
                ) = self._resume_source_state(spec=spec, source_farm=source_farm)
                LOGGER.info(
                    "[%s source %s] resume | checkpoint and source threshold verified",
                    spec.model_name,
                    source_farm,
                )
            else:
                fix_random_seed(self.seed)
                model = spec.factory()
                self._validate_adapter_model(adapter, model)
                fit_label = f"[{spec.model_name} source {source_farm}] training"
                with _progress_heartbeat(fit_label):
                    _, resource = _measure(lambda: adapter.fit_source(model, segments))
                fit_metadata = dict(adapter.fit_metadata(model))
                n_params = _scalar_stat(model, "estimate_n_params")
                model_size_mb = _scalar_stat(model, "estimate_model_size_mb")
                with _progress_heartbeat(
                    f"[{spec.model_name} source {source_farm}] checkpoint validation"
                ):
                    checkpoint_meta = self._save_model(
                        model, adapter, segments, checkpoint
                    )
                if sha256_file(checkpoint) != checkpoint_meta["sha256"]:
                    raise RuntimeError("Checkpoint checksum changed after atomic commit.")
                model = None
                _release()

                training_row = {
                    "run_id": self.run_id,
                    "track": "shared",
                    "tracks": _json_params(self.tracks),
                    "seed": self.seed,
                    "model_name": spec.model_name,
                    "model_path": spec.model_path,
                    "model_params": _json_params(spec.effective_model_params),
                    "benchmark_adapter": spec.benchmark_adapter or "",
                    "transfer_adapter": spec.transfer_adapter,
                    "transfer_adapter_params": _json_params(
                        spec.transfer_adapter_params
                    ),
                    "transfer_fit_metadata": _json_params(fit_metadata),
                    "source_farm": source_farm,
                    "source_asset_count": int(records["asset_id"].nunique()),
                    "source_event_count": int(len(records)),
                    "source_train_rows": int(sum(map(len, segments))),
                    "source_imputer_fitted_rows": imputer.fitted_rows,
                    "imputation_scope": "source_farm_train",
                    "normalization_scope": "model_native_source_fit_only",
                    "fit_seconds_source": resource["seconds"],
                    "fit_peak_memory_mb": resource["peak_memory_mb"],
                    "fit_peak_gpu_memory_mb": resource["peak_gpu_memory_mb"],
                    "fit_cpu_usage_percent": resource["cpu_usage_percent"],
                    "estimated_n_params": n_params,
                    "estimated_model_size_mb": model_size_mb,
                    "checkpoint_path": self._relative(checkpoint),
                    "checkpoint_format": checkpoint_meta["format"],
                    "checkpoint_sha256": checkpoint_meta["sha256"],
                    "checkpoint_bytes": checkpoint_meta["bytes"],
                    "checkpoint_validation_path": self._relative(
                        Path(str(checkpoint_meta["validation_path"]))
                    ),
                    "checkpoint_validation_sha256": checkpoint_meta[
                        "validation_sha256"
                    ],
                }
                self.training_rows.append(training_row)
                self._write_ledger(self.training_rows, "training.csv")

                source_threshold, source_realized = self._source_threshold(
                    spec=spec,
                    source_farm=source_farm,
                    checkpoint=checkpoint,
                    checkpoint_meta=checkpoint_meta,
                    adapter=adapter,
                    segments=segments,
                    records=records,
                )

            for target_farm in ("A", "B", "C"):
                targets = target_records(self.manifest, target_farm, self.holdouts)
                target_total = len(targets)
                for target_position, (_, record) in enumerate(targets.iterrows(), start=1):
                    pending_protocols = [
                        protocol
                        for protocol in self.protocols
                        if not self.resume
                        or not self._completed_event(
                            spec=spec,
                            source_farm=source_farm,
                            target_farm=target_farm,
                            file_name=str(record["file_name"]),
                            protocol=protocol,
                            checkpoint_sha256=str(checkpoint_meta["sha256"]),
                        )
                    ]
                    if not pending_protocols:
                        LOGGER.info(
                            "[%s %s->%s] resume | skipped completed event %d/%d: %s",
                            spec.model_name,
                            source_farm,
                            target_farm,
                            target_position,
                            target_total,
                            record["file_name"],
                        )
                        continue
                    load_label = (
                        f"[{spec.model_name} {source_farm}->{target_farm}] "
                        f"loading target event {target_position}/{target_total}: "
                        f"{record['file_name']}"
                    )
                    with _progress_heartbeat(load_label):
                        event = self._target_event(record.to_dict())
                    for protocol in pending_protocols:
                        task_dir = (
                            self.run_dir
                            / "tasks"
                            / model_key
                            / f"source_{source_farm}"
                            / protocol
                            / f"{source_farm}_to_{target_farm}"
                            / safe_name(Path(str(record["file_name"])).stem)
                        )
                        status_path = task_dir / "status.json"
                        atomic_json(
                            {
                                "status": "running",
                                "tracks": list(self.tracks),
                                "model_name": spec.model_name,
                                "source_farm": source_farm,
                                "target_farm": target_farm,
                                "target_file_name": record["file_name"],
                                "protocol": protocol,
                            },
                            status_path,
                        )
                        try:
                            infer_label = (
                                f"[{spec.model_name} {source_farm}->{target_farm} "
                                f"{protocol}] inference {target_position}/{target_total}: "
                                f"{record['file_name']}"
                            )
                            with _progress_heartbeat(infer_label):
                                results = self._evaluate_event(
                                    spec=spec,
                                    source_farm=source_farm,
                                    target_farm=target_farm,
                                    source_records_frame=records,
                                    source_imputer=imputer,
                                    source_threshold=source_threshold,
                                    source_realized=source_realized,
                                    training_row=training_row,
                                    checkpoint=checkpoint,
                                    checkpoint_meta=checkpoint_meta,
                                    adapter=adapter,
                                    event=event,
                                    protocol=protocol,
                                )
                            self.event_rows.extend(results)
                            self._write_ledger(self.event_rows, "per_event.csv")
                            atomic_json(
                                {
                                    "status": "completed",
                                    "tracks": {
                                        result["track"]: {
                                            "prediction_path": result["prediction_path"],
                                            "prediction_sha256": result[
                                                "prediction_sha256"
                                            ],
                                        }
                                        for result in results
                                    },
                                },
                                status_path,
                            )
                            (task_dir / "failure.json").unlink(missing_ok=True)
                        except BaseException as error:
                            write_failure(
                                task_dir / "failure.json",
                                error,
                                phase="evaluate_event",
                                context={
                                    "model_name": spec.model_name,
                                    "source_farm": source_farm,
                                    "target_farm": target_farm,
                                    "target_file_name": record["file_name"],
                                    "protocol": protocol,
                                    "tracks": list(self.tracks),
                                },
                            )
                            atomic_json(
                                {
                                    "status": "failed",
                                    "model_name": spec.model_name,
                                    "source_farm": source_farm,
                                    "target_farm": target_farm,
                                    "target_file_name": record["file_name"],
                                    "protocol": protocol,
                                    "tracks": list(self.tracks),
                                    "error": str(error),
                                },
                                status_path,
                            )
                            raise
            atomic_json(
                {
                    "status": "completed",
                    "track": "shared",
                    "tracks": list(self.tracks),
                    "model_name": spec.model_name,
                    "source_farm": source_farm,
                    "checkpoint_sha256": checkpoint_meta["sha256"],
                },
                source_status,
            )
            (source_dir / "failure.json").unlink(missing_ok=True)
        except BaseException as error:
            write_failure(
                source_dir / "failure.json",
                error,
                phase="source_fit_or_transfer",
                context={"model_name": spec.model_name, "source_farm": source_farm},
            )
            atomic_json(
                {
                    "status": "failed",
                    "track": "shared",
                    "tracks": list(self.tracks),
                    "model_name": spec.model_name,
                    "source_farm": source_farm,
                    "error": str(error),
                },
                source_status,
            )
            raise
        finally:
            if model is not None:
                model = None
                _release()

    def run(self, models: Sequence[ExperimentModelSpec]) -> Dict[str, int]:
        if not models:
            raise ValueError("At least one model is required for execution.")
        model_paths = [_path_name(spec.model_name, maximum=64) for spec in models]
        if len({name.casefold() for name in model_paths}) != len(model_paths):
            raise ValueError("Model paths collide after case-insensitive sanitization.")
        self.write_plans()
        LOGGER.info(
            "run started | models=%d sources=%s protocols=%s tracks=%s",
            len(models),
            ",".join(self.source_farms),
            ",".join(self.protocols),
            ",".join(self.tracks),
        )
        for source_farm in self.source_farms:
            if self.resume:
                LOGGER.info("[source %s] loading saved source state", source_farm)
                records, segments, imputer = self._resume_source_data(source_farm)
            else:
                LOGGER.info("[source %s] preparing source training pool", source_farm)
                records, segments, imputer = self._source_data(source_farm)
            for spec in models:
                self._fit_source_model(
                    spec=spec,
                    source_farm=source_farm,
                    records=records,
                    segments=segments,
                    imputer=imputer,
                )
            del segments
            gc.collect()

        per_event = pd.DataFrame(self.event_rows)
        LOGGER.info("aggregating %d event-track rows", len(per_event))
        per_asset, per_direction = write_result_tables(self.run_dir, per_event)
        return {
            "source_fits": len(self.training_rows),
            "shared_inference_results": len(self.resource_rows),
            "event_results": len(per_event),
            "score_event_results": int(per_event["track"].eq("score").sum()),
            "label_event_results": int(per_event["track"].eq("label").sum()),
            "turbine_results": len(per_asset),
            "direction_results": len(per_direction),
        }


__all__ = ["CrossDomainEngine", "ExperimentModelSpec", "safe_name"]
