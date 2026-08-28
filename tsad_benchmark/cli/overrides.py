# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _parse_optional_json(raw: Optional[str], *, label: str) -> Optional[Any]:
    """Parse a JSON CLI value, allowing literal ``None`` / ``"None"`` / empty."""
    if raw is None:
        return None
    s = raw.strip()
    if s in ("", "None", "null"):
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for {label}: {raw!r} ({e})") from e


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *overlay* into *base* (overlay wins on conflict)."""
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _pad_to_length(lst: Optional[List[Any]], n: int, fill: Any = None) -> List[Any]:
    """Right-pad *lst* with *fill* up to length *n*; ``None`` becomes empty list."""
    out = list(lst) if lst else []
    if len(out) < n:
        out.extend([fill] * (n - len(out)))
    return out[:n]


# ---------------------------------------------------------------------------
# Section-specific merges
# ---------------------------------------------------------------------------


def apply_data_overrides(
    args: argparse.Namespace,
    data_section: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply ``--dataset-root / --limit / --rebuild-meta`` to data_config."""
    out = copy.deepcopy(data_section)
    if args.dataset_root is not None:
        out["dataset_root"] = args.dataset_root
    if args.series_limit is not None:
        out["series_limit"] = args.series_limit
    if args.rebuild_meta:
        out["rebuild_meta"] = True
    return out


def apply_model_overrides(
    args: argparse.Namespace,
    model_section: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the ``models`` list from the per-flag CLI vectors.

    Required vectors: ``--model-name``, ``--model-path`` (must be the same
    length).  Optional vectors are right-padded with ``None``.
    """
    names: List[str] = list(args.model_name)
    paths: List[str] = list(args.model_path)
    if len(names) != len(paths):
        raise ValueError(
            f"--model-name ({len(names)}) and --model-path ({len(paths)}) "
            "must be the same length."
        )
    n = len(names)

    raw_hp = _pad_to_length(args.model_hyper_params, n, fill=None)
    raw_adapter = _pad_to_length(args.adapter, n, fill=None)
    raw_expected = _pad_to_length(args.expected_output, n, fill=None)

    out = copy.deepcopy(model_section)
    out.setdefault("models", [])

    for i, (name, path) in enumerate(zip(names, paths)):
        entry: Dict[str, Any] = {
            "model_name": name,
            "model_path": path,
        }

        hp = _parse_optional_json(raw_hp[i], label=f"--model-hyper-params[{i}]")
        if hp is not None:
            if not isinstance(hp, dict):
                raise ValueError(
                    f"--model-hyper-params[{i}] must be a JSON object, got {type(hp).__name__}."
                )
            entry["model_hyper_params"] = hp

        adapter = raw_adapter[i]
        if adapter is not None and adapter not in ("", "None", "null"):
            entry["adapter"] = adapter

        expected = raw_expected[i]
        if expected is not None and expected not in ("", "None", "null"):
            entry["expected_output"] = expected

        out["models"].append(entry)

    return out


def apply_evaluation_overrides(
    args: argparse.Namespace,
    evaluation_section: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply ``--strategy-args`` (deep-merged) and ``--metrics`` to evaluation_config."""
    out = copy.deepcopy(evaluation_section)
    out.setdefault("strategy_args", {})

    overlay = _parse_optional_json(args.strategy_args, label="--strategy-args") or {}
    if not isinstance(overlay, dict):
        raise ValueError("--strategy-args must decode to a JSON object.")
    out["strategy_args"] = _deep_merge(out["strategy_args"], overlay)

    if "strategy_name" not in out["strategy_args"]:
        raise ValueError(
            "evaluation_config.strategy_args.strategy_name is missing — "
            "set it in the JSON template or via --strategy-args."
        )

    if args.metrics is not None:
        if len(args.metrics) == 1 and args.metrics[0].lower() == "all":
            out["metrics"] = "all"
        else:
            out["metrics"] = list(args.metrics)

    if getattr(args, "defer_score_vus", False):
        out["defer_score_vus"] = True

    return out


def apply_report_overrides(
    args: argparse.Namespace,
    report_section: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply ``--no-report / --report-dir`` to report_config."""
    out = copy.deepcopy(report_section)
    if args.no_report:
        out["generate_report"] = False
    if args.report_dir is not None:
        out["report_dir"] = args.report_dir
    return out
