"""Refresh the normalized analysis CSVs from the WindADBench workbook.

The workbook (``benchmark.xlsx``) is the source of truth.  This script is a
standalone re-implementation of the block parsers in
``makefig/generate_experiment_figures.py``: it carries the same block specs and
the same name normalization but drops the matplotlib / ``paper_style``
dependencies so the export can run wherever the workbook is available.

Usage::

    python refresh_from_xlsx.py --workbook path/to/benchmark.xlsx \\
                                --analysis path/to/analysis [--dry-run]

Outputs (written to ``--analysis``)::

    in_farm_metrics.csv        3 blocks  (IF-A / IF-B / IF-C)
    cross_turbine_metrics.csv  3 blocks  (CT-A / CT-B / CT-C)
    cross_farm_metrics.csv     6 blocks  (CF-A>B ... CF-C>A)
    cost_metrics.csv           3 blocks  (per-farm cost and capacity columns)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_MODELS = 36
TOL = 5e-4

FAMILY_MAP: Dict[str, str] = {
    "non-learning": "NL",
    "machine-learning": "ML",
    "deep-learning": "DL",
    "llm-based": "LLM",
    "ts pre-trained": "TSP",
    "domain llm": "DLLM",
}

# (sheet, header_row, start_row, config_value) — 0-indexed rows, as in the
# paper's figure script.
IN_FARM_SPECS: List[Tuple[int, int, str]] = [(4, 5, "A"), (42, 43, "B"), (80, 81, "C")]
CROSS_TURBINE_SPECS: List[Tuple[int, int, str]] = [
    (2, 3, "A->A"), (40, 41, "B->B"), (78, 79, "C->C"),
]
CROSS_FARM_SPECS: Dict[int, List[Tuple[int, int, str]]] = {
    3: [(1, 2, "A->B"), (40, 41, "A->C")],
    4: [(1, 2, "B->A"), (40, 41, "B->C")],
    5: [(2, 3, "C->A"), (41, 42, "C->B")],
}
COST_SPECS: List[Tuple[int, int, str]] = [(23, 24, "A"), (61, 62, "B"), (99, 100, "C")]


def normalize_name(value: object) -> str:
    """Lower-case, underscore-joined identifier with the workbook's typo fixes."""
    text = str(value).strip().lower()
    text = text.replace("affliation", "affiliation")
    text = text.replace("infei", "infer")
    text = re.sub(r"[\s\-/]+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return re.sub(r"_+", "_", text).strip("_")


def numericize(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def parse_metric_block(raw: pd.DataFrame, *, header_row: int, start_row: int,
                       config: str, config_column: str) -> pd.DataFrame:
    """Parse one 36-model metric block into a tidy frame."""
    metric_names = [normalize_name(x) for x in raw.iloc[header_row, 3:].tolist()]
    block = raw.iloc[start_row:start_row + N_MODELS, :3 + len(metric_names)].copy()
    block.columns = [config_column, "family", "model", *metric_names]
    block[config_column] = config
    block["family"] = block["family"].ffill().map(FAMILY_MAP)
    block["model"] = block["model"].map(normalize_name)
    block = numericize(block, metric_names)
    if len(block) != N_MODELS or block["model"].nunique() != N_MODELS:
        raise ValueError(f"Malformed {config_column} block {config!r}")
    if block["family"].isna().any():
        raise ValueError(f"Unmapped family label in block {config!r}")
    return block


def parse_cost_block(raw: pd.DataFrame, *, header_row: int, start_row: int,
                     farm: str) -> pd.DataFrame:
    cost_names = [normalize_name(x) for x in raw.iloc[header_row, 3:11].tolist()]
    block = raw.iloc[start_row:start_row + N_MODELS, :11].copy()
    block.columns = ["farm", "family", "model", *cost_names]
    block["farm"] = farm
    block["family"] = block["family"].ffill().map(FAMILY_MAP)
    block["model"] = block["model"].map(normalize_name)
    block = numericize(block, cost_names)
    if len(block) != N_MODELS or block["model"].nunique() != N_MODELS:
        raise ValueError(f"Malformed cost block {farm!r}")
    return block


def load_all(workbook: Path) -> Dict[str, pd.DataFrame]:
    """Return the four normalized tables keyed by output file stem."""
    in_farm = pd.concat(
        [parse_metric_block(pd.read_excel(workbook, sheet_name=0, header=None),
                            header_row=h, start_row=s, config=farm,
                            config_column="farm")
         for h, s, farm in IN_FARM_SPECS], ignore_index=True)
    cross_turbine = pd.concat(
        [parse_metric_block(pd.read_excel(workbook, sheet_name=2, header=None),
                            header_row=h, start_row=s, config=direction,
                            config_column="direction")
         for h, s, direction in CROSS_TURBINE_SPECS], ignore_index=True)
    cross_farm_frames: List[pd.DataFrame] = []
    for sheet_index, blocks in CROSS_FARM_SPECS.items():
        raw = pd.read_excel(workbook, sheet_name=sheet_index, header=None)
        for header, start, direction in blocks:
            cross_farm_frames.append(
                parse_metric_block(raw, header_row=header, start_row=start,
                                   config=direction, config_column="direction"))
    costs = pd.concat(
        [parse_cost_block(pd.read_excel(workbook, sheet_name=1, header=None),
                          header_row=h, start_row=s, farm=farm)
         for h, s, farm in COST_SPECS], ignore_index=True)
    return {
        "in_farm_metrics": in_farm,
        "cross_turbine_metrics": cross_turbine,
        "cross_farm_metrics": pd.concat(cross_farm_frames, ignore_index=True),
        "cost_metrics": costs,
    }


def diff_against(existing: Path, fresh: pd.DataFrame) -> Optional[Tuple[int, int]]:
    """Return (changed_cells, changed_rows) versus the CSV on disk, if present."""
    if not existing.is_file():
        return None
    old = pd.read_csv(existing)
    key = [c for c in ("farm", "direction") if c in old.columns] + ["model"]
    numeric = [c for c in old.columns if c not in (*key, "family")]
    merged = old.merge(fresh, on=key, suffixes=("_old", "_new"), how="outer")
    if len(merged) != len(old):
        raise ValueError(f"{existing.name}: key sets differ between old and new")
    cells, rows = 0, set()
    for column in numeric:
        a, b = merged[f"{column}_old"], merged[f"{column}_new"]
        changed = ((a.isna() != b.isna()) | ((a - b).abs() > TOL)).fillna(False)
        cells += int(changed.sum())
        rows.update(merged.loc[changed, key].apply(tuple, axis=1))
    return cells, len(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the diff without writing any CSV")
    args = parser.parse_args(argv)

    if not args.workbook.is_file():
        logger.error("Workbook not found: %s", args.workbook)
        return 1
    args.analysis.mkdir(parents=True, exist_ok=True)

    tables = load_all(args.workbook)
    for stem, frame in tables.items():
        target = args.analysis / f"{stem}.csv"
        delta = diff_against(target, frame)
        if delta is None:
            logger.info("%-26s new file (%d rows)", target.name, len(frame))
        elif delta[0] == 0:
            logger.info("%-26s unchanged", target.name)
        else:
            logger.info("%-26s %d cells changed across %d rows",
                        target.name, delta[0], delta[1])
        if not args.dry_run:
            frame.to_csv(target, index=False)
    if args.dry_run:
        logger.info("dry run — nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
