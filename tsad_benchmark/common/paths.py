# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path

#: Absolute path of the package itself (``.../tsad_benchmark``).
PACKAGE_DIR: Path = Path(__file__).resolve().parent.parent

#: Repository root (``.../dataset``), where ``config/``, ``scripts/`` and
#: ``results/`` sit alongside the package.
ROOT_DIR: Path = PACKAGE_DIR.parent

#: Directory holding the JSON strategy templates (``unfixed_label.json`` etc.).
CONFIG_DIR: Path = ROOT_DIR / "config"

#: Directory holding the launch scripts (``scripts/run_benchmark.py`` and the
#: per-baseline ``.sh`` files).
SCRIPTS_DIR: Path = ROOT_DIR / "scripts"

#: Default destination for raw evaluation CSVs and generated reports.
RESULTS_DIR: Path = ROOT_DIR / "results"

#: Default location of the wind-farm metadata index.
DEFAULT_META_CSV: Path = ROOT_DIR / "WIND_AD_META.csv"


def resolve_config_path(name_or_path: str) -> Path:
    """
    Resolve a config-template reference to an absolute path.

    The CLI accepts either a bare template name (``"unfixed_label.json"``,
    looked up under :data:`CONFIG_DIR`) or an explicit relative/absolute
    path.  Both forms collapse here to a single :class:`Path`.
    """
    p = Path(name_or_path)
    if p.is_absolute() and p.exists():
        return p
    candidate = CONFIG_DIR / name_or_path
    if candidate.exists():
        return candidate
    if p.exists():
        return p.resolve()
    raise FileNotFoundError(
        f"Config template not found: {name_or_path!r}. "
        f"Looked under {CONFIG_DIR} and {Path.cwd()}."
    )


def ensure_dir(path: os.PathLike) -> Path:
    """Create *path* (and parents) if missing; return as :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
