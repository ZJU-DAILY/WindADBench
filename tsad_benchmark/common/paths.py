from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR: Path = Path(__file__).resolve().parent.parent


ROOT_DIR: Path = PACKAGE_DIR.parent


CONFIG_DIR: Path = ROOT_DIR / "config"


SCRIPTS_DIR: Path = ROOT_DIR / "scripts"


RESULTS_DIR: Path = ROOT_DIR / "results"


DEFAULT_META_CSV: Path = ROOT_DIR / "WIND_AD_META.csv"


def resolve_config_path(name_or_path: str) -> Path:
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
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
