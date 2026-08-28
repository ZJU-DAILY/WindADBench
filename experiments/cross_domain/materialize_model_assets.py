
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.cross_domain.model_catalog import entries


ROOT = Path(__file__).resolve().parent


def _config(entry: dict) -> dict:
    model = {
        key: value
        for key, value in entry.items()
        if key not in {"category", "slug"} and not (key == "transfer_adapter_params" and not value)
    }
    return {"recommend_hyper_params": {"seed": 2026}, "models": [model]}


def _launcher(entry: dict) -> str:
    category = entry["category"]
    slug = entry["slug"]
    return f"""#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../../../../.." && pwd)"
PYTHON_BIN="${{PYTHON_BIN:-python}}"
SOURCE_FARM="${{1:-${{SOURCE_FARM:-}}}}"
case "$SOURCE_FARM" in
  A|B|C) ;;
  *) echo "Usage: $0 A|B|C (or set SOURCE_FARM)" >&2; exit 2 ;;
esac
RUN_ID="${{RUN_ID:-${{SOURCE_FARM}}}}"
OUTPUT_DIR="$ROOT_DIR/experiments/cross_domain/outputs/{category}/{slug}"
LOG_DIR="$ROOT_DIR/experiments/cross_domain/logs/{category}/{slug}"
LOG_PATH="$LOG_DIR/$RUN_ID.log"
RUN_MODE=(--overwrite)
TEE_MODE=()
if [[ "${{RESUME:-0}}" == "1" ]]; then
  RUN_MODE=(--resume)
  TEE_MODE=(-a)
fi
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$ROOT_DIR"

{{
  echo "[cross-domain launcher v7] model={slug} protocol=strict_zero_shot source=$SOURCE_FARM"
  echo "[cross-domain launcher v7] run_id=$RUN_ID mode=${{RUN_MODE[0]#--}}"
  echo "[cross-domain launcher v7] log=$LOG_PATH"
  echo "[cross-domain launcher v7] results=$OUTPUT_DIR/$RUN_ID"

  "$PYTHON_BIN" -u experiments/cross_domain/run_transfer.py \\
    --config experiments/cross_domain/configs/fixed_holdout.json \\
    --models-config experiments/cross_domain/configs/models/{category}/{slug}.json \\
    --output-dir "$OUTPUT_DIR" \\
    --source-farm "$SOURCE_FARM" \\
    --run-id "$RUN_ID" \\
    "${{RUN_MODE[@]}}"
}} 2>&1 | tee "${{TEE_MODE[@]}}" "$LOG_PATH"
"""


def main() -> None:
    catalog = entries()
    if len(catalog) != 36:
        raise RuntimeError(f"Expected 36 models, found {len(catalog)}.")
    paths = set()
    launchers = []
    for entry in catalog:
        key = (entry["category"], entry["slug"])
        if key in paths:
            raise RuntimeError(f"Duplicate model asset path: {key}.")
        paths.add(key)
        config_path = ROOT / "configs" / "models" / key[0] / f"{key[1]}.json"
        script_path = ROOT / "scripts" / "baselines" / key[0] / f"{key[1]}.sh"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(_config(entry), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        script_path.write_text(_launcher(entry), encoding="utf-8", newline="\n")
        launchers.append(script_path.relative_to(ROOT).as_posix())

    all_script = ROOT / "scripts" / "run_all.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"',
        'cd "$ROOT_DIR"',
        "",
    ]
    lines.extend(
        f'bash "experiments/cross_domain/{path}" {farm}'
        for path in launchers
        for farm in ("A", "B", "C")
    )
    all_script.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
