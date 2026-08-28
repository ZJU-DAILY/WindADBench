#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_FARM="${1:-${SOURCE_FARM:-}}"
case "$SOURCE_FARM" in
  A|B|C) ;;
  *) echo "Usage: $0 A|B|C (or set SOURCE_FARM)" >&2; exit 2 ;;
esac
RUN_ID="${RUN_ID:-${SOURCE_FARM}}"
OUTPUT_DIR="$ROOT_DIR/experiments/cross_domain/outputs/machine_learning/eif"
LOG_DIR="$ROOT_DIR/experiments/cross_domain/logs/machine_learning/eif"
LOG_PATH="$LOG_DIR/$RUN_ID.log"
RUN_MODE=(--overwrite)
TEE_MODE=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RUN_MODE=(--resume)
  TEE_MODE=(-a)
fi
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$ROOT_DIR"

{
  echo "[cross-domain launcher v7] model=eif protocol=strict_zero_shot source=$SOURCE_FARM"
  echo "[cross-domain launcher v7] run_id=$RUN_ID mode=${RUN_MODE[0]#--}"
  echo "[cross-domain launcher v7] log=$LOG_PATH"
  echo "[cross-domain launcher v7] results=$OUTPUT_DIR/$RUN_ID"

  "$PYTHON_BIN" -u experiments/cross_domain/run_transfer.py \
    --config experiments/cross_domain/configs/fixed_holdout.json \
    --models-config experiments/cross_domain/configs/models/machine_learning/eif.json \
    --output-dir "$OUTPUT_DIR" \
    --source-farm "$SOURCE_FARM" \
    --run-id "$RUN_ID" \
    "${RUN_MODE[@]}"
} 2>&1 | tee "${TEE_MODE[@]}" "$LOG_PATH"
