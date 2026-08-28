#!/usr/bin/env bash
# Recompute deferred VUS metrics for score-track result CSVs.
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS="${1:-8}"

python scripts/recompute_metrics.py \
  --all \
  --track score \
  --metrics vus_pr vus_roc \
  --workers "${WORKERS}" \
  --report
