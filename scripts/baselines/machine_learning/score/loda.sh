#!/usr/bin/env bash
# Baseline: LODA (machine_learning) — track: score
# Status: IMPLEMENTED — runs end-to-end out of the box.
set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         LODA \
  --model-path         tsad_benchmark.baselines.machine_learning.loda.LODAModel \
  --model-hyper-params '{"n_bins": 10, "n_random_cuts": 100, "contamination": 0.05}' \
  --save-path          results/machine_learning/score/loda.csv \
  --defer-score-vus \
  --report-dir         results/machine_learning/score/loda_report
