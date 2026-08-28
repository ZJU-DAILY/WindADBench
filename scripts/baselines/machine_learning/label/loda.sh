#!/usr/bin/env bash
# Baseline: LODA (machine_learning) — track: label
# Status: IMPLEMENTED — runs end-to-end out of the box.
set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         LODA \
  --model-path         tsad_benchmark.baselines.machine_learning.loda.LODAModel \
  --model-hyper-params '{"n_bins": 10, "n_random_cuts": 100, "contamination": 0.05}' \
  --save-path          results/machine_learning/label/loda.csv \
  --report-dir         results/machine_learning/label/loda_report
