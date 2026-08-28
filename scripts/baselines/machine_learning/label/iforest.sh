#!/usr/bin/env bash
# Baseline: IForest (machine_learning) — track: label
# Status: IMPLEMENTED — runs end-to-end out of the box.
set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         IForest \
  --model-path         tsad_benchmark.baselines.machine_learning.iforest.IForestModel \
  --model-hyper-params '{"n_estimators": 200, "contamination": 0.05}' \
  --save-path          results/machine_learning/label/iforest.csv \
  --report-dir         results/machine_learning/label/iforest_report
