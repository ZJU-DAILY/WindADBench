#!/usr/bin/env bash
# Baseline: KNN (machine_learning) — track: label
# Status: IMPLEMENTED — runs end-to-end out of the box.
set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         KNN \
  --model-path         tsad_benchmark.baselines.machine_learning.knn.KNNModel \
  --model-hyper-params '{"n_neighbors": 5, "method": "largest", "contamination": 0.05}' \
  --save-path          results/machine_learning/label/knn.csv \
  --report-dir         results/machine_learning/label/knn_report
