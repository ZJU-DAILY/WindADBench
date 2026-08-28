#!/usr/bin/env bash
# Baseline: PCA (machine_learning) — track: label
# Status: IMPLEMENTED — runs end-to-end out of the box.
set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         PCA \
  --model-path         tsad_benchmark.baselines.machine_learning.pca.PCAModel \
  --model-hyper-params '{"weighted": true, "standardization": true, "contamination": 0.05}' \
  --save-path          results/machine_learning/label/pca.csv \
  --report-dir         results/machine_learning/label/pca_report
