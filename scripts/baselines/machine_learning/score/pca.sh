#!/usr/bin/env bash
# Baseline: PCA (machine_learning) — track: score
# Status: IMPLEMENTED — runs end-to-end out of the box.
set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         PCA \
  --model-path         tsad_benchmark.baselines.machine_learning.pca.PCAModel \
  --model-hyper-params '{"weighted": true, "standardization": true, "contamination": 0.05}' \
  --save-path          results/machine_learning/score/pca.csv \
  --defer-score-vus \
  --report-dir         results/machine_learning/score/pca_report
