#!/usr/bin/env bash
# Baseline: LOF (non_learning) — track: score
# Status: IMPLEMENTED — runs end-to-end out of the box.
set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         LOF \
  --model-path         tsad_benchmark.baselines.non_learning.lof.LOFModel \
  --model-hyper-params '{"n_neighbors": 20, "contamination": 0.05}' \
  --save-path          results/non_learning/score/lof.csv \
  --defer-score-vus \
  --report-dir         results/non_learning/score/lof_report
