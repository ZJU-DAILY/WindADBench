#!/usr/bin/env bash
# Baseline: CBLOF (non_learning) — track: score
# Status: IMPLEMENTED — runs end-to-end out of the box.
set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         CBLOF \
  --model-path         tsad_benchmark.baselines.non_learning.cblof.CBLOFModel \
  --model-hyper-params '{"n_clusters": 8, "contamination": 0.05}' \
  --save-path          results/non_learning/score/cblof.csv \
  --defer-score-vus \
  --report-dir         results/non_learning/score/cblof_report
