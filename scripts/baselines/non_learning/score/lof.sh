#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --defer-score-vus \
  --model-name         LOF \
  --model-path         tsad_benchmark.baselines.non_learning.lof.LOFModel \
  --model-hyper-params '{"n_neighbors": 20, "contamination": 0.05}' \
  --save-path          results/non_learning/score/lof.csv \
  --report-dir         results/non_learning/score/lof_report
