#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         D3R \
  --model-path         tsad_benchmark.baselines.deep_learning.d3r.D3RModel \
  --model-hyper-params '{"win_size": 96, "num_epochs": 5}' \
  --save-path          results/deep_learning/score/d3r.csv \
  --defer-score-vus \
  --report-dir         results/deep_learning/score/d3r_report
