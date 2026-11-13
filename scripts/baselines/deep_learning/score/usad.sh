#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --defer-score-vus \
  --model-name         USAD \
  --model-path         tsad_benchmark.baselines.deep_learning.usad.USADModel \
  --model-hyper-params '{"win_size": 12, "hidden_dim": 64, "num_epochs": 10}' \
  --save-path          results/deep_learning/score/usad.csv \
  --report-dir         results/deep_learning/score/usad_report
