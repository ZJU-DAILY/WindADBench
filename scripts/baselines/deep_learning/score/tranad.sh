#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         TranAD \
  --model-path         tsad_benchmark.baselines.deep_learning.tranad.TranADModel \
  --model-hyper-params '{"win_size": 10, "lr": 0.001, "num_epochs": 5}' \
  --save-path          results/deep_learning/score/tranad.csv \
  --defer-score-vus \
  --report-dir         results/deep_learning/score/tranad_report
