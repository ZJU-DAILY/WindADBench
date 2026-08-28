#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         LSTMED \
  --model-path         tsad_benchmark.baselines.deep_learning.lstmed.LSTMEDModel \
  --model-hyper-params '{"hidden_dim": 64, "num_epochs": 10, "batch_size": 64}' \
  --save-path          results/deep_learning/score/lstmed.csv \
  --defer-score-vus \
  --report-dir         results/deep_learning/score/lstmed_report
