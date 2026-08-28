#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         LSTMED \
  --model-path         tsad_benchmark.baselines.deep_learning.lstmed.LSTMEDModel \
  --model-hyper-params '{"hidden_dim": 64, "num_epochs": 10, "batch_size": 64}' \
  --save-path          results/deep_learning/label/lstmed.csv \
  --report-dir         results/deep_learning/label/lstmed_report
