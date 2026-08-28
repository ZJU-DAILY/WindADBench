#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         AutoEncoder \
  --model-path         tsad_benchmark.baselines.deep_learning.ae.AutoEncoderModel \
  --model-hyper-params '{"hidden_dim": 64, "num_epochs": 10}' \
  --save-path          results/deep_learning/score/ae.csv \
  --defer-score-vus \
  --report-dir         results/deep_learning/score/ae_report
