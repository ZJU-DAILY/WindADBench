#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         USAD \
  --model-path         tsad_benchmark.baselines.deep_learning.usad.USADModel \
  --model-hyper-params '{"win_size": 48, "hidden_dim": 64, "num_epochs": 30}' \
  --save-path          results/deep_learning/label/usad.csv \
  --report-dir         results/deep_learning/label/usad_report
