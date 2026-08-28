#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         OmniAnomaly \
  --model-path         tsad_benchmark.baselines.deep_learning.omnianomaly.OmniAnomalyModel \
  --model-hyper-params '{"win_size": 100, "latent_dim": 8, "num_epochs": 10}' \
  --save-path          results/deep_learning/label/omnianomaly.csv \
  --report-dir         results/deep_learning/label/omnianomaly_report
