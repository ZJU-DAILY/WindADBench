#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         AnomalyTransformer \
  --model-path         tsad_benchmark.baselines.deep_learning.anomaly_transformer.AnomalyTransformerModel \
  --model-hyper-params '{"win_size": 100, "lr": 0.0001, "num_epochs": 3}' \
  --save-path          results/deep_learning/label/anomaly_transformer.csv \
  --report-dir         results/deep_learning/label/anomaly_transformer_report
