#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         SARAD \
  --model-path         tsad_benchmark.baselines.deep_learning.sarad.SARADModel \
  --model-hyper-params '{"win_size": 100, "batch_size": 8, "train_stride": 8, "num_epochs": 3, "model_size": 256, "num_layers": 2, "num_heads": 4, "detector_size": 64}' \
  --save-path          results/deep_learning/label/sarad.csv \
  --report-dir         results/deep_learning/label/sarad_report
