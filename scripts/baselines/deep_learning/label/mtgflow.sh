#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         MTGFlow \
  --model-path         tsad_benchmark.baselines.deep_learning.mtgflow.MTGFlowModel \
  --model-hyper-params '{"win_size": 60, "batch_size": 64, "num_epochs": 40, "train_stride": 10}' \
  --save-path          results/deep_learning/label/mtgflow.csv \
  --report-dir         results/deep_learning/label/mtgflow_report
