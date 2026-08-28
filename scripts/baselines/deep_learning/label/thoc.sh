#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         THOC \
  --model-path         tsad_benchmark.baselines.deep_learning.thoc.THOCModel \
  --model-hyper-params '{"win_size": 100, "num_epochs": 5}' \
  --save-path          results/deep_learning/label/thoc.csv \
  --report-dir         results/deep_learning/label/thoc_report
