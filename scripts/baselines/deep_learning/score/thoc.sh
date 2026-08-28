#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         THOC \
  --model-path         tsad_benchmark.baselines.deep_learning.thoc.THOCModel \
  --model-hyper-params '{"win_size": 100, "num_epochs": 5}' \
  --save-path          results/deep_learning/score/thoc.csv \
  --defer-score-vus \
  --report-dir         results/deep_learning/score/thoc_report
