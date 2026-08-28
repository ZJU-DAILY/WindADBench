#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         DUET \
  --model-path         tsad_benchmark.baselines.deep_learning.duet.DUETAnomalyModel \
  --model-hyper-params '{"win_size": 96, "num_epochs": 3, "num_experts": 4, "k": 1, "CI": true, "batch_size": 64}' \
  --save-path          results/deep_learning/score/duet.csv \
  --defer-score-vus \
  --report-dir         results/deep_learning/score/duet_report
