#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --defer-score-vus \
  --model-name         TimesNet \
  --model-path         tsad_benchmark.baselines.deep_learning.timesnet.TimesNetModel \
  --model-hyper-params '{"win_size": 100, "batch_size": 64, "d_model": 64, "d_ff": 64, "e_layers": 2, "top_k": 5, "num_epochs": 3, "score_protocol": "independent_window_v1", "score_batch_size": 1}' \
  --save-path          results/independent_window_v1/deep_learning/score/timesnet.csv \
  --report-dir         results/independent_window_v1/deep_learning/score/timesnet_report \
  "$@"
