#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         TimesNet \
  --model-path         tsad_benchmark.baselines.deep_learning.timesnet.TimesNetModel \
  --model-hyper-params '{"win_size": 100, "d_model": 64, "d_ff": 64, "e_layers": 2, "top_k": 5, "num_epochs": 3}' \
  --save-path          results/deep_learning/score/timesnet.csv \
  --defer-score-vus \
  --report-dir         results/deep_learning/score/timesnet_report
