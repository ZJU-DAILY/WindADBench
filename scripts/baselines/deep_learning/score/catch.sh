#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         CATCH \
  --model-path         tsad_benchmark.baselines.deep_learning.catch.CATCHAnomalyModel \
  --model-hyper-params '{"win_size": 96, "patch_size": 16, "patch_stride": 16, "num_epochs": 3, "batch_size": 16, "d_model": 16, "d_ff": 64, "cf_dim": 32, "head_dim": 16, "n_heads": 2, "auxi_lambda": 0.1, "dc_lambda": 0.1}' \
  --save-path          results/deep_learning/score/catch.csv \
  --defer-score-vus \
  --report-dir         results/deep_learning/score/catch_report
