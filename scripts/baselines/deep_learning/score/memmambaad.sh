#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         MemMambaAD \
  --model-path         tsad_benchmark.baselines.deep_learning.memmambaad.MemMambaADModel \
  --model-hyper-params '{"d_model": 128, "num_epochs": 5}' \
  --save-path          results/deep_learning/score/memmambaad.csv \
  --defer-score-vus \
  --report-dir         results/deep_learning/score/memmambaad_report
