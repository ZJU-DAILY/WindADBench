#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         MemMambaAD \
  --model-path         tsad_benchmark.baselines.deep_learning.memmambaad.MemMambaADModel \
  --model-hyper-params '{"d_model": 128, "num_epochs": 5}' \
  --save-path          results/deep_learning/label/memmambaad.csv \
  --report-dir         results/deep_learning/label/memmambaad_report
