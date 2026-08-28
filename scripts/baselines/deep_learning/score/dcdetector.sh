#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         DCdetector \
  --model-path         tsad_benchmark.baselines.deep_learning.dcdetector.DCdetectorModel \
  --model-hyper-params '{"win_size": 105, "patch_size": [3, 5, 7], "num_epochs": 3, "batch_size": 16}' \
  --save-path          results/deep_learning/score/dcdetector.csv \
  --defer-score-vus \
  --report-dir         results/deep_learning/score/dcdetector_report
