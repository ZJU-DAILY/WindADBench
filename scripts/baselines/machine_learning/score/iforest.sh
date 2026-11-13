#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --defer-score-vus \
  --model-name         IForest \
  --model-path         tsad_benchmark.baselines.machine_learning.iforest.IForestModel \
  --model-hyper-params '{"n_estimators": 200, "contamination": 0.05}' \
  --save-path          results/machine_learning/score/iforest.csv \
  --report-dir         results/machine_learning/score/iforest_report
