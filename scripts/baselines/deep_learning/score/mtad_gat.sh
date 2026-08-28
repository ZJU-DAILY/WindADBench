#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         MTAD-GAT \
  --model-path         tsad_benchmark.baselines.deep_learning.mtad_gat.MTADGATModel \
  --model-hyper-params '{"win_size": 100, "batch_size": 64, "num_epochs": 5, "max_features": 256}' \
  --save-path          results/deep_learning/score/mtad_gat.csv \
  --defer-score-vus \
  --report-dir         results/deep_learning/score/mtad_gat_report
