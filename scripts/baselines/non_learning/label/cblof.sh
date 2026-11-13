#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         CBLOF \
  --model-path         tsad_benchmark.baselines.non_learning.cblof.CBLOFModel \
  --model-hyper-params '{"n_clusters": 8, "contamination": 0.05}' \
  --save-path          results/non_learning/label/cblof.csv \
  --report-dir         results/non_learning/label/cblof_report
