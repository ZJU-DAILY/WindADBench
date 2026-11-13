#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --defer-score-vus \
  --model-name         KMeans \
  --model-path         tsad_benchmark.baselines.machine_learning.kmeans.KMeansModel \
  --model-hyper-params '{"n_clusters": 20, "window_size": 50}' \
  --save-path          results/machine_learning/score/kmeans.csv \
  --report-dir         results/machine_learning/score/kmeans_report
