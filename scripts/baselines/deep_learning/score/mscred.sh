#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         MSCRED \
  --model-path         tsad_benchmark.baselines.deep_learning.mscred.MSCREDModel \
  --model-hyper-params '{"signature_scales": [10, 30, 60], "step_max": 5, "gap_time": 10, "feature_mode": "sensor_avg", "sensor_stats": ["avg"], "batch_size": 32, "num_epochs": 5, "lr": 0.0002}' \
  --save-path          results/deep_learning/score/mscred.csv \
  --defer-score-vus \
  --report-dir         results/deep_learning/score/mscred_report
