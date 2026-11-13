#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --defer-score-vus \
  --model-name         Torsk \
  --model-path         tsad_benchmark.baselines.machine_learning.torsk.TorskModel \
  --model-hyper-params '{"reservoir_dim": 256, "spectral_radius": 0.9, "leak_rate": 0.3, "ridge_lambda": 1e-4, "transient": 200}' \
  --save-path          results/machine_learning/score/torsk.csv \
  --report-dir         results/machine_learning/score/torsk_report
