#!/usr/bin/env bash
# Baseline: Torsk (machine_learning) -- track: label
set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         Torsk \
  --model-path         tsad_benchmark.baselines.machine_learning.torsk.TorskModel \
  --model-hyper-params '{"reservoir_dim": 256, "spectral_radius": 0.9, "leak_rate": 0.3, "ridge_lambda": 1e-4, "transient": 200}' \
  --save-path          results/machine_learning/label/torsk.csv \
  --report-dir         results/machine_learning/label/torsk_report
