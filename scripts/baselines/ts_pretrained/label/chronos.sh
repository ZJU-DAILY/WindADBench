#!/usr/bin/env bash
# Baseline: Chronos (ts_pretrained) - track: label
set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         Chronos \
  --model-path         tsad_benchmark.baselines.ts_pretrained.chronos.ChronosModel \
  --model-hyper-params '{"model_id": "models/chrones-bolt-base", "local_files_only": true, "context_length": 96, "prediction_length": 1, "batch_size": 64, "forecast_batch_size": 2048, "device": "cuda"}' \
  --save-path          results/ts_pretrained/label/chronos.csv \
  --report-dir         results/ts_pretrained/label/chronos_report
