#!/usr/bin/env bash
# Baseline: DADA (ts_pretrained) - track: label
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         DADA \
  --model-path         tsad_benchmark.baselines.ts_pretrained.dada.DADAModel \
  --model-hyper-params '{"model_id": "models/DADA", "local_files_only": true, "mode": "zero_shot", "win_size": 100, "batch_size": 16, "norm": false, "score_mode": "mse", "copies": 10, "device": "cuda"}' \
  --save-path          results/ts_pretrained/label/dada.csv \
  --report-dir         results/ts_pretrained/label/dada_report
