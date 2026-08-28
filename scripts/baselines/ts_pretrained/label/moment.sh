#!/usr/bin/env bash
# Baseline: MOMENT (ts_pretrained) - track: label
set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         MOMENT \
  --model-path         tsad_benchmark.baselines.ts_pretrained.moment.MOMENTModel \
  --model-hyper-params '{"model_id": "models/MOMENT-1-large", "local_files_only": true, "win_size": 96, "batch_size": 8, "fine_tune_epochs": 3, "fine_tune_batch_size": 8, "fine_tune_lr": 0.0001, "fine_tune_step": 1, "fine_tune_sample_rate": 0.05, "fine_tune_val_ratio": 0.2, "fine_tune_val_sample_rate": 0.05, "fine_tune_patience": 3, "train_score_sample_rate": 0.05, "device": "cuda"}' \
  --save-path          results/ts_pretrained/label/moment.csv \
  --report-dir         results/ts_pretrained/label/moment_report
