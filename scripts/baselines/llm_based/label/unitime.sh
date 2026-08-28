#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         UniTime \
  --model-path         tsad_benchmark.baselines.llm_based.unitime.UniTimeModel \
  --model-hyper-params '{"model_path": "models/gpt2", "local_files_only": true, "win_size": 96, "num_epochs": 10, "patience": 10, "batch_size": 16, "sampling_rate": 0.05, "val_sample_rate": 0.05, "train_score_sample_rate": 0.05, "sampling_strategy": "uniform", "stride": 16, "max_token_num": 17, "max_backcast_len": 96, "max_forecast_len": 0, "device": "cuda"}' \
  --save-path          results/llm_based/label/unitime.csv \
  --report-dir         results/llm_based/label/unitime_report
