#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         GPT4TS \
  --model-path         tsad_benchmark.baselines.llm_based.gpt4ts.GPT4TSModel \
  --model-hyper-params '{"backbone": "models/gpt2", "local_files_only": true, "win_size": 100, "num_epochs": 10, "batch_size": 64, "sampling_rate": 0.05, "sampling_strategy": "uniform", "gpt_layers": 3, "d_ff": 32, "lradj": "type1", "device": "cuda", "max_features": 768, "feature_select": "train_variance"}' \
  --save-path          results/llm_based/score/gpt4ts.csv \
  --defer-score-vus \
  --report-dir         results/llm_based/score/gpt4ts_report
