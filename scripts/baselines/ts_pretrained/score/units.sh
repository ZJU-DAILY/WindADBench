#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --defer-score-vus \
  --model-name         UniTS \
  --model-path         tsad_benchmark.baselines.ts_pretrained.units.UniTSModel \
  --model-hyper-params '{"checkpoint_path": "models/UniTS/units_x32_pretrain_checkpoint.pth", "local_files_only": true, "mode": "zero_shot", "win_size": 96, "device": "cuda"}' \
  --save-path          results/ts_pretrained/score/units.csv \
  --report-dir         results/ts_pretrained/score/units_report
