#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         GDN \
  --model-path         tsad_benchmark.baselines.deep_learning.gdn.GDNModel \
  --model-hyper-params '{"win_size": 48, "slide_stride": 6, "num_epochs": 30, "patience": 5, "topk": 15, "eval_topk": 3, "dim": 64, "batch_size": 64, "out_layer_num": 1, "out_layer_inter_dim": 256, "val_ratio": 0.1, "decay": 0.0, "label_rule": "percentile"}' \
  --save-path          results/deep_learning/label/gdn.csv \
  --report-dir         results/deep_learning/label/gdn_report
