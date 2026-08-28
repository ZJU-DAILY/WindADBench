#!/usr/bin/env bash
# Baseline: HBOS (non_learning) — track: label
# Status: IMPLEMENTED — runs end-to-end out of the box.
set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         HBOS \
  --model-path         tsad_benchmark.baselines.non_learning.hbos.HBOSModel \
  --model-hyper-params '{"n_bins": 10, "contamination": 0.05}' \
  --save-path          results/non_learning/label/hbos.csv \
  --report-dir         results/non_learning/label/hbos_report
