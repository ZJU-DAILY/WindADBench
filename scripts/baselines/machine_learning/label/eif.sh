#!/usr/bin/env bash
# Baseline: EIF (machine_learning) -- track: label
set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         EIF \
  --model-path         tsad_benchmark.baselines.machine_learning.eif.EIFModel \
  --model-hyper-params '{"n_estimators": 200, "sample_size": 256}' \
  --save-path          results/machine_learning/label/eif.csv \
  --report-dir         results/machine_learning/label/eif_report
