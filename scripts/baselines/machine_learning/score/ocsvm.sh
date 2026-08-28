#!/usr/bin/env bash
# Baseline: OCSVM (machine_learning) — track: score
# Status: IMPLEMENTED — runs end-to-end out of the box.
set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         OCSVM \
  --model-path         tsad_benchmark.baselines.machine_learning.ocsvm.OCSVMModel \
  --model-hyper-params '{"kernel": "rbf", "nu": 0.05, "gamma": "auto", "contamination": 0.05}' \
  --save-path          results/machine_learning/score/ocsvm.csv \
  --defer-score-vus \
  --report-dir         results/machine_learning/score/ocsvm_report
