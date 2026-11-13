#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python - <<'PY'
import os, sys
if os.environ.get("TSAD_ALLOW_CPU", "").strip() in ("1", "true", "True"):
    print("[preflight] TSAD_ALLOW_CPU=1 — skipping CUDA check")
    raise SystemExit(0)
try:
    import torch
except Exception as e:
    print(f"[preflight] ERROR: cannot import torch: {e}", file=sys.stderr)
    raise SystemExit(2)
print(f"[preflight] torch={torch.__version__} cuda.is_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("[preflight] ERROR: CUDA not available. Reinstall CUDA torch or set TSAD_ALLOW_CPU=1.", file=sys.stderr)
    raise SystemExit(3)
print(f"[preflight] GPU: {torch.cuda.get_device_name(0)}")
PY

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         DeepPoint \
  --model-path         tsad_benchmark.baselines.machine_learning.deeppoint.DeepPointModel \
  --model-hyper-params '{"enable_threshold": true, "score_protocol": "source_frozen_v1"}' \
  --save-path          results/source_frozen_v1/deep_learning/label/deeppoint.csv \
  --report-dir         results/source_frozen_v1/deep_learning/label/deeppoint_report \
  "$@"
