#!/usr/bin/env bash
# Back-compat shim (old path): domain_llm -> finetune_llm
set -euo pipefail
cd "$(dirname "$0")/../../../.."
bash scripts/baselines/finetune_llm/label/rpcl_tcne_mts_llm.sh
