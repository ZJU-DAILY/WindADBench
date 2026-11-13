#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_label.json \
  --model-name         RPCL-TCNE-MTS-LLM \
  --model-path         tsad_benchmark.baselines.finetune_llm.rpcl_tcne_mts_llm.RPCLTCNEMTSLLMModel \
  --model-hyper-params '{"backbone": "models/gpt2", "local_files_only": true, "win_size": 22, "train_stride": 4, "batch_size": 512, "num_epochs": 6, "lr": 0.0004, "tcne_hidden_dim": 44, "tcne_blocks": 10, "kernel_size": 3, "dropout": 0.2, "lora_rank": 4}' \
  --save-path          results/finetune_llm/label/rpcl_tcne_mts_llm.csv \
  --report-dir         results/finetune_llm/label/rpcl_tcne_mts_llm_report
