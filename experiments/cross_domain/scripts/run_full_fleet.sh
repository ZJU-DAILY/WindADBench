#!/usr/bin/env bash

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
script="$root/experiments/cross_domain/scripts/run_full_fleet.sh"
cross="$root/experiments/cross_domain"
model="${1:-}"
if [[ ! "$model" =~ ^[a-z0-9_]+/[a-z0-9_]+$ || ! -f "$cross/configs/models/$model.json" ]]; then
  echo "Usage: bash $script CATEGORY/MODEL [--resume] [--foreground]" >&2
  exit 2
fi
shift
resume=0
foreground=0
for option in "$@"; do
  case "$option" in
    --resume) resume=1 ;;
    --foreground) foreground=1 ;;
    *) echo "Unknown option: $option" >&2; exit 2 ;;
  esac
done

if [[ "$model" == "deep_learning/gdn" ]]; then

  export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi

if [[ "$model" == "machine_learning/kmeans" ]]; then

  export OMP_NUM_THREADS=1
fi

output_dir="$cross/outputs_full_fleet/$model"
log_dir="$cross/logs_full_fleet/$model"
session="fullfleet_${model##*/}"
python_bin="$(command -v "${PYTHON_BIN:-python}")"
workers="${VUS_WORKERS:-1}"
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo "VUS_WORKERS must be a positive integer." >&2; exit 2; }
for farm in A B C; do
  if [[ -e "$output_dir/$farm" && "$resume" == 0 ]]; then
    echo "$output_dir/$farm already exists. Use --resume to continue this model." >&2
    exit 1
  fi
done

if [[ "$foreground" == 0 ]]; then

  unset_args=()
  set_args=()
  for key in CUDA_VISIBLE_DEVICES LD_LIBRARY_PATH PYTHONPATH CONDA_PREFIX HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; do
    if [[ -v "$key" ]]; then set_args+=("$key=${!key}"); else unset_args+=(-u "$key"); fi
  done
  if [[ "$model" == "deep_learning/gdn" ]]; then
    set_args+=("CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG")
  fi
  if [[ "$model" == "machine_learning/kmeans" ]]; then
    set_args+=("OMP_NUM_THREADS=$OMP_NUM_THREADS")
  fi
  worker=(env "${unset_args[@]}" "PATH=$PATH" "PYTHON_BIN=$python_bin" "VUS_WORKERS=$workers"
          "${set_args[@]}" bash "$script" "$model" --foreground)
  if [[ "$resume" == 1 ]]; then worker+=(--resume); fi
  printf -v command '%q ' "${worker[@]}"
  tmux new-session -d -s "$session" -c "$root" "$command"
  echo "Started $session. Attach with: tmux attach -t $session"
  echo "Logs: $log_dir/{A,B,C}.log"
  exit 0
fi

cd "$root"
mkdir -p "$log_dir"
for farm in A B C; do
  mode=()
  if [[ "$resume" == 1 && -d "$output_dir/$farm" ]]; then mode+=(--resume); fi
  echo "[$model] source=$farm targets=A,B,C log=$log_dir/$farm.log"
  "$python_bin" -u "$cross/run_transfer.py" \
    --config "$cross/configs/full_fleet_cross_farm.json" \
    --models-config "$cross/configs/models/$model.json" \
    --output-dir "$output_dir" --source-farm "$farm" --run-id "$farm" \
    "${mode[@]}" 2>&1 | tee -a "$log_dir/$farm.log"
  "$python_bin" -u "$cross/recompute_score_vus.py" \
    --run-dir "$output_dir/$farm" --workers "$workers" --exact-fast \
    2>&1 | tee -a "$log_dir/$farm.log"
done

"$python_bin" -u -m experiments.cross_domain.collect_results \
  --run-dirs "$output_dir/A" "$output_dir/B" "$output_dir/C" \
  --output-dir "$output_dir/summary"
