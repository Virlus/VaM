#!/usr/bin/env bash
set -euo pipefail

# Split trainer launcher for the real async RECAP split closed loop.
# Run one copy with ROLE=action and one copy with ROLE=value on two shared-CPFS servers.
# SPLIT_ROUND_STEPS defines the policy round boundary; NUM_TRAIN_STEPS caps the total training budget. The default
# live test publishes at steps 200, 400, 600, and 800; blocking on the rollout/ingest cycle does not consume the
# budget.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${ROLE:?Set ROLE to action or value.}"
if [[ "${ROLE}" != "action" && "${ROLE}" != "value" ]]; then
  echo "ROLE must be 'action' or 'value', got '${ROLE}'" >&2
  exit 2
fi

CONFIG_NAME="pi05_aloha_async_recap_split_${ROLE}"
: "${RUN_NAME:?Set RUN_NAME for this async run.}"
: "${SPLIT_ROOT:?Set SPLIT_ROOT for checkpoints, datasets, and logs.}"
: "${CLOUD_ROOT:?Set CLOUD_ROOT for this async run control-plane registry.}"
: "${HF_HOME:?Set HF_HOME for Hugging Face caches.}"
: "${SPLIT_ROUND_STEPS:?Set SPLIT_ROUND_STEPS for the policy boundary.}"
: "${NUM_TRAIN_STEPS:?Set NUM_TRAIN_STEPS for the total training budget.}"
: "${ALOHA_ASYNC_BASE_PARAMS:?Set ALOHA_ASYNC_BASE_PARAMS to the initial full checkpoint params path.}"
: "${ALOHA_ASYNC_NORM_ASSETS_DIR:?Set ALOHA_ASYNC_NORM_ASSETS_DIR to the normalization asset root.}"
DATA_HF_LEROBOT_HOME="${DATA_HF_LEROBOT_HOME:-${SPLIT_ROOT}/lerobot/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${SPLIT_ROOT}/logs/${RUN_NAME}}"

mkdir -p "${CLOUD_ROOT}/dataset_views" "${LOG_DIR}"

export RUN_NAME
export SPLIT_ROOT
export CLOUD_ROOT
export DATA_HF_LEROBOT_HOME
export HF_LEROBOT_HOME="${DATA_HF_LEROBOT_HOME}"
export HF_HOME
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# The default action and value workloads fragment the H20 allocator badly enough to OOM on their first step when
# allocation is fully demand-driven. Use bounded multiheap preallocation by default, while preserving caller overrides.
if [[ "${XLA_FLAGS:-}" != *"--xla_multiheap_size_constraint_per_heap"* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_multiheap_size_constraint_per_heap=1073741824"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_CLIENT_MEM_FRACTION="${XLA_CLIENT_MEM_FRACTION:-0.90}"
# Default live test budget: four 200-step rollout rounds, including step 800. Read at config import time by
# src/vam/training/config; an explicit NUM_TRAIN_STEPS or --num-train-steps in "$@" still overrides it.
export SPLIT_ROUND_STEPS
export NUM_TRAIN_STEPS
echo "[train:${ROLE}] run=${RUN_NAME} round_steps=${SPLIT_ROUND_STEPS} num_train_steps=${NUM_TRAIN_STEPS}"
if [[ -z "${WANDB_API_KEY:-}" && -n "${TMUX:-}" ]] && command -v tmux >/dev/null 2>&1; then
  tmux_wandb_key="$(tmux show-environment -g WANDB_API_KEY 2>/dev/null | sed 's/^WANDB_API_KEY=//')"
  if [[ -n "${tmux_wandb_key}" && "${tmux_wandb_key}" != -* ]]; then
    export WANDB_API_KEY="${tmux_wandb_key}"
  fi
  unset tmux_wandb_key
fi

latest_name="${ROLE}_latest.json"
echo "[train:${ROLE}] waiting for initial dataset view ${latest_name}..."
until [[ -f "${CLOUD_ROOT}/dataset_views/${latest_name}" && -f "${CLOUD_ROOT}/dataset_views/action_latest.json" ]]; do
  sleep 10
done

# The action view is the source of truth for the canonical offline repo whose
# normalization stats are shared by both action and value roles.
if [[ -z "${ALOHA_ASYNC_OFFLINE_REPO_ID:-}" ]]; then
  ALOHA_ASYNC_OFFLINE_REPO_ID="$("${TRAIN_PYTHON:-python}" scripts/resolve_async_offline_repo.py \
    "${CLOUD_ROOT}/dataset_views/action_latest.json")"
fi
export ALOHA_ASYNC_OFFLINE_REPO_ID
export ALOHA_ASYNC_NORM_ASSETS_DIR
export ALOHA_ASYNC_NORM_STATS_PATH="${ALOHA_ASYNC_NORM_STATS_PATH:-${ALOHA_ASYNC_NORM_ASSETS_DIR}/${ALOHA_ASYNC_OFFLINE_REPO_ID}/norm_stats.json}"

if [[ ! -d "${DATA_HF_LEROBOT_HOME}/${ALOHA_ASYNC_OFFLINE_REPO_ID}" ]]; then
  echo "Offline repo is not visible in the run LeRobot home: ${DATA_HF_LEROBOT_HOME}/${ALOHA_ASYNC_OFFLINE_REPO_ID}" >&2
  echo "Start run_online.py configs/example.yaml --component ingest first so it can link the shared offline repo," >&2
  echo "or set DATA_HF_LEROBOT_HOME explicitly." >&2
  exit 2
fi
if [[ "${ALOHA_ASYNC_NORM_STATS_PATH}" != gs://* && ! -f "${ALOHA_ASYNC_NORM_STATS_PATH}" ]]; then
  echo "Norm stats not found: ${ALOHA_ASYNC_NORM_STATS_PATH}" >&2
  exit 2
fi
if [[ "${ALOHA_ASYNC_BASE_PARAMS}" != gs://* && ! -e "${ALOHA_ASYNC_BASE_PARAMS}" ]]; then
  echo "Base checkpoint params not found: ${ALOHA_ASYNC_BASE_PARAMS}" >&2
  exit 2
fi

echo "[train:${ROLE}] offline_repo=${ALOHA_ASYNC_OFFLINE_REPO_ID}"
echo "[train:${ROLE}] lerobot_home=${DATA_HF_LEROBOT_HOME}"
echo "[train:${ROLE}] norm_stats=${ALOHA_ASYNC_NORM_STATS_PATH}"
echo "[train:${ROLE}] base_params=${ALOHA_ASYNC_BASE_PARAMS}"

mode_args=(--overwrite)
if [[ "${RESUME:-0}" == "1" ]]; then
  mode_args=(--resume)
fi

"${TRAIN_PYTHON:-python}" scripts/train_async.py "${CONFIG_NAME}" \
  "${mode_args[@]}" \
  "$@" \
  2>&1 | tee -a "${LOG_DIR}/train_async_${ROLE}.log"
