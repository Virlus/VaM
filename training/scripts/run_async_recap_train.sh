#!/usr/bin/env bash
set -euo pipefail

# Minimal launcher for the real async RECAP trainer.
# Runtime defaults live in src/vam/training/config.py:
#   pi05_aloha_async_recap
#
# Fresh retrain:
#   export RUN_NAME=... CLOUD_ROOT=... CHECKPOINT_BASE_DIR=...
#   export HF_LEROBOT_HOME=... ALOHA_ASYNC_BASE_PARAMS=... ALOHA_ASYNC_NORM_ASSETS_DIR=...
#   bash scripts/run_async_recap_train.sh
#
# Resume the same run after an interruption:
#   RESUME=1 bash scripts/run_async_recap_train.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_NAME="pi05_aloha_async_recap"
: "${RUN_NAME:?Set RUN_NAME for this async run.}"
: "${CLOUD_ROOT:?Set CLOUD_ROOT for this async run control-plane registry.}"
: "${CHECKPOINT_BASE_DIR:?Set CHECKPOINT_BASE_DIR for Orbax checkpoints.}"
: "${HF_LEROBOT_HOME:?Set HF_LEROBOT_HOME for LeRobot datasets.}"
: "${ALOHA_ASYNC_BASE_PARAMS:?Set ALOHA_ASYNC_BASE_PARAMS to the initial full checkpoint params path.}"
: "${ALOHA_ASYNC_NORM_ASSETS_DIR:?Set ALOHA_ASYNC_NORM_ASSETS_DIR to the normalization asset root.}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/output/${RUN_NAME}/logs}"
export RUN_NAME CLOUD_ROOT CHECKPOINT_BASE_DIR
export ALOHA_ASYNC_BASE_PARAMS ALOHA_ASYNC_NORM_ASSETS_DIR

mkdir -p "${CLOUD_ROOT}/dataset_views" "${LOG_DIR}"

export HF_LEROBOT_HOME
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "[train] waiting for initial dataset views..."
until [[ -f "${CLOUD_ROOT}/dataset_views/action_latest.json" && -f "${CLOUD_ROOT}/dataset_views/value_latest.json" ]]; do
  sleep 10
done

if [[ -z "${ALOHA_ASYNC_OFFLINE_REPO_ID:-}" ]]; then
  ALOHA_ASYNC_OFFLINE_REPO_ID="$(python scripts/resolve_async_offline_repo.py \
    "${CLOUD_ROOT}/dataset_views/action_latest.json")"
fi
export ALOHA_ASYNC_OFFLINE_REPO_ID
export ALOHA_ASYNC_NORM_STATS_PATH="${ALOHA_ASYNC_NORM_STATS_PATH:-${ALOHA_ASYNC_NORM_ASSETS_DIR}/${ALOHA_ASYNC_OFFLINE_REPO_ID}/norm_stats.json}"

if [[ "${ALOHA_ASYNC_NORM_STATS_PATH}" != gs://* && ! -f "${ALOHA_ASYNC_NORM_STATS_PATH}" ]]; then
  echo "Norm stats not found: ${ALOHA_ASYNC_NORM_STATS_PATH}" >&2
  exit 2
fi
if [[ "${ALOHA_ASYNC_BASE_PARAMS}" != gs://* && ! -e "${ALOHA_ASYNC_BASE_PARAMS}" ]]; then
  echo "Base checkpoint params not found: ${ALOHA_ASYNC_BASE_PARAMS}" >&2
  exit 2
fi

echo "[train] offline_repo=${ALOHA_ASYNC_OFFLINE_REPO_ID}"
echo "[train] norm_stats=${ALOHA_ASYNC_NORM_STATS_PATH}"
echo "[train] base_params=${ALOHA_ASYNC_BASE_PARAMS}"

mode_args=(--overwrite)
publisher_args=()
if [[ "${RESUME:-0}" == "1" ]]; then
  mode_args=(--resume)
  if [[ -f "${CLOUD_ROOT}/checkpoints/latest.json" ]]; then
    policy_id="$(python - "${CLOUD_ROOT}/checkpoints/latest.json" <<'PY'
import json
import sys
from pathlib import Path

print(int(json.loads(Path(sys.argv[1]).read_text()).get("policy_id", 0)))
PY
)"
    publisher_args=(--async-training.checkpoint-publisher.policy-id "${policy_id}")
  fi
fi

uv run scripts/train_async.py "${CONFIG_NAME}" \
  "${mode_args[@]}" \
  "${publisher_args[@]}" \
  2>&1 | tee -a "${LOG_DIR}/train_async.log"
