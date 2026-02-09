#!/usr/bin/env bash
set -e

SCORE_JSON=${SCORE_JSON:-'{"geneval":{"torch_device":"cuda","config_path":"/abs/path/to/mmdet_config.py","ckpt_root":"/abs/path/to/mask2former_ckpt_dir","object_names_path":"/abs/path/to/object_names.txt"}}'}
DEVICE=${DEVICE:-cuda}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-18085}
NUM_REPLICAS=${NUM_REPLICAS:-}
NUM_GPUS=${NUM_GPUS:-1}
MAX_ONGOING_REQUESTS=${MAX_ONGOING_REQUESTS:-}
RAY_PORT=${RAY_PORT:-6379}
RANK=${RANK:-${NODE_RANK:-0}}
RAY_HEAD_ADDR=${RAY_HEAD_ADDR:-${MASTER_ADDR:-}}

if [ "${RANK}" = "0" ]; then
  ray start --head --port="${RAY_PORT}" >/dev/null
else
  if [ -z "${RAY_HEAD_ADDR}" ]; then
    echo "MASTER_ADDR (or RAY_HEAD_ADDR) is required on worker ranks." >&2
    exit 1
  fi
  ray start --address "${RAY_HEAD_ADDR}:${RAY_PORT}" >/dev/null
  tail -f /dev/null
  exit 0
fi

replica_args=()
if [ -n "${NUM_REPLICAS}" ]; then
  replica_args+=(--num-replicas "${NUM_REPLICAS}")
fi
ongoing_args=()
if [ -n "${MAX_ONGOING_REQUESTS}" ]; then
  ongoing_args+=(--max-ongoing-requests "${MAX_ONGOING_REQUESTS}")
fi

python -m deploy.deploy \
  --ray-address auto \
  --score-json "${SCORE_JSON}" \
  --device "${DEVICE}" \
  "${replica_args[@]}" \
  "${ongoing_args[@]}" \
  --num-gpus "${NUM_GPUS}" \
  --host "${HOST}" \
  --port "${PORT}"
