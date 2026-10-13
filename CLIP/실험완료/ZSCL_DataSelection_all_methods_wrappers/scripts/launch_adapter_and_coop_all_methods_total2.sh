#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

BASE_OUT=${BASE_OUT:-"runs/zscl_texture_all_methods"}
ADAPTER_GPU=${ADAPTER_GPU:-0}
COOP_GPU=${COOP_GPU:-1}

# 이 launcher는 adapter wrapper 1개 + coop wrapper 1개를 동시에 실행함.
# 전체 동시 학습 프로세스를 2개로 맞추기 위해 각 wrapper 내부 PARALLEL=1로 고정함.
mkdir -p "${BASE_OUT}/logs"

echo "===== launch Adapter + CoOp all methods ====="
echo "BASE_OUT   : ${BASE_OUT}"
echo "ADAPTER_GPU: ${ADAPTER_GPU}"
echo "COOP_GPU   : ${COOP_GPU}"
echo "inner PARALLEL: 1 each, total 2 processes"

CUDA_VISIBLE_DEVICES="${ADAPTER_GPU}" \
BASE_OUT="${BASE_OUT}" \
PARALLEL=1 \
bash scripts/run_all_adapter_zscl_methods.sh \
> "${BASE_OUT}/logs/adapter_all_methods.log" 2>&1 &
ADAPTER_PID=$!

CUDA_VISIBLE_DEVICES="${COOP_GPU}" \
BASE_OUT="${BASE_OUT}" \
PARALLEL=1 \
bash scripts/run_all_coop_zscl_methods.sh \
> "${BASE_OUT}/logs/coop_all_methods.log" 2>&1 &
COOP_PID=$!

echo "adapter pid: ${ADAPTER_PID}"
echo "coop pid   : ${COOP_PID}"
echo "adapter log: ${BASE_OUT}/logs/adapter_all_methods.log"
echo "coop log   : ${BASE_OUT}/logs/coop_all_methods.log"

wait "${ADAPTER_PID}" || echo "[경고] adapter wrapper 비정상 종료"
wait "${COOP_PID}" || echo "[경고] coop wrapper 비정상 종료"

echo "===== both wrappers done ====="
