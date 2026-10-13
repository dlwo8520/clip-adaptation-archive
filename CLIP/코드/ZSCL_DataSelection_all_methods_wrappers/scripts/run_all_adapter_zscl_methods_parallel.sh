#!/usr/bin/env bash
# 4개 method를 동시에 백그라운드 launch.
# 각 method 안에서는 SEED_PARALLEL 개 seed를 병렬로 돌림.
# 총 동시 process 수 = METHODS 개수 × SEED_PARALLEL.
set -uo pipefail
cd "$(dirname "$0")/.."

# GPU 바인딩 (외부에서 override 가능)
# 기본값: GPU 1번 사용
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# prototype 제거한 method 목록
METHODS=${METHODS:-"random teacher_confident teacher_margin teacher_uncertain"}

BASE_OUT=${BASE_OUT:-"runs/zscl_texture_all_methods"}
SEED_START=${SEED_START:-0}
NUM_SAMPLE_SEEDS=${NUM_SAMPLE_SEEDS:-100}
SEED_PARALLEL=${SEED_PARALLEL:-2}      # 한 method 안에서 동시에 돌릴 seed 수
CANDIDATE_FACTOR=${CANDIDATE_FACTOR:-2}

mkdir -p "${BASE_OUT}/adapter"

echo "===== run all Adapter ZSCL methods in PARALLEL ====="
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo "BASE_OUT             : ${BASE_OUT}"
echo "METHODS              : ${METHODS}"
echo "SEED_PARALLEL/method : ${SEED_PARALLEL}"
echo "TOTAL CONCURRENT     : $(echo ${METHODS} | wc -w) methods × ${SEED_PARALLEL} seeds"
echo "SEED_START           : ${SEED_START}"
echo "NUM_SAMPLE_SEEDS     : ${NUM_SAMPLE_SEEDS}"

method_pids=()

for METHOD in ${METHODS}; do
    ROOT_OUT="${BASE_OUT}/adapter/${METHOD}"
    mkdir -p "${ROOT_OUT}"
    LOG_FILE="${ROOT_OUT}/_method_wrapper.log"

    echo "[launch] method=${METHOD} -> ${LOG_FILE}"

    (
        PARALLEL="${SEED_PARALLEL}" \
        SEED_START="${SEED_START}" \
        NUM_SAMPLE_SEEDS="${NUM_SAMPLE_SEEDS}" \
        ROOT_OUT="${ROOT_OUT}" \
        SELECTION_METHOD="${METHOD}" \
        CANDIDATE_FACTOR="${CANDIDATE_FACTOR}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        bash scripts/run_texture_adapter_zscl_100seeds_gpu1.sh
    ) > "${LOG_FILE}" 2>&1 &

    method_pids+=($!)
done

echo ""
echo "===== ${#method_pids[@]} methods running in parallel. waiting... ====="

fail=0

for pid in "${method_pids[@]}"; do
    if ! wait "${pid}"; then
        echo "[경고] method PID=${pid} 비정상 종료"
        fail=1
    fi
done

echo ""
echo "===== all methods done (fail=${fail}) ====="
echo "results: ${BASE_OUT}/adapter"