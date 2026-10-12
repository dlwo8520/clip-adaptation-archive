#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 6개 configuration을 병렬로 동시에 실행함.
# (selection method 3종) × (WC on/off 2종) = 6
#
# - teacher_margin     + WC off
# - teacher_margin     + WC on
# - teacher_uncertain  + WC off
# - teacher_uncertain  + WC on
# - teacher_confident  + WC off   (※ 코드에서는 teacher_confident 인자명 그대로)
# - teacher_confident  + WC on
#
# 각 config는 sample seed 0..99 까지 순차 실행 (PARALLEL_PER_CONFIG로 조정 가능).
# 동시에 6개 config가 백그라운드 프로세스로 돌아감.
#
# 사용 예:
#   BASE_OUT=runs/coop_refdistill_6configs \
#   TARGET_ROOT=/data/datasets/texture \
#   REFERENCE_ROOT=/data/datasets/imagenet_1k/val \
#   GPU_LIST="0 0 0 1 1 1" \
#   bash scripts/launch_6_configs_parallel.sh
#
# 변수:
#   BASE_OUT              : 출력 루트 (기본 runs/coop_refdistill_6configs)
#   TARGET_NAME/ROOT      : target dataset (DTD 기본)
#   REFERENCE_NAME/ROOT   : reference dataset (ImageNet 기본)
#   PRESERVE_SPECS        : preserve dataset (imagenet+cifar100 기본)
#   SEED_START            : 시작 seed (기본 0)
#   NUM_SAMPLE_SEEDS      : 총 seed 개수 (기본 100)
#   PARALLEL_PER_CONFIG   : 각 config 내부에서 동시에 돌릴 seed 수 (기본 1)
#   LAMBDA_DISTILL        : reference distill 계수 (기본 1.0)
#   LAMBDA_WC             : WC-on config에서 사용할 wc 계수 (기본 0.1)
#   GPU_LIST              : 공백으로 구분한 6개의 GPU id (기본 "0 0 0 0 0 0", 즉 GPU 0에 몰아넣음)
#   MAX_CONCURRENT_CONFIGS: 동시에 띄울 config 수 (기본 6, 즉 전부 동시). 단일 GPU OOM 우려 시 2 또는 3 권장.
#   EPOCHS, EVAL_EVERY, SHOTS_LIST 등 (run_coop_refdistill_one_config.sh와 동일)

set -uo pipefail

cd "$(dirname "$0")/.."

# ===== 공통 환경변수 =====
BASE_OUT=${BASE_OUT:-"runs/coop_refdistill_6configs"}
TARGET_NAME=${TARGET_NAME:-"texture"}
TARGET_ROOT=${TARGET_ROOT:-"/data/datasets/texture"}
REFERENCE_NAME=${REFERENCE_NAME:-"imagenet"}
REFERENCE_ROOT=${REFERENCE_ROOT:?"REFERENCE_ROOT 필요. ImageFolder 형식의 reference image dataset 경로를 넘기세요. 가능하면 preserve eval set과 다른 split을 권장."}
PRESERVE_SPECS=${PRESERVE_SPECS:-"imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test"}

SEED_START=${SEED_START:-0}
NUM_SAMPLE_SEEDS=${NUM_SAMPLE_SEEDS:-100}
PARALLEL_PER_CONFIG=${PARALLEL_PER_CONFIG:-1}

LAMBDA_DISTILL=${LAMBDA_DISTILL:-1.0}
LAMBDA_WC=${LAMBDA_WC:-0.1}

EPOCHS=${EPOCHS:-50}
EVAL_EVERY=${EVAL_EVERY:-5}
SHOTS_LIST=${SHOTS_LIST:-"1 2 4 8 16"}
MAX_REFERENCE_IMAGES=${MAX_REFERENCE_IMAGES:-10000}
DISTILL_NUM_IMG=${DISTILL_NUM_IMG:-64}
DISTILL_NUM_CLS=${DISTILL_NUM_CLS:-64}
SELECTION_CANDIDATE_FACTOR=${SELECTION_CANDIDATE_FACTOR:-2}

# 6개 GPU 할당 (기본: 모두 GPU 0). 환경에 맞게 조정 권장.
GPU_LIST=${GPU_LIST:-"0 0 0 0 0 0"}
read -r -a GPU_ARR <<< "${GPU_LIST}"
if [ "${#GPU_ARR[@]}" -ne 6 ]; then
    echo "[error] GPU_LIST는 6개의 값으로 이루어져야 함 (현재 ${#GPU_ARR[@]}개): '${GPU_LIST}'" >&2
    exit 2
fi

# 동시에 실행할 config 수 제한 (단일 GPU 환경에서 OOM 방지용).
MAX_CONCURRENT_CONFIGS=${MAX_CONCURRENT_CONFIGS:-6}
if [ "${MAX_CONCURRENT_CONFIGS}" -lt 1 ] || [ "${MAX_CONCURRENT_CONFIGS}" -gt 6 ]; then
    echo "[error] MAX_CONCURRENT_CONFIGS는 1~6 사이여야 함 (현재 ${MAX_CONCURRENT_CONFIGS})" >&2
    exit 2
fi

mkdir -p "${BASE_OUT}/logs"

# ===== 6개 config 정의 =====
# 형식: "tag selection_method use_wc"
CONFIGS=(
    "method__teacher_margin__wc_off    teacher_margin    off"
    "method__teacher_margin__wc_on     teacher_margin    on"
    "method__teacher_uncertain__wc_off teacher_uncertain off"
    "method__teacher_uncertain__wc_on  teacher_uncertain on"
    "method__teacher_confident__wc_off teacher_confident off"
    "method__teacher_confident__wc_on  teacher_confident on"
)

# 안전장치: 6개여야 함
if [ "${#CONFIGS[@]}" -ne 6 ]; then
    echo "[error] CONFIGS 개수 이상: ${#CONFIGS[@]}" >&2
    exit 2
fi

echo "===== launch_6_configs_parallel.sh ====="
echo "BASE_OUT             : ${BASE_OUT}"
echo "TARGET_NAME / ROOT   : ${TARGET_NAME} / ${TARGET_ROOT}"
echo "REFERENCE_NAME/ROOT  : ${REFERENCE_NAME} / ${REFERENCE_ROOT}"
echo "PRESERVE_SPECS       : ${PRESERVE_SPECS}"
echo "SEED_START           : ${SEED_START}"
echo "NUM_SAMPLE_SEEDS     : ${NUM_SAMPLE_SEEDS}"
echo "PARALLEL_PER_CONFIG  : ${PARALLEL_PER_CONFIG}"
echo "LAMBDA_DISTILL       : ${LAMBDA_DISTILL}"
echo "LAMBDA_WC            : ${LAMBDA_WC}"
echo "GPU_LIST             : ${GPU_LIST}"
echo "MAX_CONCURRENT_CONFIGS: ${MAX_CONCURRENT_CONFIGS}"
echo "EPOCHS / EVAL_EVERY  : ${EPOCHS} / ${EVAL_EVERY}"
echo "SHOTS_LIST           : ${SHOTS_LIST}"
echo "MAX_REFERENCE_IMAGES : ${MAX_REFERENCE_IMAGES}"
echo "DISTILL_NUM_IMG/CLS  : ${DISTILL_NUM_IMG} / ${DISTILL_NUM_CLS}"
echo ""

pids=()
labels=()

# 동시 실행 중인 background process pid를 추적해서, MAX_CONCURRENT_CONFIGS만큼만 동시에 띄움.
running_pids=()

for i in "${!CONFIGS[@]}"; do
    # 동시 실행 한도 도달 시 하나가 끝날 때까지 대기
    while [ "${#running_pids[@]}" -ge "${MAX_CONCURRENT_CONFIGS}" ]; do
        new_running=()
        for rp in "${running_pids[@]}"; do
            if kill -0 "${rp}" 2>/dev/null; then
                new_running+=("${rp}")
            fi
        done
        running_pids=("${new_running[@]}")
        if [ "${#running_pids[@]}" -ge "${MAX_CONCURRENT_CONFIGS}" ]; then
            sleep 5
        fi
    done

    read -r tag method use_wc <<< "${CONFIGS[$i]}"
    gpu="${GPU_ARR[$i]}"

    ROOT_OUT="${BASE_OUT}/${tag}"
    LOG_PATH="${BASE_OUT}/logs/${tag}.log"
    mkdir -p "${ROOT_OUT}"

    echo "[launch ${i}] tag=${tag} method=${method} use_wc=${use_wc} gpu=${gpu}"
    echo "             ROOT_OUT=${ROOT_OUT}"
    echo "             LOG_PATH=${LOG_PATH}"

    BASE_OUT="${BASE_OUT}" \
    ROOT_OUT="${ROOT_OUT}" \
    SELECTION_METHOD="${method}" \
    USE_WC="${use_wc}" \
    LAMBDA_DISTILL="${LAMBDA_DISTILL}" \
    LAMBDA_WC="${LAMBDA_WC}" \
    SEED_START="${SEED_START}" \
    NUM_SAMPLE_SEEDS="${NUM_SAMPLE_SEEDS}" \
    PARALLEL="${PARALLEL_PER_CONFIG}" \
    TARGET_NAME="${TARGET_NAME}" \
    TARGET_ROOT="${TARGET_ROOT}" \
    REFERENCE_NAME="${REFERENCE_NAME}" \
    REFERENCE_ROOT="${REFERENCE_ROOT}" \
    PRESERVE_SPECS="${PRESERVE_SPECS}" \
    GPU_ID="${gpu}" \
    EPOCHS="${EPOCHS}" \
    EVAL_EVERY="${EVAL_EVERY}" \
    SHOTS_LIST="${SHOTS_LIST}" \
    MAX_REFERENCE_IMAGES="${MAX_REFERENCE_IMAGES}" \
    DISTILL_NUM_IMG="${DISTILL_NUM_IMG}" \
    DISTILL_NUM_CLS="${DISTILL_NUM_CLS}" \
    SELECTION_CANDIDATE_FACTOR="${SELECTION_CANDIDATE_FACTOR}" \
    bash scripts/run_coop_refdistill_one_config.sh \
    > "${LOG_PATH}" 2>&1 &

    pid=$!
    pids+=("${pid}")
    running_pids+=("${pid}")
    labels+=("${tag} (pid=${pid}, gpu=${gpu})")
    echo "             pid=${pid}"
    echo ""
done

echo "===== 6 background processes launched ====="
for label in "${labels[@]}"; do
    echo "  - ${label}"
done
echo ""
echo "각 config의 진행은 ${BASE_OUT}/logs/<tag>.log 에서 확인."
echo "개별 seed 진행은 ${BASE_OUT}/<tag>/sample_seed_<seed>/run.log 에서 확인."
echo ""
echo "[info] 모든 background process가 끝날 때까지 대기 중..."

fail_count=0
for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    label="${labels[$i]}"
    if wait "${pid}"; then
        echo "[done] ${label}"
    else
        echo "[fail] ${label}"
        fail_count=$(( fail_count + 1 ))
    fi
done

echo ""
if (( fail_count > 0 )); then
    echo "===== 종료 (실패한 config ${fail_count}개) ====="
    exit 1
else
    echo "===== 모든 config 정상 종료 ====="
fi
