#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 6개 configuration을 GPU 한 장(기본 GPU 1)에서 한 번에 하나씩 순차로 실행함.
# (selection method 3종) × (WC on/off 2종) = 6
#
# - teacher_margin     + WC off
# - teacher_margin     + WC on
# - teacher_entropy    + WC off
# - teacher_entropy    + WC on
# - teacher_confident  + WC off
# - teacher_confident  + WC on
#
# 한 config 안에서도 seed 0..99 가 PARALLEL_PER_CONFIG (기본 1) 단위로 진행됨.
# 즉 기본값에서는 "한 번에 1 process"만 GPU를 점유함.
#
# parallel 버전과의 차이:
#   - background 실행 없음. 한 config가 끝나야 다음 config 시작.
#   - GPU도 한 장만 사용 (GPU_ID, 기본 1).
#   - 단일 GPU 환경 / OOM 우려 / 디버깅 시 권장.
#
# 사용 예:
#   BASE_OUT=runs/coop_refdistill_6configs_seq \
#   TARGET_ROOT=/data/datasets/texture \
#   REFERENCE_ROOT=/data/datasets/imagenet_1k/val \
#   bash scripts/launch_6_configs_sequential.sh
#
# 변수: launch_6_configs_parallel.sh와 동일. 단 GPU_LIST 대신 GPU_ID 하나만 받음.
#   GPU_ID                : 사용할 GPU id (기본 1)
#   BASE_OUT              : 출력 루트 (기본 runs/coop_refdistill_6configs_seq)
#   TARGET_NAME/ROOT      : target dataset (DTD 기본)
#   REFERENCE_NAME/ROOT   : reference dataset (ImageNet 기본)
#   PRESERVE_SPECS        : preserve dataset (imagenet+cifar100 기본)
#   SEED_START            : 시작 seed (기본 0)
#   NUM_SAMPLE_SEEDS      : 총 seed 개수 (기본 100)
#   PARALLEL_PER_CONFIG   : 한 config 내부에서 동시에 돌릴 seed 수 (기본 1, 단일 GPU에선 1 권장)
#   LAMBDA_DISTILL        : reference distill 계수 (기본 1.0)
#   LAMBDA_WC             : WC-on config의 wc 계수 (기본 0.1)
#   EPOCHS, EVAL_EVERY, SHOTS_LIST 등 : run_coop_refdistill_one_config.sh와 동일

set -uo pipefail

cd "$(dirname "$0")/.."

# ===== 공통 환경변수 =====
BASE_OUT=${BASE_OUT:-"runs/coop_refdistill_6configs_seq"}
TARGET_NAME=${TARGET_NAME:-"texture"}
TARGET_ROOT=${TARGET_ROOT:-"/data/datasets/texture"}
REFERENCE_NAME=${REFERENCE_NAME:-"imagenet"}
REFERENCE_ROOT=${REFERENCE_ROOT:?"REFERENCE_ROOT 필요. ImageFolder 형식의 reference image dataset 경로를 넘기세요."}
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

# 단일 GPU 사용 (기본 GPU 1)
GPU_ID=${GPU_ID:-1}

mkdir -p "${BASE_OUT}/logs"

# ===== 6개 config 정의 =====
# 형식: "tag selection_method use_wc"
CONFIGS=(
    "method__teacher_margin__wc_off    teacher_margin    off"
    "method__teacher_margin__wc_on     teacher_margin    on"
    "method__teacher_entropy__wc_off   teacher_entropy   off"
    "method__teacher_entropy__wc_on    teacher_entropy   on"
    "method__teacher_confident__wc_off teacher_confident off"
    "method__teacher_confident__wc_on  teacher_confident on"
)

if [ "${#CONFIGS[@]}" -ne 6 ]; then
    echo "[error] CONFIGS 개수 이상: ${#CONFIGS[@]}" >&2
    exit 2
fi

echo "===== launch_6_configs_sequential.sh ====="
echo "BASE_OUT             : ${BASE_OUT}"
echo "TARGET_NAME / ROOT   : ${TARGET_NAME} / ${TARGET_ROOT}"
echo "REFERENCE_NAME/ROOT  : ${REFERENCE_NAME} / ${REFERENCE_ROOT}"
echo "PRESERVE_SPECS       : ${PRESERVE_SPECS}"
echo "SEED_START           : ${SEED_START}"
echo "NUM_SAMPLE_SEEDS     : ${NUM_SAMPLE_SEEDS}"
echo "PARALLEL_PER_CONFIG  : ${PARALLEL_PER_CONFIG}"
echo "LAMBDA_DISTILL       : ${LAMBDA_DISTILL}"
echo "LAMBDA_WC            : ${LAMBDA_WC}"
echo "GPU_ID               : ${GPU_ID}"
echo "EPOCHS / EVAL_EVERY  : ${EPOCHS} / ${EVAL_EVERY}"
echo "SHOTS_LIST           : ${SHOTS_LIST}"
echo "MAX_REFERENCE_IMAGES : ${MAX_REFERENCE_IMAGES}"
echo "DISTILL_NUM_IMG/CLS  : ${DISTILL_NUM_IMG} / ${DISTILL_NUM_CLS}"
echo ""

start_ts=$(date +%s)
fail_tags=()
done_tags=()

for i in "${!CONFIGS[@]}"; do
    read -r tag method use_wc <<< "${CONFIGS[$i]}"

    ROOT_OUT="${BASE_OUT}/${tag}"
    LOG_PATH="${BASE_OUT}/logs/${tag}.log"
    mkdir -p "${ROOT_OUT}"

    cfg_start_ts=$(date +%s)
    echo "===== [$(( i + 1 ))/6] config 시작: ${tag} ====="
    echo "       method=${method} use_wc=${use_wc} gpu=${GPU_ID}"
    echo "       ROOT_OUT=${ROOT_OUT}"
    echo "       LOG_PATH=${LOG_PATH}"

    set +e
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
    GPU_ID="${GPU_ID}" \
    EPOCHS="${EPOCHS}" \
    EVAL_EVERY="${EVAL_EVERY}" \
    SHOTS_LIST="${SHOTS_LIST}" \
    MAX_REFERENCE_IMAGES="${MAX_REFERENCE_IMAGES}" \
    DISTILL_NUM_IMG="${DISTILL_NUM_IMG}" \
    DISTILL_NUM_CLS="${DISTILL_NUM_CLS}" \
    SELECTION_CANDIDATE_FACTOR="${SELECTION_CANDIDATE_FACTOR}" \
    bash scripts/run_coop_refdistill_one_config.sh \
    > "${LOG_PATH}" 2>&1
    rc=$?
    set -e

    cfg_end_ts=$(date +%s)
    elapsed=$(( cfg_end_ts - cfg_start_ts ))
    if [ "${rc}" -eq 0 ]; then
        echo "[done] ${tag}  (소요 ${elapsed}s)"
        done_tags+=("${tag}")
    else
        echo "[fail] ${tag}  (rc=${rc}, 소요 ${elapsed}s)  → ${LOG_PATH} 확인"
        fail_tags+=("${tag}")
    fi
    echo ""
done

end_ts=$(date +%s)
total=$(( end_ts - start_ts ))

echo "===== 전체 종료 (총 ${total}s) ====="
echo "성공: ${#done_tags[@]} / 실패: ${#fail_tags[@]}"
for t in "${done_tags[@]}"; do echo "  [done] ${t}"; done
for t in "${fail_tags[@]}"; do echo "  [fail] ${t}"; done

if [ "${#fail_tags[@]}" -gt 0 ]; then
    exit 1
fi
