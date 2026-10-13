#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# GIFT-inspired 실험을 위한 4개 configuration을 launch.
#
# 본 launcher는 selection method를 하나(SELECTION_METHOD; default teacher_margin)로 고정한 채
# preservation 방법만 4가지를 비교하기 위한 것.  selection method를 바꾸려면 환경변수로 변경.
#
# 4 configs:
#   A) baseline_refdistill         : 기존 ZSCL real-reference distillation (변경 없음)
#   B) synthetic_zscl_like         : synthetic image를 reference처럼만 쓰고 ITA 없음
#                                    (α>0, β=0, AWC off; refdistill과의 fair 비교용)
#   C) gift_synthetic              : synthetic matched pair에 대해 contrastive distill + ITA
#                                    (α=1.0, β=0.25, AWC off; 본 GIFT-inspired)
#   D) gift_synthetic_awc          : C + Adaptive Weight Consolidation on CoOp ctx
#
# 본 launcher는 background process로 4개를 띄움. 단일 GPU 환경이면 MAX_CONCURRENT_CONFIGS를
# 낮춰 OOM을 피하세요.
#
# 사용 예 (가장 일반적):
#   BASE_OUT=runs/coop_gift_4configs \
#   TARGET_ROOT=/data/datasets/texture \
#   REFERENCE_ROOT=/data/datasets/imagenet_1k/train \
#   GIFT_SYNTHETIC_ROOT=/data/datasets/gift_synthetic_dtd \
#   PRESERVE_SPECS="imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test" \
#   GPU_LIST="0 0 1 1" \
#   NUM_SAMPLE_SEEDS=100 \
#   bash scripts/launch_gift_configs.sh
#
# 변수:
#   BASE_OUT, TARGET_NAME/ROOT, PRESERVE_SPECS    : run_coop_gift_one_config.sh와 동일
#   REFERENCE_NAME / REFERENCE_ROOT               : refdistill 경로 (config A에 필수)
#   GIFT_SYNTHETIC_ROOT / GIFT_METADATA           : gift_synthetic 경로 (B/C/D에 필수)
#   SELECTION_METHOD                              : default teacher_margin
#   GPU_LIST                                       : "g0 g1 g2 g3" (4개)
#   MAX_CONCURRENT_CONFIGS                        : default 4 (단일 GPU면 2)
#   SEED_START / NUM_SAMPLE_SEEDS / PARALLEL_PER_CONFIG / EPOCHS / EVAL_EVERY / SHOTS_LIST
#   LAMBDA_DISTILL / LAMBDA_WC
#   GIFT_ALPHA / GIFT_BETA / GIFT_LAMBDA_AWC / GIFT_BATCH_SIZE / GIFT_MAX_IMAGES
#   SYNTHETIC_ZSCL_ALPHA / SYNTHETIC_ZSCL_BETA   : config B 전용 (default 1.0 / 0.0)

set -uo pipefail

cd "$(dirname "$0")/.."

# ===== 공통 환경변수 =====
BASE_OUT=${BASE_OUT:-"runs/coop_gift_4configs"}
TARGET_NAME=${TARGET_NAME:-"texture"}
TARGET_ROOT=${TARGET_ROOT:-"/data/datasets/texture"}
PRESERVE_SPECS=${PRESERVE_SPECS:-"imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test"}

# refdistill 경로용
REFERENCE_NAME=${REFERENCE_NAME:-"imagenet"}
REFERENCE_ROOT=${REFERENCE_ROOT:-""}
MAX_REFERENCE_IMAGES=${MAX_REFERENCE_IMAGES:-10000}
DISTILL_NUM_IMG=${DISTILL_NUM_IMG:-64}
DISTILL_NUM_CLS=${DISTILL_NUM_CLS:-64}

# gift_synthetic 경로용
GIFT_SYNTHETIC_ROOT=${GIFT_SYNTHETIC_ROOT:-""}
GIFT_METADATA=${GIFT_METADATA:-""}
GIFT_ALPHA=${GIFT_ALPHA:-1.0}
GIFT_BETA=${GIFT_BETA:-0.25}
GIFT_BATCH_SIZE=${GIFT_BATCH_SIZE:-64}
GIFT_LAMBDA_AWC=${GIFT_LAMBDA_AWC:-0.1}
GIFT_MAX_IMAGES=${GIFT_MAX_IMAGES:-0}

# config B (synthetic ZSCL-like ablation): α만 두고 β=0
SYNTHETIC_ZSCL_ALPHA=${SYNTHETIC_ZSCL_ALPHA:-1.0}
SYNTHETIC_ZSCL_BETA=${SYNTHETIC_ZSCL_BETA:-0.0}

# 공통
LAMBDA_DISTILL=${LAMBDA_DISTILL:-1.0}
LAMBDA_WC=${LAMBDA_WC:-0.0}
SELECTION_METHOD=${SELECTION_METHOD:-"teacher_margin"}
SELECTION_CANDIDATE_FACTOR=${SELECTION_CANDIDATE_FACTOR:-2}

SEED_START=${SEED_START:-0}
NUM_SAMPLE_SEEDS=${NUM_SAMPLE_SEEDS:-100}
PARALLEL_PER_CONFIG=${PARALLEL_PER_CONFIG:-1}
EPOCHS=${EPOCHS:-50}
EVAL_EVERY=${EVAL_EVERY:-5}
SHOTS_LIST=${SHOTS_LIST:-"1 2 4 8 16"}

GPU_LIST=${GPU_LIST:-"0 0 0 0"}
read -r -a GPU_ARR <<< "${GPU_LIST}"
if [ "${#GPU_ARR[@]}" -ne 4 ]; then
    echo "[error] GPU_LIST는 4개의 값이어야 함 (현재 ${#GPU_ARR[@]}개): '${GPU_LIST}'" >&2
    exit 2
fi

MAX_CONCURRENT_CONFIGS=${MAX_CONCURRENT_CONFIGS:-4}
if [ "${MAX_CONCURRENT_CONFIGS}" -lt 1 ] || [ "${MAX_CONCURRENT_CONFIGS}" -gt 4 ]; then
    echo "[error] MAX_CONCURRENT_CONFIGS는 1~4 사이여야 함 (현재 ${MAX_CONCURRENT_CONFIGS})" >&2
    exit 2
fi

mkdir -p "${BASE_OUT}/logs"

# ===== 4개 config 정의 =====
# 형식: "tag preserve_method extra_kv..."
# extra_kv는 같은 line의 env override를 한 줄로 표현 (parse는 launcher가 직접).
CONFIGS=(
    "A__baseline_refdistill   refdistill"
    "B__synthetic_zscl_like   gift_synthetic"
    "C__gift_synthetic        gift_synthetic"
    "D__gift_synthetic_awc    gift_synthetic"
)

if [ "${#CONFIGS[@]}" -ne 4 ]; then
    echo "[error] CONFIGS 개수가 4가 아님: ${#CONFIGS[@]}" >&2
    exit 2
fi

# 사전 검증: refdistill 경로에는 REFERENCE_ROOT, gift_synthetic 경로에는 GIFT_SYNTHETIC_ROOT 필요
if [ -z "${REFERENCE_ROOT}" ]; then
    echo "[error] config A (baseline_refdistill) 를 위해 REFERENCE_ROOT 필요" >&2
    exit 2
fi
if [ -z "${GIFT_SYNTHETIC_ROOT}" ]; then
    echo "[error] config B/C/D (gift_synthetic) 를 위해 GIFT_SYNTHETIC_ROOT 필요" >&2
    exit 2
fi

echo "===== launch_gift_configs.sh ====="
echo "BASE_OUT             : ${BASE_OUT}"
echo "TARGET_NAME / ROOT   : ${TARGET_NAME} / ${TARGET_ROOT}"
echo "PRESERVE_SPECS       : ${PRESERVE_SPECS}"
echo "REFERENCE_ROOT       : ${REFERENCE_ROOT}"
echo "GIFT_SYNTHETIC_ROOT  : ${GIFT_SYNTHETIC_ROOT}"
echo "SELECTION_METHOD     : ${SELECTION_METHOD}"
echo "SEED_START / NUM     : ${SEED_START} / ${NUM_SAMPLE_SEEDS}"
echo "PARALLEL_PER_CONFIG  : ${PARALLEL_PER_CONFIG}"
echo "GPU_LIST             : ${GPU_LIST}"
echo "MAX_CONCURRENT_CONFIGS: ${MAX_CONCURRENT_CONFIGS}"
echo "EPOCHS / EVAL_EVERY  : ${EPOCHS} / ${EVAL_EVERY}"
echo "SHOTS_LIST           : ${SHOTS_LIST}"
echo "LAMBDA_DISTILL/WC    : ${LAMBDA_DISTILL} / ${LAMBDA_WC}"
echo "GIFT_ALPHA/BETA      : ${GIFT_ALPHA} / ${GIFT_BETA}"
echo "GIFT_LAMBDA_AWC      : ${GIFT_LAMBDA_AWC}"
echo "SYNTH_ZSCL ALPHA/BETA: ${SYNTHETIC_ZSCL_ALPHA} / ${SYNTHETIC_ZSCL_BETA}"
echo ""

pids=()
labels=()
running_pids=()

for i in "${!CONFIGS[@]}"; do
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

    read -r tag pmethod <<< "${CONFIGS[$i]}"
    gpu="${GPU_ARR[$i]}"

    ROOT_OUT="${BASE_OUT}/${tag}"
    LOG_PATH="${BASE_OUT}/logs/${tag}.log"
    mkdir -p "${ROOT_OUT}"

    echo "[launch ${i}] tag=${tag} preserve_method=${pmethod} gpu=${gpu}"
    echo "             ROOT_OUT=${ROOT_OUT}"
    echo "             LOG_PATH=${LOG_PATH}"

    # config 별 env override (alpha/beta/awc 결정)
    case "${tag}" in
        A__baseline_refdistill)
            CFG_ALPHA="${GIFT_ALPHA}"
            CFG_BETA="${GIFT_BETA}"
            CFG_USE_AWC="off"
            ;;
        B__synthetic_zscl_like)
            # synthetic을 쓰지만 ITA 없음 (β=0). α만으로 reference-style distillation 흉내.
            CFG_ALPHA="${SYNTHETIC_ZSCL_ALPHA}"
            CFG_BETA="${SYNTHETIC_ZSCL_BETA}"
            CFG_USE_AWC="off"
            ;;
        C__gift_synthetic)
            CFG_ALPHA="${GIFT_ALPHA}"
            CFG_BETA="${GIFT_BETA}"
            CFG_USE_AWC="off"
            ;;
        D__gift_synthetic_awc)
            CFG_ALPHA="${GIFT_ALPHA}"
            CFG_BETA="${GIFT_BETA}"
            CFG_USE_AWC="on"
            ;;
        *)
            echo "[error] 알 수 없는 tag: ${tag}" >&2
            exit 2
            ;;
    esac

    BASE_OUT="${BASE_OUT}" \
    ROOT_OUT="${ROOT_OUT}" \
    PRESERVE_METHOD="${pmethod}" \
    SELECTION_METHOD="${SELECTION_METHOD}" \
    SELECTION_CANDIDATE_FACTOR="${SELECTION_CANDIDATE_FACTOR}" \
    LAMBDA_DISTILL="${LAMBDA_DISTILL}" \
    LAMBDA_WC="${LAMBDA_WC}" \
    REFERENCE_NAME="${REFERENCE_NAME}" \
    REFERENCE_ROOT="${REFERENCE_ROOT}" \
    MAX_REFERENCE_IMAGES="${MAX_REFERENCE_IMAGES}" \
    DISTILL_NUM_IMG="${DISTILL_NUM_IMG}" \
    DISTILL_NUM_CLS="${DISTILL_NUM_CLS}" \
    GIFT_SYNTHETIC_ROOT="${GIFT_SYNTHETIC_ROOT}" \
    GIFT_METADATA="${GIFT_METADATA}" \
    GIFT_ALPHA="${CFG_ALPHA}" \
    GIFT_BETA="${CFG_BETA}" \
    GIFT_BATCH_SIZE="${GIFT_BATCH_SIZE}" \
    GIFT_MAX_IMAGES="${GIFT_MAX_IMAGES}" \
    USE_AWC="${CFG_USE_AWC}" \
    GIFT_LAMBDA_AWC="${GIFT_LAMBDA_AWC}" \
    SEED_START="${SEED_START}" \
    NUM_SAMPLE_SEEDS="${NUM_SAMPLE_SEEDS}" \
    PARALLEL="${PARALLEL_PER_CONFIG}" \
    TARGET_NAME="${TARGET_NAME}" \
    TARGET_ROOT="${TARGET_ROOT}" \
    PRESERVE_SPECS="${PRESERVE_SPECS}" \
    GPU_ID="${gpu}" \
    EPOCHS="${EPOCHS}" \
    EVAL_EVERY="${EVAL_EVERY}" \
    SHOTS_LIST="${SHOTS_LIST}" \
    bash scripts/run_coop_gift_one_config.sh \
    > "${LOG_PATH}" 2>&1 &

    pid=$!
    pids+=("${pid}")
    running_pids+=("${pid}")
    labels+=("${tag} (pid=${pid}, gpu=${gpu})")
    echo "             pid=${pid}"
    echo ""
done

echo "===== ${#CONFIGS[@]} background processes launched ====="
for label in "${labels[@]}"; do
    echo "  - ${label}"
done

echo ""
echo "각 config의 진행: ${BASE_OUT}/logs/<tag>.log"
echo "개별 seed 진행 : ${BASE_OUT}/<tag>/sample_seed_<seed>/run.log"
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
