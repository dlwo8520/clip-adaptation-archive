#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 한 configuration (PRESERVE_METHOD × SELECTION_METHOD × WC / AWC 옵션) 에 대해
# sample seed SEED_START..(SEED_START + NUM_SAMPLE_SEEDS - 1) 까지 실행.
#
# 호출 변수:
#   ROOT_OUT          : 출력 루트 (e.g. runs/coop_gift/<config_tag>)
#   PRESERVE_METHOD   : refdistill | gift_synthetic   (필수)
#   SELECTION_METHOD  : teacher_margin | teacher_entropy | teacher_confident
#
#   # refdistill 경로용
#   LAMBDA_DISTILL    : default 1.0
#   LAMBDA_WC         : default 0.0 (ZSCL Eq.(5) raw weight consolidation)
#   REFERENCE_NAME / REFERENCE_ROOT
#   MAX_REFERENCE_IMAGES (default 10000)
#   DISTILL_NUM_IMG / DISTILL_NUM_CLS (default 64 / 64)
#
#   # gift_synthetic 경로용
#   GIFT_SYNTHETIC_ROOT : 합성 dataset 경로 (필수 when PRESERVE_METHOD=gift_synthetic)
#   GIFT_METADATA       : (선택) metadata.jsonl 경로 (없으면 root/metadata.jsonl)
#   GIFT_ALPHA / GIFT_BETA : default 1.0 / 0.25
#   GIFT_BATCH_SIZE     : default 64
#   USE_AWC             : "on" or "off" (default off)
#   GIFT_LAMBDA_AWC     : default 0.1 (USE_AWC=on일 때만)
#   GIFT_MAX_IMAGES     : default 0 (0이면 전부 사용)
#
#   # 공통
#   SEED_START          : default 0
#   NUM_SAMPLE_SEEDS    : default 100
#   PARALLEL            : 한 config 내 동시 seed 실행 수 (default 1)
#   TARGET_NAME / TARGET_ROOT : default texture / /data/datasets/texture
#   PRESERVE_SPECS      : default imagenet+cifar100
#   GPU_ID              : default 0
#   EPOCHS              : default 50
#   EVAL_EVERY          : default 5
#   SHOTS_LIST          : default "1 2 4 8 16"
#
# 결과 구조:
#   ${ROOT_OUT}/sample_seed_<seed>/{args.json, curves.jsonl, results.json, ...}

set -uo pipefail

cd "$(dirname "$0")/.."

# ===== 환경변수 기본값 =====
ROOT_OUT=${ROOT_OUT:?"ROOT_OUT 필요"}
PRESERVE_METHOD=${PRESERVE_METHOD:?"PRESERVE_METHOD 필요 (refdistill | gift_synthetic)"}
SELECTION_METHOD=${SELECTION_METHOD:?"SELECTION_METHOD 필요"}

LAMBDA_DISTILL=${LAMBDA_DISTILL:-1.0}
LAMBDA_WC=${LAMBDA_WC:-0.0}

REFERENCE_NAME=${REFERENCE_NAME:-"imagenet"}
REFERENCE_ROOT=${REFERENCE_ROOT:-""}
MAX_REFERENCE_IMAGES=${MAX_REFERENCE_IMAGES:-10000}
DISTILL_NUM_IMG=${DISTILL_NUM_IMG:-64}
DISTILL_NUM_CLS=${DISTILL_NUM_CLS:-64}

GIFT_SYNTHETIC_ROOT=${GIFT_SYNTHETIC_ROOT:-""}
GIFT_METADATA=${GIFT_METADATA:-""}
GIFT_ALPHA=${GIFT_ALPHA:-1.0}
GIFT_BETA=${GIFT_BETA:-0.25}
GIFT_BATCH_SIZE=${GIFT_BATCH_SIZE:-64}
GIFT_MAX_IMAGES=${GIFT_MAX_IMAGES:-0}
USE_AWC=${USE_AWC:-"off"}
GIFT_LAMBDA_AWC=${GIFT_LAMBDA_AWC:-0.1}

SEED_START=${SEED_START:-0}
NUM_SAMPLE_SEEDS=${NUM_SAMPLE_SEEDS:-100}
PARALLEL=${PARALLEL:-1}

TARGET_NAME=${TARGET_NAME:-"texture"}
TARGET_ROOT=${TARGET_ROOT:-"/data/datasets/texture"}
PRESERVE_SPECS=${PRESERVE_SPECS:-"imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test"}

GPU_ID=${GPU_ID:-0}
EPOCHS=${EPOCHS:-50}
EVAL_EVERY=${EVAL_EVERY:-5}
SHOTS_LIST=${SHOTS_LIST:-"1 2 4 8 16"}
SELECTION_CANDIDATE_FACTOR=${SELECTION_CANDIDATE_FACTOR:-2}

# preserve method별 필수 인자 검증
if [ "${PRESERVE_METHOD}" = "refdistill" ]; then
    if [ -z "${REFERENCE_ROOT}" ]; then
        echo "[error] PRESERVE_METHOD=refdistill 인데 REFERENCE_ROOT가 비어 있음" >&2
        exit 2
    fi
elif [ "${PRESERVE_METHOD}" = "gift_synthetic" ]; then
    if [ -z "${GIFT_SYNTHETIC_ROOT}" ]; then
        echo "[error] PRESERVE_METHOD=gift_synthetic 인데 GIFT_SYNTHETIC_ROOT가 비어 있음" >&2
        exit 2
    fi
else
    echo "[error] PRESERVE_METHOD는 refdistill | gift_synthetic 중 하나여야 함 (현재=${PRESERVE_METHOD})" >&2
    exit 2
fi

# USE_AWC -> effective gift_lambda_awc 결정 (gift_synthetic에서만 의미 있음)
EFFECTIVE_GIFT_LAMBDA_AWC="0.0"
USE_AWC_FLAG=""
if [ "${PRESERVE_METHOD}" = "gift_synthetic" ]; then
    if [ "${USE_AWC}" = "on" ]; then
        USE_AWC_FLAG="--gift-use-awc"
        EFFECTIVE_GIFT_LAMBDA_AWC="${GIFT_LAMBDA_AWC}"
    elif [ "${USE_AWC}" = "off" ]; then
        USE_AWC_FLAG=""
        EFFECTIVE_GIFT_LAMBDA_AWC="0.0"
    else
        echo "[error] USE_AWC는 on/off 중 하나여야 함 (현재=${USE_AWC})" >&2
        exit 2
    fi
fi

mkdir -p "${ROOT_OUT}"

echo "===== run_coop_gift_one_config.sh ====="
echo "ROOT_OUT             : ${ROOT_OUT}"
echo "PRESERVE_METHOD      : ${PRESERVE_METHOD}"
echo "SELECTION_METHOD     : ${SELECTION_METHOD}"
echo "LAMBDA_DISTILL       : ${LAMBDA_DISTILL} (refdistill 경로용)"
echo "LAMBDA_WC            : ${LAMBDA_WC} (raw WC)"
if [ "${PRESERVE_METHOD}" = "refdistill" ]; then
    echo "REFERENCE_ROOT       : ${REFERENCE_ROOT}"
    echo "MAX_REFERENCE_IMAGES : ${MAX_REFERENCE_IMAGES}"
    echo "DISTILL_NUM_IMG/CLS  : ${DISTILL_NUM_IMG} / ${DISTILL_NUM_CLS}"
else
    echo "GIFT_SYNTHETIC_ROOT  : ${GIFT_SYNTHETIC_ROOT}"
    echo "GIFT_ALPHA/BETA      : ${GIFT_ALPHA} / ${GIFT_BETA}"
    echo "GIFT_BATCH_SIZE      : ${GIFT_BATCH_SIZE}"
    echo "USE_AWC              : ${USE_AWC} (effective gift_lambda_awc=${EFFECTIVE_GIFT_LAMBDA_AWC})"
    echo "GIFT_MAX_IMAGES      : ${GIFT_MAX_IMAGES}"
fi
echo "SEED_START           : ${SEED_START}"
echo "NUM_SAMPLE_SEEDS     : ${NUM_SAMPLE_SEEDS}"
echo "PARALLEL             : ${PARALLEL}"
echo "TARGET_NAME / ROOT   : ${TARGET_NAME} / ${TARGET_ROOT}"
echo "PRESERVE_SPECS       : ${PRESERVE_SPECS}"
echo "GPU_ID               : ${GPU_ID}"
echo "EPOCHS / EVAL_EVERY  : ${EPOCHS} / ${EVAL_EVERY}"
echo "SHOTS_LIST           : ${SHOTS_LIST}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

SEED_END=$(( SEED_START + NUM_SAMPLE_SEEDS - 1 ))
batch_num=0

for batch_start in $(seq "${SEED_START}" "${PARALLEL}" "${SEED_END}"); do
    batch_end=$(( batch_start + PARALLEL - 1 ))
    if (( batch_end > SEED_END )); then
        batch_end=${SEED_END}
    fi

    batch_num=$(( batch_num + 1 ))
    echo ""
    echo "===== batch ${batch_num}: seed ${batch_start} ~ ${batch_end} 시작 ====="

    pids=()
    for sample_seed in $(seq "${batch_start}" "${batch_end}"); do
        OUT_DIR="${ROOT_OUT}/sample_seed_${sample_seed}"

        if [ -f "${OUT_DIR}/results.json" ]; then
            echo "  [skip] 이미 완료: sample_seed=${sample_seed}"
            continue
        fi

        mkdir -p "${OUT_DIR}"
        echo "  [start] sample_seed=${sample_seed} -> ${OUT_DIR}"

        # ----- preserve method 별 인자 구성 -----
        EXTRA_FLAGS=()
        if [ "${PRESERVE_METHOD}" = "refdistill" ]; then
            EXTRA_FLAGS+=(
                --reference-name "${REFERENCE_NAME}"
                --reference-root "${REFERENCE_ROOT}"
                --max-reference-images "${MAX_REFERENCE_IMAGES}"
                --reference-precompute-seed 0
                --lambda-distill "${LAMBDA_DISTILL}"
                --distill-num-image-samples "${DISTILL_NUM_IMG}"
                --distill-num-class-samples "${DISTILL_NUM_CLS}"
            )
        else
            EXTRA_FLAGS+=(
                --gift-synthetic-root "${GIFT_SYNTHETIC_ROOT}"
                --gift-alpha "${GIFT_ALPHA}"
                --gift-beta "${GIFT_BETA}"
                --gift-batch-size "${GIFT_BATCH_SIZE}"
                --gift-max-images "${GIFT_MAX_IMAGES}"
                --gift-precompute-seed 0
                --gift-num-workers 2
                --gift-lambda-awc "${EFFECTIVE_GIFT_LAMBDA_AWC}"
            )
            if [ -n "${GIFT_METADATA}" ]; then
                EXTRA_FLAGS+=(--gift-metadata "${GIFT_METADATA}")
            fi
            if [ -n "${USE_AWC_FLAG}" ]; then
                EXTRA_FLAGS+=("${USE_AWC_FLAG}")
            fi
        fi

        # shellcheck disable=SC2086
        python train_blue_target_coop_gift.py \
            --preserve-method "${PRESERVE_METHOD}" \
            --target-name "${TARGET_NAME}" \
            --target-root "${TARGET_ROOT}" \
            --preserve ${PRESERVE_SPECS} \
            --model ViT-B-32-quickgelu \
            --pretrained openai \
            --shots-list ${SHOTS_LIST} \
            --n-ctx 16 \
            --epochs "${EPOCHS}" \
            --eval-every "${EVAL_EVERY}" \
            --lr 0.002 \
            --batch-size 128 \
            --num-workers 2 \
            --device cuda \
            --seed 42 \
            --prompt-seed 777 \
            --sample-seed "${sample_seed}" \
            --selection-method "${SELECTION_METHOD}" \
            --selection-candidate-factor "${SELECTION_CANDIDATE_FACTOR}" \
            --lambda-wc "${LAMBDA_WC}" \
            "${EXTRA_FLAGS[@]}" \
            --cache-features \
            --output-dir "${OUT_DIR}" \
            > "${OUT_DIR}/run.log" 2>&1 &

        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "${pid}" || echo "  [경고] PID=${pid} 비정상 종료"
    done

    echo "===== batch ${batch_num} 완료 ====="
done

echo ""
echo "===== ${ROOT_OUT} 의 모든 seed 종료 ====="
