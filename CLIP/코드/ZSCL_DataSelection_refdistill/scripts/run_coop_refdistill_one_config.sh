#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 한 configuration (선택된 selection_method × WC on/off)에 대해
# sample seed 0..N-1 까지 실행함.
#
# 호출 변수:
#   ROOT_OUT          : 출력 루트 (e.g. runs/coop_refdistill_6configs/<config_tag>)
#   SELECTION_METHOD  : teacher_margin | teacher_uncertain | teacher_confident
#   USE_WC            : "on" 또는 "off"
#   LAMBDA_DISTILL    : reference distillation 계수 (기본 1.0)
#   LAMBDA_WC         : weight consolidation 계수 (USE_WC=on일 때 사용, 기본 0.1)
#   SEED_START        : 시작 seed (기본 0)
#   NUM_SAMPLE_SEEDS  : seed 개수 (기본 100)
#   PARALLEL          : 동일 config 내 동시 seed 실행 수 (기본 1)
#   TARGET_NAME       : target dataset 이름 (기본 texture)
#   TARGET_ROOT       : target dataset 경로
#   REFERENCE_NAME    : reference dataset 이름 (기본 imagenet)
#   REFERENCE_ROOT    : reference dataset 경로 (필수)
#   PRESERVE_SPECS    : preserve dataset spec string (기본 imagenet+cifar100)
#   GPU_ID            : 사용할 GPU id (기본 0)
#   EPOCHS            : 학습 epoch (기본 50)
#   EVAL_EVERY        : 평가 주기 (기본 5)
#   SHOTS_LIST        : "1 2 4 8 16"
#
# 결과 구조:
#   ${ROOT_OUT}/sample_seed_<seed>/{args.json, curves.jsonl, results.json, ...}

set -uo pipefail

cd "$(dirname "$0")/.."

# ===== 환경변수 기본값 =====
ROOT_OUT=${ROOT_OUT:?"ROOT_OUT 필요"}
SELECTION_METHOD=${SELECTION_METHOD:?"SELECTION_METHOD 필요"}
USE_WC=${USE_WC:-"off"}
LAMBDA_DISTILL=${LAMBDA_DISTILL:-1.0}
LAMBDA_WC=${LAMBDA_WC:-0.1}
SEED_START=${SEED_START:-0}
NUM_SAMPLE_SEEDS=${NUM_SAMPLE_SEEDS:-100}
PARALLEL=${PARALLEL:-1}

TARGET_NAME=${TARGET_NAME:-"texture"}
TARGET_ROOT=${TARGET_ROOT:-"/data/datasets/texture"}
REFERENCE_NAME=${REFERENCE_NAME:-"imagenet"}
REFERENCE_ROOT=${REFERENCE_ROOT:?"REFERENCE_ROOT 필요 (ImageFolder 형식의 reference 이미지 경로)"}
PRESERVE_SPECS=${PRESERVE_SPECS:-"imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test"}

GPU_ID=${GPU_ID:-0}
EPOCHS=${EPOCHS:-50}
EVAL_EVERY=${EVAL_EVERY:-5}
SHOTS_LIST=${SHOTS_LIST:-"1 2 4 8 16"}
MAX_REFERENCE_IMAGES=${MAX_REFERENCE_IMAGES:-10000}
DISTILL_NUM_IMG=${DISTILL_NUM_IMG:-64}
DISTILL_NUM_CLS=${DISTILL_NUM_CLS:-64}
SELECTION_CANDIDATE_FACTOR=${SELECTION_CANDIDATE_FACTOR:-2}

# WC on/off에 따라 실제 lambda_wc 결정
if [ "${USE_WC}" = "on" ]; then
    EFFECTIVE_LAMBDA_WC="${LAMBDA_WC}"
elif [ "${USE_WC}" = "off" ]; then
    EFFECTIVE_LAMBDA_WC="0.0"
else
    echo "[error] USE_WC는 on/off 둘 중 하나여야 함. 현재=${USE_WC}" >&2
    exit 2
fi

mkdir -p "${ROOT_OUT}"

echo "===== run_coop_refdistill_one_config.sh ====="
echo "ROOT_OUT             : ${ROOT_OUT}"
echo "SELECTION_METHOD     : ${SELECTION_METHOD}"
echo "USE_WC               : ${USE_WC} (effective lambda_wc=${EFFECTIVE_LAMBDA_WC})"
echo "LAMBDA_DISTILL       : ${LAMBDA_DISTILL}"
echo "SEED_START           : ${SEED_START}"
echo "NUM_SAMPLE_SEEDS     : ${NUM_SAMPLE_SEEDS}"
echo "PARALLEL             : ${PARALLEL}"
echo "TARGET_NAME          : ${TARGET_NAME}"
echo "TARGET_ROOT          : ${TARGET_ROOT}"
echo "REFERENCE_NAME       : ${REFERENCE_NAME}"
echo "REFERENCE_ROOT       : ${REFERENCE_ROOT}"
echo "PRESERVE_SPECS       : ${PRESERVE_SPECS}"
echo "GPU_ID               : ${GPU_ID}"
echo "EPOCHS               : ${EPOCHS}"
echo "EVAL_EVERY           : ${EVAL_EVERY}"
echo "SHOTS_LIST           : ${SHOTS_LIST}"
echo "MAX_REFERENCE_IMAGES : ${MAX_REFERENCE_IMAGES}"
echo "DISTILL_NUM_IMG      : ${DISTILL_NUM_IMG}"
echo "DISTILL_NUM_CLS      : ${DISTILL_NUM_CLS}"

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

        # shellcheck disable=SC2086
        python train_blue_target_coop_zscl_refdistill.py \
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
            --reference-name "${REFERENCE_NAME}" \
            --reference-root "${REFERENCE_ROOT}" \
            --max-reference-images "${MAX_REFERENCE_IMAGES}" \
            --reference-precompute-seed 0 \
            --lambda-distill "${LAMBDA_DISTILL}" \
            --distill-num-image-samples "${DISTILL_NUM_IMG}" \
            --distill-num-class-samples "${DISTILL_NUM_CLS}" \
            --lambda-wc "${EFFECTIVE_LAMBDA_WC}" \
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
