#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# GIFT-inspired synthetic-pair distillation 빠른 smoke test.
#
# 두 단계:
#   1) (필요 시) 매우 작은 synthetic 데이터셋을 placeholder 모드로 생성
#      (Stable Diffusion 불필요).
#   2) target=DTD, preserve=imagenet+cifar100, gift_synthetic + AWC 켠 상태로
#      sample_seed 0 하나만 epochs=2, shots-list="1 4"로 매우 짧게 학습.
#
# 통과 후 본 실행은 scripts/launch_gift_configs.sh 로.

set -uo pipefail

cd "$(dirname "$0")/.."

# ===== 사용자 설정 (대부분 기본값으로 충분) =====
BASE_OUT=${BASE_OUT:-"runs/smoke_coop_gift"}
TARGET_NAME=${TARGET_NAME:-"texture"}
TARGET_ROOT=${TARGET_ROOT:?"TARGET_ROOT 필요 (e.g. /data/datasets/texture)"}
PRESERVE_SPECS=${PRESERVE_SPECS:-"imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test"}
GPU_ID=${GPU_ID:-0}

# synthetic 데이터셋: 학습용 입력. 미리 만들었으면 그 경로를 SYNTHETIC_ROOT로 지정.
SYNTHETIC_ROOT=${SYNTHETIC_ROOT:-"${BASE_OUT}/synthetic_gift_smoke"}
BASE_CLASSNAMES_FILE=${BASE_CLASSNAMES_FILE:-"gift_classnames/imagenet_classes_subset.txt"}
TARGET_CLASSNAMES_FILE=${TARGET_CLASSNAMES_FILE:-"gift_classnames/dtd_classes.txt"}
TARGET_SOURCE_TAG=${TARGET_SOURCE_TAG:-"target_dtd"}
NUM_PER_CLASS=${NUM_PER_CLASS:-2}
SYNTH_MODE=${SYNTH_MODE:-"placeholder"}   # "placeholder" or "stable_diffusion"
SYNTH_OVERWRITE=${SYNTH_OVERWRITE:-"1"}    # 1이면 기존 metadata 덮어쓰기

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p "${BASE_OUT}"

echo "===== smoke test (GIFT) 시작 ====="
echo "BASE_OUT              : ${BASE_OUT}"
echo "TARGET_ROOT           : ${TARGET_ROOT}"
echo "SYNTHETIC_ROOT        : ${SYNTHETIC_ROOT}"
echo "BASE_CLASSNAMES_FILE  : ${BASE_CLASSNAMES_FILE}"
echo "TARGET_CLASSNAMES_FILE: ${TARGET_CLASSNAMES_FILE}"
echo "NUM_PER_CLASS         : ${NUM_PER_CLASS}"
echo "SYNTH_MODE            : ${SYNTH_MODE}"
echo "GPU_ID                : ${GPU_ID}"

# ===== 1) synthetic 데이터셋 생성 =====
GEN_FLAGS=()
if [ "${SYNTH_OVERWRITE}" = "1" ]; then
    GEN_FLAGS+=(--overwrite)
fi

echo ""
echo "===== [1/2] synthetic 데이터셋 생성 ====="
python scripts/generate_synthetic_dataset.py \
    --out-dir "${SYNTHETIC_ROOT}" \
    --base-classnames-file "${BASE_CLASSNAMES_FILE}" \
    --target-classnames-file "${TARGET_CLASSNAMES_FILE}" \
    --target-source-tag "${TARGET_SOURCE_TAG}" \
    --num-per-class "${NUM_PER_CLASS}" \
    --mode "${SYNTH_MODE}" \
    "${GEN_FLAGS[@]}"

echo ""
echo "===== [2/2] 학습 smoke test ====="
OUT_DIR="${BASE_OUT}/smoke_seed_0"
mkdir -p "${OUT_DIR}"

python train_blue_target_coop_gift.py \
    --preserve-method gift_synthetic \
    --target-name "${TARGET_NAME}" \
    --target-root "${TARGET_ROOT}" \
    --preserve ${PRESERVE_SPECS} \
    --model ViT-B-32-quickgelu \
    --pretrained openai \
    --shots-list 1 4 \
    --n-ctx 16 \
    --epochs 2 \
    --eval-every 2 \
    --lr 0.002 \
    --batch-size 64 \
    --num-workers 2 \
    --device cuda \
    --seed 42 \
    --prompt-seed 777 \
    --sample-seed 0 \
    --selection-method teacher_margin \
    --selection-candidate-factor 2 \
    --gift-synthetic-root "${SYNTHETIC_ROOT}" \
    --gift-alpha 1.0 \
    --gift-beta 0.25 \
    --gift-batch-size 16 \
    --gift-num-workers 2 \
    --gift-use-awc \
    --gift-lambda-awc 0.1 \
    --gift-precompute-seed 0 \
    --gift-verify-paths \
    --cache-features \
    --output-dir "${OUT_DIR}"

echo ""
echo "===== smoke test 완료 ====="
echo "결과 확인: ${OUT_DIR}/results.json"
echo "          ${OUT_DIR}/curves.jsonl"
echo "          ${OUT_DIR}/gift_distill_meta.json"
echo ""
echo "  curves.jsonl 의 train.train_loss_cd_raw, train_loss_ita_raw, train_loss_awc_raw 가"
echo "  실제로 값이 찍히는지 확인하세요. 0만 찍힌다면 무엇인가 잘못된 것."
