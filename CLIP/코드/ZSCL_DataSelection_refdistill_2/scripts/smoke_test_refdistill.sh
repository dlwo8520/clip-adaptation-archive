#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 빠른 smoke test:
# - 6개 config 중 하나(teacher_margin + WC on)만 골라서 sample_seed 0 하나만 짧게 돌려봄.
# - epochs=2, eval_every=2, shots-list="1 4"로 매우 짧게.
# - reference image도 1000장만 사용.
# 학습이 끝까지 도는지 + 모든 loss component가 정상 계산되는지 확인할 때 사용.
#
# 통과 후에 본 실행은 scripts/launch_6_configs_parallel.sh 로.

set -uo pipefail

cd "$(dirname "$0")/.."

BASE_OUT=${BASE_OUT:-"runs/smoke_coop_refdistill"}
TARGET_NAME=${TARGET_NAME:-"texture"}
TARGET_ROOT=${TARGET_ROOT:-"/data/datasets/texture"}
REFERENCE_NAME=${REFERENCE_NAME:-"imagenet"}
REFERENCE_ROOT=${REFERENCE_ROOT:?"REFERENCE_ROOT 필요"}
PRESERVE_SPECS=${PRESERVE_SPECS:-"imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test"}
GPU_ID=${GPU_ID:-0}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

OUT_DIR="${BASE_OUT}/smoke_seed_0"
mkdir -p "${OUT_DIR}"

echo "===== smoke test 시작 ====="
echo "OUT_DIR        : ${OUT_DIR}"
echo "TARGET_ROOT    : ${TARGET_ROOT}"
echo "REFERENCE_ROOT : ${REFERENCE_ROOT}"
echo "GPU_ID         : ${GPU_ID}"

# shellcheck disable=SC2086
python train_blue_target_coop_zscl_refdistill.py \
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
    --reference-name "${REFERENCE_NAME}" \
    --reference-root "${REFERENCE_ROOT}" \
    --max-reference-images 1000 \
    --reference-precompute-seed 0 \
    --lambda-distill 1.0 \
    --distill-num-image-samples 32 \
    --distill-num-class-samples 32 \
    --lambda-wc 0.1 \
    --cache-features \
    --output-dir "${OUT_DIR}"

echo ""
echo "===== smoke test 완료 ====="
echo "결과 확인: ${OUT_DIR}/results.json"
