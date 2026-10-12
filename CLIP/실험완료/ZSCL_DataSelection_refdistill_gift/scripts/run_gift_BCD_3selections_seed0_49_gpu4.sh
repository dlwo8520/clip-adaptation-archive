#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# ===== 공통 설정 =====
GPU_ID=3
SEED_START=0
NUM_SAMPLE_SEEDS=50
PARALLEL=1

TARGET_NAME=texture
TARGET_ROOT=/data/datasets/texture
GIFT_SYNTHETIC_ROOT=${RESEARCH_WORKSPACE}/projects/ZSCL_DataSelection_refdistill_gift/data/gift_synthetic_dtd
PRESERVE_SPECS="imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test"

EPOCHS=50
EVAL_EVERY=5
SHOTS_LIST="1 2 4 8 16"

BASE_OUT=runs/coop_gift_BCD_3selection_seed0_49_gpu4

# 데이터 선택 방법 3개
SELECTION_METHODS=(
  # teacher_confident
  teacher_entropy
  teacher_margin
)

# B, C, D만 실행
CONFIG_TAGS=(
  B__synthetic_zscl_like
  C__gift_synthetic
  D__gift_synthetic_awc
)

mkdir -p "${BASE_OUT}/logs"

for SELECTION_METHOD in "${SELECTION_METHODS[@]}"; do
  for CONFIG_TAG in "${CONFIG_TAGS[@]}"; do

    case "${CONFIG_TAG}" in
      B__synthetic_zscl_like)
        GIFT_ALPHA=1.0
        GIFT_BETA=0.0
        USE_AWC=off
        GIFT_LAMBDA_AWC=0.0
        ;;

      C__gift_synthetic)
        GIFT_ALPHA=1.0
        GIFT_BETA=0.25
        USE_AWC=off
        GIFT_LAMBDA_AWC=0.0
        ;;

      D__gift_synthetic_awc)
        GIFT_ALPHA=1.0
        GIFT_BETA=0.25
        USE_AWC=on
        GIFT_LAMBDA_AWC=0.1
        ;;

      *)
        echo "[error] unknown config: ${CONFIG_TAG}"
        exit 2
        ;;
    esac

    ROOT_OUT="${BASE_OUT}/${SELECTION_METHOD}/${CONFIG_TAG}"
    LOG_PATH="${BASE_OUT}/logs/${SELECTION_METHOD}__${CONFIG_TAG}.log"

    echo ""
    echo "============================================================"
    echo "[START] selection=${SELECTION_METHOD} config=${CONFIG_TAG}"
    echo "ROOT_OUT=${ROOT_OUT}"
    echo "LOG_PATH=${LOG_PATH}"
    echo "============================================================"

    ROOT_OUT="${ROOT_OUT}" \
    PRESERVE_METHOD=gift_synthetic \
    SELECTION_METHOD="${SELECTION_METHOD}" \
    GIFT_SYNTHETIC_ROOT="${GIFT_SYNTHETIC_ROOT}" \
    GIFT_ALPHA="${GIFT_ALPHA}" \
    GIFT_BETA="${GIFT_BETA}" \
    USE_AWC="${USE_AWC}" \
    GIFT_LAMBDA_AWC="${GIFT_LAMBDA_AWC}" \
    SEED_START="${SEED_START}" \
    NUM_SAMPLE_SEEDS="${NUM_SAMPLE_SEEDS}" \
    PARALLEL="${PARALLEL}" \
    TARGET_NAME="${TARGET_NAME}" \
    TARGET_ROOT="${TARGET_ROOT}" \
    PRESERVE_SPECS="${PRESERVE_SPECS}" \
    GPU_ID="${GPU_ID}" \
    EPOCHS="${EPOCHS}" \
    EVAL_EVERY="${EVAL_EVERY}" \
    SHOTS_LIST="${SHOTS_LIST}" \
    bash scripts/run_coop_gift_one_config.sh \
      > "${LOG_PATH}" 2>&1

    echo "[DONE] selection=${SELECTION_METHOD} config=${CONFIG_TAG}"

  done
done

echo ""
echo "All requested runs finished."
