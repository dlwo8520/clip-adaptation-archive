#!/usr/bin/env bash
set -euo pipefail

# 이 스크립트 위치 기준으로 프로젝트 루트로 이동
cd "$(dirname "$0")/.."

# 기본 GPU는 1번
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

ROOT_OUT="runs/blue_texture_preserve_100seeds"

SEED_START=${SEED_START:-0}
NUM_SAMPLE_SEEDS=${NUM_SAMPLE_SEEDS:-100}

echo "===== 100 sample-seed texture experiment ====="
echo "project root: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "ROOT_OUT=${ROOT_OUT}"
echo "SEED_START=${SEED_START}"
echo "NUM_SAMPLE_SEEDS=${NUM_SAMPLE_SEEDS}"

mkdir -p "${ROOT_OUT}"

for sample_seed in $(seq ${SEED_START} $((SEED_START + NUM_SAMPLE_SEEDS - 1))); do
    OUT_DIR="${ROOT_OUT}/sample_seed_${sample_seed}"
    mkdir -p "${OUT_DIR}"

    echo ""
    echo "===== sample_seed=${sample_seed} 시작 ====="
    echo "output_dir=${OUT_DIR}"

    python train_blue_target_coop.py \
      --target-name texture \
      --target-root /data/datasets/texture \
      --preserve imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test \
      --model ViT-B-32-quickgelu \
      --pretrained openai \
      --shots-list 1 2 4 8 16 \
      --n-ctx 16 \
      --epochs 50 \
      --eval-every 5 \
      --lr 0.002 \
      --batch-size 128 \
      --num-workers 4 \
      --device cuda \
      --seed 42 \
      --prompt-seed 777 \
      --sample-seed "${sample_seed}" \
      --cache-features \
      --output-dir "${OUT_DIR}" \
      2>&1 | tee "${OUT_DIR}/run.log"

    echo "===== sample_seed=${sample_seed} 종료 ====="
done

echo "===== 전체 종료 ====="
echo "results root: ${ROOT_OUT}"

