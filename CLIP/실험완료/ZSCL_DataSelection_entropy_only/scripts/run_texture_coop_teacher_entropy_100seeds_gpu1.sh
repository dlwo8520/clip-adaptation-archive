#!/usr/bin/env bash
set -euo pipefail

# Entropy-only CoOp experiment.
# GPU 1을 쓰고 싶으면 기본값 그대로 실행하면 됨.
# 다른 GPU를 쓰려면 예: CUDA_VISIBLE_DEVICES=2 bash scripts/run_texture_coop_teacher_entropy_100seeds_gpu1.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

NUM_SAMPLE_SEEDS="${NUM_SAMPLE_SEEDS:-100}"
CANDIDATE_FACTOR="${CANDIDATE_FACTOR:-2}"

for SAMPLE_SEED in $(seq 0 $((NUM_SAMPLE_SEEDS - 1))); do
  OUT_DIR="runs/zscl_texture_entropy_only_gpu1/coop/teacher_entropy/sample_seed_${SAMPLE_SEED}"

  if [ -f "${OUT_DIR}/results.json" ]; then
    echo "[SKIP] already done: sample_seed_${SAMPLE_SEED}"
    continue
  fi

  echo "===== CoOp teacher_entropy / sample_seed=${SAMPLE_SEED} / CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ====="

  python train_blue_target_coop_zscl.py \
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
    --num-workers 2 \
    --device cuda \
    --seed 42 \
    --prompt-seed 777 \
    --sample-seed "${SAMPLE_SEED}" \
    --selection-method teacher_entropy \
    --selection-candidate-factor "${CANDIDATE_FACTOR}" \
    --cache-features \
    --output-dir "${OUT_DIR}"
done
