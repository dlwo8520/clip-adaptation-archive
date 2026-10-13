#!/usr/bin/env bash
set -euo pipefail

NUM_SAMPLE_SEEDS="${NUM_SAMPLE_SEEDS:-100}"

for SAMPLE_SEED in $(seq 0 $((NUM_SAMPLE_SEEDS - 1))); do
  OUT_DIR="runs/zscl_texture_uncertain_only_gpu1/coop/teacher_uncertain/sample_seed_${SAMPLE_SEED}"

  if [ -f "${OUT_DIR}/results.json" ]; then
    echo "[SKIP] already done: sample_seed_${SAMPLE_SEED}"
    continue
  fi

  echo "===== CoOp teacher_uncertain / sample_seed=${SAMPLE_SEED} ====="

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
    --selection-method teacher_uncertain \
    --selection-candidate-factor 2 \
    --cache-features \
    --output-dir "${OUT_DIR}"
done
