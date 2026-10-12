#!/usr/bin/env bash
set -euo pipefail

# 빠른 동작 확인용. 실제 논문용 결과로 쓰면 안 됨.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

python train_blue_target_coop_zscl.py \
  --target-name texture \
  --target-root /data/datasets/texture \
  --preserve imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test \
  --model ViT-B-32-quickgelu \
  --pretrained openai \
  --shots-list 1 \
  --n-ctx 16 \
  --epochs 2 \
  --eval-every 1 \
  --lr 0.002 \
  --batch-size 128 \
  --num-workers 2 \
  --device cuda \
  --seed 42 \
  --prompt-seed 777 \
  --sample-seed 0 \
  --selection-method teacher_entropy \
  --selection-candidate-factor 2 \
  --cache-features \
  --max-eval-samples 512 \
  --output-dir runs/smoke_texture_entropy_only/coop/teacher_entropy/sample_seed_0
