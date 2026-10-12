#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

python train_blue_target_cocoop.py \
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
  --batch-size 4 \
  --num-workers 4 \
  --device cuda \
  --seed 42 \
  --prompt-seed 777 \
  --sample-seed ${SAMPLE_SEED:-42} \
  --cache-features \
  --cocoop-class-chunk-size 128 \
  --cocoop-eval-batch-size 1 \
  --output-dir runs/cocoop_texture_preserve
