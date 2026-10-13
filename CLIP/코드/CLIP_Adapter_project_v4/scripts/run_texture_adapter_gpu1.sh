#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python train_blue_target_adapter.py \
  --target-name texture \
  --target-root /data/datasets/texture \
  --preserve imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test \
  --model ViT-B-32-quickgelu \
  --pretrained openai \
  --shots-list 1 2 4 8 16 \
  --adapter-hidden-ratio 0.25 \
  --adapter-alpha-init 0.6 \
  --adapter-seed 777 \
  --epochs 50 \
  --eval-every 5 \
  --lr 0.001 \
  --batch-size 128 \
  --num-workers 4 \
  --device cuda \
  --seed 42 \
  --sample-seed ${SAMPLE_SEED:-42} \
  --cache-features \
  --output-dir runs/adapter_texture_preserve
