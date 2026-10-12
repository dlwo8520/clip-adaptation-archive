#!/bin/bash
# DTD/texture target adaptation 실험
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clip
cd ${RESEARCH_WORKSPACE}/projects/blue_target_coop_project_v2_prompt_seed
export CUDA_VISIBLE_DEVICES=1
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("visible device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("visible device 0:", torch.cuda.get_device_name(0))
PY
python train_blue_target_coop.py \
  --target-name texture \
  --target-root /data/datasets/texture \
  --split-test-ratio 0.5 \
  --preserve imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test \
  --shots-list 1 2 4 8 16 \
  --epochs 50 \
  --eval-every 5 \
  --batch-size 128 \
  --num-workers 4 \
  --device cuda \
  --seed 42 \
  --prompt-seed 777 \
  --cache-features \
  --output-dir runs/blue_texture_preserve
python plot_blue_target_results.py \
  --curves runs/blue_texture_preserve/curves.jsonl \
  --baseline-json runs/blue_texture_preserve/zero_shot_baselines.json \
  --out-dir runs/blue_texture_preserve/figures


