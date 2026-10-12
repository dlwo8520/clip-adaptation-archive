#!/bin/bash
# GPU 1번 smoke test
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
python -m py_compile src/utils.py
python -m py_compile src/data.py
python -m py_compile src/coop_model.py
python -m py_compile src/eval.py
python -m py_compile train_blue_target_coop.py
python -m py_compile plot_blue_target_results.py
python train_blue_target_coop.py \
  --target-name cifar10_smoke \
  --target-train /data/datasets/cifar10/train \
  --target-test /data/datasets/cifar10/test \
  --preserve cifar100:/data/datasets/cifar100/test \
  --shots-list 1 \
  --epochs 1 \
  --eval-every 1 \
  --batch-size 64 \
  --num-workers 2 \
  --device cuda \
  --seed 42 \
  --prompt-seed 777 \
  --max-eval-samples 256 \
  --cache-features \
  --output-dir runs/smoke_gpu1
