#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

OUT_DIR=${OUT_DIR:-runs/twoteacher_texture_diag}
TEACHER_RUNS_ROOT=${TEACHER_RUNS_ROOT:-runs/blue_texture_preserve_100seeds}
LAMBDA_WC=${LAMBDA_WC:-0.0}

echo "===== two-teacher texture diagnostic ====="
echo "project root: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OUT_DIR=${OUT_DIR}"
echo "TEACHER_RUNS_ROOT=${TEACHER_RUNS_ROOT}"
echo "LAMBDA_WC=${LAMBDA_WC}"

python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("visible device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("visible device 0:", torch.cuda.get_device_name(0))
PY

python train_twoteacher_selected_coop.py \
  --target-name texture \
  --target-root /data/datasets/texture \
  --split-test-ratio 0.5 \
  --preserve imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test \
  --teacher-runs-root "${TEACHER_RUNS_ROOT}" \
  --teacher-project-root . \
  --teacher-shots 1 2 4 8 16 \
  --teacher-top-k 3 \
  --student-shots 1 2 4 8 16 \
  --pairing-mode matched \
  --require-both-correct \
  --lambda-clip 1.0 \
  --lambda-teacher 1.0 \
  --lambda-similarity 1.0 \
  --lambda-wc "${LAMBDA_WC}" \
  --model ViT-B-32-quickgelu \
  --pretrained openai \
  --n-ctx 16 \
  --epochs 50 \
  --eval-every 5 \
  --lr 0.002 \
  --batch-size 128 \
  --selection-batch-size 256 \
  --num-workers 4 \
  --device cuda \
  --seed 42 \
  --prompt-seed 777 \
  --selection-seed 2026 \
  --cache-features \
  --output-dir "${OUT_DIR}"

echo "===== done ====="
echo "results: ${OUT_DIR}/results.json"
echo "curves: ${OUT_DIR}/curves.jsonl"
