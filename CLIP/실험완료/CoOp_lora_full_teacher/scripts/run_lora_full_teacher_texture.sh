#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

OUT_DIR=${OUT_DIR:-runs/lora_full_teacher_twoteacher_texture}
LORA_TARGET=${LORA_TARGET:-visual}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
LORA_EPOCHS=${LORA_EPOCHS:-20}
LORA_LR=${LORA_LR:-1e-4}

echo "===== LoRA full-data teacher -> two-teacher selected Student CoOp ====="
echo "project root: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OUT_DIR=${OUT_DIR}"
echo "LORA_TARGET=${LORA_TARGET}"
echo "LORA_RANK=${LORA_RANK}"
echo "LORA_ALPHA=${LORA_ALPHA}"
echo "LORA_DROPOUT=${LORA_DROPOUT}"
echo "LORA_EPOCHS=${LORA_EPOCHS}"
echo "LORA_LR=${LORA_LR}"

python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("visible device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("visible device 0:", torch.cuda.get_device_name(0))
PY

python train_lora_full_teacher_twoteacher_coop.py \
  --target-name texture \
  --target-root /data/datasets/texture \
  --split-test-ratio 0.5 \
  --preserve imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test \
  --model ViT-B-32-quickgelu \
  --pretrained openai \
  --lora-target "${LORA_TARGET}" \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --lora-epochs "${LORA_EPOCHS}" \
  --lora-lr "${LORA_LR}" \
  --lora-batch-size 128 \
  --student-shots 1 2 4 8 16 \
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
  --require-both-correct \
  --cache-features \
  --output-dir "${OUT_DIR}"

echo "===== done ====="
echo "results: ${OUT_DIR}/results.json"
echo "curves: ${OUT_DIR}/curves.jsonl"
echo "teacher eval: ${OUT_DIR}/lora_teacher/teacher_eval.json"
