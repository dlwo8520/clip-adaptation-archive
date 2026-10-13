#!/usr/bin/env bash
# 빠른 sanity check용 smoke test.
# - sample_seed=0, shots=[1, 2], epochs=2, eval_every=1
# - text_drift selection이 정상 작동하는지 + text_drift_scores.pt가 저장되는지 확인.

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUT_DIR=${OUT_DIR:-/tmp/textdrift_smoke}
VOCAB_SIZE=${VOCAB_SIZE:-100}   # smoke test에서는 vocab을 작게

mkdir -p "${OUT_DIR}"

echo "===== text_drift smoke test ====="
echo "OUT_DIR    : ${OUT_DIR}"
echo "VOCAB_SIZE : ${VOCAB_SIZE}"

python train_blue_target_coop_zscl.py \
  --target-name texture \
  --target-root /data/datasets/texture \
  --preserve imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test \
  --model ViT-B-32-quickgelu --pretrained openai \
  --shots-list 1 2 --epochs 2 --eval-every 1 \
  --n-ctx 16 \
  --lr 0.002 --batch-size 128 --num-workers 2 --device cuda \
  --seed 42 --prompt-seed 777 --sample-seed 0 \
  --selection-method text_drift \
  --selection-candidate-factor 2 \
  --text-drift-vocab-size "${VOCAB_SIZE}" \
  --text-drift-vocab-seed 12345 \
  --text-drift-step-size 0.01 \
  --text-drift-templates default \
  --text-drift-chunk-size 1024 \
  --cache-features \
  --output-dir "${OUT_DIR}"

echo ""
echo "===== smoke test outputs ====="
ls -la "${OUT_DIR}"
echo ""
echo "selection_meta.json:"
cat "${OUT_DIR}/selection_meta.json" 2>/dev/null || echo "  (missing)"
