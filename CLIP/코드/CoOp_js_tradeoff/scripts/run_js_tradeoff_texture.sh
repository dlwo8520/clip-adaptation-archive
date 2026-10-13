#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

SELECTION_MODE=${SELECTION_MODE:-js_q1}
SAMPLE_SEED=${SAMPLE_SEED:-0}
OUT_DIR=${OUT_DIR:-runs/js_tradeoff_texture/${SELECTION_MODE}_seed_${SAMPLE_SEED}}
SHOTS_LIST=${SHOTS_LIST:-"1 2 4 8 16"}
ANCHOR_RATIO=${ANCHOR_RATIO:-0.2}
ANCHOR_PER_CLASS=${ANCHOR_PER_CLASS:-0}
ANCHOR_DIST_MODE=${ANCHOR_DIST_MODE:-global}
ANCHOR_MODE=${ANCHOR_MODE:-global}
NUM_JS_BINS=${NUM_JS_BINS:-4}
BALANCE_MODE=${BALANCE_MODE:-true_label}
EXCLUDE_ANCHOR=${EXCLUDE_ANCHOR:-1}

EXCLUDE_FLAG=""
if [[ "${EXCLUDE_ANCHOR}" == "1" ]]; then
  EXCLUDE_FLAG="--exclude-anchor"
fi

echo "===== JS target-preservation trade-off CoOp ====="
echo "project root: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SELECTION_MODE=${SELECTION_MODE}"
echo "SAMPLE_SEED=${SAMPLE_SEED}"
echo "SHOTS_LIST=${SHOTS_LIST}"
echo "NUM_JS_BINS=${NUM_JS_BINS}"
echo "OUT_DIR=${OUT_DIR}"

mkdir -p "${OUT_DIR}"

python train_js_tradeoff_coop.py \
  --target-name texture \
  --target-root /data/datasets/texture \
  --preserve imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test \
  --model ViT-B-32-quickgelu \
  --pretrained openai \
  --shots-list ${SHOTS_LIST} \
  --n-ctx 16 \
  --epochs 50 \
  --eval-every 5 \
  --lr 0.002 \
  --batch-size 128 \
  --num-workers 4 \
  --device cuda \
  --seed 42 \
  --prompt-seed 777 \
  --sample-seed "${SAMPLE_SEED}" \
  --selection-mode "${SELECTION_MODE}" \
  --anchor-ratio "${ANCHOR_RATIO}" \
  --anchor-per-class "${ANCHOR_PER_CLASS}" \
  --anchor-mode "${ANCHOR_MODE}" \
  --anchor-dist-mode "${ANCHOR_DIST_MODE}" \
  --num-js-bins "${NUM_JS_BINS}" \
  --balance-mode "${BALANCE_MODE}" \
  ${EXCLUDE_FLAG} \
  --cache-features \
  --output-dir "${OUT_DIR}" \
  2>&1 | tee "${OUT_DIR}/run.log"

echo "===== done ====="
echo "out_dir: ${OUT_DIR}"
