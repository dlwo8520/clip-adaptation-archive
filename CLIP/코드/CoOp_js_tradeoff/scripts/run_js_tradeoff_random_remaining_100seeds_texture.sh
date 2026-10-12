#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

ROOT_OUT=${ROOT_OUT:-runs/js_tradeoff_random_remaining_100seeds}
SEED_START=${SEED_START:-0}
NUM_SAMPLE_SEEDS=${NUM_SAMPLE_SEEDS:-100}
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

echo "===== JS trade-off random remaining-data few-shot, 100 seeds ====="
echo "project root: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "ROOT_OUT=${ROOT_OUT}"
echo "SEED_START=${SEED_START}"
echo "NUM_SAMPLE_SEEDS=${NUM_SAMPLE_SEEDS}"
echo "SHOTS_LIST=${SHOTS_LIST}"

mkdir -p "${ROOT_OUT}"

for sample_seed in $(seq "${SEED_START}" $((SEED_START + NUM_SAMPLE_SEEDS - 1))); do
  OUT_DIR="${ROOT_OUT}/sample_seed_${sample_seed}"
  mkdir -p "${OUT_DIR}"

  echo ""
  echo "===== random sample_seed=${sample_seed} ====="
  echo "output_dir=${OUT_DIR}"

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
    --sample-seed "${sample_seed}" \
    --selection-mode random \
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

  echo "===== random sample_seed=${sample_seed} done ====="
done

echo "===== all done ====="
echo "results root: ${ROOT_OUT}"
