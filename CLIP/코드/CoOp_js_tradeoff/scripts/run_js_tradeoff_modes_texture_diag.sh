#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

ROOT_OUT=${ROOT_OUT:-runs/js_tradeoff_texture_diag}
SAMPLE_SEED=${SAMPLE_SEED:-0}
SHOTS_LIST=${SHOTS_LIST:-"1 2 4 8 16"}
MODES=${MODES:-"random anchor_entropy js_q1 js_q2 js_q3 js_q4"}

echo "===== JS trade-off modes diagnostic ====="
echo "project root: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "ROOT_OUT=${ROOT_OUT}"
echo "SAMPLE_SEED=${SAMPLE_SEED}"
echo "SHOTS_LIST=${SHOTS_LIST}"
echo "MODES=${MODES}"

mkdir -p "${ROOT_OUT}"

for mode in ${MODES}; do
  OUT_DIR="${ROOT_OUT}/${mode}"
  echo ""
  echo "===== mode=${mode} ====="
  SELECTION_MODE="${mode}" \
  SAMPLE_SEED="${SAMPLE_SEED}" \
  SHOTS_LIST="${SHOTS_LIST}" \
  OUT_DIR="${OUT_DIR}" \
  EXCLUDE_ANCHOR=1 \
  bash scripts/run_js_tradeoff_texture.sh
done

echo "===== all modes done ====="
echo "results root: ${ROOT_OUT}"
