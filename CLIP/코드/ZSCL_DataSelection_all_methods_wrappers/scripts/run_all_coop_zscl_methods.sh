#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

# 5개 data-selection method 전체 실행
METHODS=${METHODS:-"random teacher_confident teacher_margin prototype teacher_uncertain"}

# 결과 폴더 구조:
# ${BASE_OUT}/coop/<method>/sample_seed_<seed>/...
BASE_OUT=${BASE_OUT:-"runs/zscl_texture_all_methods"}

# 동시에 adapter wrapper도 같이 돌릴 경우, 전체 동시 실행 수를 2개로 맞추려면 PARALLEL=1 권장.
# coop만 단독 실행하면 PARALLEL=2로 올려도 됨.
PARALLEL=${PARALLEL:-1}
SEED_START=${SEED_START:-0}
NUM_SAMPLE_SEEDS=${NUM_SAMPLE_SEEDS:-100}
CANDIDATE_FACTOR=${CANDIDATE_FACTOR:-2}

mkdir -p "${BASE_OUT}/coop"

echo "===== run all CoOp ZSCL methods ====="
echo "project root        : $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<script default>}"
echo "BASE_OUT            : ${BASE_OUT}"
echo "METHODS             : ${METHODS}"
echo "PARALLEL            : ${PARALLEL}"
echo "SEED_START          : ${SEED_START}"
echo "NUM_SAMPLE_SEEDS    : ${NUM_SAMPLE_SEEDS}"
echo "CANDIDATE_FACTOR    : ${CANDIDATE_FACTOR}"

for METHOD in ${METHODS}; do
  ROOT_OUT="${BASE_OUT}/coop/${METHOD}"
  mkdir -p "${ROOT_OUT}"

  echo ""
  echo "===== [CoOp] start method=${METHOD} ====="
  echo "ROOT_OUT=${ROOT_OUT}"

  PARALLEL="${PARALLEL}" \
  SEED_START="${SEED_START}" \
  NUM_SAMPLE_SEEDS="${NUM_SAMPLE_SEEDS}" \
  ROOT_OUT="${ROOT_OUT}" \
  SELECTION_METHOD="${METHOD}" \
  CANDIDATE_FACTOR="${CANDIDATE_FACTOR}" \
  bash scripts/run_texture_coop_zscl_100seeds_gpu1.sh

  echo "===== [CoOp] done method=${METHOD} ====="
done

echo ""
echo "===== all CoOp methods done ====="
echo "results: ${BASE_OUT}/coop"
