#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# CLIP-Adapter texture 실험을 단일 GPU에서 N개 sample-seed씩 병렬 실행
#
# 동작:
#   GPU 한 장(기본 GPU 1번)에 NUM_PARALLEL개 process를 동시에 띄움.
#   batch가 끝나면 다음 batch 진행.
#
# 사용 예:
#   # GPU 1번에 5 process 동시, seed 0~99 (총 100 seed)
#   bash scripts/run_texture_adapter_parallel.sh
#
#   # GPU 0번에 5 process 동시
#   GPU=0 bash scripts/run_texture_adapter_parallel.sh
#
#   # 동시 process 수 변경 (메모리 OOM이면 3으로 줄이기 등)
#   NUM_PARALLEL=3 bash scripts/run_texture_adapter_parallel.sh
#
#   # seed 50부터 10개만
#   SEED_START=50 NUM_SAMPLE_SEEDS=10 bash scripts/run_texture_adapter_parallel.sh
# -----------------------------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")/.."

# 사용할 단일 GPU id
GPU=${GPU:-1}

# GPU 1장에서 동시에 띄울 process 수
NUM_PARALLEL=${NUM_PARALLEL:-5}

ROOT_OUT=${ROOT_OUT:-"runs/adapter_texture_preserve_parallel"}
SEED_START=${SEED_START:-0}
NUM_SAMPLE_SEEDS=${NUM_SAMPLE_SEEDS:-100}

mkdir -p "${ROOT_OUT}"

echo "===== single-GPU parallel CLIP-Adapter texture experiment ====="
echo "project root  : $(pwd)"
echo "GPU           : ${GPU}"
echo "NUM_PARALLEL  : ${NUM_PARALLEL}   (동일 GPU에서 동시에 띄울 process 수)"
echo "ROOT_OUT      : ${ROOT_OUT}"
echo "SEED RANGE    : [${SEED_START}, $((SEED_START + NUM_SAMPLE_SEEDS - 1))]   (total=${NUM_SAMPLE_SEEDS})"
echo "================================================================"

run_one_seed() {
    local sample_seed=$1
    local OUT_DIR="${ROOT_OUT}/sample_seed_${sample_seed}"
    mkdir -p "${OUT_DIR}"

    echo "[$(date '+%H:%M:%S')] [gpu=${GPU}] sample_seed=${sample_seed} START"

    CUDA_VISIBLE_DEVICES=${GPU} python train_blue_target_adapter.py \
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
      --sample-seed "${sample_seed}" \
      --cache-features \
      --output-dir "${OUT_DIR}" \
      > "${OUT_DIR}/run.log" 2>&1

    echo "[$(date '+%H:%M:%S')] [gpu=${GPU}] sample_seed=${sample_seed} DONE"
}

# NUM_PARALLEL개씩 묶어서 동시에 launch -> wait -> 다음 batch
launched_in_batch=0
for sample_seed in $(seq ${SEED_START} $((SEED_START + NUM_SAMPLE_SEEDS - 1))); do
    run_one_seed ${sample_seed} &

    launched_in_batch=$((launched_in_batch + 1))

    if (( launched_in_batch % NUM_PARALLEL == 0 )); then
        wait
        launched_in_batch=0
    fi
done

# 마지막 남은 process 대기
wait

echo "===== all done ====="
echo "results root: ${ROOT_OUT}"