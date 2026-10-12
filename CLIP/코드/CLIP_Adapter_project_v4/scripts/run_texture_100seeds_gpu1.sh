#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

ROOT_OUT=${ROOT_OUT:-"runs/adapter_texture_preserve_100seeds"}
SEED_START=${SEED_START:-0}
NUM_SAMPLE_SEEDS=${NUM_SAMPLE_SEEDS:-100}
PARALLEL=${PARALLEL:-10}

mkdir -p "${ROOT_OUT}"

echo "===== 100 sample-seed texture CLIP-Adapter experiment (parallel=${PARALLEL}) ====="
echo "project root: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "ROOT_OUT=${ROOT_OUT}"
echo "SEED_START=${SEED_START}"
echo "NUM_SAMPLE_SEEDS=${NUM_SAMPLE_SEEDS}"
echo "PARALLEL=${PARALLEL}"

SEED_END=$(( SEED_START + NUM_SAMPLE_SEEDS - 1 ))
batch_num=0

for batch_start in $(seq ${SEED_START} ${PARALLEL} ${SEED_END}); do
    batch_end=$(( batch_start + PARALLEL - 1 ))
    if (( batch_end > SEED_END )); then
        batch_end=${SEED_END}
    fi

    batch_num=$(( batch_num + 1 ))
    echo ""
    echo "===== batch ${batch_num}: seed ${batch_start} ~ ${batch_end} 병렬 시작 ====="

    pids=()
    for sample_seed in $(seq ${batch_start} ${batch_end}); do
        OUT_DIR="${ROOT_OUT}/sample_seed_${sample_seed}"
        mkdir -p "${OUT_DIR}"
        echo "  [start] sample_seed=${sample_seed}"

        python train_blue_target_adapter.py \
          --target-name texture \
          --target-root /data/datasets/texture \
          --preserve imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test \
          --model ViT-B-32-quickgelu \
          --pretrained openai \
          --shots-list 1 2 4 8 16 \
          --adapter-hidden-ratio 0.25 \
          --adapter-alpha-init 0.2 \
          --adapter-seed 777 \
          --epochs 50 \
          --eval-every 5 \
          --lr 0.001 \
          --batch-size 128 \
          --num-workers 2 \
          --device cuda \
          --seed 42 \
          --sample-seed "${sample_seed}" \
          --cache-features \
          --output-dir "${OUT_DIR}" \
          > "${OUT_DIR}/run.log" 2>&1 &

        pids+=($!)
    done

    # 이 batch의 모든 프로세스가 끝날 때까지 대기
    for pid in "${pids[@]}"; do
        wait "${pid}" || echo "  [경고] PID=${pid} 비정상 종료"
    done

    echo "===== batch ${batch_num} 완료 ====="
done

echo ""
echo "===== 전체 종료 ====="
echo "results root: ${ROOT_OUT}"