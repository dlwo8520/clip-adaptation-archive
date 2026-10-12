#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# ===== 설정 =====
ROOT_OUT=${ROOT_OUT:-"runs/coop_textdrift_texture_100seeds"}
SEED_START=${SEED_START:-0}
NUM_SAMPLE_SEEDS=${NUM_SAMPLE_SEEDS:-100}
PARALLEL=${PARALLEL:-2}
CANDIDATE_FACTOR=${CANDIDATE_FACTOR:-2}

# text_drift 전용 옵션
VOCAB_FILE=${VOCAB_FILE:-""}        # 비우면 torchvision ImageNet-1K fallback
VOCAB_SIZE=${VOCAB_SIZE:-1000}      # -1이면 전체 vocab 사용
VOCAB_SEED=${VOCAB_SEED:-12345}
STEP_SIZE=${STEP_SIZE:-0.01}
TEMPLATES=${TEMPLATES:-"default"}   # default | robust
CHUNK_SIZE=${CHUNK_SIZE:-1024}

mkdir -p "${ROOT_OUT}"

echo "===== CoOp + text_drift selection: 100 sample-seed (parallel=${PARALLEL}) ====="
echo "project root        : $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "ROOT_OUT            : ${ROOT_OUT}"
echo "SEED_START          : ${SEED_START}"
echo "NUM_SAMPLE_SEEDS    : ${NUM_SAMPLE_SEEDS}"
echo "PARALLEL            : ${PARALLEL}"
echo "CANDIDATE_FACTOR    : ${CANDIDATE_FACTOR}"
echo ""
echo "[text_drift options]"
echo "VOCAB_FILE          : ${VOCAB_FILE:-<torchvision ImageNet-1K fallback>}"
echo "VOCAB_SIZE          : ${VOCAB_SIZE}"
echo "VOCAB_SEED          : ${VOCAB_SEED}"
echo "STEP_SIZE           : ${STEP_SIZE}"
echo "TEMPLATES           : ${TEMPLATES}"
echo "CHUNK_SIZE          : ${CHUNK_SIZE}"

# vocab-file 인자를 조건부 추가
VOCAB_FILE_ARG=""
if [ -n "${VOCAB_FILE}" ]; then
    VOCAB_FILE_ARG="--text-drift-vocab-file ${VOCAB_FILE}"
fi

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

        if [ -f "${OUT_DIR}/results.json" ]; then
            echo "  [SKIP] already done: sample_seed_${sample_seed}"
            continue
        fi

        echo "  [start] sample_seed=${sample_seed}"

        python train_blue_target_coop_zscl.py \
          --target-name texture \
          --target-root /data/datasets/texture \
          --preserve imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test \
          --model ViT-B-32-quickgelu \
          --pretrained openai \
          --shots-list 1 2 4 8 16 \
          --n-ctx 16 \
          --epochs 50 \
          --eval-every 5 \
          --lr 0.002 \
          --batch-size 128 \
          --num-workers 2 \
          --device cuda \
          --seed 42 \
          --prompt-seed 777 \
          --sample-seed "${sample_seed}" \
          --selection-method text_drift \
          --selection-candidate-factor "${CANDIDATE_FACTOR}" \
          --text-drift-vocab-size "${VOCAB_SIZE}" \
          --text-drift-vocab-seed "${VOCAB_SEED}" \
          --text-drift-step-size "${STEP_SIZE}" \
          --text-drift-templates "${TEMPLATES}" \
          --text-drift-chunk-size "${CHUNK_SIZE}" \
          ${VOCAB_FILE_ARG} \
          --cache-features \
          --output-dir "${OUT_DIR}" \
          > "${OUT_DIR}/run.log" 2>&1 &

        pids+=($!)
    done

    if [ ${#pids[@]} -gt 0 ]; then
        for pid in "${pids[@]}"; do
            wait "${pid}" || echo "  [경고] PID=${pid} 비정상 종료"
        done
    fi

    echo "===== batch ${batch_num} 완료 ====="
done

echo ""
echo "===== 전체 종료 ====="
echo "results root: ${ROOT_OUT}"
