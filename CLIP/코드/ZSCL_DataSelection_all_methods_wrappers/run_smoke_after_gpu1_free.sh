#!/usr/bin/env bash
set -euo pipefail

GPU_ID=1
CHECK_INTERVAL=60
STABLE_EMPTY_CHECKS=3

echo "===== wait until GPU ${GPU_ID} is free from my processes ====="
echo "user: $USER"
echo "check interval: ${CHECK_INTERVAL}s"
echo "need consecutive empty checks: ${STABLE_EMPTY_CHECKS}"

empty_count=0

while true; do
  my_gpu_pids=""

  # GPU 1에 올라간 compute PID들을 확인
  pids=$(nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)

  for pid in $pids; do
    owner=$(ps -o user= -p "$pid" 2>/dev/null | xargs || true)
    cmd=$(ps -o args= -p "$pid" 2>/dev/null || true)

    if [ "$owner" = "$USER" ]; then
      my_gpu_pids="${my_gpu_pids}
PID=${pid} USER=${owner} CMD=${cmd}"
    fi
  done

  if [ -z "$my_gpu_pids" ]; then
    empty_count=$((empty_count + 1))
    echo "[$(date)] GPU ${GPU_ID}: no my process (${empty_count}/${STABLE_EMPTY_CHECKS})"

    if [ "$empty_count" -ge "$STABLE_EMPTY_CHECKS" ]; then
      break
    fi
  else
    empty_count=0
    echo "[$(date)] GPU ${GPU_ID}: still running my process:"
    echo "$my_gpu_pids"
  fi

  sleep "$CHECK_INTERVAL"
done

echo "===== GPU ${GPU_ID} is free. start smoke test ====="

CUDA_VISIBLE_DEVICES=1 python train_blue_target_adapter_zscl.py \
  --target-name texture \
  --target-root /data/datasets/texture \
  --preserve imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test \
  --model ViT-B-32-quickgelu --pretrained openai \
  --shots-list 1 --epochs 2 --eval-every 1 \
  --adapter-hidden-ratio 0.25 --adapter-alpha-init 0.6 --adapter-seed 777 \
  --lr 1e-5 --batch-size 32 --num-workers 2 --device cuda \
  --seed 42 --sample-seed 0 \
  --selection-method random --selection-candidate-factor 2 \
  --output-dir /tmp/zscl_smoke_test

echo "===== smoke test done ====="
