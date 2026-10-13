#!/bin/bash
set -e
echo "===== texture ====="
find /data/datasets/texture -maxdepth 2 -type d | sort | head -100 || true
echo ""
echo "===== ImageNet ====="
find /data/datasets/imagenet_1k -maxdepth 2 -type d | sort | head -50 || true
echo ""
echo "===== CIFAR100 ====="
find /data/datasets/cifar100 -maxdepth 2 -type d | sort | head -50 || true
echo ""
echo "===== SUN ====="
find /data/datasets/SUN -maxdepth 2 -type d | sort | head -50 || true
echo ""
echo "===== GPU status ====="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
