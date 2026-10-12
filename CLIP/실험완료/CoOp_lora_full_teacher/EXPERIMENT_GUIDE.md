# 실험 설명서: Blue Target CoOp Adaptation

## 1. 실험 배경

CLIP은 자연어 prompt를 이용해 zero-shot classification을 수행할 수 있음. 그러나 CLIP 논문 Figure 5를 보면 모든 dataset에서 잘하는 것은 아님. EuroSAT, KITTI Distance, PatchCamelyon, GTSRB, CLEVRCounts, DTD, Flowers102, RESISC45, FGVCAircraft 같은 dataset에서는 ResNet50 feature linear probe보다 낮은 성능을 보임.

이 실험은 바로 이 파란색 dataset들을 target으로 삼음.

## 2. 실험 질문

```text
CLIP이 zero-shot에서 약한 dataset에 CoOp few-shot adaptation을 하면,
target 성능은 좋아지지만,
다른 dataset에서의 기존 zero-shot-style 성능은 낮아지는가?
```

여기서 target은 적응 대상이고, preserve는 보존성 평가 대상임.

## 3. 실험 구조

### Target adaptation dataset

우선순위는 다음과 같음.

```text
1. DTD / texture
2. EuroSAT
3. RESISC45
4. GTSRB
5. FGVCAircraft
6. Flowers102
```

현재 서버에 `/data/datasets/texture`가 있으므로 1차 실험은 texture를 권장함.

### Preserve evaluation dataset

```text
1. ImageNet val
2. CIFAR100 test
3. SUN or Places
```

ImageNet은 일반 객체 인식 능력 보존을 보는 기준임. CIFAR100은 저해상도 일반 객체 보존을 보는 보조 기준임. SUN/Places는 scene recognition 보존을 보는 기준임.

## 4. 실험에서 학습되는 것

학습되는 것은 CoOp prompt context뿐임.

```text
[V]1 [V]2 ... [V]M [CLASS]
```

여기서 `[V]1 ... [V]M`만 학습되고, CLIP image encoder와 text encoder는 고정됨.

즉 이 실험은 CLIP 전체 fine-tuning이 아님.

## 5. 측정 지표

각 target dataset T, preserve dataset P, shot 수 s에 대해 다음을 측정함.

```text
Target Delta:
Acc(CoOp trained on T, evaluated on T) - Acc(Original CLIP zero-shot on T)

Preserve Delta:
Acc(CoOp trained on T, evaluated on P) - Acc(Original CLIP zero-shot on P)
```

보고 싶은 패턴은 다음임.

```text
Target Delta > 0
Preserve Delta < 0
```

즉 target은 좋아지는데 기존 zero-shot-style 성능은 떨어지는 패턴임.

## 6. 저장 파일

실험 결과는 다음 구조로 저장됨.

```text
runs/blue_texture_preserve/
├── args.json
├── zero_shot_baselines.json
├── prompt_initialization.json
├── prompt_initialization.pt
├── results.json
├── curves.jsonl
├── shot_1/
│   ├── history.json
│   └── coop_prompt_final.pt
├── shot_2/
│   ├── history.json
│   └── coop_prompt_final.pt
└── figures/
```

### zero_shot_baselines.json

원본 CLIP의 target/preserve 성능임.

**중요: epoch 0과 zero-shot baseline은 다른 값임**

- `zero_shot_baselines.json`의 값 = original CLIP zero-shot 성능 (hand-crafted prompt template 사용)
- `curves.jsonl`의 epoch 0 값 = 동일하게 초기화된 CoOp prompt context로 평가한 값
  - hand-crafted prompt가 아닌 random initialization된 learnable context이므로 zero-shot과 다른 값임
  - 모든 shot 실험은 같은 `prompt_seed`에서 출발하므로 epoch 0 결과는 shot 간 동일해야 함

따라서 결과 해석은 accuracy 자체보다 `delta_from_zero_shot`을 중심으로 봐야 하고, 그래프에는 dashed line으로 zero-shot baseline이 함께 표시됨.

### curves.jsonl

epoch별 성능 변화가 한 줄씩 저장됨. 그래프 만들 때 가장 중요함.

### results.json

shot별 마지막 결과 요약임.

### coop_prompt_final.pt

shot별 학습된 prompt vector checkpoint임.

## 7. 권장 실험 순서

### 0단계: GPU 확인

```bash
export CUDA_VISIBLE_DEVICES=1
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
PY
```

`device_count`가 1이어야 함.

### 1단계: smoke test

```bash
bash scripts/run_smoke_gpu1.sh
```

이건 코드가 도는지 확인하는 용도임. 결과 해석용 아님.

### 2단계: texture 실험

```bash
bash scripts/run_texture_gpu1.sh
```

### 3단계: 그래프 생성

```bash
python plot_blue_target_results.py \
  --curves runs/blue_texture_preserve/curves.jsonl \
  --out-dir runs/blue_texture_preserve/figures
```

## 8. 결과 해석

### 좋은 결과 패턴

```text
target accuracy 증가
ImageNet/CIFAR100/SUN 중 일부 accuracy 감소
```

이 경우 target adaptation과 zero-shot preservation 사이에 trade-off가 관찰된 것임.

### 애매한 결과 패턴

```text
target accuracy 증가
preserve accuracy도 증가
```

이 경우 CoOp이 오히려 generalization에도 도움을 준 것일 수 있음. CoOp 논문에서도 domain generalization이 좋아지는 경우가 보고되므로, 이 결과도 실패라고 보긴 어려움.

### 나쁜 결과 패턴

```text
target accuracy 증가 없음
preserve accuracy 감소
```

이 경우 prompt 학습이 target에도 도움을 못 주고 일반 성능만 망친 것이므로 hyperparameter나 dataset split을 의심해야 함.

## 9. 한계

- ImageNet class folder가 WordNet ID일 경우 class name mapping이 틀리면 zero-shot baseline이 왜곡될 수 있음.
- texture만으로 전체 주장을 하기엔 약함.
- 최소 3 seed 평균이 필요함.
- target dataset을 DTD, EuroSAT, GTSRB, RESISC45 등으로 늘려야 함.
- CoOp은 항상 zero-shot 성능을 떨어뜨리는 방법이 아님. 특정 조건에서 trade-off가 생기는지 보는 실험으로 써야 함.
