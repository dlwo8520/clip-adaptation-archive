# Blue Target CoOp Project

CLIP 논문 Figure 5에서 zero-shot CLIP이 약하게 나온 파란색 계열 dataset을 target으로 잡고, CoOp few-shot adaptation을 수행한 뒤, 기존 zero-shot-style 성능이 다른 dataset에서 얼마나 변하는지 확인하는 실험 코드임.

## 실험 목적

이 실험은 다음 질문을 확인함.

> CLIP이 zero-shot에서 약한 dataset에 CoOp few-shot adaptation을 하면, target 성능은 올라가지만 ImageNet/CIFAR100/SUN 같은 기존 dataset의 zero-shot-style 성능은 떨어지는가?

## 기본 구조

- target dataset: DTD/texture, EuroSAT, RESISC45, GTSRB, FGVCAircraft, Flowers102 등
- preserve dataset: ImageNet val, CIFAR100 test, SUN/Places 등
- adaptation: CLIP 전체 fine-tuning이 아니라 CoOp prompt context만 학습
- 저장: json, jsonl, checkpoint, graph

## 파일 구조

```text
blue_target_coop_project/
├── README.md
├── EXPERIMENT_GUIDE.md
├── requirements.txt
├── train_blue_target_coop.py
├── plot_blue_target_results.py
├── src/
│   ├── utils.py
│   ├── data.py
│   ├── coop_model.py
│   └── eval.py
└── scripts/
    ├── check_dataset_structure.sh
    ├── run_smoke_gpu1.sh
    └── run_texture_gpu1.sh
```

## 서버에 복사

```bash
cd ${RESEARCH_WORKSPACE}/projects/clip_zs_drop
unzip blue_target_coop_project.zip
cp -r blue_target_coop_project/* .
```

## GPU 1번으로 실행

```bash
export CUDA_VISIBLE_DEVICES=1
```

PyTorch 안에서는 물리 GPU 1번이 `cuda:0`으로 보임. 그래서 코드 실행은 `--device cuda`로 하면 됨.

## smoke test

```bash
bash scripts/run_smoke_gpu1.sh
```

## DTD/texture 1차 실험

```bash
bash scripts/run_texture_gpu1.sh
```

내부에서 `--prompt-seed 777`로 모든 shot 실험이 동일한 CoOp prompt 초기값에서 출발함. 실행 후 `prompt_initialization.json`과 `prompt_initialization.pt`가 생성됨.

## 그래프 생성

```bash
python plot_blue_target_results.py \
  --curves runs/blue_texture_preserve/curves.jsonl \
  --baseline-json runs/blue_texture_preserve/zero_shot_baselines.json \
  --out-dir runs/blue_texture_preserve/figures
```

`--baseline-json`을 함께 넘기면 그래프에 original CLIP zero-shot baseline이 dashed line으로 표시됨. epoch 0 값과 zero-shot baseline이 다르다는 점을 그래프에서 바로 확인 가능.

## 결과 해석 시 주의

- `curves.jsonl`의 epoch 0 = initialized CoOp prompt 평가값 (≠ original zero-shot)
- `zero_shot_baselines.json`의 값 = original CLIP zero-shot 평가값
- 두 값은 다르며, 비교 기준은 항상 `delta_from_zero_shot` (=accuracy − zero_shot_baseline)
