# ZSCL-CoOp + Reference Distillation

기존 CoOp + ZSCL-aware data selection 파이프라인 위에
**Zheng et al., CVPR 2023 ("Preventing Zero-Shot Transfer Degradation in Continual
Learning of Vision-Language Models", ZSCL)** 의
*reference dataset 기반 feature-space distillation* 과
*parameter-space weight consolidation* 을 추가한 버전.

## 1. 변경 요약

기존 (`train_blue_target_coop_zscl.py`):

```
L = CE(target support set, CoOp(ctx) text features)
```

신규 (`train_blue_target_coop_zscl_refdistill.py`):

```
L = CE(target support, CoOp(ctx) text features)                       # 기존 그대로
  + λ_distill · ( L_lwf_img + L_lwf_txt )    on reference (image, classname) 풀
  + λ_wc      · ||ctx - ctx_init||²          (옵션, --lambda-wc > 0 일 때)
```

* `L_lwf_img`, `L_lwf_txt` 는 ZSCL 논문 Eq.(3),(4) 에 대응.
  CoOp 셋팅에서는 image encoder가 freeze이므로 reference image feature와
  teacher text feature를 **한 번만 캐싱**하고, 매 step student text feature만
  새로 계산함.
* `L_wc` 는 ZSCL Eq.(5)를 CoOp ctx에 적용 (학습되는 파라미터가 ctx뿐).

## 2. 디렉토리 구조

```
ZSCL_DataSelection_refdistill/
├── README.md                                # 이 파일
├── requirements.txt
├── train_blue_target_coop_zscl.py           # 기존 파이프라인 (비교용, 변경 없음)
├── train_blue_target_coop_zscl_refdistill.py  # 신규 학습 스크립트
├── src/
│   ├── coop_model.py            # 기존
│   ├── data.py                  # 기존
│   ├── eval.py                  # 기존
│   ├── sample_selection.py      # 기존 (margin/entropy/confident 모두 유지)
│   ├── utils.py                 # 기존
│   └── reference_distill.py     # 신규: CoOpReferenceDistiller + WC 함수
└── scripts/
    ├── run_coop_refdistill_one_config.sh   # 한 config × 100 seed 실행
    ├── launch_6_configs_parallel.sh        # 6 config 병렬 실행 (마스터)
    └── smoke_test_refdistill.sh            # 본 실행 전 검증용 짧은 테스트
```

## 3. 6개 config

실험 구성: **selection method 3종 × WC on/off 2종 = 6**

| tag                                | selection_method  | use_wc |
|------------------------------------|-------------------|--------|
| method__teacher_margin__wc_off     | teacher_margin    | off    |
| method__teacher_margin__wc_on      | teacher_margin    | on     |
| method__teacher_entropy__wc_off    | teacher_entropy   | off    |
| method__teacher_entropy__wc_on     | teacher_entropy   | on     |
| method__teacher_confident__wc_off  | teacher_confident | off    |
| method__teacher_confident__wc_on   | teacher_confident | on     |

* 각 config: sample seed 0..99 = 100 run, 총 600 run.

## 4. 실행 방법

### 4-1. (먼저) smoke test

```bash
REFERENCE_ROOT=/data/datasets/imagenet_1k/val \
TARGET_ROOT=/data/datasets/texture \
GPU_ID=0 \
bash scripts/smoke_test_refdistill.sh
```

* `runs/smoke_coop_refdistill/smoke_seed_0/` 에 결과 생성.
* `curves.jsonl` 의 `train.train_loss_distill_raw`, `train.train_loss_wc_raw`가
  실제로 값이 찍히는지 확인하면 됨.
* 한 번 통과하면 본 실행으로.

### 4-2. (본 실행) 6 config 병렬

```bash
BASE_OUT=runs/coop_refdistill_6configs \
TARGET_NAME=texture \
TARGET_ROOT=/data/datasets/texture \
REFERENCE_NAME=imagenet \
REFERENCE_ROOT=/data/datasets/imagenet_1k/val \
PRESERVE_SPECS="imagenet:/data/datasets/imagenet_1k/val cifar100:/data/datasets/cifar100/test" \
GPU_LIST="0 0 0 1 1 1" \
PARALLEL_PER_CONFIG=1 \
NUM_SAMPLE_SEEDS=100 \
bash scripts/launch_6_configs_parallel.sh
```

* `GPU_LIST` 는 공백으로 구분된 **6개** GPU id. 위 예시는 GPU 0에 3개, GPU 1에 3개 분산.
* `MAX_CONCURRENT_CONFIGS` (기본 6): 동시에 띄울 config 수 제한. **단일 GPU 환경에서 OOM 방지용**.
* `PARALLEL_PER_CONFIG` 는 한 config 안에서 동시에 돌릴 seed 수.
  실제 동시 학습 프로세스 수 = min(`MAX_CONCURRENT_CONFIGS`, 6) × `PARALLEL_PER_CONFIG`.
* 이미 완료된 seed (`results.json` 존재) 는 자동 skip 됨.

#### 단일 GPU (예: GPU 4번만 사용 가능) 실행 예시

```bash
# 한 번에 2개 config씩만 동시 실행 → 총 동시 프로세스 2개로 OOM 방지
BASE_OUT=runs/coop_refdistill_6configs \
TARGET_ROOT=/data/datasets/texture \
REFERENCE_ROOT=/data/datasets/imagenet_1k/val \
GPU_LIST="4 4 4 4 4 4" \
MAX_CONCURRENT_CONFIGS=2 \
PARALLEL_PER_CONFIG=1 \
NUM_SAMPLE_SEEDS=100 \
bash scripts/launch_6_configs_parallel.sh
```

* GPU 메모리 여유에 따라 `MAX_CONCURRENT_CONFIGS` 조정:
  - 24GB (3090/4090): 2 또는 3
  - 48GB (A6000/L40): 3 또는 4
  - 80GB (A100/H100): 6 (전부 동시)
* 6개를 무조건 동시에 띄우려면 `MAX_CONCURRENT_CONFIGS=6` (기본값).
* 가장 보수적으로 가려면 `MAX_CONCURRENT_CONFIGS=1` (config들이 완전 순차 실행, seed는 PARALLEL_PER_CONFIG만큼 병렬).

결과 구조:

```
runs/coop_refdistill_6configs/
├── logs/
│   ├── method__teacher_margin__wc_off.log
│   ├── ... (6개)
├── method__teacher_margin__wc_off/
│   ├── sample_seed_0/
│   │   ├── args.json
│   │   ├── zero_shot_baselines.json
│   │   ├── reference_distill_meta.json
│   │   ├── prompt_initialization.{json,pt}
│   │   ├── selection_meta.json
│   │   ├── curves.jsonl
│   │   ├── results.json
│   │   ├── run.log
│   │   └── shot_1/, shot_2/, shot_4/, shot_8/, shot_16/
│   ├── sample_seed_1/
│   ├── ...
│   └── sample_seed_99/
├── method__teacher_margin__wc_on/
└── ... (총 6개)
```

## 5. 학습 loss 추적

`curves.jsonl` 각 행의 `train` 필드:

```json
{
  "train_loss":             전체 합 (CE + λ·distill + λ_wc·wc),
  "train_loss_ce":          target CE만,
  "train_loss_distill_raw": λ 곱하기 전의 (L_lwf_img + L_lwf_txt),
  "train_loss_wc_raw":      λ 곱하기 전의 ||ctx-ctx_init||²,
  "train_acc":              ...
}
```

이걸 보면 distill / wc 가 학습 후반에도 살아있는지, 혹은 너무 작거나 너무 커서
무의미한지 확인 가능.

## 6. λ 튜닝 메모

* `--lambda-distill`: ZSCL 논문 부록 기준 λ=1 권장. (기본값)
* `--lambda-wc`: CoOp ctx는 파라미터 수가 적어 (n_ctx × ctx_dim ≈ 16×512)
  ZSCL의 원래 CLIP 전체 weight consolidation과는 크기 스케일이 다름.
  기본 0.1로 시작하고, `curves.jsonl` 의 `train_loss_wc_raw` 가 CE 대비
  너무 크거나 작으면 조정.

## 7. Reference dataset 선택 시 leakage 주의

`REFERENCE_ROOT` 와 `PRESERVE_SPECS` 의 imagenet 경로가 같으면,
**preservation 평가에 쓰는 바로 그 이미지로 distillation을 한 셈** 이 됨.
ImageNet val 50k를 reference로 쓰면 preserve eval (역시 ImageNet val)에 직접적인
정보 누수. 학습 스크립트가 같은 경로를 감지하면 stderr에 경고 출력함.

권장:
* ImageNet train의 일부 (`/data/datasets/imagenet_1k/train`)
* 별도 split (예: val 50k를 split하여 25k는 reference, 25k는 preserve eval)
* 혹은 완전히 다른 일반 이미지 셋

본 코드는 path만 다르면 그대로 동작함. classname은 `--reference-classnames` 로
override 가능.

## 8. 기존 코드 호환성

* 기존 `train_blue_target_coop_zscl.py` 는 그대로 두었음 (regression test 등 용).
* `--lambda-distill 0 --lambda-wc 0` 으로 새 스크립트를 호출하면
  기존 학습과 거의 동일하게 동작 (단, sample_seed 기반 selection은 그대로).
