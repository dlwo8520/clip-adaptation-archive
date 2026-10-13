# GIFT-Inspired Synthetic-Data Preservation 구현 노트

본 문서는 기존 ZSCL reference distillation 기반 CoOp few-shot 코드베이스
(`ZSCL_DataSelection_refdistill_2`)에 GIFT (CVPR 2025, Wu et al.) 의 합성
데이터 기반 보존 방법을 **새 옵션으로 추가**한 결과를 설명합니다.

본 구현의 핵심 원칙은 다음과 같습니다.

- 기존 ZSCL refdistill 경로는 **삭제하지 않음**. 두 방법을 `--preserve-method`
  플래그 하나로 선택하도록 함.
- DTD 1/2/4/8/16-shot, 100 sample seed, CoOp prompt-only fine-tuning, 그리고
  sample selection (teacher_confident / teacher_margin / teacher_entropy) 구조는
  그대로 유지.
- 학습 시점에서 Stable Diffusion 호출이 **필요 없도록** 함. 합성 이미지는
  사전(offline) 생성한 폴더만 있으면 됨. 선택적으로 SD 모드도 제공함.
- GIFT 논문의 **full reproduction** 은 아님. CoOp 한정 환경에 맞게 단순화된
  GIFT-inspired 구현임 (자세한 차이는 본문 §2 참조).

---

## 1. 변경/추가된 파일 요약

### 새로 추가된 파일

| 파일 | 역할 |
|---|---|
| `train_blue_target_coop_gift.py` | 새 학습 진입점. `--preserve-method refdistill | gift_synthetic` 으로 두 보존 방식 선택. |
| `src/gift_dataset.py` | `metadata.jsonl` 기반 synthetic image-text pair Dataset (`SyntheticImageTextDataset`). |
| `src/gift_distill.py` | GIFT 핵심 모듈: contrastive distillation (CD), image-text alignment (ITA), adaptive weight consolidation (AWC) (CoOp ctx 한정). |
| `scripts/generate_synthetic_dataset.py` | placeholder / Stable Diffusion 두 모드로 학습용 합성 이미지를 미리 생성. |
| `scripts/smoke_test_gift.sh` | placeholder 합성 데이터 생성 후 `gift_synthetic + AWC` 로 2 에폭 빠른 검증. |
| `scripts/run_coop_gift_one_config.sh` | 하나의 (preserve-method, selection-method, USE_AWC) config를 100 seed 까지 실행. |
| `scripts/launch_gift_configs.sh` | 4-config 그리드 (A: baseline refdistill, B: synthetic ZSCL-like β=0, C: GIFT, D: GIFT+AWC) 동시 실행. |
| `gift_classnames/imagenet_classes_subset.txt` | 기본 base 컨셉 풀 (20 클래스, 빠른 실험용). |
| `gift_classnames/dtd_classes.txt` | DTD 47 텍스처 클래스명. target source 프롬프트용. |
| `README_GIFT.md` | 본 문서. |

### 변경되지 않은 파일

- `train_blue_target_coop_zscl.py` (원래 baseline)
- `train_blue_target_coop_zscl_refdistill.py` (기존 ZSCL refdistill 학습)
- `src/coop_model.py`, `src/reference_distill.py`, `src/sample_selection.py`,
  `src/data.py`, `src/eval.py`, `src/utils.py`
- 기존 `scripts/smoke_test_refdistill.sh`,
  `scripts/run_coop_refdistill_one_config.sh`,
  `scripts/launch_6_configs_*.sh`

즉, **이전 실험과 결과는 모두 그대로 재현 가능**합니다.

---

## 2. 본 구현이 GIFT 논문과 어떻게 다른가

| 측면 | GIFT 논문 | 본 구현 |
|---|---|---|
| 학습 대상 파라미터 | CLIP image+text encoder 전체 fine-tuning | **CoOp prompt context (`ctx`) 만** 학습. 백본은 frozen. |
| 학습 task 구조 | MTIL (11 task sequence) | **단일 target task** (DTD) + 100 sample seed × 1/2/4/8/16 shot. |
| Teacher 모델 | 이전 task 의 학습된 모델 θ^(t-1) | 원본 CLIP (raw prompt). CoOp 설정에서는 `ctx` 만 다르므로 θ^(t-1) ≈ original CLIP 과 동일. |
| 합성 이미지 생성 | Stable Diffusion 매 task 마다 재생성, 폐기 | **사전(offline) 생성 후 디스크 보관**. 학습 중 SD 호출 없음. placeholder 모드도 지원. |
| Contrastive matrix | M = CLIP image_emb @ text_emb.T (B×B) | 동일. **단, image encoder 가 frozen 이므로 student/teacher image feature 가 같음** → text feature 의 변화만 distill. |
| Image-text alignment | LITA = CE(Mt, I_B) row + column | 동일. |
| AWC scope | CLIP backbone 전체 파라미터 | **CoOp `ctx` 한정**. Fisher = (∂L_distill/∂ctx)². reference = init `ctx`. |
| α, β, AWC λ 기본값 | α=1.0, β=0.25, AWC λ 별도 튜닝 | 동일. λ 는 명시적으로 켜야 함 (`--gift-use-awc`). |
| 합성 이미지 수 | 논문 권장 1K/task | 사용자 선택. smoke test 는 NUM_PER_CLASS=2 정도로 빠르게. |
| Base concept pool 갱신 | task 가 진행됨에 따라 누적 | 단일 target setting 이므로 base + target_dtd 두 source 만 사전 생성. |

요약하면: **CoOp prompt-only 환경에 맞게 단순화된 GIFT-inspired 구현**이며,
- 이미지 encoder가 frozen → student/teacher image features 가 동일.
- 따라서 distillation 은 사실상 *teacher text features* (원본 prompt 기반) 와
  *student text features* (CoOp ctx 기반) 간의 정렬을 보존하는 효과를 가짐.
- 이는 GIFT 의 "synthetic image-text pair 의 broad distribution + alignment"
  핵심 아이디어를 그대로 살리면서, 학습 비용을 prompt tuning 수준으로 낮춤.

---

## 3. 기존 ZSCL refdistill 과의 비교

| 항목 | 기존 (`refdistill`) | 새 옵션 (`gift_synthetic`) |
|---|---|---|
| Reference 데이터 | 실제 이미지셋 (ImageNet 등) + class name | 합성 이미지셋 (Stable Diffusion 생성) + 짝지어진 prompt |
| Distillation 행렬 | (image features) × (template-averaged text features) | (matched image features) × (matched prompt-derived text features) |
| Image-Text Alignment loss | 없음 | 있음 (β · LITA, 대각이 positive) |
| 추가 정규화 | 선택적 vanilla WC (`--lambda-wc`) | 선택적 AWC (`--gift-use-awc`, `--gift-lambda-awc`) |
| 손실 함수 | L_CE + λ_dist·(L_img + L_txt) + λ_wc·L_WC | L_CE + α·L_CD + β·L_ITA + λ_AWC·L_AWC |
| 입력 메타 | `--reference-root`, `--reference-classnames` | `--gift-synthetic-root`, `--gift-metadata` |

두 경로는 **상호 배타적이 아니라 직교적**입니다. `gift_synthetic` 모드에서도
기존 `--lambda-wc` (vanilla WC) 와 새 `--gift-use-awc` 를 동시에 켤 수 있어
ablation 비교가 자유롭습니다.

---

## 4. 합성 데이터 준비

### 4.1 디렉토리 / 메타데이터 형식

```
<synthetic_root>/
├── images/
│   ├── base_goldfish_0000.png
│   ├── base_tiger_0000.png
│   ├── target_dtd_banded_0000.png
│   └── ...
└── metadata.jsonl
```

`metadata.jsonl` 의 각 라인은 다음 형식의 JSON 객체입니다.

```json
{"path": "images/base_goldfish_0000.png", "class_name": "goldfish", "prompt": "a photo of a goldfish.", "source": "base", "seed": 123}
{"path": "images/target_dtd_banded_0000.png", "class_name": "banded", "prompt": "a photo of a banded texture.", "source": "target_dtd", "seed": 456}
```

필수 필드: `path`, `class_name`, `prompt`. 선택 필드: `source`, `seed`.

### 4.2 placeholder 모드 (Stable Diffusion 없이 빠른 검증용)

`scripts/generate_synthetic_dataset.py` 를 `--mode placeholder` 로 실행하면,
class name 의 sha256 해시로 결정되는 sinusoidal 패턴 + 노이즈 이미지를 생성합니다.
**Stable Diffusion 패키지가 설치되어 있지 않아도 동작**하므로 smoke test 에 적합합니다.

```bash
python scripts/generate_synthetic_dataset.py \
    --mode placeholder \
    --out-dir ./synthetic_gift_placeholder \
    --base-classnames-file ./gift_classnames/imagenet_classes_subset.txt \
    --target-classnames-file ./gift_classnames/dtd_classes.txt \
    --target-source-tag target_dtd \
    --target-template "a photo of a {} texture." \
    --num-per-class-base 5 \
    --num-per-class-target 5 \
    --base-seed 2024 \
    --overwrite
```

### 4.3 Stable Diffusion 모드 (실제 GIFT-style 실험용)

`diffusers` 가 설치되어 있어야 합니다.

```bash
pip install diffusers accelerate transformers

python scripts/generate_synthetic_dataset.py \
    --mode stable_diffusion \
    --out-dir ./synthetic_gift_sd \
    --base-classnames-file ./gift_classnames/imagenet_classes_subset.txt \
    --target-classnames-file ./gift_classnames/dtd_classes.txt \
    --target-source-tag target_dtd \
    --target-template "a photo of a {} texture." \
    --num-per-class-base 20 \
    --num-per-class-target 20 \
    --sd-model runwayml/stable-diffusion-v1-5 \
    --sd-steps 30 \
    --sd-guidance 7.5 \
    --base-seed 2024 \
    --overwrite
```

생성된 `<out_dir>` 폴더의 경로를 학습 스크립트의 `--gift-synthetic-root` 에 그대로 전달합니다.

---

## 5. 학습 실행 방법

### 5.1 새 인자 (gift 전용)

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--preserve-method` | `refdistill` | `refdistill` 또는 `gift_synthetic`. |
| `--gift-synthetic-root` | None | 합성 이미지 + metadata.jsonl 의 root 디렉토리. `gift_synthetic` 일 때 필수. |
| `--gift-metadata` | `<root>/metadata.jsonl` | 메타데이터 경로 override. |
| `--gift-alpha` | 1.0 | L_CD weight. |
| `--gift-beta` | 0.25 | L_ITA weight. |
| `--gift-batch-size` | 64 | 매 step 에서 synthetic batch 크기 (paper default). |
| `--gift-num-workers` | 4 | 사전 teacher feature 추출 workers. |
| `--gift-max-images` | 0 | >0 이면 그 만큼 random subsample. |
| `--gift-precompute-seed` | 0 | subsample 의 seed. |
| `--gift-store-cpu` | flag | teacher feature 를 CPU 에 저장하여 GPU 메모리 절약. |
| `--gift-use-awc` | flag | AWC 사용 여부. |
| `--gift-lambda-awc` | 0.0 | AWC 강도. AWC 가 켜져 있어도 이 값이 0 이면 효과 없음 → smoke test 에서는 0.1 정도. |
| `--gift-awc-eps` | 0.0 | Fisher floor. |
| `--gift-verify-paths` | flag | metadata 의 모든 path 가 실제 존재하는지 사전 검사. |

기존 `--lambda-wc` (vanilla WC) 도 `gift_synthetic` 모드에서 그대로 동작합니다.

### 5.2 Smoke test

```bash
export TARGET_ROOT=/path/to/dtd       # 필수
export BASE_OUT=./smoke_outputs_gift
export GPU_ID=0
bash scripts/smoke_test_gift.sh
```

이 스크립트는 자동으로
1. `./smoke_outputs_gift/synthetic` 에 placeholder 합성 데이터 생성,
2. 2 epoch / shots {1,4} / sample_seed 0 으로 `gift_synthetic + AWC` 학습 실행,
3. 결과 `curves.jsonl` 에서 `train_loss_cd_raw`, `train_loss_ita_raw`,
   `train_loss_awc_raw` 가 nonzero 인지 검증.

### 5.3 단일 config / 100 sample seed

```bash
# config C: GIFT-inspired (α=1, β=0.25, no AWC)
export TARGET_ROOT=/path/to/dtd
export GIFT_SYNTHETIC_ROOT=./synthetic_gift_sd
export BASE_OUT=./outputs_gift
export PRESERVE_METHOD=gift_synthetic
export SELECTION_METHOD=teacher_margin
export USE_AWC=off
export LAMBDA_AWC=0.0
export SHOTS_LIST="1 2 4 8 16"
export SAMPLE_SEED_START=0
export SAMPLE_SEED_END=99
export PARALLEL=4
export GPU_ID=0

bash scripts/run_coop_gift_one_config.sh
```

이미 결과가 있는 (shot, sample_seed) 조합은 **자동 skip** 합니다 (`results.json` 존재 여부로 판단).

### 5.4 4-config 그리드 동시 실행

```bash
export TARGET_ROOT=/path/to/dtd
export REFERENCE_ROOT=/path/to/imagenet           # config A 용
export GIFT_SYNTHETIC_ROOT=./synthetic_gift_sd    # config B,C,D 용
export BASE_OUT=./outputs_gift_grid
export GPU_LIST="0 1 2 3"            # 4 GPU 동시
export MAX_CONCURRENT_CONFIGS=4
export SHOTS_LIST="1 2 4 8 16"
export SAMPLE_SEED_START=0
export SAMPLE_SEED_END=99
export SELECTION_METHOD=teacher_margin
export PARALLEL=4                     # config 내 동시 seed 수

bash scripts/launch_gift_configs.sh
```

이 실행은 4 configs × 5 shots × 100 sample_seeds = **2000 run** 을 만듭니다.
config 정의는 다음과 같습니다.

| Config | 의미 | preserve-method | α | β | AWC |
|---|---|---|---|---|---|
| A | baseline refdistill | refdistill | (n/a) | (n/a) | off |
| B | synthetic ZSCL-like ablation | gift_synthetic | 1.0 | **0.0** | off |
| C | **GIFT-inspired** | gift_synthetic | 1.0 | 0.25 | off |
| D | **GIFT-inspired + AWC** | gift_synthetic | 1.0 | 0.25 | **on**, λ=0.1 |

이 그리드는 "ZSCL refdistill (real) vs synthetic-only ablation vs full GIFT vs
full GIFT+AWC" 를 같은 base/target/selection/shot/seed 조건에서 정밀 비교할 수 있게 합니다.

---

## 6. 로그 / 출력 형식

각 run 의 `<output>/<shot>shots/seed<sample_seed>/` 폴더에:

- `args.json` — 모든 인자 + 메타정보
- `gift_distill_meta.json` (gift_synthetic 만) — synthetic dataset 통계
  - `metadata_path`, `num_synthetic_images`, `num_active_after_subsample`,
    `num_unique_classes_active`, `source_counts`, `num_base_concepts`,
    `num_target_concepts`, `alpha`, `beta`, `lambda_awc`, ...
- `curves.jsonl` — epoch별 라인:
  - 공통: `train_loss_ce`, `train_loss_total`, `dtd_acc`, `cifar100_acc`,
    `imagenet_acc`, ...
  - refdistill: `train_loss_distill_raw`, `train_loss_wc_raw`
  - gift_synthetic: `train_loss_cd_raw`, `train_loss_ita_raw`,
    `train_loss_awc_raw`, `train_fisher_l1`, `train_fisher_max`
- `results.json` — 최종 성능 (DTD, CIFAR100, ImageNet 등)

---

## 7. 결과 비교 시 가장 먼저 확인할 것

1. **smoke test 의 `curves.jsonl`** 에서 `train_loss_cd_raw`,
   `train_loss_ita_raw`, `train_loss_awc_raw` 가 모두 nonzero 인지 — GIFT 손실 경로가
   실제로 발화하는지 검증.
2. config A vs C 의 `results.json` 에서 **DTD acc 와 CIFAR100/ImageNet acc 의 trade-off**.
   - 합성 데이터만으로 zero-shot transfer (CIFAR100, ImageNet) 가 ZSCL refdistill 만큼
     보존되는지 확인.
3. config B vs C — **β·L_ITA 의 효과**.
4. config C vs D — **AWC 의 효과**. CoOp 환경에서는 ctx 만 학습하므로 AWC 효과가
   GIFT 논문보다 약할 수 있음 (paper limitation 으로 명시되어 있음).
5. selection method (teacher_margin vs teacher_entropy vs teacher_confident) 와 preserve
   method 간의 교호작용 — 6-config grid (selection × WC) 와 4-config grid (preserve)
   를 함께 사용하면 12+ 조합 비교 가능.

---

## 8. AWC 구현의 한계 / 안전 모드

GIFT 논문의 AWC 는 매 step Fisher = (∂L_KD/∂θ)² 를 계산하여
backbone 전체에 적용합니다. 본 구현은 **CoOp `ctx` 만 학습 대상**이므로

- Fisher 는 `torch.autograd.grad(L_distill, ctx, retain_graph=True,
  create_graph=False)` 로 계산하고 detach 함.
- 따라서 **2차 미분 그래프를 만들지 않음** → 메모리 안정.
- 단점: Fisher 자체는 미분 가능하지 않으므로, GIFT 논문에서 이론적으로 의도한 "Fisher
  가 학습 dynamics 에 따라 부드럽게 변하는" 효과는 1-step 근사로만 살아 있음.
- `ctx` 가 forward graph 에 등장하지 않는 코너 케이스 (`allow_unused=True`) 에서는
  zero Fisher 를 반환하여 손실은 0 이 되도록 처리.

비활성화: `--gift-use-awc` 를 빼거나 `--gift-lambda-awc 0` 으로 둠.

---

## 9. 자주 마주칠 수 있는 오류

- `metadata.jsonl` 의 어떤 라인에 필수 필드가 빠진 경우 → `gift_dataset.py` 에서
  **줄 번호 + 누락된 필드명** 을 명시한 `ValueError` 발생.
- `--gift-synthetic-root` 와 기존 `--preserve` (CIFAR100/ImageNet) 가
  **같은 디렉토리** 를 가리키는 경우 → 누설 위험. 코드가 경고를 출력함.
- `--gift-batch-size` 가 actual unique synthetic image 수보다 큰 경우 → DataLoader
  가 자연스럽게 drop_last/sampling 처리. 매 epoch 다른 batch 가 들어옴.
- placeholder 모드에서는 의미 있는 visual content 가 없으므로 **실제 보존 성능 검증용
  으로는 절대 쓰지 말 것**. smoke test 전용.

---

## 10. 짧은 한국어 요약

- **무엇이 바뀌었나?** 기존 ZSCL refdistill 코드는 그대로 두고, GIFT-inspired
  synthetic data 기반 보존을 `--preserve-method gift_synthetic` 옵션으로 추가함.
  `train_blue_target_coop_gift.py` 가 새 진입점.
- **왜 바뀌었나?** real reference image (ImageNet 등) 대신 stable diffusion
  으로 생성한 image-prompt 매칭 쌍으로 zero-shot transfer ability 를 보존하기
  위함. GIFT 논문의 CD + ITA + AWC 손실을 CoOp prompt tuning 환경에 맞게 단순화한
  형태로 이식.
- **어떻게 실행하나?**
  1. `scripts/generate_synthetic_dataset.py` 로 합성 데이터 사전 생성 (placeholder 또는 SD).
  2. `scripts/smoke_test_gift.sh` 로 빠른 검증 (2 epoch).
  3. `scripts/launch_gift_configs.sh` 로 4-config × 5 shot × 100 seed grid 실행.
- **무엇을 먼저 봐야 하나?** smoke test 의 `curves.jsonl` 에서
  `train_loss_cd_raw`, `train_loss_ita_raw`, `train_loss_awc_raw` 가 nonzero 인지
  확인. 그 다음 config A vs C 에서 DTD acc 와 CIFAR100/ImageNet acc 의 trade-off
  비교.
