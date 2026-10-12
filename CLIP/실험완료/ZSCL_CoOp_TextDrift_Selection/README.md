# ZSCL CoOp + Text-Drift Data Selection

CoOp few-shot adaptation에서 **임의 데이터셋에 대한 일반 zero-shot 능력 보존**을
목표로 하는 data selection 구현입니다. 평가 데이터셋(ImageNet, CIFAR100 등)에
selection이 종속되는 한계를 우회하기 위해, image dataset을 일절 사용하지 않고
**vocabulary V (예: ImageNet-1K classnames)에 대한 text feature drift만** 측정합니다.

## 핵심 아이디어

PDS 논문(Gu et al., ICLR 2025)의 Theorem 2.1 Eq.6은 다음을 보입니다:

$$\text{score}(x_n) \propto \lambda_{t+1}^\top \nabla l(x_n, \theta_t)$$

여기서 $\lambda_{t+1}$은 downstream loss $J(\theta)$의 gradient를 backprop으로
누적한 "목표 방향 벡터"입니다.

본 구현은:

- **$J(\theta) = \frac{1}{|V|} \sum_v \|\phi(v;\text{ctx}) - \phi(v;\text{ctx}_0)\|^2$**
  로 정의 (vocabulary V의 평균 text feature drift).
- 한-스텝 근사: $\lambda_{t+1} \approx -\nabla J(\theta_0)$.
- 결과: **"이 sample로 ctx를 한 번 학습시키면 V의 text feature가 얼마나 흔들리는가"**
  를 직접 측정.

CoOp이 학습하는 것은 ctx 하나뿐이고 그 ctx가 모든 클래스의 text feature를
만들어내므로, V 전체의 평균 drift가 **"이 sample이 일반 prompt를 망가뜨리는 정도"**
를 잘 근사합니다. **image reference가 필요 없다**는 것이 핵심 — CLIP 학습 분포
(WebImageText)에 직접 접근할 수 없는 상황의 우회책입니다.

## 추가된 selection method

기존 5개 method (`random`, `teacher_confident`, `teacher_margin`, `prototype`,
`teacher_uncertain`)에 새로운 옵션이 추가되었습니다:

```
--selection-method text_drift
```

이 method가 선택되면 `compute_text_drift_scores`가 호출되어 train pool 전체에
대해 drift score를 계산합니다 (한 번만, 모든 sample_seed에서 재사용).

## 파일 구조

```
ZSCL_CoOp_TextDrift_Selection/
├── README.md
├── requirements.txt
├── train_blue_target_coop_zscl.py     # MODIFIED: text_drift 분기 추가
├── src/
│   ├── coop_model.py                   # 변경 없음
│   ├── data.py                         # 변경 없음
│   ├── eval.py                         # 변경 없음
│   ├── utils.py                        # 변경 없음
│   ├── sample_selection.py             # MODIFIED: text_drift 분기 추가
│   └── text_drift_selection.py         # NEW: 핵심 구현
└── scripts/
    ├── run_smoke_textdrift.sh          # 빠른 sanity check
    └── run_texture_coop_textdrift_100seeds.sh   # 100 sample-seed 실험 launcher
```

## 핵심 코드 흐름 (`src/text_drift_selection.py`)

1. **Vocabulary V 로드** — 기본 ImageNet-1K classnames (torchvision fallback).
   파일로 override 가능.
2. **Prompt pieces 사전 계산** — V × templates 조합에 대해 `prefix`/`suffix`/
   `tokenized`를 미리 만들어 둠 (ctx 독립).
3. **Reference text features** — `base_ctx`에서의 $\phi_V(\text{base\_ctx})$를
   한 번만 계산.
4. **Per-sample drift loop**:
   - `ctx_param` = base_ctx (leaf, requires_grad)
   - target text features at ctx_param + per-sample CE loss + autograd
   - `g_n` = ∂loss / ∂ctx
   - `ctx_pert` = base_ctx − step_size × g_n
   - $\phi_V(\text{ctx\_pert})$ 계산
   - $\text{drift}(x_n) = \frac{1}{|V|} \sum_v \|\phi_V^{\text{pert}}(v) - \phi_V^{\text{base}}(v)\|^2$
5. **drift_score** = −drift_value (높을수록 보존 우수)
6. **Nested top-K selection** — 기존 ZSCL 프레임워크를 그대로 재사용.
   클래스마다 score 상위 K = max_shot × candidate_factor 후보를 만들고,
   `sample_seed`로 후보 풀 안에서 셔플 → 1-shot ⊂ 2-shot ⊂ … ⊂ 16-shot 보장.

## CLI 사용 예

### 단일 실행

```bash
python train_blue_target_coop_zscl.py \
  --target-name texture \
  --target-root /data/datasets/texture \
  --preserve imagenet:/data/datasets/imagenet_1k/val \
             cifar100:/data/datasets/cifar100/test \
  --model ViT-B-32-quickgelu --pretrained openai \
  --shots-list 1 2 4 8 16 \
  --n-ctx 16 --epochs 50 --eval-every 5 \
  --lr 0.002 --batch-size 128 \
  --seed 42 --prompt-seed 777 --sample-seed 0 \
  --selection-method text_drift \
  --selection-candidate-factor 2 \
  --text-drift-vocab-size 1000 \
  --text-drift-step-size 0.01 \
  --text-drift-templates default \
  --cache-features \
  --output-dir runs/coop_textdrift/sample_seed_0
```

### text_drift 전용 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--text-drift-vocab-file` | None | vocabulary 파일 경로. 없으면 torchvision ImageNet-1K. |
| `--text-drift-vocab-size` | -1 | vocab을 random subsample. -1이면 전체. |
| `--text-drift-vocab-seed` | 12345 | vocab subsample용 seed. 모든 sample_seed에서 동일하게 유지 권장. |
| `--text-drift-step-size` | 0.01 | ctx_pert step. ranking에는 영향 없으나 작을수록 linearization 정확. |
| `--text-drift-templates` | default | `default`: `["{}."]` (CoOp 학습 형식). `robust`: 4개 템플릿 평균. |
| `--text-drift-chunk-size` | 1024 | text encoder forward chunk 크기 (메모리 조절). |

### 100-seed 실험

```bash
ROOT_OUT=runs/zscl_coop_textdrift \
PARALLEL=2 \
NUM_SAMPLE_SEEDS=100 \
VOCAB_SIZE=1000 \
STEP_SIZE=0.01 \
bash scripts/run_texture_coop_textdrift_100seeds.sh
```

기존 5개 method와 비교하기 위해 같은 `ROOT_OUT` 아래 method별 폴더로 정리하면
plotting code(`plot_adapter_accuracy_results.py` 등)에서 그대로 비교 가능합니다.

## 결과 파일

각 `sample_seed_N` 폴더에 다음이 생성됩니다:

- `args.json` — 실험 설정
- `prompt_initialization.json/.pt` — base_ctx (모든 shot 공유)
- `selection_meta.json` — text_drift 메타정보 (vocab_size, step_size, drift 통계)
- `text_drift_scores.pt` — 모든 candidate의 drift_value/drift_score (분석용)
- `zero_shot_baselines.json` — 원본 CLIP zero-shot baseline
- `curves.jsonl` — epoch별 target/preserve accuracy (그래프용)
- `shot_K/coop_prompt_final.pt` — 학습된 prompt
- `shot_K/history.json` — 해당 shot의 학습 기록
- `results.json` — 전체 요약

## 계산 비용

ViT-B-32, V=1000, DTD train pool ~3760 sample 기준:

- 사전 계산: ImageNet-1K 1000 prompts forward ≈ 0.3s (1회)
- 후보별 drift: 1 backward (target 47 prompts) + 1 forward (V 1000 prompts)
  ≈ 0.3s/candidate
- **총 selection 비용 ≈ 20–25분** (1 GPU, V100/A100 기준).
- 이 비용은 **sample_seed에 무관**하므로 100-seed 실험에서도 1번만 발생.

## 기존 5개 method와의 비교 가설

| Method | $J(\theta)$ 정의 | 예상 동작 |
|---|---|---|
| random | (없음) | baseline |
| teacher_confident | DTD CE의 1-step에서 작은 \|gradient\| | teacher가 잘 푸는 prototypical sample 선호 — preserve가 *간접적*으로 좋아질 수 있음 |
| teacher_margin | (위와 유사) | decision boundary에서 안정적 |
| prototype | DTD class prototype 거리 | DTD class 평균 표현에 가까움 |
| teacher_uncertain | (대조군) | preserve가 더 많이 망가짐 |
| **text_drift** | **V의 text feature 평균 drift** | **preserve를 *직접* 보존 목적으로 사용** |

`teacher_confident`는 우연히 preserve를 보호할 수도 있지만, `text_drift`는
**보존을 명시적으로 selection 목표에 포함**한 것이 핵심 차이입니다.

## 한계와 후속 실험

1. **단일 V 의존성**: ImageNet-1K classnames에 selection이 편향될 수 있음.
   - mitigation: `--text-drift-vocab-file`로 더 broad한 vocab 사용 (ImageNet-21K,
     WordNet nouns, GPT 합성 명사 등).
2. **한-스텝 근사**: PDS의 long-range λ를 1-step gradient로 단순화함.
   - PDS Table 5에 따르면 long-range가 single-step보다 우월하나, few-shot scale
     에서는 우월성이 줄어들 가능성. ablation 필요.
3. **DTD 클래스가 V와 겹칠 가능성**: DTD는 texture라 일반 명사와 거의 안 겹치지만,
   다른 target에서는 V에서 target classnames를 제외하는 후처리 필요할 수 있음.

## 참고문헌

- **PDS**: Gu et al. *Data Selection via Optimal Control for Language Models.*
  ICLR 2025. https://arxiv.org/abs/2410.07064
- **ZSCL**: Zheng et al. *Preventing Zero-Shot Transfer Degradation in Continual
  Learning of Vision-Language Models.* ICCV 2023.
- **CoOp**: Zhou et al. *Learning to Prompt for Vision-Language Models.*
  IJCV 2022.
