# CoCoOp Added Guide

이 버전은 기존 `CoOp _` 프로젝트에 CoCoOp 실험 코드를 추가한 버전임.

## 추가/수정된 파일

### 수정
- `src/coop_model.py`
  - `CoCoOpPromptLearner` 추가
  - `build_cocoop_logits(...)` 추가

### 추가
- `train_blue_target_cocoop.py`
  - 기존 `train_blue_target_coop.py`와 같은 데이터 로딩, few-shot sampling, 결과 저장 포맷 사용
  - 모델만 CoCoOp 구조로 변경
- `scripts/run_texture_cocoop_gpu1.sh`
  - 단일 sample seed용 CoCoOp 실행 스크립트
- `scripts/run_texture_cocoop_100seeds_gpu1.sh`
  - sample_seed 0~99 반복 실행용 CoCoOp 스크립트

## 기존 그래프 코드 재사용

CoCoOp 결과도 기존과 같은 구조로 저장됨.

- `curves.jsonl`
- `results.json`
- `zero_shot_baselines.json`
- `shot_*/history.json`

따라서 기존 그래프 코드를 그대로 사용할 수 있음.

예:

```bash
python plot_seed_aggregate_results.py \
  --root-dir runs/cocoop_texture_preserve_100seeds
```

## 실행 전 주의점

CoCoOp은 이미지마다 prompt가 달라지는 구조임.
그래서 CoOp처럼 class별 text feature를 한 번 만들고 재사용할 수 없음.
특히 ImageNet 1000-class preserve evaluation은 CoOp보다 훨씬 느림.

먼저 1~3개 seed로 smoke test 권장.

```bash
cd ${RESEARCH_WORKSPACE}/projects/blue_target_coop_project_v2_prompt_seed

rm -rf runs/cocoop_texture_preserve_100seeds
CUDA_VISIBLE_DEVICES=1 NUM_SAMPLE_SEEDS=3 bash scripts/run_texture_cocoop_100seeds_gpu1.sh
```

## 전체 100 seed 실행

```bash
cd ${RESEARCH_WORKSPACE}/projects/blue_target_coop_project_v2_prompt_seed

rm -rf runs/cocoop_texture_preserve_100seeds
CUDA_VISIBLE_DEVICES=1 NUM_SAMPLE_SEEDS=100 nohup bash scripts/run_texture_cocoop_100seeds_gpu1.sh > texture_cocoop_100seeds_master.log 2>&1 &
echo "PID=$!"
```

## 진행 확인

```bash
tail -f texture_cocoop_100seeds_master.log
find runs/cocoop_texture_preserve_100seeds -name curves.jsonl | wc -l
nvidia-smi
```

## CoOp과 다른 점

CoOp:
- 모든 이미지에 대해 같은 prompt context 사용
- static text classifier 생성 가능

CoCoOp:
- image feature를 meta-net에 넣어 이미지별 prompt bias 생성
- 이미지마다 text classifier가 달라짐
- evaluation이 훨씬 느림

## 주요 argument

- `--cocoop-class-chunk-size`
  - 한 번에 text encoder로 처리할 class 수
  - ImageNet OOM 방지용
  - 기본 128

- `--cocoop-eval-batch-size`
  - CoCoOp evaluation에서 한 번에 처리할 image feature 수
  - 기본 1
  - 늘리면 빠르지만 GPU memory 사용량 증가

- `--batch-size`
  - CoCoOp training batch size
  - 공식 CoCoOp은 batch size 1을 주로 사용함
  - 여기서는 기본 스크립트에서 4로 설정함

## 결과 비교 아이디어

CoOp 결과:
- `runs/blue_texture_preserve_100seeds`

CoCoOp 결과:
- `runs/cocoop_texture_preserve_100seeds`

두 결과의 `seed_epoch_summary.csv`를 비교하면 됨.
