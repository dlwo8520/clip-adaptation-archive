# ZSCL Data Selection - Entropy Only Codepack

이 코드팩은 기존 `teacher_uncertain = -confidence` 방식 대신 `teacher_entropy` 방식만 남긴 버전임.

## 변경 요약

- `src/sample_selection.py`
  - `entropy = -sum p(c|x) log p(c|x)` 계산 추가.
  - selection score를 `teacher_data["entropy"]`로 고정.
  - `teacher_confident`, `teacher_margin`, `prototype`, 기존 `teacher_uncertain` 제거.
  - nested 구조 유지: `1-shot ⊂ 2-shot ⊂ 4-shot ⊂ 8-shot ⊂ 16-shot`.

- `train_blue_target_coop_zscl.py`
  - `--selection-method`는 `teacher_entropy`만 허용.
  - `selection_meta.json`에 `mean_entropy`, `mean_normalized_entropy`, `mean_gt_confidence` 저장.
  - margin 관련 저장값 제거.

- `scripts/`
  - entropy 전용 실행 스크립트만 포함.

## 사용 전 위치

압축을 풀고 프로젝트 폴더로 이동.

```bash
cd ZSCL_DataSelection_entropy_only
```

## 빠른 동작 확인

```bash
bash scripts/run_texture_coop_teacher_entropy_smoke_gpu1.sh
```

주의: smoke 결과는 논문/보고용으로 쓰면 안 됨. 코드가 도는지 확인하는 용도임.

## 100 sample_seed 순차 실행

```bash
bash scripts/run_texture_coop_teacher_entropy_100seeds_gpu1.sh
```

기본 설정:

- GPU: `CUDA_VISIBLE_DEVICES=1`
- sample_seed: `0~99`
- selection method: `teacher_entropy`
- candidate factor: `2`

## 100 sample_seed 병렬 실행

```bash
bash scripts/run_texture_coop_teacher_entropy_100seeds_gpu1_parallel.sh
```

기본 병렬 개수는 5개임.

병렬 개수를 줄이려면:

```bash
PARALLEL=3 bash scripts/run_texture_coop_teacher_entropy_100seeds_gpu1_parallel.sh
```

10개 seed만 먼저 돌리려면:

```bash
NUM_SAMPLE_SEEDS=10 bash scripts/run_texture_coop_teacher_entropy_100seeds_gpu1_parallel.sh
```

GPU를 바꾸려면:

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/run_texture_coop_teacher_entropy_100seeds_gpu1_parallel.sh
```

## 결과 저장 위치

```text
runs/zscl_texture_entropy_only_gpu1/coop/teacher_entropy/sample_seed_{SAMPLE_SEED}/
```

주요 파일:

```text
args.json
selection_meta.json
zero_shot_baselines.json
curves.jsonl
results.json
shot_1/history.json
shot_2/history.json
shot_4/history.json
shot_8/history.json
shot_16/history.json
```

## 해석상 주의

entropy가 높은 샘플은 teacher가 헷갈리는 샘플임. 따라서 target adaptation에는 강한 신호가 될 수 있지만, preserve 성능 관점에서는 더 불안정할 수도 있음. 이 방법은 “좋은 보존 선택법”이라기보다 `불확실 샘플을 entropy로 정의했을 때 어떤 trade-off가 생기는지` 확인하는 실험에 가까움.
