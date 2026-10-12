# LoRA Full-Data Teacher Two-Teacher Selection

This experiment trains one LoRA CLIP teacher on the full target train pool, then
uses original CLIP and the LoRA teacher together to select few-shot support sets
for a Student CoOp model.

The LoRA teacher is **not** few-shot. It sees the target train pool. The Student
CoOp is still trained with selected 1/2/4/8/16-shot supports.

## Question

```text
If a strong full-data LoRA teacher and original CLIP agree on target train-pool
images, are those images good few-shot supports for Student CoOp?
```

## Pipeline

```text
target train pool
  -> train LoRA CLIP teacher
  -> score each train-pool image with original CLIP + LoRA teacher
  -> select class-balanced 1/2/4/8/16 supports
  -> train Student CoOp on each selected support
```

The score is:

```text
score =
  lambda_clip       * p_original_clip[true_label]
+ lambda_lora       * p_lora_teacher[true_label]
+ lambda_similarity * (1 - normalized_JS(p_original_clip, p_lora_teacher))
```

The default runner also uses:

```text
argmax(original CLIP) == true_label
argmax(LoRA teacher) == true_label
```

as a preferred strict filter. If a class does not have enough strict candidates,
the script fills the remainder with score-ranked samples unless
`--strict-selection` is passed.

## Run

```bash
CUDA_VISIBLE_DEVICES=5 bash scripts/run_lora_full_teacher_texture.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=5 \
OUT_DIR=runs/lora_full_teacher_twoteacher_texture_visual_r8 \
LORA_TARGET=visual \
LORA_RANK=8 \
LORA_ALPHA=16 \
LORA_EPOCHS=20 \
LORA_LR=1e-4 \
bash scripts/run_lora_full_teacher_texture.sh
```

To adapt both visual and text Linear layers:

```bash
LORA_TARGET=both bash scripts/run_lora_full_teacher_texture.sh
```

## Output

```text
runs/lora_full_teacher_twoteacher_texture/
├── args.json
├── zero_shot_baselines.json
├── curves.jsonl
├── results.json
├── lora_teacher/
│   ├── lora_config.json
│   ├── teacher_history.json
│   ├── teacher_eval.json
│   └── lora_teacher_final.pt
├── selected_support/
│   ├── candidate_score_summary.json
│   ├── shot_1_selection.json
│   ├── shot_2_selection.json
│   ├── shot_4_selection.json
│   ├── shot_8_selection.json
│   └── shot_16_selection.json
├── shot_1/
├── shot_2/
├── shot_4/
├── shot_8/
└── shot_16/
```

Compare this run against:

- random few-shot CoOp 100-seed mean/IQR
- previous CoOp-teacher two-teacher selection
- LoRA full-data teacher two-teacher selection
