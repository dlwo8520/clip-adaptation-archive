# Two-Teacher Selected CoOp

This package adds `train_twoteacher_selected_coop.py` to the existing CoOp code.

The experiment asks whether a support set is better when it is selected by both:

- original CLIP zero-shot confidence
- a strong fine-tuned CoOp teacher
- similarity between their prediction distributions

## Teacher Selection

The script reads an existing multi-seed run directory:

```text
runs/blue_texture_preserve_100seeds/sample_seed_*/results.json
```

For each teacher shot in `1 2 4 8 16`, it selects the top `--teacher-top-k`
checkpoints by final target accuracy.

Default diagnostic size:

```text
5 teacher shots x top 3 teachers x matched student shot = 15 runs
```

Full grid:

```bash
--pairing-mode full
```

This runs:

```text
5 teacher shots x top 3 teachers x 5 student shots = 75 runs
```

## Score

For image `x` with true label `y`:

```text
p0 = original CLIP zero-shot distribution
p1 = fine-tuned CoOp teacher distribution

score =
  lambda_clip       * p0[y]
+ lambda_teacher    * p1[y]
+ lambda_similarity * (1 - normalized_JS(p0, p1))
```

Use `--require-both-correct` to first prefer samples where both models predict
the true class. If a class does not have enough strict samples, the script fills
the remainder with the best score-ranked samples unless `--strict-selection` is
also passed.

Use `--exclude-teacher-support` when you want to avoid selecting the same
few-shot samples that trained the teacher.

## Main Output

```text
runs/twoteacher_texture_diag/
├── selected_teachers.json
├── zero_shot_baselines.json
├── prompt_initialization.json
├── prompt_initialization.pt
├── curves.jsonl
├── results.json
└── teacher_shot_*_rank_*_seed_*/
    ├── candidate_score_summary.json
    └── student_shot_*/
        ├── selection.json
        ├── history.json
        └── coop_prompt_final.pt
```

`selection.json` is the key audit file. It stores selected image paths, labels,
scores, CLIP confidence, teacher confidence, distribution similarity, and
whether each model predicted the true class.

## Quick Run

After the original 100-seed run exists:

```bash
bash scripts/run_twoteacher_texture_diag.sh
```

If your teacher results live elsewhere:

```bash
TEACHER_RUNS_ROOT=/path/to/blue_texture_preserve_100seeds \
OUT_DIR=runs/twoteacher_texture_diag \
bash scripts/run_twoteacher_texture_diag.sh
```
