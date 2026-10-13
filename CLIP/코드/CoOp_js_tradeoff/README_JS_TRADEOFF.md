# JS-Based Target-Preservation Trade-Off CoOp Diagnostic

This package extends the initial-CLIP entropy anchor experiment into a
JS-quantile trade-off analysis.

It is not a two-teacher method. It uses only the initial CLIP zero-shot
prediction distribution over the target train pool.

## Question

Does the distance from the entropy-anchor distribution explain a
target/preservation trade-off?

The experiment asks whether samples with larger
`JS(P_A, p_0(x))` produce different target and preserve behavior after Student
CoOp adaptation.

## Method

1. Score every target train-pool image with initial CLIP:

   `p_0(x) = softmax(tau f(x) T_zs^T)`

2. Compute entropy:

   `H(x) = -sum_c p_0(c|x) log p_0(c|x)`

3. Build entropy anchor set `D_A` from the high-entropy samples.

4. Build the anchor distribution:

   `P_A = mean_{x in D_A} p_0(x)`

5. Compute JS distance to the anchor distribution:

   `S_JS(x) = JS(P_A, p_0(x))`

6. Split samples into JS quantile bins:

   - `js_q1`: lowest JS group
   - `js_q2`: low-mid JS group
   - `js_q3`: high-mid JS group
   - `js_q4`: highest JS group

7. For each bin, build a class-balanced support set and train Student CoOp.

Inside each JS quantile, samples are chosen randomly per class. This is
intentional: the experiment tests the effect of the JS interval, not an
additional within-bin ranking rule.

With `--balance-mode true_label`, this is a label-aware diagnostic experiment.

## Selection Modes

Supported `--selection-mode` values:

- `random`
- `anchor_entropy`
- `js_q1`
- `js_q2`
- `js_q3`
- `js_q4`
- `agreement` and `disagreement` are also kept for compatibility with the
  previous endpoint experiment.

## Fallback

If a class does not have enough samples inside a requested JS bin, the code
fills from neighboring bins and records the fallback.

Examples:

- `js_q4`: `Q4 -> Q3 -> Q2 -> Q1`
- `js_q1`: `Q1 -> Q2 -> Q3 -> Q4`
- `js_q2`: `Q2 -> Q1 -> Q3 -> Q4`
- `js_q3`: `Q3 -> Q4 -> Q2 -> Q1`

The fallback is saved in:

- `shot_{k}/selection.json`
- `shot_{k}/score_stats.json`

## Main Outputs

Each run writes:

- `args.json`
- `zero_shot_baselines.json`
- `anchor_set.json`
- `anchor_distribution.json`
- `js_bins.json`
- `train_pool_score_overview.json`
- `curves.jsonl`
- `results.json`
- `shot_{k}/history.json`
- `shot_{k}/selection.json`
- `shot_{k}/score_stats.json`
- `shot_{k}/coop_prompt_final.pt`

`js_bins.json` is important. It records per-bin JS, entropy, confidence, sample
count, and true/pseudo label histograms.

## Run Six-Mode Diagnostic

```bash
cd ${RESEARCH_WORKSPACE}/projects/CoOp_js_tradeoff

CUDA_VISIBLE_DEVICES=5 \
ROOT_OUT=runs/js_tradeoff_texture_diag \
SAMPLE_SEED=0 \
SHOTS_LIST="1 2 4 8 16" \
bash scripts/run_js_tradeoff_modes_texture_diag.sh
```

For a quick 16-shot first check:

```bash
CUDA_VISIBLE_DEVICES=5 \
ROOT_OUT=runs/js_tradeoff_texture_diag_16shot \
SAMPLE_SEED=0 \
SHOTS_LIST="16" \
bash scripts/run_js_tradeoff_modes_texture_diag.sh
```

## Single Mode

```bash
CUDA_VISIBLE_DEVICES=5 \
SELECTION_MODE=js_q4 \
SAMPLE_SEED=0 \
OUT_DIR=runs/js_tradeoff_texture/js_q4_seed_0 \
bash scripts/run_js_tradeoff_texture.sh
```

## Random Remaining-Data 100 Seeds

This is useful as a random baseline from `D \\ D_A`.

```bash
CUDA_VISIBLE_DEVICES=5 \
ROOT_OUT=runs/js_tradeoff_random_remaining_100seeds \
SEED_START=0 \
NUM_SAMPLE_SEEDS=100 \
SHOTS_LIST="1 2 4 8 16" \
bash scripts/run_js_tradeoff_random_remaining_100seeds_texture.sh
```

## Plot Results

For the six-mode diagnostic:

```bash
python plot_js_tradeoff_results.py \
  --root-dir runs/js_tradeoff_texture_diag \
  --out-dir plots/js_tradeoff_texture_diag \
  --final-epoch 50 \
  --ylim 0.30 0.75
```

If you also ran the random 100-seed baseline, include it:

```bash
python plot_js_tradeoff_results.py \
  --root-dir runs/js_tradeoff_texture_diag \
  --random-root runs/js_tradeoff_random_remaining_100seeds \
  --out-dir plots/js_tradeoff_texture_diag_with_random100 \
  --seed-start 0 \
  --seed-end 99 \
  --final-epoch 50 \
  --ylim 0.30 0.75
```

The plot script writes:

- `epoch_curves/`
- `tradeoff/`
- `js_tradeoff_flat.csv`
- `js_tradeoff_summary.csv`

## Options

- `ANCHOR_RATIO=0.2`
- `ANCHOR_MODE=global`
- `ANCHOR_DIST_MODE=global`
- `NUM_JS_BINS=4`
- `BALANCE_MODE=true_label`
- `EXCLUDE_ANCHOR=1`

If global anchors remove too many samples from a class, try class-balanced
anchors:

```bash
ANCHOR_MODE=true_label \
ANCHOR_RATIO=0.2 \
bash scripts/run_js_tradeoff_modes_texture_diag.sh
```

or exact per-class anchors:

```bash
ANCHOR_MODE=true_label \
ANCHOR_PER_CLASS=12 \
bash scripts/run_js_tradeoff_modes_texture_diag.sh
```

## Interpretation

Compare each bin:

`Q_r -> (delta_target, delta_preserve)`

Possible outcomes:

- Higher JS reduces preserve degradation.
- Higher JS hurts both target and preserve, suggesting outlier/noisy samples.
- Middle JS bins give the best trade-off.

ImageNet and CIFAR100 are used only as preserve evaluation datasets in the
provided scripts.
