# JS-quantile support trade-off — source

**Category:** Trade-off diagnostics

**Repository path:** `CLIP/코드/CoOp_js_tradeoff/`

## Purpose

Explain high-entropy anchors and JS-quantile support bins.

## Evidence and status

implemented and graph-derived records; raw results.json absent.

## Method identity

- Experiment ID: `CLIP-012`
- Implemented method: High-entropy anchor and JS-quantile support bins.
- Question: Diagnose sample-distribution trade-off.
- Baseline: original CLIP and package-specific random adaptation.
- Package role: source variant.

The folder name `실험완료` is historical labeling. Completion follows nonempty run evidence, not the directory label. Duplicate source or result copies are not independent runs.

## Read first

- [CLIP/코드/CoOp_js_tradeoff/README_JS_TRADEOFF.md](README_JS_TRADEOFF.md)
- [CLIP/코드/CoOp_js_tradeoff/scripts/run_js_tradeoff_random_remaining_100seeds_texture.sh](scripts/run_js_tradeoff_random_remaining_100seeds_texture.sh)
- [CLIP/코드/CoOp_js_tradeoff/scripts/run_js_tradeoff_modes_texture_diag.sh](scripts/run_js_tradeoff_modes_texture_diag.sh)
- [experiments/registry.json](../../../experiments/registry.json)
- [docs/EXPERIMENTS.md](../../../docs/EXPERIMENTS.md)
- [docs/RESULTS.md](../../../docs/RESULTS.md)

## Contents

| Entry | Role |
|---|---|
| [README_JS_TRADEOFF.md](README_JS_TRADEOFF.md) | Preserved source or evidence file |
| [plot_js_tradeoff_results.py](plot_js_tradeoff_results.py) | Preserved source or evidence file |
| [plots](plots) | Recorded visual summaries |
| [requirements.txt](requirements.txt) | Preserved source or evidence file |
| [scripts](scripts) | Launch, analysis or validation commands |
| [src](src) | Method implementation |
| [train_blue_target_coop.py](train_blue_target_coop.py) | Preserved source or evidence file |
| [train_js_tradeoff_coop.py](train_js_tradeoff_coop.py) | Preserved source or evidence file |

## Category navigation

[Full categorized repository map](../../../docs/CATEGORIES.md)
