# Prompt text-drift selection — run-archive

**Category:** Support selection

**Repository path:** `CLIP/실험완료/ZSCL_CoOp_TextDrift_Selection/`

## Purpose

Explain one-step prompt perturbation and vocabulary-feature drift selection.

## Evidence and status

100 nonempty support-seed result records found.

## Method identity

- Experiment ID: `CLIP-004`
- Implemented method: One-step prompt-gradient perturbation and vocabulary text-feature drift.
- Question: Select supports with smaller text drift.
- Baseline: original CLIP and package-specific random adaptation.
- Package role: historical experiment package.

The folder name `실험완료` is historical labeling. Completion follows nonempty run evidence, not the directory label. Duplicate source or result copies are not independent runs.

## Read first

- [CLIP/실험완료/ZSCL_CoOp_TextDrift_Selection/README.md](README.md)
- [CLIP/실험완료/ZSCL_CoOp_TextDrift_Selection/scripts/run_texture_coop_textdrift_100seeds.sh](scripts/run_texture_coop_textdrift_100seeds.sh)
- [CLIP/실험완료/ZSCL_CoOp_TextDrift_Selection/scripts/run_smoke_textdrift.sh](scripts/run_smoke_textdrift.sh)
- [experiments/registry.json](../../../experiments/registry.json)
- [docs/EXPERIMENTS.md](../../../docs/EXPERIMENTS.md)
- [docs/RESULTS.md](../../../docs/RESULTS.md)

## Contents

| Entry | Role |
|---|---|
| [README.md](README.md) | Original package overview |
| [[new]_plot_accuracy_results.py](%5Bnew%5D_plot_accuracy_results.py) | Preserved source or evidence file |
| [plot_adapter_accuracy_results.py](plot_adapter_accuracy_results.py) | Preserved source or evidence file |
| [plot_seed_aggregate_results.py](plot_seed_aggregate_results.py) | Preserved source or evidence file |
| [requirements.txt](requirements.txt) | Preserved source or evidence file |
| [runs](runs) | Saved analysis or execution evidence; completion is record-specific |
| [scripts](scripts) | Launch, analysis or validation commands |
| [src](src) | Method implementation |
| [train_blue_target_coop_zscl.py](train_blue_target_coop_zscl.py) | Preserved source or evidence file |

## Category navigation

[Full categorized repository map](../../../docs/CATEGORIES.md)
