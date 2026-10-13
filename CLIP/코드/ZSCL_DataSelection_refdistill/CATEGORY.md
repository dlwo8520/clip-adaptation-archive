# Reference distribution distillation — source

**Category:** Reference preservation

**Repository path:** `CLIP/코드/ZSCL_DataSelection_refdistill/`

## Purpose

Explain reference image/text distillation and optional context anchoring.

## Evidence and status

source/scripts present; recoverable completion must be checked per artifact.

## Method identity

- Experiment ID: `CLIP-006`
- Implemented method: Reference text/image distribution distillation plus optional context anchoring.
- Question: Preserve reference predictions.
- Baseline: original CLIP and package-specific random adaptation.
- Package role: source variant.

The folder name `실험완료` is historical labeling. Completion follows nonempty run evidence, not the directory label. Duplicate source or result copies are not independent runs.

## Read first

- [CLIP/코드/ZSCL_DataSelection_refdistill/README.md](README.md)
- [CLIP/코드/ZSCL_DataSelection_refdistill/scripts/launch_6_configs_parallel.sh](scripts/launch_6_configs_parallel.sh)
- [CLIP/코드/ZSCL_DataSelection_refdistill/scripts/run_coop_refdistill_one_config.sh](scripts/run_coop_refdistill_one_config.sh)
- [experiments/registry.json](../../../experiments/registry.json)
- [docs/EXPERIMENTS.md](../../../docs/EXPERIMENTS.md)
- [docs/RESULTS.md](../../../docs/RESULTS.md)

## Contents

| Entry | Role |
|---|---|
| [README.md](README.md) | Original package overview |
| [[new]_plot_accuracy_results.py](%5Bnew%5D_plot_accuracy_results.py) | Preserved source or evidence file |
| [requirements.txt](requirements.txt) | Preserved source or evidence file |
| [scripts](scripts) | Launch, analysis or validation commands |
| [src](src) | Method implementation |
| [train_blue_target_coop_zscl.py](train_blue_target_coop_zscl.py) | Preserved source or evidence file |
| [train_blue_target_coop_zscl_refdistill.py](train_blue_target_coop_zscl_refdistill.py) | Preserved source or evidence file |

## Category navigation

[Full categorized repository map](../../../docs/CATEGORIES.md)
