# Confidence, margin and prototype selection — source

**Category:** Support selection

**Repository path:** `CLIP/코드/ZSCL_DataSelection_all_methods_wrappers/`

## Purpose

Catalogue support-ranking wrappers and their surviving evidence.

## Evidence and status

empty local run records; graph summaries survive for several methods.

## Method identity

- Experiment ID: `CLIP-003`
- Implemented method: Confidence, margin, prototype, negative-confidence support ranking.
- Question: Choose stable support sets.
- Baseline: original CLIP and package-specific random adaptation.
- Package role: source variant.

The folder name `실험완료` is historical labeling. Completion follows nonempty run evidence, not the directory label. Duplicate source or result copies are not independent runs.

## Read first

- [CLIP/코드/ZSCL_DataSelection_all_methods_wrappers/run_smoke_after_gpu1_free.sh](run_smoke_after_gpu1_free.sh)
- [CLIP/코드/ZSCL_DataSelection_all_methods_wrappers/scripts/run_texture_adapter_zscl_100seeds_gpu1.sh](scripts/run_texture_adapter_zscl_100seeds_gpu1.sh)
- [experiments/registry.json](../../../experiments/registry.json)
- [docs/EXPERIMENTS.md](../../../docs/EXPERIMENTS.md)
- [docs/RESULTS.md](../../../docs/RESULTS.md)

## Contents

| Entry | Role |
|---|---|
| [launch_all_gpu1.log](launch_all_gpu1.log) | Preserved source or evidence file |
| [plot_adapter_accuracy_results.py](plot_adapter_accuracy_results.py) | Preserved source or evidence file |
| [requirements.txt](requirements.txt) | Preserved source or evidence file |
| [run_smoke_after_gpu1_free.sh](run_smoke_after_gpu1_free.sh) | Preserved source or evidence file |
| [scripts](scripts) | Launch, analysis or validation commands |
| [src](src) | Method implementation |
| [train_blue_target_coop_zscl.py](train_blue_target_coop_zscl.py) | Preserved source or evidence file |

## Category navigation

[Full categorized repository map](../../../docs/CATEGORIES.md)
