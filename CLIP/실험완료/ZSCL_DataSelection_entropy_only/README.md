# Predictive-entropy selection — run-archive

**Category:** Support selection

**Repository path:** `CLIP/실험완료/ZSCL_DataSelection_entropy_only/`

## Purpose

Distinguish predictive entropy from negative-confidence uncertainty.

## Evidence and status

empty local result records; standalone tables survive.

## Method identity

- Experiment ID: `CLIP-005`
- Implemented method: Predictive entropy rather than negative confidence.
- Question: Test uncertainty criteria.
- Baseline: original CLIP and package-specific random adaptation.
- Package role: historical experiment package.

The folder name `실험완료` is historical labeling. Completion follows nonempty run evidence, not the directory label. Duplicate source or result copies are not independent runs.

## Read first

- [CLIP/실험완료/ZSCL_DataSelection_entropy_only/README_ENTROPY_ONLY.md](README_ENTROPY_ONLY.md)
- [CLIP/실험완료/ZSCL_DataSelection_entropy_only/scripts/run_texture_coop_teacher_entropy_100seeds_gpu1.sh](scripts/run_texture_coop_teacher_entropy_100seeds_gpu1.sh)
- [CLIP/실험완료/ZSCL_DataSelection_entropy_only/scripts/run_texture_coop_teacher_entropy_smoke_gpu1.sh](scripts/run_texture_coop_teacher_entropy_smoke_gpu1.sh)
- [experiments/registry.json](../../../experiments/registry.json)
- [docs/EXPERIMENTS.md](../../../docs/EXPERIMENTS.md)
- [docs/RESULTS.md](../../../docs/RESULTS.md)

## Contents

| Entry | Role |
|---|---|
| [README_ENTROPY_ONLY.md](README_ENTROPY_ONLY.md) | Preserved source or evidence file |
| [[new]_plot_accuracy_results.py](%5Bnew%5D_plot_accuracy_results.py) | Preserved source or evidence file |
| [plot_adapter_accuracy_results.py](plot_adapter_accuracy_results.py) | Preserved source or evidence file |
| [requirements.txt](requirements.txt) | Preserved source or evidence file |
| [runs](runs) | Saved analysis or execution evidence; completion is record-specific |
| [scripts](scripts) | Launch, analysis or validation commands |
| [src](src) | Method implementation |
| [train_blue_target_coop_zscl.py](train_blue_target_coop_zscl.py) | Preserved source or evidence file |

## Category navigation

[Full categorized repository map](../../../docs/CATEGORIES.md)
