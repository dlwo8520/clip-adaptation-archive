# CoOp prompt adaptation — run-archive

**Category:** Prompt and adapter baselines

**Repository path:** `CLIP/실험완료/CoOp/`

## Purpose

Explain shared-prompt adaptation and its target/preservation trade-off.

## Evidence and status

nonempty raw base results; duplicated copies exist.

## Method identity

- Experiment ID: `CLIP-001`
- Implemented method: CoOp prompt-only adaptation.
- Question: Test target/preservation trade-off.
- Baseline: original CLIP and package-specific random adaptation.
- Package role: historical experiment package.

The folder name `실험완료` is historical labeling. Completion follows nonempty run evidence, not the directory label. Duplicate source or result copies are not independent runs.

## Read first

- [CLIP/실험완료/CoOp/README.md](README.md)
- [CLIP/실험완료/CoOp/EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)
- [CLIP/실험완료/CoOp/scripts/run_smoke_gpu1.sh](scripts/run_smoke_gpu1.sh)
- [CLIP/실험완료/CoOp/scripts/check_dataset_structure.sh](scripts/check_dataset_structure.sh)
- [experiments/registry.json](../../../experiments/registry.json)
- [docs/EXPERIMENTS.md](../../../docs/EXPERIMENTS.md)
- [docs/RESULTS.md](../../../docs/RESULTS.md)

## Contents

| Entry | Role |
|---|---|
| [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) | Preserved source or evidence file |
| [README.md](README.md) | Original package overview |
| [[new]_plot_accuracy_results.py](%5Bnew%5D_plot_accuracy_results.py) | Preserved source or evidence file |
| [analyze_seed_tradeoff.py](analyze_seed_tradeoff.py) | Preserved source or evidence file |
| [dataset_structure_check.log](dataset_structure_check.log) | Preserved source or evidence file |
| [plot_16shot_best_cases.py](plot_16shot_best_cases.py) | Preserved source or evidence file |
| [plot_best_preserve_seed_case.py](plot_best_preserve_seed_case.py) | Preserved source or evidence file |
| [plot_blue_target_results.py](plot_blue_target_results.py) | Preserved source or evidence file |
| [plot_coop_accuracy_spaghetti_filtered.py](plot_coop_accuracy_spaghetti_filtered.py) | Preserved source or evidence file |
| [plot_coop_filtered_epochs.py](plot_coop_filtered_epochs.py) | Preserved source or evidence file |
| [plot_seed_aggregate_results.py](plot_seed_aggregate_results.py) | Preserved source or evidence file |
| [requirements.txt](requirements.txt) | Preserved source or evidence file |
| [runs](runs) | Saved analysis or execution evidence; completion is record-specific |
| [scripts](scripts) | Launch, analysis or validation commands |
| [smoke_run.log](smoke_run.log) | Preserved source or evidence file |
| [src](src) | Method implementation |
| [train_blue_target_coop.py](train_blue_target_coop.py) | Preserved source or evidence file |

## Category navigation

[Full categorized repository map](../../../docs/CATEGORIES.md)
