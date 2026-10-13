# Original and adapted two-teacher agreement — run-archive

**Category:** Teacher-guided selection

**Repository path:** `CLIP/실험완료/CoOp_twoteacher_selected/`

## Purpose

Explain support selection using original CLIP and an adapted CoOp teacher.

## Evidence and status

nonempty selected-student run results.

## Method identity

- Experiment ID: `CLIP-009`
- Implemented method: Original CLIP plus adapted CoOp teacher agreement/JS score.
- Question: Improve support quality.
- Baseline: original CLIP and package-specific random adaptation.
- Package role: historical experiment package.

The folder name `실험완료` is historical labeling. Completion follows nonempty run evidence, not the directory label. Duplicate source or result copies are not independent runs.

## Read first

- [CLIP/실험완료/CoOp_twoteacher_selected/README.md](README.md)
- [CLIP/실험완료/CoOp_twoteacher_selected/EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)
- [CLIP/실험완료/CoOp_twoteacher_selected/TWOTEACHER_EXPERIMENT.md](TWOTEACHER_EXPERIMENT.md)
- [CLIP/실험완료/CoOp_twoteacher_selected/scripts/run_smoke_gpu1.sh](scripts/run_smoke_gpu1.sh)
- [CLIP/실험완료/CoOp_twoteacher_selected/scripts/check_dataset_structure.sh](scripts/check_dataset_structure.sh)
- [experiments/registry.json](../../../experiments/registry.json)
- [docs/EXPERIMENTS.md](../../../docs/EXPERIMENTS.md)
- [docs/RESULTS.md](../../../docs/RESULTS.md)

## Contents

| Entry | Role |
|---|---|
| [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) | Preserved source or evidence file |
| [README.md](README.md) | Original package overview |
| [TWOTEACHER_EXPERIMENT.md](TWOTEACHER_EXPERIMENT.md) | Preserved source or evidence file |
| [analyze_seed_tradeoff.py](analyze_seed_tradeoff.py) | Preserved source or evidence file |
| [plot_twoteacher_vs_random_coop.py](plot_twoteacher_vs_random_coop.py) | Preserved source or evidence file |
| [requirements.txt](requirements.txt) | Preserved source or evidence file |
| [runs](runs) | Saved analysis or execution evidence; completion is record-specific |
| [scripts](scripts) | Launch, analysis or validation commands |
| [src](src) | Method implementation |
| [train_blue_target_coop.py](train_blue_target_coop.py) | Preserved source or evidence file |
| [train_twoteacher_selected_coop.py](train_twoteacher_selected_coop.py) | Preserved source or evidence file |

## Category navigation

[Full categorized repository map](../../../docs/CATEGORIES.md)
