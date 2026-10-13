# Two-teacher selection with prompt WC — run-archive

**Category:** Teacher-guided selection

**Repository path:** `CLIP/실험완료/CoOp_twoteacher_selected_wc/`

## Purpose

Explain prompt consolidation and keep different WC strengths separate.

## Evidence and status

nonempty WC-sweep results; different lambdas are distinct conditions.

## Method identity

- Experiment ID: `CLIP-010`
- Implemented method: Two-teacher selection plus prompt weight consolidation.
- Question: Reduce shared-context drift.
- Baseline: original CLIP and package-specific random adaptation.
- Package role: historical experiment package.

The folder name `실험완료` is historical labeling. Completion follows nonempty run evidence, not the directory label. Duplicate source or result copies are not independent runs.

## Read first

- [CLIP/실험완료/CoOp_twoteacher_selected_wc/README.md](README.md)
- [CLIP/실험완료/CoOp_twoteacher_selected_wc/EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)
- [CLIP/실험완료/CoOp_twoteacher_selected_wc/TWOTEACHER_EXPERIMENT.md](TWOTEACHER_EXPERIMENT.md)
- [CLIP/실험완료/CoOp_twoteacher_selected_wc/scripts/run_smoke_gpu1.sh](scripts/run_smoke_gpu1.sh)
- [CLIP/실험완료/CoOp_twoteacher_selected_wc/scripts/check_dataset_structure.sh](scripts/check_dataset_structure.sh)
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
| [plot_wc_lambda_spaghetti_all.py](plot_wc_lambda_spaghetti_all.py) | Preserved source or evidence file |
| [plot_wc_lambda_sweep.py](plot_wc_lambda_sweep.py) | Preserved source or evidence file |
| [plots](plots) | Recorded visual summaries |
| [requirements.txt](requirements.txt) | Preserved source or evidence file |
| [runs](runs) | Saved analysis or execution evidence; completion is record-specific |
| [scripts](scripts) | Launch, analysis or validation commands |
| [src](src) | Method implementation |
| [train_blue_target_coop.py](train_blue_target_coop.py) | Preserved source or evidence file |
| [train_twoteacher_selected_coop.py](train_twoteacher_selected_coop.py) | Preserved source or evidence file |

## Category navigation

[Full categorized repository map](../../../docs/CATEGORIES.md)
