# Full-data LoRA teacher for support selection — run-archive

**Category:** Teacher-guided selection

**Repository path:** `CLIP/실험완료/CoOp_lora_full_teacher/`

## Purpose

Explain the stronger teacher and its additional full-target training resource.

## Evidence and status

nonempty LoRA/student results; additional full-data resource.

## Method identity

- Experiment ID: `CLIP-011`
- Implemented method: Full-data LoRA teacher plus CLIP agreement.
- Question: Test a stronger full-data selector.
- Baseline: original CLIP and package-specific random adaptation.
- Package role: historical experiment package.

The folder name `실험완료` is historical labeling. Completion follows nonempty run evidence, not the directory label. Duplicate source or result copies are not independent runs.

## Read first

- [CLIP/실험완료/CoOp_lora_full_teacher/README.md](README.md)
- [CLIP/실험완료/CoOp_lora_full_teacher/EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)
- [CLIP/실험완료/CoOp_lora_full_teacher/TWOTEACHER_EXPERIMENT.md](TWOTEACHER_EXPERIMENT.md)
- [CLIP/실험완료/CoOp_lora_full_teacher/scripts/run_smoke_gpu1.sh](scripts/run_smoke_gpu1.sh)
- [CLIP/실험완료/CoOp_lora_full_teacher/scripts/check_dataset_structure.sh](scripts/check_dataset_structure.sh)
- [experiments/registry.json](../../../experiments/registry.json)
- [docs/EXPERIMENTS.md](../../../docs/EXPERIMENTS.md)
- [docs/RESULTS.md](../../../docs/RESULTS.md)

## Contents

| Entry | Role |
|---|---|
| [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) | Preserved source or evidence file |
| [LORA_FULL_TEACHER_EXPERIMENT.md](LORA_FULL_TEACHER_EXPERIMENT.md) | Preserved source or evidence file |
| [README.md](README.md) | Original package overview |
| [TWOTEACHER_EXPERIMENT.md](TWOTEACHER_EXPERIMENT.md) | Preserved source or evidence file |
| [[new2]_plot_accuracy_results_gift_auto_tradeoff.py](%5Bnew2%5D_plot_accuracy_results_gift_auto_tradeoff.py) | Preserved source or evidence file |
| [analyze_seed_tradeoff.py](analyze_seed_tradeoff.py) | Preserved source or evidence file |
| [plot_lora_full_teacher_results.py](plot_lora_full_teacher_results.py) | Preserved source or evidence file |
| [plot_lora_random_wc_compare.py](plot_lora_random_wc_compare.py) | Preserved source or evidence file |
| [plots](plots) | Recorded visual summaries |
| [requirements.txt](requirements.txt) | Preserved source or evidence file |
| [runs](runs) | Saved analysis or execution evidence; completion is record-specific |
| [scripts](scripts) | Launch, analysis or validation commands |
| [src](src) | Method implementation |
| [train_blue_target_coop.py](train_blue_target_coop.py) | Preserved source or evidence file |
| [train_lora_full_teacher_twoteacher_coop.py](train_lora_full_teacher_twoteacher_coop.py) | Preserved source or evidence file |
| [train_twoteacher_selected_coop.py](train_twoteacher_selected_coop.py) | Preserved source or evidence file |

## Category navigation

[Full categorized repository map](../../../docs/CATEGORIES.md)
