# Reference distillation with WC controls — source

**Category:** Reference preservation

**Repository path:** `CLIP/코드/ZSCL_DataSelection_refdistill_2/`

## Purpose

Separate three selection criteria and WC-on/off reference-preservation conditions.

## Evidence and status

local result metadata absent/empty; standalone graph tables survive.

## Method identity

- Experiment ID: `CLIP-007`
- Implemented method: Three selection criteria × WC on/off.
- Question: Evaluate distillation and anchoring.
- Baseline: original CLIP and package-specific random adaptation.
- Package role: source variant.

The folder name `실험완료` is historical labeling. Completion follows nonempty run evidence, not the directory label. Duplicate source or result copies are not independent runs.

## Read first

- [CLIP/코드/ZSCL_DataSelection_refdistill_2/README.md](README.md)
- [CLIP/코드/ZSCL_DataSelection_refdistill_2/scripts/launch_6_configs_parallel.sh](scripts/launch_6_configs_parallel.sh)
- [CLIP/코드/ZSCL_DataSelection_refdistill_2/scripts/run_coop_refdistill_one_config.sh](scripts/run_coop_refdistill_one_config.sh)
- [experiments/registry.json](../../../experiments/registry.json)
- [docs/EXPERIMENTS.md](../../../docs/EXPERIMENTS.md)
- [docs/RESULTS.md](../../../docs/RESULTS.md)

## Contents

| Entry | Role |
|---|---|
| [README.md](README.md) | Original package overview |
| [requirements.txt](requirements.txt) | Preserved source or evidence file |
| [scripts](scripts) | Launch, analysis or validation commands |
| [src](src) | Method implementation |
| [train_blue_target_coop_zscl.py](train_blue_target_coop_zscl.py) | Preserved source or evidence file |
| [train_blue_target_coop_zscl_refdistill.py](train_blue_target_coop_zscl_refdistill.py) | Preserved source or evidence file |

## Category navigation

[Full categorized repository map](../../../docs/CATEGORIES.md)
