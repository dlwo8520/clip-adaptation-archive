# Categorized repository map

Folders are grouped by research method. Each guide briefly states purpose and evidence status. File-list messages use `method / artifact type` categories for code, configs, records, tables, figures and notes.

## Prompt and adapter baselines

| Path | Purpose / guide |
|---|---|
| `CLIP/실험완료/CoOp/` | [CoOp prompt adaptation — run-archive](../CLIP/%EC%8B%A4%ED%97%98%EC%99%84%EB%A3%8C/CoOp/CATEGORY.md) |
| `CLIP/코드/CoOp/` | [CoOp prompt adaptation — source](../CLIP/%EC%BD%94%EB%93%9C/CoOp/CATEGORY.md) |
| `CLIP/실험완료/CLIP_Adapter_project_v4/` | [CLIP-Adapter, CoOp and CoCoOp — run-archive](../CLIP/%EC%8B%A4%ED%97%98%EC%99%84%EB%A3%8C/CLIP_Adapter_project_v4/CATEGORY.md) |
| `CLIP/코드/CLIP_Adapter_project_v4/` | [CLIP-Adapter, CoOp and CoCoOp — source](../CLIP/%EC%BD%94%EB%93%9C/CLIP_Adapter_project_v4/CATEGORY.md) |

## Support selection

| Path | Purpose / guide |
|---|---|
| `CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers/` | [Confidence, margin and prototype selection — run-archive](../CLIP/%EC%8B%A4%ED%97%98%EC%99%84%EB%A3%8C/ZSCL_DataSelection_all_methods_wrappers/README.md) |
| `CLIP/코드/ZSCL_DataSelection_all_methods_wrappers/` | [Confidence, margin and prototype selection — source](../CLIP/%EC%BD%94%EB%93%9C/ZSCL_DataSelection_all_methods_wrappers/README.md) |
| `CLIP/실험완료/ZSCL_CoOp_TextDrift_Selection/` | [Prompt text-drift selection — run-archive](../CLIP/%EC%8B%A4%ED%97%98%EC%99%84%EB%A3%8C/ZSCL_CoOp_TextDrift_Selection/CATEGORY.md) |
| `CLIP/실험완료/ZSCL_DataSelection_entropy_only/` | [Predictive-entropy selection — run-archive](../CLIP/%EC%8B%A4%ED%97%98%EC%99%84%EB%A3%8C/ZSCL_DataSelection_entropy_only/README.md) |

## Reference preservation

| Path | Purpose / guide |
|---|---|
| `CLIP/코드/ZSCL_DataSelection_refdistill/` | [Reference distribution distillation — source](../CLIP/%EC%BD%94%EB%93%9C/ZSCL_DataSelection_refdistill/CATEGORY.md) |
| `CLIP/실험완료/ZSCL_DataSelection_refdistill_2/` | [Reference distillation with WC controls — run-archive](../CLIP/%EC%8B%A4%ED%97%98%EC%99%84%EB%A3%8C/ZSCL_DataSelection_refdistill_2/CATEGORY.md) |
| `CLIP/코드/ZSCL_DataSelection_refdistill_2/` | [Reference distillation with WC controls — source](../CLIP/%EC%BD%94%EB%93%9C/ZSCL_DataSelection_refdistill_2/CATEGORY.md) |
| `CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/` | [GIFT-inspired synthetic-reference variants — run-archive](../CLIP/%EC%8B%A4%ED%97%98%EC%99%84%EB%A3%8C/ZSCL_DataSelection_refdistill_gift/CATEGORY.md) |

## Teacher-guided selection

| Path | Purpose / guide |
|---|---|
| `CLIP/실험완료/CoOp_twoteacher_selected/` | [Original and adapted two-teacher agreement — run-archive](../CLIP/%EC%8B%A4%ED%97%98%EC%99%84%EB%A3%8C/CoOp_twoteacher_selected/CATEGORY.md) |
| `CLIP/코드/CoOp_twoteacher_selected/` | [Original and adapted two-teacher agreement — source](../CLIP/%EC%BD%94%EB%93%9C/CoOp_twoteacher_selected/CATEGORY.md) |
| `CLIP/실험완료/CoOp_twoteacher_selected_wc/` | [Two-teacher selection with prompt WC — run-archive](../CLIP/%EC%8B%A4%ED%97%98%EC%99%84%EB%A3%8C/CoOp_twoteacher_selected_wc/CATEGORY.md) |
| `CLIP/실험완료/CoOp_lora_full_teacher/` | [Full-data LoRA teacher for support selection — run-archive](../CLIP/%EC%8B%A4%ED%97%98%EC%99%84%EB%A3%8C/CoOp_lora_full_teacher/CATEGORY.md) |

## Trade-off diagnostics

| Path | Purpose / guide |
|---|---|
| `CLIP/코드/CoOp_js_tradeoff/` | [JS-quantile support trade-off — source](../CLIP/%EC%BD%94%EB%93%9C/CoOp_js_tradeoff/README.md) |

## Graph evidence / Reference preservation

| Path | Purpose / guide |
|---|---|
| `CLIP/그래프/4_CoOp__teacher_confident__wc_off/` | [CoOp reference-distillation graphs: confidence, WC off](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/4_CoOp__teacher_confident__wc_off/README.md) |
| `CLIP/그래프/4_CoOp__teacher_confident__wc_on/` | [CoOp reference-distillation graphs: confidence, WC on](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/4_CoOp__teacher_confident__wc_on/README.md) |
| `CLIP/그래프/4_CoOp__teacher_entropy__wc_off/` | [CoOp reference-distillation graphs: entropy, WC off](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/4_CoOp__teacher_entropy__wc_off/README.md) |
| `CLIP/그래프/4_CoOp__teacher_entropy__wc_on/` | [CoOp reference-distillation graphs: entropy, WC on](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/4_CoOp__teacher_entropy__wc_on/README.md) |
| `CLIP/그래프/4_CoOp__teacher_margin__wc_off/` | [CoOp reference-distillation graphs: margin, WC off](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/4_CoOp__teacher_margin__wc_off/README.md) |
| `CLIP/그래프/4_CoOp__teacher_margin__wc_on/` | [CoOp reference-distillation graphs: margin, WC on](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/4_CoOp__teacher_margin__wc_on/README.md) |
| `CLIP/그래프/5_teacher_confident_B__synthetic_zscl_like/` | [Synthetic-reference graphs: confident_B, B: synthetic ZSCL-like](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/5_teacher_confident_B__synthetic_zscl_like/README.md) |
| `CLIP/그래프/5_teacher_confident_C__gift_synthetic/` | [Synthetic-reference graphs: confident_C, C: GIFT synthetic](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/5_teacher_confident_C__gift_synthetic/README.md) |
| `CLIP/그래프/5_teacher_confident_D__gift_synthetic_awc/` | [Synthetic-reference graphs: confident_D, D: GIFT synthetic AWC](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/5_teacher_confident_D__gift_synthetic_awc/README.md) |
| `CLIP/그래프/5_teacher_entropy_B__synthetic_zscl_like/` | [Synthetic-reference graphs: entropy_B, B: synthetic ZSCL-like](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/5_teacher_entropy_B__synthetic_zscl_like/README.md) |
| `CLIP/그래프/5_teacher_entropy_C__gift_synthetic/` | [Synthetic-reference graphs: entropy_C, C: GIFT synthetic](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/5_teacher_entropy_C__gift_synthetic/README.md) |
| `CLIP/그래프/5_teacher_entropy_D__gift_synthetic_awc/` | [Synthetic-reference graphs: entropy_D, D: GIFT synthetic AWC](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/5_teacher_entropy_D__gift_synthetic_awc/README.md) |
| `CLIP/그래프/5_teacher_margin_B__synthetic_zscl_like/` | [Synthetic-reference graphs: margin_B, B: synthetic ZSCL-like](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/5_teacher_margin_B__synthetic_zscl_like/README.md) |
| `CLIP/그래프/5_teacher_margin_C__gift_synthetic/` | [Synthetic-reference graphs: margin_C, C: GIFT synthetic](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/5_teacher_margin_C__gift_synthetic/README.md) |
| `CLIP/그래프/5_teacher_margin_D__gift_synthetic_awc/` | [Synthetic-reference graphs: margin_D, D: GIFT synthetic AWC](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/5_teacher_margin_D__gift_synthetic_awc/README.md) |

## Graph evidence / Prompt and adapter baselines

| Path | Purpose / guide |
|---|---|
| `CLIP/그래프/CoOp/` | [CoOp base graphs](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/CoOp/README.md) |
| `CLIP/그래프/adapter/` | [Visual adapter base graphs](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/adapter/README.md) |
| `CLIP/그래프/휴지통/` | [Legacy graph copies](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/%ED%9C%B4%EC%A7%80%ED%86%B5/README.md) |

## Graph evidence / Support selection

| Path | Purpose / guide |
|---|---|
| `CLIP/그래프/CoOp_confidenct/` | [CoOp confidence-selection graphs](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/CoOp_confidenct/README.md) |
| `CLIP/그래프/CoOp_margin/` | [CoOp margin-selection graphs](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/CoOp_margin/README.md) |
| `CLIP/그래프/CoOp_textdrift/` | [CoOp text-drift graphs](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/CoOp_textdrift/README.md) |
| `CLIP/그래프/CoOp_uncertain/` | [CoOp uncertainty-selection graphs](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/CoOp_uncertain/README.md) |
| `CLIP/그래프/adapter_teacher_confident/` | [Visual adapter confidence-selection graphs](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/adapter_teacher_confident/README.md) |
| `CLIP/그래프/adapter_teacher_margin/` | [Visual adapter margin-selection graphs](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/adapter_teacher_margin/README.md) |
| `CLIP/그래프/adapter_teacher_uncertain/` | [Visual adapter uncertainty-selection graphs](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/adapter_teacher_uncertain/README.md) |

## CLIP archive sections

| Path | Purpose / guide |
|---|---|
| `CLIP/코드/` | [CLIP source variants](../CLIP/%EC%BD%94%EB%93%9C/README.md) |
| `CLIP/실험완료/` | [Historical CLIP experiment packages](../CLIP/%EC%8B%A4%ED%97%98%EC%99%84%EB%A3%8C/README.md) |
| `CLIP/그래프/` | [CLIP graph and table collections](../CLIP/%EA%B7%B8%EB%9E%98%ED%94%84/README.md) |
| `CLIP/` | [CLIP source, run and graph archive](../CLIP/README.md) |

## Documentation

| Path | Purpose / guide |
|---|---|
| `docs/` | [Research navigation and reproducibility](README.md) |

## Environments

| Path | Purpose / guide |
|---|---|
| `environments/` | [Observed software environments](../environments/README.md) |

## Experiment management

| Path | Purpose / guide |
|---|---|
| `experiments/` | [Experiment registry and evidence cards](../experiments/README.md) |

## Provenance

| Path | Purpose / guide |
|---|---|
| `provenance/` | [Source lineage, hashes and excluded artifacts](../provenance/README.md) |

## Result indexes

| Path | Purpose / guide |
|---|---|
| `results/` | [Derived result tables and evidence indexes](../results/README.md) |

## Research utilities

| Path | Purpose / guide |
|---|---|
| `tools/` | [Restoration, indexing and validation utilities](../tools/README.md) |

## Root files

| File | Role |
|---|---|
| [.gitignore](../.gitignore) | Excludes heavy runtime artifacts, caches and credentials |
| [README.md](../README.md) | Research overview, methods, current evidence and category navigation |
| [THIRD_PARTY.md](../THIRD_PARTY.md) | Attribution and boundaries between upstream methods and local adaptations |
| [requirements.txt](../requirements.txt) | Dependency entry point; select the appropriate package-specific environment |

## Reading the evidence

Use original run records for scientific claims. Category snapshot commits organize existing files; they do not represent new experiments. Earlier history remains reachable through a merge, and preserved source/result blobs are unchanged.
