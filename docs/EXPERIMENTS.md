# CLIP experiment catalogue

Experiment families are distinct source packages. Each row links original documents and commands. Completion is inferred from usable content, never from `실험완료` in a pathname.

| ID | Family / implemented approach | Question | Evidence status |
|---|---|---|---|
| CLIP-001 | CoOp: CoOp prompt-only adaptation | Test target/preservation trade-off | nonempty raw base results; duplicated copies exist |
| CLIP-002 | CLIP_Adapter_project_v4: CLIP visual adapter, CoOp and CoCoOp implementations | Compare adaptation mechanisms | nonempty adapter/base evidence; CoCoOp code is not proof of a completed run |
| CLIP-003 | ZSCL_DataSelection_all_methods_wrappers: Confidence, margin, prototype, negative-confidence support ranking | Choose stable support sets | empty local run records; graph summaries survive for several methods |
| CLIP-004 | ZSCL_CoOp_TextDrift_Selection: One-step prompt-gradient perturbation and vocabulary text-feature drift | Select supports with smaller text drift | 100 nonempty support-seed result records found |
| CLIP-005 | ZSCL_DataSelection_entropy_only: Predictive entropy rather than negative confidence | Test uncertainty criteria | empty local result records; standalone tables survive |
| CLIP-006 | ZSCL_DataSelection_refdistill: Reference text/image distribution distillation plus optional context anchoring | Preserve reference predictions | source/scripts present; recoverable completion must be checked per artifact |
| CLIP-007 | ZSCL_DataSelection_refdistill_2: Three selection criteria × WC on/off | Evaluate distillation and anchoring | local result metadata absent/empty; standalone graph tables survive |
| CLIP-008 | ZSCL_DataSelection_refdistill_gift: Synthetic reference and GIFT-inspired teacher/reference variants | Test synthetic preservation resources | source/scripts and graph-only summaries; full original run metadata not recovered |
| CLIP-009 | CoOp_twoteacher_selected: Original CLIP plus adapted CoOp teacher agreement/JS score | Improve support quality | nonempty selected-student run results |
| CLIP-010 | CoOp_twoteacher_selected_wc: Two-teacher selection plus prompt weight consolidation | Reduce shared-context drift | nonempty WC-sweep results; different lambdas are distinct conditions |
| CLIP-011 | CoOp_lora_full_teacher: Full-data LoRA teacher plus CLIP agreement | Test a stronger full-data selector | nonempty LoRA/student results; additional full-data resource |
| CLIP-012 | CoOp_js_tradeoff: High-entropy anchor and JS-quantile support bins | Diagnose sample-distribution trade-off | implemented and graph-derived records; raw results.json absent |

## Source, setting and command routing

### CLIP-001 — CoOp

Baseline: original CLIP and package-specific random adaptation. Question: Test target/preservation trade-off. Evidence: nonempty raw base results; duplicated copies exist.

Source packages: [CLIP/그래프/CoOp](../CLIP/그래프/CoOp), [CLIP/실험완료/CoOp](../CLIP/실험완료/CoOp), [CLIP/코드/CoOp](../CLIP/코드/CoOp).

Method/setup/limitations: [CLIP/실험완료/CoOp/README.md](../CLIP/실험완료/CoOp/README.md), [CLIP/실험완료/CoOp/EXPERIMENT_GUIDE.md](../CLIP/실험완료/CoOp/EXPERIMENT_GUIDE.md), [CLIP/코드/CoOp/README.md](../CLIP/코드/CoOp/README.md), [CLIP/코드/CoOp/EXPERIMENT_GUIDE.md](../CLIP/코드/CoOp/EXPERIMENT_GUIDE.md).

Recorded launchers: [CLIP/실험완료/CoOp/scripts/run_smoke_gpu1.sh](../CLIP/실험완료/CoOp/scripts/run_smoke_gpu1.sh), [CLIP/실험완료/CoOp/scripts/check_dataset_structure.sh](../CLIP/실험완료/CoOp/scripts/check_dataset_structure.sh), [CLIP/실험완료/CoOp/scripts/run_texture_100seeds_gpu1.sh](../CLIP/실험완료/CoOp/scripts/run_texture_100seeds_gpu1.sh), [CLIP/실험완료/CoOp/scripts/run_texture_gpu1.sh](../CLIP/실험완료/CoOp/scripts/run_texture_gpu1.sh), [CLIP/코드/CoOp/scripts/run_smoke_gpu1.sh](../CLIP/코드/CoOp/scripts/run_smoke_gpu1.sh), [CLIP/코드/CoOp/scripts/check_dataset_structure.sh](../CLIP/코드/CoOp/scripts/check_dataset_structure.sh), [CLIP/코드/CoOp/scripts/run_texture_100seeds_gpu1.sh](../CLIP/코드/CoOp/scripts/run_texture_100seeds_gpu1.sh), [CLIP/코드/CoOp/scripts/run_texture_gpu1.sh](../CLIP/코드/CoOp/scripts/run_texture_gpu1.sh). Use explicit data roots and an unused output directory; do not launch all historical queues by default.

### CLIP-002 — CLIP_Adapter_project_v4

Baseline: original CLIP and package-specific random adaptation. Question: Compare adaptation mechanisms. Evidence: nonempty adapter/base evidence; CoCoOp code is not proof of a completed run.

Source packages: [CLIP/실험완료/CLIP_Adapter_project_v4](../CLIP/실험완료/CLIP_Adapter_project_v4), [CLIP/코드/CLIP_Adapter_project_v4](../CLIP/코드/CLIP_Adapter_project_v4).

Method/setup/limitations: [CLIP/실험완료/CLIP_Adapter_project_v4/README.md](../CLIP/실험완료/CLIP_Adapter_project_v4/README.md), [CLIP/실험완료/CLIP_Adapter_project_v4/FIXES.md](../CLIP/실험완료/CLIP_Adapter_project_v4/FIXES.md), [CLIP/실험완료/CLIP_Adapter_project_v4/EXPERIMENT_GUIDE.md](../CLIP/실험완료/CLIP_Adapter_project_v4/EXPERIMENT_GUIDE.md), [CLIP/실험완료/CLIP_Adapter_project_v4/COCOOP_ADDED_GUIDE.md](../CLIP/실험완료/CLIP_Adapter_project_v4/COCOOP_ADDED_GUIDE.md), [CLIP/코드/CLIP_Adapter_project_v4/README.md](../CLIP/코드/CLIP_Adapter_project_v4/README.md), [CLIP/코드/CLIP_Adapter_project_v4/FIXES.md](../CLIP/코드/CLIP_Adapter_project_v4/FIXES.md), [CLIP/코드/CLIP_Adapter_project_v4/EXPERIMENT_GUIDE.md](../CLIP/코드/CLIP_Adapter_project_v4/EXPERIMENT_GUIDE.md), [CLIP/코드/CLIP_Adapter_project_v4/COCOOP_ADDED_GUIDE.md](../CLIP/코드/CLIP_Adapter_project_v4/COCOOP_ADDED_GUIDE.md).

Recorded launchers: [CLIP/실험완료/CLIP_Adapter_project_v4/scripts/run_smoke_gpu1.sh](../CLIP/실험완료/CLIP_Adapter_project_v4/scripts/run_smoke_gpu1.sh), [CLIP/실험완료/CLIP_Adapter_project_v4/scripts/check_dataset_structure.sh](../CLIP/실험완료/CLIP_Adapter_project_v4/scripts/check_dataset_structure.sh), [CLIP/실험완료/CLIP_Adapter_project_v4/scripts/run_texture_cocoop_gpu1.sh](../CLIP/실험완료/CLIP_Adapter_project_v4/scripts/run_texture_cocoop_gpu1.sh), [CLIP/실험완료/CLIP_Adapter_project_v4/scripts/run_texture_adapter_100seeds_gpu1.sh](../CLIP/실험완료/CLIP_Adapter_project_v4/scripts/run_texture_adapter_100seeds_gpu1.sh), [CLIP/실험완료/CLIP_Adapter_project_v4/scripts/run_texture_adapter_gpu1.sh](../CLIP/실험완료/CLIP_Adapter_project_v4/scripts/run_texture_adapter_gpu1.sh), [CLIP/실험완료/CLIP_Adapter_project_v4/scripts/run_texture_100seeds_gpu1.sh](../CLIP/실험완료/CLIP_Adapter_project_v4/scripts/run_texture_100seeds_gpu1.sh), [CLIP/실험완료/CLIP_Adapter_project_v4/scripts/run_texture_gpu1.sh](../CLIP/실험완료/CLIP_Adapter_project_v4/scripts/run_texture_gpu1.sh), [CLIP/실험완료/CLIP_Adapter_project_v4/scripts/run_texture_adapter_parallel.sh](../CLIP/실험완료/CLIP_Adapter_project_v4/scripts/run_texture_adapter_parallel.sh). Use explicit data roots and an unused output directory; do not launch all historical queues by default.

### CLIP-003 — ZSCL_DataSelection_all_methods_wrappers

Baseline: original CLIP and package-specific random adaptation. Question: Choose stable support sets. Evidence: empty local run records; graph summaries survive for several methods.

Source packages: [CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers](../CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers), [CLIP/코드/ZSCL_DataSelection_all_methods_wrappers](../CLIP/코드/ZSCL_DataSelection_all_methods_wrappers).

Method/setup/limitations: .

Recorded launchers: [CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers/run_smoke_after_gpu1_free.sh](../CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers/run_smoke_after_gpu1_free.sh), [CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers/scripts/run_texture_adapter_zscl_100seeds_gpu1.sh](../CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers/scripts/run_texture_adapter_zscl_100seeds_gpu1.sh), [CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers/scripts/run_all_adapter_zscl_methods_parallel.sh](../CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers/scripts/run_all_adapter_zscl_methods_parallel.sh), [CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers/scripts/run_all_adapter_zscl_methods.sh](../CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers/scripts/run_all_adapter_zscl_methods.sh), [CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers/scripts/run_all_coop_zscl_methods.sh](../CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers/scripts/run_all_coop_zscl_methods.sh), [CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers/scripts/run_texture_coop_zscl_100seeds_gpu1.sh](../CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers/scripts/run_texture_coop_zscl_100seeds_gpu1.sh), [CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers/scripts/launch_adapter_and_coop_all_methods_total2.sh](../CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers/scripts/launch_adapter_and_coop_all_methods_total2.sh), [CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers/scripts/run_texture_coop_teacher_uncertain_100seeds_gpu1.sh](../CLIP/실험완료/ZSCL_DataSelection_all_methods_wrappers/scripts/run_texture_coop_teacher_uncertain_100seeds_gpu1.sh). Use explicit data roots and an unused output directory; do not launch all historical queues by default.

### CLIP-004 — ZSCL_CoOp_TextDrift_Selection

Baseline: original CLIP and package-specific random adaptation. Question: Select supports with smaller text drift. Evidence: 100 nonempty support-seed result records found.

Source packages: [CLIP/실험완료/ZSCL_CoOp_TextDrift_Selection](../CLIP/실험완료/ZSCL_CoOp_TextDrift_Selection).

Method/setup/limitations: [CLIP/실험완료/ZSCL_CoOp_TextDrift_Selection/README.md](../CLIP/실험완료/ZSCL_CoOp_TextDrift_Selection/README.md).

Recorded launchers: [CLIP/실험완료/ZSCL_CoOp_TextDrift_Selection/scripts/run_texture_coop_textdrift_100seeds.sh](../CLIP/실험완료/ZSCL_CoOp_TextDrift_Selection/scripts/run_texture_coop_textdrift_100seeds.sh), [CLIP/실험완료/ZSCL_CoOp_TextDrift_Selection/scripts/run_smoke_textdrift.sh](../CLIP/실험완료/ZSCL_CoOp_TextDrift_Selection/scripts/run_smoke_textdrift.sh). Use explicit data roots and an unused output directory; do not launch all historical queues by default.

### CLIP-005 — ZSCL_DataSelection_entropy_only

Baseline: original CLIP and package-specific random adaptation. Question: Test uncertainty criteria. Evidence: empty local result records; standalone tables survive.

Source packages: [CLIP/실험완료/ZSCL_DataSelection_entropy_only](../CLIP/실험완료/ZSCL_DataSelection_entropy_only).

Method/setup/limitations: [CLIP/실험완료/ZSCL_DataSelection_entropy_only/README_ENTROPY_ONLY.md](../CLIP/실험완료/ZSCL_DataSelection_entropy_only/README_ENTROPY_ONLY.md).

Recorded launchers: [CLIP/실험완료/ZSCL_DataSelection_entropy_only/scripts/run_texture_coop_teacher_entropy_100seeds_gpu1.sh](../CLIP/실험완료/ZSCL_DataSelection_entropy_only/scripts/run_texture_coop_teacher_entropy_100seeds_gpu1.sh), [CLIP/실험완료/ZSCL_DataSelection_entropy_only/scripts/run_texture_coop_teacher_entropy_smoke_gpu1.sh](../CLIP/실험완료/ZSCL_DataSelection_entropy_only/scripts/run_texture_coop_teacher_entropy_smoke_gpu1.sh), [CLIP/실험완료/ZSCL_DataSelection_entropy_only/scripts/run_texture_coop_teacher_entropy_100seeds_gpu1_parallel.sh](../CLIP/실험완료/ZSCL_DataSelection_entropy_only/scripts/run_texture_coop_teacher_entropy_100seeds_gpu1_parallel.sh). Use explicit data roots and an unused output directory; do not launch all historical queues by default.

### CLIP-006 — ZSCL_DataSelection_refdistill

Baseline: original CLIP and package-specific random adaptation. Question: Preserve reference predictions. Evidence: source/scripts present; recoverable completion must be checked per artifact.

Source packages: [CLIP/코드/ZSCL_DataSelection_refdistill](../CLIP/코드/ZSCL_DataSelection_refdistill).

Method/setup/limitations: [CLIP/코드/ZSCL_DataSelection_refdistill/README.md](../CLIP/코드/ZSCL_DataSelection_refdistill/README.md).

Recorded launchers: [CLIP/코드/ZSCL_DataSelection_refdistill/scripts/launch_6_configs_parallel.sh](../CLIP/코드/ZSCL_DataSelection_refdistill/scripts/launch_6_configs_parallel.sh), [CLIP/코드/ZSCL_DataSelection_refdistill/scripts/run_coop_refdistill_one_config.sh](../CLIP/코드/ZSCL_DataSelection_refdistill/scripts/run_coop_refdistill_one_config.sh), [CLIP/코드/ZSCL_DataSelection_refdistill/scripts/smoke_test_refdistill.sh](../CLIP/코드/ZSCL_DataSelection_refdistill/scripts/smoke_test_refdistill.sh). Use explicit data roots and an unused output directory; do not launch all historical queues by default.

### CLIP-007 — ZSCL_DataSelection_refdistill_2

Baseline: original CLIP and package-specific random adaptation. Question: Evaluate distillation and anchoring. Evidence: local result metadata absent/empty; standalone graph tables survive.

Source packages: [CLIP/실험완료/ZSCL_DataSelection_refdistill_2](../CLIP/실험완료/ZSCL_DataSelection_refdistill_2), [CLIP/코드/ZSCL_DataSelection_refdistill_2](../CLIP/코드/ZSCL_DataSelection_refdistill_2).

Method/setup/limitations: [CLIP/실험완료/ZSCL_DataSelection_refdistill_2/README.md](../CLIP/실험완료/ZSCL_DataSelection_refdistill_2/README.md), [CLIP/코드/ZSCL_DataSelection_refdistill_2/README.md](../CLIP/코드/ZSCL_DataSelection_refdistill_2/README.md).

Recorded launchers: [CLIP/실험완료/ZSCL_DataSelection_refdistill_2/scripts/launch_6_configs_parallel.sh](../CLIP/실험완료/ZSCL_DataSelection_refdistill_2/scripts/launch_6_configs_parallel.sh), [CLIP/실험완료/ZSCL_DataSelection_refdistill_2/scripts/run_coop_refdistill_one_config.sh](../CLIP/실험완료/ZSCL_DataSelection_refdistill_2/scripts/run_coop_refdistill_one_config.sh), [CLIP/실험완료/ZSCL_DataSelection_refdistill_2/scripts/launch_6_configs_sequential.sh](../CLIP/실험완료/ZSCL_DataSelection_refdistill_2/scripts/launch_6_configs_sequential.sh), [CLIP/실험완료/ZSCL_DataSelection_refdistill_2/scripts/smoke_test_refdistill.sh](../CLIP/실험완료/ZSCL_DataSelection_refdistill_2/scripts/smoke_test_refdistill.sh), [CLIP/코드/ZSCL_DataSelection_refdistill_2/scripts/launch_6_configs_parallel.sh](../CLIP/코드/ZSCL_DataSelection_refdistill_2/scripts/launch_6_configs_parallel.sh), [CLIP/코드/ZSCL_DataSelection_refdistill_2/scripts/run_coop_refdistill_one_config.sh](../CLIP/코드/ZSCL_DataSelection_refdistill_2/scripts/run_coop_refdistill_one_config.sh), [CLIP/코드/ZSCL_DataSelection_refdistill_2/scripts/launch_6_configs_sequential.sh](../CLIP/코드/ZSCL_DataSelection_refdistill_2/scripts/launch_6_configs_sequential.sh), [CLIP/코드/ZSCL_DataSelection_refdistill_2/scripts/smoke_test_refdistill.sh](../CLIP/코드/ZSCL_DataSelection_refdistill_2/scripts/smoke_test_refdistill.sh). Use explicit data roots and an unused output directory; do not launch all historical queues by default.

### CLIP-008 — ZSCL_DataSelection_refdistill_gift

Baseline: original CLIP and package-specific random adaptation. Question: Test synthetic preservation resources. Evidence: source/scripts and graph-only summaries; full original run metadata not recovered.

Source packages: [CLIP/실험완료/ZSCL_DataSelection_refdistill_gift](../CLIP/실험완료/ZSCL_DataSelection_refdistill_gift).

Method/setup/limitations: [CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/README.md](../CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/README.md), [CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/README_GIFT.md](../CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/README_GIFT.md).

Recorded launchers: [CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/scripts/run_coop_gift_one_config.sh](../CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/scripts/run_coop_gift_one_config.sh), [CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/scripts/launch_6_configs_parallel.sh](../CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/scripts/launch_6_configs_parallel.sh), [CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/scripts/run_coop_refdistill_one_config.sh](../CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/scripts/run_coop_refdistill_one_config.sh), [CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/scripts/launch_gift_configs.sh](../CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/scripts/launch_gift_configs.sh), [CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/scripts/run_gift_BCD_3selections_seed0_49_gpu4.sh](../CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/scripts/run_gift_BCD_3selections_seed0_49_gpu4.sh), [CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/scripts/launch_6_configs_sequential.sh](../CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/scripts/launch_6_configs_sequential.sh), [CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/scripts/smoke_test_gift.sh](../CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/scripts/smoke_test_gift.sh), [CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/scripts/smoke_test_refdistill.sh](../CLIP/실험완료/ZSCL_DataSelection_refdistill_gift/scripts/smoke_test_refdistill.sh). Use explicit data roots and an unused output directory; do not launch all historical queues by default.

### CLIP-009 — CoOp_twoteacher_selected

Baseline: original CLIP and package-specific random adaptation. Question: Improve support quality. Evidence: nonempty selected-student run results.

Source packages: [CLIP/실험완료/CoOp_twoteacher_selected](../CLIP/실험완료/CoOp_twoteacher_selected), [CLIP/코드/CoOp_twoteacher_selected](../CLIP/코드/CoOp_twoteacher_selected).

Method/setup/limitations: [CLIP/실험완료/CoOp_twoteacher_selected/README.md](../CLIP/실험완료/CoOp_twoteacher_selected/README.md), [CLIP/실험완료/CoOp_twoteacher_selected/EXPERIMENT_GUIDE.md](../CLIP/실험완료/CoOp_twoteacher_selected/EXPERIMENT_GUIDE.md), [CLIP/실험완료/CoOp_twoteacher_selected/TWOTEACHER_EXPERIMENT.md](../CLIP/실험완료/CoOp_twoteacher_selected/TWOTEACHER_EXPERIMENT.md), [CLIP/코드/CoOp_twoteacher_selected/README.md](../CLIP/코드/CoOp_twoteacher_selected/README.md), [CLIP/코드/CoOp_twoteacher_selected/EXPERIMENT_GUIDE.md](../CLIP/코드/CoOp_twoteacher_selected/EXPERIMENT_GUIDE.md), [CLIP/코드/CoOp_twoteacher_selected/TWOTEACHER_EXPERIMENT.md](../CLIP/코드/CoOp_twoteacher_selected/TWOTEACHER_EXPERIMENT.md).

Recorded launchers: [CLIP/실험완료/CoOp_twoteacher_selected/scripts/run_smoke_gpu1.sh](../CLIP/실험완료/CoOp_twoteacher_selected/scripts/run_smoke_gpu1.sh), [CLIP/실험완료/CoOp_twoteacher_selected/scripts/check_dataset_structure.sh](../CLIP/실험완료/CoOp_twoteacher_selected/scripts/check_dataset_structure.sh), [CLIP/실험완료/CoOp_twoteacher_selected/scripts/run_twoteacher_texture_diag.sh](../CLIP/실험완료/CoOp_twoteacher_selected/scripts/run_twoteacher_texture_diag.sh), [CLIP/실험완료/CoOp_twoteacher_selected/scripts/run_texture_100seeds_gpu1.sh](../CLIP/실험완료/CoOp_twoteacher_selected/scripts/run_texture_100seeds_gpu1.sh), [CLIP/실험완료/CoOp_twoteacher_selected/scripts/run_texture_gpu1.sh](../CLIP/실험완료/CoOp_twoteacher_selected/scripts/run_texture_gpu1.sh), [CLIP/코드/CoOp_twoteacher_selected/scripts/run_smoke_gpu1.sh](../CLIP/코드/CoOp_twoteacher_selected/scripts/run_smoke_gpu1.sh), [CLIP/코드/CoOp_twoteacher_selected/scripts/check_dataset_structure.sh](../CLIP/코드/CoOp_twoteacher_selected/scripts/check_dataset_structure.sh), [CLIP/코드/CoOp_twoteacher_selected/scripts/run_twoteacher_texture_diag.sh](../CLIP/코드/CoOp_twoteacher_selected/scripts/run_twoteacher_texture_diag.sh). Use explicit data roots and an unused output directory; do not launch all historical queues by default.

### CLIP-010 — CoOp_twoteacher_selected_wc

Baseline: original CLIP and package-specific random adaptation. Question: Reduce shared-context drift. Evidence: nonempty WC-sweep results; different lambdas are distinct conditions.

Source packages: [CLIP/실험완료/CoOp_twoteacher_selected_wc](../CLIP/실험완료/CoOp_twoteacher_selected_wc).

Method/setup/limitations: [CLIP/실험완료/CoOp_twoteacher_selected_wc/README.md](../CLIP/실험완료/CoOp_twoteacher_selected_wc/README.md), [CLIP/실험완료/CoOp_twoteacher_selected_wc/EXPERIMENT_GUIDE.md](../CLIP/실험완료/CoOp_twoteacher_selected_wc/EXPERIMENT_GUIDE.md), [CLIP/실험완료/CoOp_twoteacher_selected_wc/TWOTEACHER_EXPERIMENT.md](../CLIP/실험완료/CoOp_twoteacher_selected_wc/TWOTEACHER_EXPERIMENT.md).

Recorded launchers: [CLIP/실험완료/CoOp_twoteacher_selected_wc/scripts/run_smoke_gpu1.sh](../CLIP/실험완료/CoOp_twoteacher_selected_wc/scripts/run_smoke_gpu1.sh), [CLIP/실험완료/CoOp_twoteacher_selected_wc/scripts/check_dataset_structure.sh](../CLIP/실험완료/CoOp_twoteacher_selected_wc/scripts/check_dataset_structure.sh), [CLIP/실험완료/CoOp_twoteacher_selected_wc/scripts/run_twoteacher_texture_diag.sh](../CLIP/실험완료/CoOp_twoteacher_selected_wc/scripts/run_twoteacher_texture_diag.sh), [CLIP/실험완료/CoOp_twoteacher_selected_wc/scripts/run_texture_100seeds_gpu1.sh](../CLIP/실험완료/CoOp_twoteacher_selected_wc/scripts/run_texture_100seeds_gpu1.sh), [CLIP/실험완료/CoOp_twoteacher_selected_wc/scripts/run_texture_gpu1.sh](../CLIP/실험완료/CoOp_twoteacher_selected_wc/scripts/run_texture_gpu1.sh). Use explicit data roots and an unused output directory; do not launch all historical queues by default.

### CLIP-011 — CoOp_lora_full_teacher

Baseline: original CLIP and package-specific random adaptation. Question: Test a stronger full-data selector. Evidence: nonempty LoRA/student results; additional full-data resource.

Source packages: [CLIP/실험완료/CoOp_lora_full_teacher](../CLIP/실험완료/CoOp_lora_full_teacher).

Method/setup/limitations: [CLIP/실험완료/CoOp_lora_full_teacher/README.md](../CLIP/실험완료/CoOp_lora_full_teacher/README.md), [CLIP/실험완료/CoOp_lora_full_teacher/EXPERIMENT_GUIDE.md](../CLIP/실험완료/CoOp_lora_full_teacher/EXPERIMENT_GUIDE.md), [CLIP/실험완료/CoOp_lora_full_teacher/TWOTEACHER_EXPERIMENT.md](../CLIP/실험완료/CoOp_lora_full_teacher/TWOTEACHER_EXPERIMENT.md), [CLIP/실험완료/CoOp_lora_full_teacher/LORA_FULL_TEACHER_EXPERIMENT.md](../CLIP/실험완료/CoOp_lora_full_teacher/LORA_FULL_TEACHER_EXPERIMENT.md).

Recorded launchers: [CLIP/실험완료/CoOp_lora_full_teacher/scripts/run_smoke_gpu1.sh](../CLIP/실험완료/CoOp_lora_full_teacher/scripts/run_smoke_gpu1.sh), [CLIP/실험완료/CoOp_lora_full_teacher/scripts/check_dataset_structure.sh](../CLIP/실험완료/CoOp_lora_full_teacher/scripts/check_dataset_structure.sh), [CLIP/실험완료/CoOp_lora_full_teacher/scripts/run_twoteacher_texture_diag.sh](../CLIP/실험완료/CoOp_lora_full_teacher/scripts/run_twoteacher_texture_diag.sh), [CLIP/실험완료/CoOp_lora_full_teacher/scripts/run_texture_100seeds_gpu1.sh](../CLIP/실험완료/CoOp_lora_full_teacher/scripts/run_texture_100seeds_gpu1.sh), [CLIP/실험완료/CoOp_lora_full_teacher/scripts/run_texture_gpu1.sh](../CLIP/실험완료/CoOp_lora_full_teacher/scripts/run_texture_gpu1.sh), [CLIP/실험완료/CoOp_lora_full_teacher/scripts/run_lora_full_teacher_texture.sh](../CLIP/실험완료/CoOp_lora_full_teacher/scripts/run_lora_full_teacher_texture.sh). Use explicit data roots and an unused output directory; do not launch all historical queues by default.

### CLIP-012 — CoOp_js_tradeoff

Baseline: original CLIP and package-specific random adaptation. Question: Diagnose sample-distribution trade-off. Evidence: implemented and graph-derived records; raw results.json absent.

Source packages: [CLIP/코드/CoOp_js_tradeoff](../CLIP/코드/CoOp_js_tradeoff).

Method/setup/limitations: [CLIP/코드/CoOp_js_tradeoff/README_JS_TRADEOFF.md](../CLIP/코드/CoOp_js_tradeoff/README_JS_TRADEOFF.md).

Recorded launchers: [CLIP/코드/CoOp_js_tradeoff/scripts/run_js_tradeoff_random_remaining_100seeds_texture.sh](../CLIP/코드/CoOp_js_tradeoff/scripts/run_js_tradeoff_random_remaining_100seeds_texture.sh), [CLIP/코드/CoOp_js_tradeoff/scripts/run_js_tradeoff_modes_texture_diag.sh](../CLIP/코드/CoOp_js_tradeoff/scripts/run_js_tradeoff_modes_texture_diag.sh), [CLIP/코드/CoOp_js_tradeoff/scripts/run_js_tradeoff_texture.sh](../CLIP/코드/CoOp_js_tradeoff/scripts/run_js_tradeoff_texture.sh). Use explicit data roots and an unused output directory; do not launch all historical queues by default.
