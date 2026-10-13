# Recorded results and evidence boundaries

All accuracy values below are percentages. These are 16-shot / epoch-50 graph summaries, not a certified matched leaderboard. Each row retains the recorded support-seed count. Exact split/resource/config matching is unavailable for several graph-only packages.

| Graph group | DTD target | ImageNet preserve | CIFAR-100 preserve | Recorded support seeds |
|---|---:|---:|---:|---:|
| 4_CoOp__teacher_confident__wc_off | 56.766 | 60.076 | 64.796 | 100 |
| 4_CoOp__teacher_confident__wc_on | 58.058 | 60.038 | 64.709 | 100 |
| 4_CoOp__teacher_entropy__wc_off | 59.323 | 59.953 | 64.960 | 100 |
| 4_CoOp__teacher_entropy__wc_on | 60.721 | 59.871 | 64.796 | 100 |
| 4_CoOp__teacher_margin__wc_off | 57.159 | 60.097 | 64.922 | 100 |
| 4_CoOp__teacher_margin__wc_on | 58.520 | 60.079 | 64.862 | 100 |
| 5_teacher_confident_B__synthetic_zscl_like | 55.391 | 56.522 | 62.720 | 50 |
| 5_teacher_confident_C__gift_synthetic | 55.698 | 56.824 | 62.394 | 50 |
| 5_teacher_confident_D__gift_synthetic_awc | 55.698 | 56.824 | 62.394 | 50 |
| 5_teacher_entropy_B__synthetic_zscl_like | 57.053 | 55.163 | 61.951 | 49 |
| 5_teacher_entropy_C__gift_synthetic | 57.447 | 55.612 | 61.894 | 50 |
| 5_teacher_entropy_D__gift_synthetic_awc | 57.447 | 55.612 | 61.894 | 50 |
| 5_teacher_margin_B__synthetic_zscl_like | 55.045 | 57.037 | 62.932 | 50 |
| 5_teacher_margin_C__gift_synthetic | 55.695 | 56.944 | 62.725 | 49 |
| 5_teacher_margin_D__gift_synthetic_awc | 55.690 | 56.922 | 62.726 | 50 |
| CoOp | 64.378 | 43.976 | 51.658 | 100 |
| CoOp_confidenct | 62.036 | 50.730 | 56.986 | 100 |
| CoOp_margin | 62.520 | 49.519 | 55.664 | 100 |
| CoOp_textdrift | 63.455 | 42.901 | 50.738 | 100 |
| CoOp_uncertain | 63.800 | 43.324 | 51.722 | 100 |
| adapter | 51.691 | 56.759 | 62.049 | 100 |
| adapter_teacher_confident | 42.694 | 56.957 | 62.340 | 100 |
| adapter_teacher_margin | 42.551 | 56.790 | 62.354 | 100 |
| adapter_teacher_uncertain | 43.067 | 56.551 | 62.241 | 100 |

Original base zero-shot: texture 42.482%, ImageNet 60.500%, CIFAR-100 64.930%. Do not infer a graph-specific delta without checking its recorded baseline and evaluation setup. All graph rows and source paths are in [graph_epoch50_shot16.csv](../results/graph_epoch50_shot16.csv).

635 archived `results.json` files are empty. This count includes copies; it is not a failed-run count. No numbers were recovered from those files. The wrappers and entropy package have empty local raw records while standalone graph tables survive; reference-distillation and GIFT graph summaries likewise do not reconstruct missing original args/checkpoints. Text-drift and base CoOp have nonempty raw records.

Graph groups beginning `4_` and `5_` retain their historical labels. The shown `5_` final rows record 49–50 support seeds, while many base and `4_` tables record 100. Equal epoch/shot/seed-count does not establish equal teacher/reference resources. Trash-folder graph versions are retained but excluded here.

The [raw index](../results/raw_final_results.csv) records every parseable non-smoke final result with its source. `duplicate_file=true` means a byte-identical `results.json` was already encountered; do not count it again as a replicate. Packages and standalone plots may still contain logically duplicated runs with different serialization. Selected best cases and post-hoc teacher choices are diagnostics, not unbiased population averages.

JS-quantile diagnostics have an implemented trainer and nonempty plotted summaries in `CLIP/코드/CoOp_js_tradeoff/plots/`, but no surviving `results.json`. Inspect the plot summary, fallback counts and teacher/resource boundaries before comparing it with the base experiments.
