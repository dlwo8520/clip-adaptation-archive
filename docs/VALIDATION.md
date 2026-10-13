# Packaging validation — 2026-10-12 KST

- Python syntax and export hashes were checked with `tools/validate_export.py`.
- 165 copied original Python files were compared structurally after normalizing only path literals and the added workspace-path helper: zero algorithm AST differences. See `provenance/source_logic_validation.json`.
- All 90 preserved shell scripts passed `bash -n`. This establishes shell syntax, not successful runtime environment activation or dataset availability.
- The copied base CoOp trainer successfully imported and produced `--help` in the inspected `clip` environment. No CLIP weights were loaded for inference/training.
- The standalone index tool regenerated 9,045 final metric rows from readable non-smoke raw records, plus 72 final graph-summary rows and the 12-family record index. The rows are metrics and repeated origins, not 9,045 independent experiments.
- Obvious token/private-key patterns and personal home-prefix leaks were scanned. Original and exported hashes remain separate; model/path-based identities were not reconstructed.

Not verified: every legacy launcher, dataset/class mapping in every archive, teacher checkpoint availability, historical package versions, GPU training, full accuracy reproduction, empty run metadata, graph-only settings or seed-independent generalization. No code is described as fully working solely because it compiled or returned help.
