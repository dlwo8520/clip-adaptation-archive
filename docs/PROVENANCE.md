# Preservation and provenance

This export was prepared on 2026-10-12 KST from a distributed local research workspace. The original files and original Git repositories were retained. Source directories keep their original relative layout because follow-up learners import sibling projects and inherit earlier JSON configurations. No loss, optimizer, selection rule, architecture or evaluation calculation was refactored for presentation.

`provenance/file_manifest.csv` maps each exported file to its original relative path, original SHA-256, export SHA-256 and transformation. Identical hashes in `duplicates.csv` identify byte duplicates, not proof that two experiments are equivalent. Historical source variants are intentionally retained. `dependency_references.csv` records inspected path/import sites. `audit.json` records empty/invalid JSON and static import coverage.

Only path-bearing Python literals were changed to derive their workspace from the repository location or `RESEARCH_WORKSPACE`. Other exported text replaces personal home prefixes with `${RESEARCH_WORKSPACE}`. Frozen JSON/CSV run records retain this marker as evidence; they must not be treated as directly runnable configuration. Path-dependent source digests and sample IDs from original runs remain historical evidence, and will not necessarily match a relocated rerun. Original and exported hashes are deliberately distinct.

Large CSV, JSONL and log files are gzip compressed with a deterministic gzip header. `tools/restore_tables.py` restores them and checks their uncompressed hashes, refusing to replace an existing different file. Restored tables are ignored by Git; their compressed versions remain tracked. No numeric entries were rewritten. Source snapshot code/config/document occurrences are represented by `snapshot_manifest.csv` and deduplicated `snapshot_objects/`; recursive copies of output folders inside those snapshots are omitted. Snapshot objects preserve source evidence and are not runnable packages.

`excluded_files.csv` lists excluded checkpoints, feature arrays, generated images, dataset symlinks, paper PDFs, caches, environments, process locks and other artifacts. Directory exclusions mean a whole generated/environment subtree, not a per-file inventory of that subtree. Empty result files are retained and reported; their names are not evidence of completed runs. `audit.json` is the pre-compression inspection; the final manifests and validation report describe the packaged files.

Token/private-key patterns are checked without printing credential values. A negative pattern scan is not a proof of absence of every possible secret. Personal home paths are removed from the export. Original local artifacts are available in the source workspace and were not deleted. GPU jobs were neither started nor stopped for this organization task.

## Large raw evidence kept outside Git

`provenance/external_evidence.csv` lists verbose logs and compressed diagnostic tables larger than 1 MiB kept in the sibling `_local-evidence/REPOSITORY/` directory. Their exported/original hashes and exact relative locations are preserved. They are not pushed by normal Git commands. This keeps Git usable while retaining the raw files locally. Core summaries, config metadata, comparison tables, selected per-sample evidence and figures remain versioned. Files listed here are optional external artifacts when browsing old source report links, just like excluded checkpoints.

`python tools/restore_tables.py --external-root /path/to/_local-evidence/REPOSITORY` restores the local diagnostic evidence. Transfer that directory separately if moving machines. No raw evidence was deleted from the original research workspace.
