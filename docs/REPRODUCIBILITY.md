# Reproducing what the archive supports

Install Python and a matching PyTorch/torchvision build for your hardware, then the package dependencies. The observed `clip` environment (Python 3.10.20, OpenCLIP 3.3.0) is recorded in `environments/clip.json`; this is the inspected environment, not proof that every archived run used these exact versions. Each original requirements file is preserved. Use `environments/requirements-clip.txt` as a consolidation, not a guarantee that historical packages work on every newer release.

```bash
python tools/restore_tables.py
python tools/rebuild_indexes.py
python tools/validate_export.py
```

For a base CoOp rerun, provide actual ImageFolder datasets and a writable new output directory:

```bash
cd CLIP/실험완료/CoOp
python train_blue_target_coop.py --help
CUDA_VISIBLE_DEVICES=0 python train_blue_target_coop.py   --target-name texture --target-root /path/to/texture   --preserve imagenet:/path/to/imagenet/val cifar100:/path/to/cifar100/test   --model ViT-B-32-quickgelu --pretrained openai   --shots-list 1 2 4 8 16 --n-ctx 16 --epochs 50 --eval-every 5   --lr 0.002 --batch-size 128 --seed 42 --prompt-seed 777 --sample-seed 0   --output-dir /path/to/new-run
```

Inspect `src/data.py` for auto classwise split logic and class-name mapping; explicit train/test folders are safer when reproducing a recorded split. Target dataset, preservation subsets, full-data teachers and reference images must match the selected source experiment. Two-teacher runs need historical teacher checkpoint files that were not committed. Resolve the `--teacher-runs-root` explicitly. Generic `/data/datasets` script defaults are historical conventions, not included data.

`${RESEARCH_WORKSPACE}` in frozen run metadata means an anonymized former home prefix. Python path literals resolve their workspace root at runtime; legacy shell scripts require environment/path adaptation and are retained as historical commands. The model/calculation logic was not changed. Source snapshots and original prompt checkpoints are not recreated from result files.

Packaging verification covers syntax, readable evidence, export hashes, index reconstruction and selected CLI/import checks as detailed in `docs/VALIDATION.md`. It does not establish that every archived GPU training run currently works or reproduce its accuracy. Empty local runs and graph-only evidence cannot be fully reproduced from the archive alone.
