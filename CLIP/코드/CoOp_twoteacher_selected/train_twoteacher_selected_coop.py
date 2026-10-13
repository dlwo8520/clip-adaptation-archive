# -*- coding: utf-8 -*-
"""
Two-teacher selected CoOp experiment.

This script tests whether a support set selected by both original CLIP and a
fine-tuned CoOp teacher is better than a random few-shot support set.

Pipeline:
1. Select top-k fine-tuned CoOp teachers per teacher shot from an existing
   multi-seed run directory.
2. Score every target train-pool image with:
      original CLIP true-label confidence
      CoOp teacher true-label confidence
      prediction-distribution similarity
3. Pick class-balanced top-scored images for each student shot.
4. Train a fresh student CoOp prompt on the selected support set.
5. Save history.json, coop_prompt_final.pt, selection.json, and results.json.
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import open_clip

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
sys.path.append(str(SRC_DIR))
sys.path.append(str(CURRENT_DIR))

from coop_model import (  # noqa: E402
    CoOpPromptLearner,
    build_coop_text_features,
    build_zero_shot_text_features,
    freeze_clip_model,
)
from data import (  # noqa: E402
    build_fewshot_subset,
    build_loader,
    build_preserve_dataset,
    build_target_datasets,
    maybe_replace_imagenet_wnid_with_torchvision_classes,
)
from eval import evaluate_classifier, precompute_image_features  # noqa: E402
from 실험완료.CoOp_twoteacher_selected.train_blue_target_coop import (  # noqa: E402
    evaluate_current_prompt,
    get_templates,
    make_zero_shot_epoch0_row,
    parse_dataset_spec,
    train_one_epoch,
)
from utils import append_jsonl, ensure_dir, load_classnames_file, save_json, seed_everything  # noqa: E402


def _as_float(value) -> float:
    return float(value.detach().cpu().item() if torch.is_tensor(value) else value)


def _safe_name(value: str) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_checkpoint_path(raw_path: str, result_file: Path, project_root: Path, teacher_runs_root: Path) -> Path:
    """Resolve checkpoint paths saved by the original experiment."""
    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return path

    candidates = [
        project_root / path,
        CURRENT_DIR / path,
        teacher_runs_root.parent.parent / path,
        result_file.parent / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return project_root / path


def extract_final_metric_from_history(history_path: Path, target_epoch: Optional[int]) -> Optional[Dict]:
    if not history_path.exists():
        return None
    history = _load_json(history_path)
    if not history:
        return None
    if target_epoch is None:
        return history[-1]
    exact = [row for row in history if int(row.get("epoch", -1)) == int(target_epoch)]
    return exact[-1] if exact else None


def select_top_teachers(
    teacher_runs_root: Path,
    project_root: Path,
    teacher_shots: Sequence[int],
    top_k: int,
    teacher_epoch: Optional[int],
) -> List[Dict]:
    """
    Select top-k teachers for each shot.

    Expected input:
      runs/blue_texture_preserve_100seeds/sample_seed_*/results.json
    Fallback:
      runs/.../sample_seed_*/shot_*/history.json
    """
    teacher_runs_root = teacher_runs_root.resolve()
    project_root = project_root.resolve()
    grouped: Dict[int, List[Dict]] = defaultdict(list)

    result_files = sorted(teacher_runs_root.glob("sample_seed_*/results.json"))
    for result_file in result_files:
        seed_dir = result_file.parent
        seed_text = seed_dir.name.replace("sample_seed_", "")
        try:
            sample_seed = int(seed_text)
        except ValueError:
            sample_seed = None

        result = _load_json(result_file)
        for item in result.get("shots", []):
            shot = int(item.get("shot"))
            if shot not in teacher_shots:
                continue

            if teacher_epoch is None:
                final_row = item.get("final")
            else:
                history_path = seed_dir / f"shot_{shot}" / "history.json"
                final_row = extract_final_metric_from_history(history_path, teacher_epoch)

            if not final_row:
                continue

            target_acc = float(final_row["target"]["accuracy"])
            checkpoint_raw = item.get("checkpoint_path") or f"shot_{shot}/coop_prompt_final.pt"
            checkpoint_path = resolve_checkpoint_path(checkpoint_raw, result_file, project_root, teacher_runs_root)

            grouped[shot].append(
                {
                    "teacher_shot": shot,
                    "sample_seed": sample_seed,
                    "target_accuracy": target_acc,
                    "teacher_epoch": int(final_row.get("epoch", -1)),
                    "checkpoint_path": str(checkpoint_path),
                    "history_path": str(seed_dir / f"shot_{shot}" / "history.json"),
                    "result_path": str(result_file),
                    "final": final_row,
                }
            )

    if not grouped:
        for history_path in sorted(teacher_runs_root.glob("sample_seed_*/shot_*/history.json")):
            shot = int(history_path.parent.name.replace("shot_", ""))
            if shot not in teacher_shots:
                continue
            sample_seed = int(history_path.parent.parent.name.replace("sample_seed_", ""))
            row = extract_final_metric_from_history(history_path, teacher_epoch)
            if not row:
                continue
            checkpoint_path = history_path.parent / "coop_prompt_final.pt"
            grouped[shot].append(
                {
                    "teacher_shot": shot,
                    "sample_seed": sample_seed,
                    "target_accuracy": float(row["target"]["accuracy"]),
                    "teacher_epoch": int(row.get("epoch", -1)),
                    "checkpoint_path": str(checkpoint_path),
                    "history_path": str(history_path),
                    "result_path": None,
                    "final": row,
                }
            )

    selected = []
    for shot in teacher_shots:
        teachers = sorted(grouped.get(int(shot), []), key=lambda x: x["target_accuracy"], reverse=True)
        for rank, teacher in enumerate(teachers[:top_k], start=1):
            teacher = dict(teacher)
            teacher["teacher_rank"] = rank
            selected.append(teacher)

    missing = [shot for shot in teacher_shots if not grouped.get(int(shot))]
    if missing:
        raise ValueError(f"No teachers found for shots: {missing} under {teacher_runs_root}")

    return selected


def fast_label_extract(dataset: Dataset) -> List[int]:
    if isinstance(dataset, Subset):
        labels = fast_label_extract(dataset.dataset)
        return [int(labels[idx]) for idx in dataset.indices]
    if hasattr(dataset, "samples"):
        return [int(sample[1]) for sample in dataset.samples]
    if hasattr(dataset, "targets"):
        return [int(target) for target in dataset.targets]
    labels = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        labels.append(int(label))
    return labels


def get_dataset_path_and_label(dataset: Dataset, local_idx: int) -> Tuple[Optional[str], int]:
    if isinstance(dataset, Subset):
        return get_dataset_path_and_label(dataset.dataset, int(dataset.indices[local_idx]))
    if hasattr(dataset, "samples"):
        path, label = dataset.samples[local_idx]
        return str(path), int(label)
    if hasattr(dataset, "imgs"):
        path, label = dataset.imgs[local_idx]
        return str(path), int(label)
    _, label = dataset[local_idx]
    return None, int(label)


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def subset_local_indices(subset: Dataset) -> Set[int]:
    if isinstance(subset, Subset):
        return {int(idx) for idx in subset.indices}
    return set(range(len(subset)))


def load_teacher_prompt(
    model,
    tokenizer,
    checkpoint_path: Path,
    n_ctx: int,
    device: torch.device,
) -> CoOpPromptLearner:
    checkpoint = safe_torch_load(checkpoint_path)
    if "ctx" not in checkpoint:
        raise KeyError(f"Checkpoint has no ctx tensor: {checkpoint_path}")
    ckpt_n_ctx = int(checkpoint.get("n_ctx", n_ctx))
    if ckpt_n_ctx != int(n_ctx):
        raise ValueError(f"n_ctx mismatch for {checkpoint_path}: checkpoint={ckpt_n_ctx}, args={n_ctx}")

    prompt_learner = CoOpPromptLearner(model, tokenizer, n_ctx=n_ctx, init_std=0.02).to(device)
    with torch.no_grad():
        prompt_learner.ctx.copy_(checkpoint["ctx"].to(device))
    prompt_learner.eval()
    return prompt_learner


def js_similarity(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Return 1 - normalized Jensen-Shannon divergence.

    JS divergence with natural logs is bounded by log(2), so this maps the
    similarity to roughly [0, 1].
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p.log() - m.log())).sum(dim=1) + 0.5 * (q * (q.log() - m.log())).sum(dim=1)
    sim = 1.0 - js / math.log(2.0)
    return sim.clamp(0.0, 1.0)


@torch.no_grad()
def score_train_pool(
    model,
    tokenizer,
    train_pool: Dataset,
    classnames: List[str],
    target_name: str,
    teacher: Dict,
    args,
    device: torch.device,
) -> List[Dict]:
    """Score every target train-pool image with the selected teacher."""
    model.eval()
    labels = fast_label_extract(train_pool)
    zero_text = build_zero_shot_text_features(
        model=model,
        tokenizer=tokenizer,
        classnames=classnames,
        templates=get_templates(target_name),
        device=device,
    )
    teacher_prompt = load_teacher_prompt(
        model=model,
        tokenizer=tokenizer,
        checkpoint_path=Path(teacher["checkpoint_path"]),
        n_ctx=args.n_ctx,
        device=device,
    )
    teacher_text = build_coop_text_features(model, teacher_prompt, classnames, device)

    loader = DataLoader(
        train_pool,
        batch_size=args.selection_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    rows: List[Dict] = []
    logit_scale = model.logit_scale.exp()
    local_offset = 0

    for images, targets in tqdm(loader, desc=f"score teacher shot={teacher['teacher_shot']} rank={teacher['teacher_rank']}"):
        batch_size = int(targets.numel())
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        p_clip = (logit_scale * image_features @ zero_text.t()).softmax(dim=1)
        p_teacher = (logit_scale * image_features @ teacher_text.t()).softmax(dim=1)

        row_index = torch.arange(batch_size, device=device)
        conf_clip = p_clip[row_index, targets]
        conf_teacher = p_teacher[row_index, targets]
        sim = js_similarity(p_clip, p_teacher)
        pred_clip = p_clip.argmax(dim=1)
        pred_teacher = p_teacher.argmax(dim=1)

        score = (
            args.lambda_clip * conf_clip
            + args.lambda_teacher * conf_teacher
            + args.lambda_similarity * sim
        )

        for i in range(batch_size):
            local_idx = local_offset + i
            label = int(targets[i].detach().cpu().item())
            if label != int(labels[local_idx]):
                raise RuntimeError("DataLoader order mismatch while scoring train pool.")
            path, _ = get_dataset_path_and_label(train_pool, local_idx)
            rows.append(
                {
                    "local_index": int(local_idx),
                    "path": path,
                    "label": label,
                    "class_name": classnames[label],
                    "score": _as_float(score[i]),
                    "conf_clip": _as_float(conf_clip[i]),
                    "conf_teacher": _as_float(conf_teacher[i]),
                    "distribution_similarity": _as_float(sim[i]),
                    "pred_clip": int(pred_clip[i].detach().cpu().item()),
                    "pred_teacher": int(pred_teacher[i].detach().cpu().item()),
                    "clip_correct": bool(pred_clip[i].detach().cpu().item() == label),
                    "teacher_correct": bool(pred_teacher[i].detach().cpu().item() == label),
                }
            )
        local_offset += batch_size

    return rows


def build_teacher_support_indices(train_pool: Dataset, teacher: Dict) -> Set[int]:
    sample_seed = teacher.get("sample_seed")
    teacher_shot = teacher.get("teacher_shot")
    if sample_seed is None or teacher_shot is None:
        return set()
    support = build_fewshot_subset(train_pool, int(teacher_shot), int(sample_seed))
    return subset_local_indices(support)


def select_balanced_by_score(
    score_rows: List[Dict],
    student_shot: int,
    class_count: int,
    require_both_correct: bool,
    excluded_indices: Optional[Set[int]],
    strict_selection: bool,
) -> Tuple[List[int], List[Dict], Dict]:
    excluded_indices = excluded_indices or set()
    by_label: Dict[int, List[Dict]] = defaultdict(list)
    for row in score_rows:
        if int(row["local_index"]) in excluded_indices:
            continue
        by_label[int(row["label"])].append(row)

    selected_indices: List[int] = []
    selected_rows: List[Dict] = []
    summary = {
        "student_shot": int(student_shot),
        "class_count": int(class_count),
        "require_both_correct": bool(require_both_correct),
        "strict_selection": bool(strict_selection),
        "excluded_count": int(len(excluded_indices)),
        "fallback_classes": [],
        "per_class": {},
    }

    for label in range(class_count):
        candidates = sorted(by_label.get(label, []), key=lambda x: float(x["score"]), reverse=True)
        if not candidates:
            raise ValueError(f"No candidates available for label={label}")

        if require_both_correct:
            primary = [row for row in candidates if row["clip_correct"] and row["teacher_correct"]]
        else:
            primary = candidates

        chosen = primary[:student_shot]
        fallback_count = 0
        if len(chosen) < student_shot:
            if strict_selection:
                raise ValueError(
                    f"label={label} has only {len(chosen)} strict candidates, need {student_shot}. "
                    "Use --allow-selection-fallback or remove --require-both-correct."
                )
            chosen_ids = {int(row["local_index"]) for row in chosen}
            fallback = [row for row in candidates if int(row["local_index"]) not in chosen_ids]
            needed = student_shot - len(chosen)
            chosen.extend(fallback[:needed])
            fallback_count = needed
            summary["fallback_classes"].append({"label": label, "class_name": candidates[0]["class_name"], "needed": needed})

        if len(chosen) < student_shot:
            raise ValueError(f"label={label} has only {len(chosen)} candidates after fallback, need {student_shot}")

        summary["per_class"][str(label)] = {
            "class_name": candidates[0]["class_name"],
            "available": len(candidates),
            "strict_available": len(primary),
            "selected": len(chosen),
            "fallback_selected": fallback_count,
            "best_score": float(candidates[0]["score"]),
            "worst_selected_score": float(chosen[-1]["score"]),
        }
        for row in chosen:
            selected_indices.append(int(row["local_index"]))
            selected_rows.append(row)

    return selected_indices, selected_rows, summary


def summarize_scores(score_rows: List[Dict]) -> Dict:
    if not score_rows:
        return {}
    scores = torch.tensor([float(row["score"]) for row in score_rows])
    clip_acc = sum(1 for row in score_rows if row["clip_correct"]) / len(score_rows)
    teacher_acc = sum(1 for row in score_rows if row["teacher_correct"]) / len(score_rows)
    both_acc = sum(1 for row in score_rows if row["clip_correct"] and row["teacher_correct"]) / len(score_rows)
    return {
        "num_candidates": len(score_rows),
        "score_mean": float(scores.mean().item()),
        "score_std": float(scores.std(unbiased=False).item()),
        "score_min": float(scores.min().item()),
        "score_max": float(scores.max().item()),
        "clip_train_pool_accuracy": float(clip_acc),
        "teacher_train_pool_accuracy": float(teacher_acc),
        "both_correct_rate": float(both_acc),
    }


def run_student_training(
    model,
    tokenizer,
    target_name: str,
    target_test_loader,
    target_classnames: List[str],
    preserve_loaders,
    preserve_classnames,
    baselines,
    target_cached,
    preserve_cached,
    selected_subset: Dataset,
    selected_indices: List[int],
    selected_rows: List[Dict],
    selection_summary: Dict,
    teacher: Dict,
    output_dir: Path,
    curves_path: Path,
    base_ctx: torch.Tensor,
    prompt_init_meta: Dict,
    args,
    device: torch.device,
) -> Dict:
    student_shot = int(selection_summary["student_shot"])
    teacher_id = f"teacher_shot_{teacher['teacher_shot']}_rank_{teacher['teacher_rank']}_seed_{teacher['sample_seed']}"
    run_dir = ensure_dir(output_dir / teacher_id / f"student_shot_{student_shot}")

    rng = random.Random(args.selection_seed)
    shuffled_indices = list(selected_indices)
    rng.shuffle(shuffled_indices)
    train_subset = Subset(selected_subset.dataset, shuffled_indices) if isinstance(selected_subset, Subset) else selected_subset
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    selection_payload = {
        "teacher": teacher,
        "student_shot": student_shot,
        "score_mode": "label_confidence_plus_js_similarity",
        "weights": {
            "lambda_clip": args.lambda_clip,
            "lambda_teacher": args.lambda_teacher,
            "lambda_similarity": args.lambda_similarity,
        },
        "require_both_correct": bool(args.require_both_correct),
        "exclude_teacher_support": bool(args.exclude_teacher_support),
        "selected_count": len(selected_indices),
        "selected_indices": selected_indices,
        "selected_samples": selected_rows,
        "summary": selection_summary,
    }
    save_json(selection_payload, run_dir / "selection.json")

    prompt_learner = CoOpPromptLearner(model, tokenizer, n_ctx=args.n_ctx, init_std=0.02).to(device)
    with torch.no_grad():
        prompt_learner.ctx.copy_(base_ctx)

    optimizer = torch.optim.SGD(
        [prompt_learner.ctx],
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    shot_history = []
    row = make_zero_shot_epoch0_row(
        target_name=target_name,
        preserve_loaders=preserve_loaders,
        baselines=baselines,
        shot=student_shot,
        optimizer=optimizer,
    )
    row["sample_seed"] = args.selection_seed
    row["split_seed"] = args.seed
    row["prompt_seed"] = args.prompt_seed
    row["teacher"] = {
        "teacher_shot": teacher["teacher_shot"],
        "teacher_rank": teacher["teacher_rank"],
        "sample_seed": teacher["sample_seed"],
        "target_accuracy": teacher["target_accuracy"],
        "checkpoint_path": teacher["checkpoint_path"],
    }
    row["selection"] = {
        "method": "two_teacher",
        "student_shot": student_shot,
        "selected_count": len(selected_indices),
    }
    shot_history.append(row)
    append_jsonl(row, curves_path)

    for epoch in range(1, args.epochs + 1):
        train_metric = train_one_epoch(
            model,
            prompt_learner,
            train_loader,
            target_classnames,
            optimizer,
            device,
            epoch,
            args.epochs,
        )
        scheduler.step()

        do_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if not do_eval:
            continue

        row = evaluate_current_prompt(
            model,
            prompt_learner,
            target_name,
            target_test_loader,
            target_classnames,
            preserve_loaders,
            preserve_classnames,
            baselines,
            device,
            student_shot,
            epoch,
            target_cached=target_cached,
            preserve_cached=preserve_cached,
        )
        row["sample_seed"] = args.selection_seed
        row["split_seed"] = args.seed
        row["prompt_seed"] = args.prompt_seed
        row["train"] = {
            "train_loss": train_metric["train_loss"],
            "train_acc": train_metric["train_acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        row["prompt_init"] = prompt_init_meta
        row["teacher"] = {
            "teacher_shot": teacher["teacher_shot"],
            "teacher_rank": teacher["teacher_rank"],
            "sample_seed": teacher["sample_seed"],
            "target_accuracy": teacher["target_accuracy"],
            "checkpoint_path": teacher["checkpoint_path"],
        }
        row["selection"] = {
            "method": "two_teacher",
            "student_shot": student_shot,
            "selected_count": len(selected_indices),
            "require_both_correct": bool(args.require_both_correct),
            "exclude_teacher_support": bool(args.exclude_teacher_support),
        }
        shot_history.append(row)
        append_jsonl(row, curves_path)
        save_json(shot_history, run_dir / "history.json")

    torch.save(
        {
            "sample_seed": args.selection_seed,
            "split_seed": args.seed,
            "shot": student_shot,
            "ctx": prompt_learner.ctx.detach().cpu(),
            "n_ctx": args.n_ctx,
            "target_classnames": target_classnames,
            "history": shot_history,
            "args": vars(args),
            "prompt_seed": args.prompt_seed,
            "same_initial_context_across_shots": True,
            "base_ctx_at_init": base_ctx.cpu(),
            "teacher": teacher,
            "selection_path": str(run_dir / "selection.json"),
        },
        run_dir / "coop_prompt_final.pt",
    )

    return {
        "teacher": teacher,
        "student_shot": student_shot,
        "selected_train_images": len(selected_indices),
        "final": shot_history[-1],
        "history_path": str(run_dir / "history.json"),
        "checkpoint_path": str(run_dir / "coop_prompt_final.pt"),
        "selection_path": str(run_dir / "selection.json"),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target-name", type=str, required=True)
    parser.add_argument("--target-root", type=str, default=None)
    parser.add_argument("--target-train", type=str, default=None)
    parser.add_argument("--target-test", type=str, default=None)
    parser.add_argument("--target-classnames", type=str, default=None)
    parser.add_argument("--split-test-ratio", type=float, default=0.5)
    parser.add_argument(
        "--preserve",
        type=str,
        nargs="*",
        default=[],
        help="name:path or name:path:classnames_txt",
    )

    parser.add_argument("--teacher-runs-root", type=str, required=True)
    parser.add_argument("--teacher-project-root", type=str, default=".")
    parser.add_argument("--teacher-shots", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--teacher-top-k", type=int, default=3)
    parser.add_argument("--teacher-epoch", type=int, default=None)
    parser.add_argument("--student-shots", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--pairing-mode", choices=["matched", "full"], default="matched")

    parser.add_argument("--require-both-correct", action="store_true")
    parser.add_argument("--strict-selection", action="store_true")
    parser.add_argument("--exclude-teacher-support", action="store_true")
    parser.add_argument("--save-all-candidate-scores", action="store_true")
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--selection-batch-size", type=int, default=256)
    parser.add_argument("--lambda-clip", type=float, default=1.0)
    parser.add_argument("--lambda-teacher", type=float, default=1.0)
    parser.add_argument("--lambda-similarity", type=float, default=1.0)

    parser.add_argument("--model", type=str, default="ViT-B-32-quickgelu")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--n-ctx", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-seed", type=int, default=777)
    parser.add_argument("--max-eval-samples", type=int, default=-1)
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--output-dir", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    output_dir = ensure_dir(args.output_dir)
    curves_path = output_dir / "curves.jsonl"
    if curves_path.exists():
        curves_path.unlink()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device: {device}")

    print("===== load CLIP =====")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    freeze_clip_model(model)
    model.eval()

    print("===== build datasets =====")
    target_train_pool, target_test_dataset, target_classnames, split_mode = build_target_datasets(
        target_root=args.target_root,
        target_train=args.target_train,
        target_test=args.target_test,
        transform=preprocess,
        split_test_ratio=args.split_test_ratio,
        seed=args.seed,
    )
    if args.target_classnames:
        target_classnames = load_classnames_file(args.target_classnames)

    target_test_loader = build_loader(
        target_test_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        max_samples=args.max_eval_samples,
    )

    preserve_loaders = {}
    preserve_classnames = {}
    preserve_info = {}
    for spec in args.preserve:
        name, path, classnames_file = parse_dataset_spec(spec)
        dataset, classnames = build_preserve_dataset(path, preprocess)
        if classnames_file:
            classnames = load_classnames_file(classnames_file)
        classnames = maybe_replace_imagenet_wnid_with_torchvision_classes(name, classnames)
        preserve_loaders[name] = build_loader(
            dataset,
            args.batch_size,
            args.num_workers,
            shuffle=False,
            max_samples=args.max_eval_samples,
        )
        preserve_classnames[name] = classnames
        preserve_info[name] = {"path": path, "num_classes": len(classnames), "num_images": len(dataset)}

    print("===== select teachers =====")
    teachers = select_top_teachers(
        teacher_runs_root=Path(args.teacher_runs_root),
        project_root=Path(args.teacher_project_root),
        teacher_shots=args.teacher_shots,
        top_k=args.teacher_top_k,
        teacher_epoch=args.teacher_epoch,
    )
    save_json(teachers, output_dir / "selected_teachers.json")
    for teacher in teachers:
        print(
            f"teacher shot={teacher['teacher_shot']} rank={teacher['teacher_rank']} "
            f"seed={teacher['sample_seed']} acc={teacher['target_accuracy']:.4f}"
        )

    print("===== original CLIP zero-shot baselines =====")
    baselines = {"target": {}, "preserve": {}}
    target_zero_text = build_zero_shot_text_features(
        model=model,
        tokenizer=tokenizer,
        classnames=target_classnames,
        templates=get_templates(args.target_name),
        device=device,
    )
    baselines["target"] = evaluate_classifier(
        model=model,
        loader=target_test_loader,
        text_features=target_zero_text,
        device=device,
        desc=f"zero-shot target={args.target_name}",
    )
    for preserve_name, loader in preserve_loaders.items():
        preserve_zero_text = build_zero_shot_text_features(
            model=model,
            tokenizer=tokenizer,
            classnames=preserve_classnames[preserve_name],
            templates=get_templates(preserve_name),
            device=device,
        )
        baselines["preserve"][preserve_name] = evaluate_classifier(
            model=model,
            loader=loader,
            text_features=preserve_zero_text,
            device=device,
            desc=f"zero-shot preserve={preserve_name}",
        )
    save_json(baselines, output_dir / "zero_shot_baselines.json")

    target_cached = None
    preserve_cached = None
    if args.cache_features:
        print("===== cache eval image features =====")
        target_cached = precompute_image_features(
            model,
            target_test_loader,
            device,
            desc=f"cache target={args.target_name}",
        )
        preserve_cached = {}
        for preserve_name, loader in preserve_loaders.items():
            preserve_cached[preserve_name] = precompute_image_features(
                model,
                loader,
                device,
                desc=f"cache preserve={preserve_name}",
            )

    print("===== common student prompt initialization =====")
    torch.manual_seed(args.prompt_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.prompt_seed)
    dummy_prompt = CoOpPromptLearner(model, tokenizer, n_ctx=args.n_ctx, init_std=0.02).to(device)
    base_ctx = dummy_prompt.ctx.detach().clone()
    save_json(
        {
            "prompt_seed": args.prompt_seed,
            "same_initial_context_across_runs": True,
            "ctx_shape": list(base_ctx.shape),
            "init_std": 0.02,
        },
        output_dir / "prompt_initialization.json",
    )
    torch.save(
        {"base_ctx": base_ctx.cpu(), "prompt_seed": args.prompt_seed, "n_ctx": args.n_ctx},
        output_dir / "prompt_initialization.pt",
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompt_init_meta = {
        "prompt_seed": args.prompt_seed,
        "same_initial_context_across_runs": True,
        "epoch_0_meaning": "original CLIP zero-shot baseline before CoOp adaptation",
        "epoch_1_and_after": "student CoOp prompt after two-teacher selected support adaptation",
    }

    all_results = {
        "args": vars(args),
        "target_name": args.target_name,
        "split_mode": split_mode,
        "preserve_info": preserve_info,
        "zero_shot_baselines": baselines,
        "prompt_init": prompt_init_meta,
        "selected_teachers": teachers,
        "runs": [],
    }

    for teacher in teachers:
        if args.pairing_mode == "matched":
            student_shots = [int(teacher["teacher_shot"])]
        else:
            student_shots = [int(s) for s in args.student_shots]

        print(
            f"===== score pool: teacher shot={teacher['teacher_shot']} "
            f"rank={teacher['teacher_rank']} seed={teacher['sample_seed']} ====="
        )
        score_rows = score_train_pool(
            model=model,
            tokenizer=tokenizer,
            train_pool=target_train_pool,
            classnames=target_classnames,
            target_name=args.target_name,
            teacher=teacher,
            args=args,
            device=device,
        )
        teacher_dir = ensure_dir(
            output_dir
            / f"teacher_shot_{teacher['teacher_shot']}_rank_{teacher['teacher_rank']}_seed_{teacher['sample_seed']}"
        )
        score_summary = summarize_scores(score_rows)
        save_json(score_summary, teacher_dir / "candidate_score_summary.json")
        if args.save_all_candidate_scores:
            save_json(score_rows, teacher_dir / "candidate_scores.json")

        excluded = build_teacher_support_indices(target_train_pool, teacher) if args.exclude_teacher_support else set()

        for student_shot in student_shots:
            selected_indices, selected_rows, selection_summary = select_balanced_by_score(
                score_rows=score_rows,
                student_shot=int(student_shot),
                class_count=len(target_classnames),
                require_both_correct=args.require_both_correct,
                excluded_indices=excluded,
                strict_selection=args.strict_selection,
            )
            selected_subset = Subset(target_train_pool, selected_indices)
            run_result = run_student_training(
                model=model,
                tokenizer=tokenizer,
                target_name=args.target_name,
                target_test_loader=target_test_loader,
                target_classnames=target_classnames,
                preserve_loaders=preserve_loaders,
                preserve_classnames=preserve_classnames,
                baselines=baselines,
                target_cached=target_cached,
                preserve_cached=preserve_cached,
                selected_subset=selected_subset,
                selected_indices=selected_indices,
                selected_rows=selected_rows,
                selection_summary=selection_summary,
                teacher=teacher,
                output_dir=output_dir,
                curves_path=curves_path,
                base_ctx=base_ctx,
                prompt_init_meta=prompt_init_meta,
                args=args,
                device=device,
            )
            all_results["runs"].append(run_result)
            save_json(all_results, output_dir / "results.json")

    print("===== done =====")
    print(f"output_dir: {output_dir}")
    print(f"results: {output_dir / 'results.json'}")
    print(f"curves: {curves_path}")


if __name__ == "__main__":
    main()
