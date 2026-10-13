# -*- coding: utf-8 -*-
"""
Full-data LoRA teacher + original CLIP two-teacher selection + Student CoOp.

Experiment:
1. Train one LoRA CLIP teacher on the full target train pool.
2. Score target train-pool images with original CLIP and the LoRA teacher.
3. Select class-balanced 1/2/4/8/16-shot supports.
4. Train fresh Student CoOp prompts on the selected supports.

This keeps the student stage identical to the previous selected-support CoOp
experiment, while replacing the fine-tuned CoOp teacher with a full-data LoRA
teacher.
"""

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn
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
    build_zero_shot_text_features,
    freeze_clip_model,
)
from data import (  # noqa: E402
    build_loader,
    build_preserve_dataset,
    build_target_datasets,
    maybe_replace_imagenet_wnid_with_torchvision_classes,
)
from eval import evaluate_classifier, precompute_image_features  # noqa: E402
from 실험완료.CoOp_lora_full_teacher.train_blue_target_coop import (  # noqa: E402
    evaluate_current_prompt,
    get_templates,
    make_zero_shot_epoch0_row,
    parse_dataset_spec,
    train_one_epoch,
)
from utils import append_jsonl, ensure_dir, load_classnames_file, save_json, seed_everything  # noqa: E402


def _as_float(value) -> float:
    return float(value.detach().cpu().item() if torch.is_tensor(value) else value)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class LoRALinear(nn.Module):
    """A frozen Linear layer with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.enabled = True

        for param in self.base.parameters():
            param.requires_grad_(False)

        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        lora_hidden = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_hidden, self.lora_B)
        return out + lora_out * self.scaling

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features


def freeze_all_parameters(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def should_lora_wrap(module_name: str, lora_target: str, module_filter: Optional[re.Pattern]) -> bool:
    if lora_target == "visual" and not module_name.startswith("visual."):
        return False
    if lora_target == "text" and module_name.startswith("visual."):
        return False
    if module_filter is not None and not module_filter.search(module_name):
        return False
    return True


def inject_lora_linear(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    lora_target: str,
    module_filter: str,
) -> List[str]:
    """Replace selected nn.Linear children with LoRALinear modules."""
    pattern = re.compile(module_filter) if module_filter else None
    replaced: List[str] = []

    def visit(parent: nn.Module, prefix: str = ""):
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LoRALinear):
                continue
            if isinstance(parent, nn.MultiheadAttention) and child_name == "out_proj":
                continue
            if isinstance(child, nn.Linear) and should_lora_wrap(full_name, lora_target, pattern):
                setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced.append(full_name)
            else:
                visit(child, full_name)

    visit(model)
    if not replaced:
        raise RuntimeError(
            f"No Linear modules were wrapped by LoRA. lora_target={lora_target}, module_filter={module_filter}"
        )
    return replaced


def iter_lora_modules(model: nn.Module) -> Iterable[LoRALinear]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    for module in iter_lora_modules(model):
        module.enabled = bool(enabled)


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    params = []
    for module in iter_lora_modules(model):
        params.extend([module.lora_A, module.lora_B])
    return params


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
            state[f"{name}.alpha"] = torch.tensor(module.alpha)
            state[f"{name}.rank"] = torch.tensor(module.rank)
    return state


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


def make_zero_shot_token_groups(tokenizer, classnames: List[str], templates: List[str], device: torch.device):
    token_groups = []
    for class_name in classnames:
        clean = class_name.replace("_", " ").replace("-", " ")
        prompts = [template.format(clean) for template in templates]
        token_groups.append(tokenizer(prompts).to(device))
    return token_groups


def build_zero_shot_text_features_trainable(model: nn.Module, token_groups: List[torch.Tensor]) -> torch.Tensor:
    """Zero-shot text features without torch.no_grad, for LoRA text training."""
    features = []
    for tokens in token_groups:
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
        class_feature = text_features.mean(dim=0)
        class_feature = F.normalize(class_feature, dim=0)
        features.append(class_feature)
    return torch.stack(features, dim=0)


def train_lora_teacher_one_epoch(
    model,
    train_loader,
    token_groups,
    optimizer,
    device,
    epoch,
    total_epochs,
) -> Dict[str, float]:
    set_lora_enabled(model, True)
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    pbar = tqdm(train_loader, desc=f"LoRA teacher epoch {epoch}/{total_epochs}")

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        text_features = build_zero_shot_text_features_trainable(model, token_groups)
        logits = model.logit_scale.exp() * image_features @ text_features.t()
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct = (preds == targets).sum().item()
        total_loss += loss.item() * targets.numel()
        total_correct += correct
        total_seen += targets.numel()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / targets.numel():.4f}")

    return {
        "train_loss": float(total_loss / max(1, total_seen)),
        "train_acc": float(total_correct / max(1, total_seen)),
    }


def evaluate_zero_shot_with_lora_state(
    model,
    tokenizer,
    target_name,
    target_loader,
    target_classnames,
    preserve_loaders,
    preserve_classnames,
    device,
    lora_enabled: bool,
) -> Dict:
    set_lora_enabled(model, lora_enabled)
    model.eval()
    result = {"target": {}, "preserve": {}}
    target_text = build_zero_shot_text_features(
        model=model,
        tokenizer=tokenizer,
        classnames=target_classnames,
        templates=get_templates(target_name),
        device=device,
    )
    result["target"] = evaluate_classifier(
        model=model,
        loader=target_loader,
        text_features=target_text,
        device=device,
        desc=f"{'LoRA' if lora_enabled else 'original'} target={target_name}",
    )

    for preserve_name, loader in preserve_loaders.items():
        preserve_text = build_zero_shot_text_features(
            model=model,
            tokenizer=tokenizer,
            classnames=preserve_classnames[preserve_name],
            templates=get_templates(preserve_name),
            device=device,
        )
        result["preserve"][preserve_name] = evaluate_classifier(
            model=model,
            loader=loader,
            text_features=preserve_text,
            device=device,
            desc=f"{'LoRA' if lora_enabled else 'original'} preserve={preserve_name}",
        )
    return result


def js_similarity(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p.log() - m.log())).sum(dim=1) + 0.5 * (q * (q.log() - m.log())).sum(dim=1)
    return (1.0 - js / math.log(2.0)).clamp(0.0, 1.0)


@torch.no_grad()
def score_train_pool_with_lora_teacher(
    model,
    tokenizer,
    train_pool: Dataset,
    classnames: List[str],
    target_name: str,
    args,
    device: torch.device,
) -> List[Dict]:
    model.eval()
    labels = fast_label_extract(train_pool)

    set_lora_enabled(model, False)
    original_text = build_zero_shot_text_features(
        model=model,
        tokenizer=tokenizer,
        classnames=classnames,
        templates=get_templates(target_name),
        device=device,
    )

    set_lora_enabled(model, True)
    lora_text = build_zero_shot_text_features(
        model=model,
        tokenizer=tokenizer,
        classnames=classnames,
        templates=get_templates(target_name),
        device=device,
    )

    loader = DataLoader(
        train_pool,
        batch_size=args.selection_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logit_scale = model.logit_scale.exp()
    rows: List[Dict] = []
    local_offset = 0

    for images, targets in tqdm(loader, desc="score train pool with original CLIP + LoRA teacher"):
        batch_size = int(targets.numel())
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        set_lora_enabled(model, False)
        original_features = model.encode_image(images)
        original_features = F.normalize(original_features, dim=-1)
        p_original = (logit_scale * original_features @ original_text.t()).softmax(dim=1)

        set_lora_enabled(model, True)
        lora_features = model.encode_image(images)
        lora_features = F.normalize(lora_features, dim=-1)
        p_lora = (logit_scale * lora_features @ lora_text.t()).softmax(dim=1)

        row_index = torch.arange(batch_size, device=device)
        conf_original = p_original[row_index, targets]
        conf_lora = p_lora[row_index, targets]
        sim = js_similarity(p_original, p_lora)
        pred_original = p_original.argmax(dim=1)
        pred_lora = p_lora.argmax(dim=1)
        score = (
            args.lambda_clip * conf_original
            + args.lambda_lora * conf_lora
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
                    "conf_original_clip": _as_float(conf_original[i]),
                    "conf_lora_teacher": _as_float(conf_lora[i]),
                    "distribution_similarity": _as_float(sim[i]),
                    "pred_original_clip": int(pred_original[i].detach().cpu().item()),
                    "pred_lora_teacher": int(pred_lora[i].detach().cpu().item()),
                    "original_clip_correct": bool(pred_original[i].detach().cpu().item() == label),
                    "lora_teacher_correct": bool(pred_lora[i].detach().cpu().item() == label),
                }
            )
        local_offset += batch_size

    return rows


def summarize_scores(score_rows: List[Dict]) -> Dict:
    scores = torch.tensor([float(row["score"]) for row in score_rows])
    original_acc = sum(1 for row in score_rows if row["original_clip_correct"]) / len(score_rows)
    lora_acc = sum(1 for row in score_rows if row["lora_teacher_correct"]) / len(score_rows)
    both_acc = sum(1 for row in score_rows if row["original_clip_correct"] and row["lora_teacher_correct"]) / len(score_rows)
    return {
        "num_candidates": len(score_rows),
        "score_mean": float(scores.mean().item()),
        "score_std": float(scores.std(unbiased=False).item()),
        "score_min": float(scores.min().item()),
        "score_max": float(scores.max().item()),
        "original_clip_train_pool_accuracy": float(original_acc),
        "lora_teacher_train_pool_accuracy": float(lora_acc),
        "both_correct_rate": float(both_acc),
    }


def select_balanced_by_score(
    score_rows: List[Dict],
    student_shot: int,
    class_count: int,
    require_both_correct: bool,
    strict_selection: bool,
) -> Tuple[List[int], List[Dict], Dict]:
    by_label: Dict[int, List[Dict]] = defaultdict(list)
    for row in score_rows:
        by_label[int(row["label"])].append(row)

    selected_indices: List[int] = []
    selected_rows: List[Dict] = []
    summary = {
        "student_shot": int(student_shot),
        "class_count": int(class_count),
        "require_both_correct": bool(require_both_correct),
        "strict_selection": bool(strict_selection),
        "fallback_classes": [],
        "per_class": {},
    }

    for label in range(class_count):
        candidates = sorted(by_label.get(label, []), key=lambda x: float(x["score"]), reverse=True)
        if not candidates:
            raise ValueError(f"No candidates available for label={label}")

        if require_both_correct:
            primary = [row for row in candidates if row["original_clip_correct"] and row["lora_teacher_correct"]]
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


def train_student_coop_for_shot(
    model,
    tokenizer,
    target_name,
    target_train_pool,
    target_test_loader,
    target_classnames,
    preserve_loaders,
    preserve_classnames,
    baselines,
    target_cached,
    preserve_cached,
    base_ctx,
    prompt_init_meta,
    selected_indices,
    selected_rows,
    selection_summary,
    output_dir: Path,
    curves_path: Path,
    args,
    device,
) -> Dict:
    set_lora_enabled(model, False)
    student_shot = int(selection_summary["student_shot"])
    shot_dir = ensure_dir(output_dir / f"shot_{student_shot}")
    train_subset = Subset(target_train_pool, selected_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    selection_payload = {
        "student_shot": student_shot,
        "score_mode": "original_clip_true_conf_plus_lora_teacher_true_conf_plus_js_similarity",
        "weights": {
            "lambda_clip": args.lambda_clip,
            "lambda_lora": args.lambda_lora,
            "lambda_similarity": args.lambda_similarity,
        },
        "require_both_correct": bool(args.require_both_correct),
        "selected_count": len(selected_indices),
        "selected_indices": selected_indices,
        "selected_samples": selected_rows,
        "summary": selection_summary,
    }
    save_json(selection_payload, output_dir / "selected_support" / f"shot_{student_shot}_selection.json")

    prompt_learner = CoOpPromptLearner(model, tokenizer, n_ctx=args.n_ctx, init_std=0.02).to(device)
    with torch.no_grad():
        prompt_learner.ctx.copy_(base_ctx)
    optimizer = torch.optim.SGD([prompt_learner.ctx], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    shot_history = []
    row = make_zero_shot_epoch0_row(
        target_name=target_name,
        preserve_loaders=preserve_loaders,
        baselines=baselines,
        shot=student_shot,
        optimizer=optimizer,
    )
    row["selection"] = {
        "method": "original_clip_plus_lora_full_teacher",
        "student_shot": student_shot,
        "selected_count": len(selected_indices),
    }
    row["split_seed"] = args.seed
    row["prompt_seed"] = args.prompt_seed
    shot_history.append(row)
    append_jsonl(row, curves_path)

    for epoch in range(1, args.epochs + 1):
        set_lora_enabled(model, False)
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

        set_lora_enabled(model, False)
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
        row["train"] = {
            "train_loss": train_metric["train_loss"],
            "train_acc": train_metric["train_acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        row["prompt_init"] = prompt_init_meta
        row["selection"] = {
            "method": "original_clip_plus_lora_full_teacher",
            "student_shot": student_shot,
            "selected_count": len(selected_indices),
            "require_both_correct": bool(args.require_both_correct),
        }
        row["split_seed"] = args.seed
        row["prompt_seed"] = args.prompt_seed
        shot_history.append(row)
        append_jsonl(row, curves_path)
        save_json(shot_history, shot_dir / "history.json")

    torch.save(
        {
            "shot": student_shot,
            "ctx": prompt_learner.ctx.detach().cpu(),
            "n_ctx": args.n_ctx,
            "target_classnames": target_classnames,
            "history": shot_history,
            "args": vars(args),
            "prompt_seed": args.prompt_seed,
            "same_initial_context_across_shots": True,
            "base_ctx_at_init": base_ctx.detach().cpu(),
            "selection_path": str(output_dir / "selected_support" / f"shot_{student_shot}_selection.json"),
        },
        shot_dir / "coop_prompt_final.pt",
    )

    return {
        "shot": student_shot,
        "selected_train_images": len(selected_indices),
        "final": shot_history[-1],
        "history_path": str(shot_dir / "history.json"),
        "checkpoint_path": str(shot_dir / "coop_prompt_final.pt"),
        "selection_path": str(output_dir / "selected_support" / f"shot_{student_shot}_selection.json"),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-name", type=str, required=True)
    parser.add_argument("--target-root", type=str, default=None)
    parser.add_argument("--target-train", type=str, default=None)
    parser.add_argument("--target-test", type=str, default=None)
    parser.add_argument("--target-classnames", type=str, default=None)
    parser.add_argument("--split-test-ratio", type=float, default=0.5)
    parser.add_argument("--preserve", type=str, nargs="*", default=[])

    parser.add_argument("--model", type=str, default="ViT-B-32-quickgelu")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=-1)
    parser.add_argument("--cache-features", action="store_true")

    parser.add_argument("--lora-target", choices=["visual", "text", "both"], default="visual")
    parser.add_argument("--lora-module-filter", type=str, default=".*")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-epochs", type=int, default=20)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--lora-weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-batch-size", type=int, default=128)

    parser.add_argument("--student-shots", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--n-ctx", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--prompt-seed", type=int, default=777)

    parser.add_argument("--require-both-correct", action="store_true")
    parser.add_argument("--strict-selection", action="store_true")
    parser.add_argument("--selection-batch-size", type=int, default=256)
    parser.add_argument("--save-all-candidate-scores", action="store_true")
    parser.add_argument("--lambda-clip", type=float, default=1.0)
    parser.add_argument("--lambda-lora", type=float, default=1.0)
    parser.add_argument("--lambda-similarity", type=float, default=1.0)

    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    ensure_dir(output_dir / "lora_teacher")
    ensure_dir(output_dir / "selected_support")
    save_json(vars(args), output_dir / "args.json")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    print("===== run info =====")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"visible cuda device count: {torch.cuda.device_count()}")
        print(f"visible cuda device 0: {torch.cuda.get_device_name(0)}")

    print("===== load CLIP and inject LoRA =====")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    model.float()
    tokenizer = open_clip.get_tokenizer(args.model)
    freeze_all_parameters(model)
    replaced = inject_lora_linear(
        model=model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        lora_target=args.lora_target,
        module_filter=args.lora_module_filter,
    )
    model.to(device)
    save_json(
        {
            "lora_target": args.lora_target,
            "lora_module_filter": args.lora_module_filter,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "wrapped_modules": replaced,
            "num_wrapped_modules": len(replaced),
            "trainable_parameters": int(sum(p.numel() for p in get_lora_parameters(model))),
        },
        output_dir / "lora_teacher" / "lora_config.json",
    )
    print(f"LoRA wrapped modules: {len(replaced)}")
    print(f"LoRA trainable parameters: {sum(p.numel() for p in get_lora_parameters(model))}")

    print("===== build datasets =====")
    target_train_pool, target_test_dataset, target_classnames, split_mode = build_target_datasets(
        target_root=args.target_root,
        target_train=args.target_train,
        target_test=args.target_test,
        transform=preprocess,
        split_test_ratio=args.split_test_ratio,
        seed=args.seed,
    )
    if args.target_classnames is not None:
        target_classnames = load_classnames_file(args.target_classnames)

    target_test_loader = build_loader(
        target_test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        max_samples=args.max_eval_samples,
    )
    lora_train_loader = DataLoader(
        target_train_pool,
        batch_size=args.lora_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
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
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            max_samples=args.max_eval_samples,
        )
        preserve_classnames[name] = classnames
        preserve_info[name] = {"path": path, "num_classes": len(classnames), "num_images": len(dataset)}

    print(f"target classes: {len(target_classnames)}")
    print(f"target train pool images: {len(target_train_pool)}")
    print(f"target test images: {len(target_test_dataset)}")

    print("===== original CLIP zero-shot baseline =====")
    baselines = evaluate_zero_shot_with_lora_state(
        model=model,
        tokenizer=tokenizer,
        target_name=args.target_name,
        target_loader=target_test_loader,
        target_classnames=target_classnames,
        preserve_loaders=preserve_loaders,
        preserve_classnames=preserve_classnames,
        device=device,
        lora_enabled=False,
    )
    save_json(baselines, output_dir / "zero_shot_baselines.json")

    print("===== train full-data LoRA teacher =====")
    set_lora_enabled(model, True)
    for param in get_lora_parameters(model):
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(get_lora_parameters(model), lr=args.lora_lr, weight_decay=args.lora_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.lora_epochs))
    token_groups = make_zero_shot_token_groups(
        tokenizer=tokenizer,
        classnames=target_classnames,
        templates=get_templates(args.target_name),
        device=device,
    )

    teacher_history = []
    for epoch in range(1, args.lora_epochs + 1):
        metric = train_lora_teacher_one_epoch(
            model=model,
            train_loader=lora_train_loader,
            token_groups=token_groups,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.lora_epochs,
        )
        scheduler.step()
        row = {"epoch": epoch, "train": {**metric, "lr": optimizer.param_groups[0]["lr"]}}
        teacher_history.append(row)
        save_json(teacher_history, output_dir / "lora_teacher" / "teacher_history.json")

    torch.save(
        {
            "lora_state_dict": get_lora_state_dict(model),
            "args": vars(args),
            "target_classnames": target_classnames,
            "wrapped_modules": replaced,
        },
        output_dir / "lora_teacher" / "lora_teacher_final.pt",
    )

    print("===== evaluate LoRA teacher =====")
    teacher_eval = evaluate_zero_shot_with_lora_state(
        model=model,
        tokenizer=tokenizer,
        target_name=args.target_name,
        target_loader=target_test_loader,
        target_classnames=target_classnames,
        preserve_loaders=preserve_loaders,
        preserve_classnames=preserve_classnames,
        device=device,
        lora_enabled=True,
    )
    save_json(teacher_eval, output_dir / "lora_teacher" / "teacher_eval.json")

    print("===== cache original CLIP eval features for student CoOp =====")
    target_cached = None
    preserve_cached = None
    if args.cache_features:
        set_lora_enabled(model, False)
        target_cached = precompute_image_features(
            model,
            target_test_loader,
            device,
            desc=f"cache target={args.target_name} original CLIP",
        )
        preserve_cached = {}
        for preserve_name, loader in preserve_loaders.items():
            preserve_cached[preserve_name] = precompute_image_features(
                model,
                loader,
                device,
                desc=f"cache preserve={preserve_name} original CLIP",
            )

    print("===== score train pool and select supports =====")
    score_rows = score_train_pool_with_lora_teacher(
        model=model,
        tokenizer=tokenizer,
        train_pool=target_train_pool,
        classnames=target_classnames,
        target_name=args.target_name,
        args=args,
        device=device,
    )
    score_summary = summarize_scores(score_rows)
    save_json(score_summary, output_dir / "selected_support" / "candidate_score_summary.json")
    if args.save_all_candidate_scores:
        save_json(score_rows, output_dir / "selected_support" / "candidate_scores.json")

    print("===== common student CoOp prompt initialization =====")
    set_lora_enabled(model, False)
    torch.manual_seed(args.prompt_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.prompt_seed)
    base_prompt_learner = CoOpPromptLearner(model, tokenizer, n_ctx=args.n_ctx, init_std=0.02).to(device)
    base_ctx = base_prompt_learner.ctx.detach().clone()
    save_json(
        {
            "prompt_seed": args.prompt_seed,
            "same_initial_context_across_shots": True,
            "ctx_shape": list(base_ctx.shape),
            "init_std": 0.02,
        },
        output_dir / "prompt_initialization.json",
    )
    torch.save({"base_ctx": base_ctx.cpu(), "prompt_seed": args.prompt_seed, "n_ctx": args.n_ctx}, output_dir / "prompt_initialization.pt")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompt_init_meta = {
        "prompt_seed": args.prompt_seed,
        "same_initial_context_across_shots": True,
        "epoch_0_meaning": "original CLIP zero-shot baseline before CoOp adaptation",
        "epoch_1_and_after": "Student CoOp prompt after selected-support adaptation",
    }

    curves_path = output_dir / "curves.jsonl"
    if curves_path.exists():
        curves_path.unlink()

    all_results = {
        "args": vars(args),
        "target_name": args.target_name,
        "split_mode": split_mode,
        "preserve_info": preserve_info,
        "zero_shot_baselines": baselines,
        "lora_teacher_eval": teacher_eval,
        "candidate_score_summary": score_summary,
        "prompt_init": prompt_init_meta,
        "shots": [],
    }

    print("===== train Student CoOp on selected supports =====")
    for student_shot in args.student_shots:
        selected_indices, selected_rows, selection_summary = select_balanced_by_score(
            score_rows=score_rows,
            student_shot=int(student_shot),
            class_count=len(target_classnames),
            require_both_correct=args.require_both_correct,
            strict_selection=args.strict_selection,
        )
        result = train_student_coop_for_shot(
            model=model,
            tokenizer=tokenizer,
            target_name=args.target_name,
            target_train_pool=target_train_pool,
            target_test_loader=target_test_loader,
            target_classnames=target_classnames,
            preserve_loaders=preserve_loaders,
            preserve_classnames=preserve_classnames,
            baselines=baselines,
            target_cached=target_cached,
            preserve_cached=preserve_cached,
            base_ctx=base_ctx,
            prompt_init_meta=prompt_init_meta,
            selected_indices=selected_indices,
            selected_rows=selected_rows,
            selection_summary=selection_summary,
            output_dir=output_dir,
            curves_path=curves_path,
            args=args,
            device=device,
        )
        all_results["shots"].append(result)
        save_json(all_results, output_dir / "results.json")

    print("===== done =====")
    print(f"output_dir: {output_dir}")
    print(f"results: {output_dir / 'results.json'}")
    print(f"curves: {curves_path}")


if __name__ == "__main__":
    main()
