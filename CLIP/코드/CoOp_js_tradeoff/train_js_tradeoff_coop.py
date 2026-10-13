# -*- coding: utf-8 -*-
"""
Initial-CLIP JS quantile trade-off selection + Student CoOp.

This diagnostic experiment uses only the initial zero-shot CLIP distribution
over the target train pool:

1. Score every train-pool image with initial CLIP.
2. Define an entropy anchor set D_A from high-entropy samples.
3. Build an anchor prediction distribution P_A.
4. Compute JS(P_A, p_0(x)) for every sample.
5. Assign samples to JS quantile bins Q1..QK.
6. Select class-balanced few-shot supports by:
   - random: random class-balanced support
   - anchor_entropy: high-entropy class-balanced support
   - js_q1: random samples from the lowest-JS bin
   - js_q2: random samples from the low-mid JS bin
   - js_q3: random samples from the high-mid JS bin
   - js_q4: random samples from the highest-JS bin
7. Train a fresh Student CoOp prompt with standard CE on the selected support.

This is a label-aware diagnostic experiment when --balance-mode true_label is
used, because true labels are used to enforce class balance during selection.
"""

import argparse
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

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
from train_blue_target_coop import (  # noqa: E402
    evaluate_current_prompt,
    get_templates,
    make_zero_shot_epoch0_row,
    parse_dataset_spec,
    train_one_epoch,
)
from utils import append_jsonl, ensure_dir, load_classnames_file, save_json, seed_everything  # noqa: E402


EPS = 1e-8
SELECTION_MODES = [
    "random",
    "anchor_entropy",
    "agreement",
    "disagreement",
    "js_q1",
    "js_q2",
    "js_q3",
    "js_q4",
]


def _as_float(value) -> float:
    return float(value.detach().cpu().item() if torch.is_tensor(value) else value)


def get_local_label(dataset: Dataset, local_idx: int) -> int:
    if isinstance(dataset, Subset):
        base_idx = int(dataset.indices[local_idx])
        base = dataset.dataset
        if hasattr(base, "samples"):
            return int(base.samples[base_idx][1])
        if hasattr(base, "targets"):
            return int(base.targets[base_idx])
    if hasattr(dataset, "samples"):
        return int(dataset.samples[local_idx][1])
    if hasattr(dataset, "targets"):
        return int(dataset.targets[local_idx])
    _, label = dataset[local_idx]
    return int(label)


def get_local_path(dataset: Dataset, local_idx: int) -> str:
    if isinstance(dataset, Subset):
        base_idx = int(dataset.indices[local_idx])
        base = dataset.dataset
        if hasattr(base, "samples"):
            return str(base.samples[base_idx][0])
        return f"{type(base).__name__}[{base_idx}]"
    if hasattr(dataset, "samples"):
        return str(dataset.samples[local_idx][0])
    return f"{type(dataset).__name__}[{local_idx}]"


def label_hist(records: Sequence[Dict], key: str) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for record in records:
        value = str(record.get(key, "missing"))
        hist[value] = hist.get(value, 0) + 1
    return dict(sorted(hist.items(), key=lambda item: item[0]))


def numeric_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": _as_float(tensor.mean()),
        "std": _as_float(tensor.std(unbiased=False)),
        "min": _as_float(tensor.min()),
        "max": _as_float(tensor.max()),
    }


def light_record(record: Dict) -> Dict:
    keys = [
        "local_idx",
        "image_path",
        "true_label",
        "true_classname",
        "pseudo_label",
        "pseudo_classname",
        "confidence",
        "entropy",
        "js_to_anchor",
        "js_bin",
        "is_anchor",
        "is_fallback",
        "fallback_from_bin",
        "target_js_bin",
        "selection_rank",
    ]
    return {key: record.get(key) for key in keys if key in record}


@torch.no_grad()
def score_train_pool_with_initial_clip(
    model,
    target_train_pool: Dataset,
    target_classnames: List[str],
    zero_text_features: torch.Tensor,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> List[Dict]:
    loader = DataLoader(
        target_train_pool,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    model.eval()
    records: List[Dict] = []
    offset = 0

    for images, targets in tqdm(loader, desc="score train pool with initial CLIP"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        logits = model.logit_scale.exp() * image_features @ zero_text_features.t()
        probs = logits.softmax(dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(EPS))).sum(dim=-1)
        confidence, pseudo = probs.max(dim=-1)

        for i in range(images.shape[0]):
            local_idx = offset + i
            true_label = int(targets[i].detach().cpu().item())
            pseudo_label = int(pseudo[i].detach().cpu().item())
            record = {
                "local_idx": int(local_idx),
                "image_path": get_local_path(target_train_pool, local_idx),
                "true_label": true_label,
                "true_classname": str(target_classnames[true_label]),
                "pseudo_label": pseudo_label,
                "pseudo_classname": str(target_classnames[pseudo_label]),
                "confidence": _as_float(confidence[i]),
                "entropy": _as_float(entropy[i]),
                "prob": [float(v) for v in probs[i].detach().cpu().tolist()],
                "logits": [float(v) for v in logits[i].detach().cpu().tolist()],
                "is_anchor": False,
            }
            records.append(record)
        offset += images.shape[0]

    return records


def select_entropy_anchor_set(
    records: List[Dict],
    anchor_ratio: float,
    anchor_per_class: int,
    anchor_mode: str,
) -> List[int]:
    if not records:
        raise ValueError("records is empty")

    if anchor_mode in {"true_label", "pseudo_label"}:
        label_key = "true_label" if anchor_mode == "true_label" else "pseudo_label"
        grouped: Dict[int, List[Dict]] = defaultdict(list)
        for record in records:
            grouped[int(record[label_key])].append(record)
        selected: List[int] = []
        for label, group in sorted(grouped.items()):
            ordered = sorted(group, key=lambda r: (-float(r["entropy"]), int(r["local_idx"])))
            k = int(anchor_per_class) if anchor_per_class > 0 else max(1, int(math.floor(float(anchor_ratio) * len(ordered))))
            if len(ordered) < k:
                raise ValueError(
                    f"anchor_mode={anchor_mode} label={label} has {len(ordered)} samples, "
                    f"but requested anchor count={k}"
                )
            selected.extend(int(r["local_idx"]) for r in ordered[:k])
        return sorted(set(selected))

    k = max(1, int(math.floor(float(anchor_ratio) * len(records))))
    ordered = sorted(records, key=lambda r: (-float(r["entropy"]), int(r["local_idx"])))
    return sorted(int(r["local_idx"]) for r in ordered[:k])


def build_anchor_distribution(records: List[Dict], anchor_indices: Iterable[int], mode: str) -> Dict:
    anchor_set = set(int(i) for i in anchor_indices)
    if not anchor_set:
        raise ValueError("anchor set is empty")

    anchor_records = [records[i] for i in sorted(anchor_set)]
    global_prob = torch.tensor([r["prob"] for r in anchor_records], dtype=torch.float32).mean(dim=0)

    if mode == "global":
        return {"mode": "global", "global": global_prob.tolist(), "by_pseudo_class": {}}

    grouped: Dict[int, List[List[float]]] = defaultdict(list)
    for record in anchor_records:
        grouped[int(record["pseudo_label"])].append(record["prob"])
    by_class = {
        str(label): torch.tensor(probs, dtype=torch.float32).mean(dim=0).tolist()
        for label, probs in grouped.items()
    }
    return {"mode": "pseudo_class", "global": global_prob.tolist(), "by_pseudo_class": by_class}


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.float().clamp_min(EPS)
    q = q.float().clamp_min(EPS)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    js = 0.5 * torch.sum(p * torch.log(p / m.clamp_min(EPS))) + 0.5 * torch.sum(q * torch.log(q / m.clamp_min(EPS)))
    return _as_float(js)


def compute_js_to_anchor(records: List[Dict], anchor_distribution: Dict) -> None:
    global_anchor = torch.tensor(anchor_distribution["global"], dtype=torch.float32)
    by_class = {
        int(label): torch.tensor(prob, dtype=torch.float32)
        for label, prob in anchor_distribution.get("by_pseudo_class", {}).items()
    }
    use_pseudo = anchor_distribution.get("mode") == "pseudo_class"
    for record in records:
        prob = torch.tensor(record["prob"], dtype=torch.float32)
        if use_pseudo:
            anchor = by_class.get(int(record["pseudo_label"]), global_anchor)
        else:
            anchor = global_anchor
        record["js_to_anchor"] = js_divergence(anchor, prob)


def assign_js_quantile_bins(records: List[Dict], num_bins: int, exclude_anchor: bool) -> Dict:
    if num_bins < 2:
        raise ValueError("--num-js-bins must be >= 2")
    candidates = [r for r in records if not (exclude_anchor and r.get("is_anchor"))]
    if not candidates:
        raise ValueError("No candidates available for JS bin assignment.")

    values = torch.tensor([float(r["js_to_anchor"]) for r in candidates], dtype=torch.float32)
    quantile_points = [i / num_bins for i in range(1, num_bins)]
    thresholds = [float(torch.quantile(values, q).item()) for q in quantile_points]

    def to_bin(value: float) -> int:
        bin_id = 1
        for threshold in thresholds:
            if value > threshold:
                bin_id += 1
            else:
                break
        return min(num_bins, max(1, bin_id))

    for record in records:
        record["js_bin"] = int(to_bin(float(record["js_to_anchor"])))

    return {
        "num_js_bins": int(num_bins),
        "exclude_anchor_for_bin_thresholds": bool(exclude_anchor),
        "quantile_points": quantile_points,
        "thresholds": thresholds,
        "candidate_count": len(candidates),
        "candidate_js": numeric_stats([float(r["js_to_anchor"]) for r in candidates]),
    }


def js_bin_fallback_order(target_bin: int, num_bins: int) -> List[int]:
    bins = list(range(1, num_bins + 1))
    if target_bin == 1:
        return bins
    if target_bin == num_bins:
        return list(reversed(bins))
    lower_first = target_bin <= (num_bins + 1) / 2

    def key_fn(bin_id: int) -> Tuple[int, int]:
        distance = abs(bin_id - target_bin)
        if lower_first:
            side = 0 if bin_id < target_bin else 1
        else:
            side = 0 if bin_id > target_bin else 1
        return (distance, side)

    return sorted(bins, key=key_fn)


def js_bin_statistics(records: List[Dict], num_bins: int, exclude_anchor: bool) -> Dict:
    stats = {
        "num_js_bins": int(num_bins),
        "exclude_anchor": bool(exclude_anchor),
        "bins": [],
    }
    for bin_id in range(1, num_bins + 1):
        group = [
            r
            for r in records
            if int(r.get("js_bin", -1)) == bin_id and not (exclude_anchor and r.get("is_anchor"))
        ]
        stats["bins"].append(
            {
                "bin_id": int(bin_id),
                "num_samples": len(group),
                "js": numeric_stats([float(r["js_to_anchor"]) for r in group]),
                "entropy": numeric_stats([float(r["entropy"]) for r in group]),
                "confidence": numeric_stats([float(r["confidence"]) for r in group]),
                "true_label_hist": label_hist(group, "true_label"),
                "pseudo_label_hist": label_hist(group, "pseudo_label"),
                "anchor_count": int(sum(1 for r in group if r.get("is_anchor"))),
            }
        )
    return stats


def parse_js_quantile_mode(selection_mode: str) -> Optional[int]:
    if not selection_mode.startswith("js_q"):
        return None
    try:
        return int(selection_mode.replace("js_q", ""))
    except ValueError:
        return None


def choose_class_balanced_records(
    records: List[Dict],
    shots: int,
    selection_mode: str,
    exclude_anchor: bool,
    balance_mode: str,
    sample_seed: int,
    num_js_bins: int,
) -> Tuple[List[int], List[Dict]]:
    if shots <= 0:
        raise ValueError("shots must be >= 1")
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"unknown selection mode: {selection_mode}")
    if balance_mode not in {"true_label", "pseudo_label"}:
        raise ValueError(f"unknown balance mode: {balance_mode}")

    rng = random.Random(sample_seed)
    balance_key = "true_label" if balance_mode == "true_label" else "pseudo_label"

    target_js_bin = parse_js_quantile_mode(selection_mode)
    if target_js_bin is not None and not (1 <= target_js_bin <= num_js_bins):
        raise ValueError(f"{selection_mode} is incompatible with num_js_bins={num_js_bins}")

    candidates: List[Dict] = []
    for record in records:
        if selection_mode in {"random", "agreement", "disagreement"} and exclude_anchor and record.get("is_anchor"):
            continue
        if target_js_bin is not None and exclude_anchor and record.get("is_anchor"):
            continue
        candidates.append(record)

    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for record in candidates:
        grouped[int(record[balance_key])].append(record)

    selected_records: List[Dict] = []
    for label, group in sorted(grouped.items()):
        group_copy = list(group)
        if selection_mode == "random":
            rng.shuffle(group_copy)
            ordered = group_copy
        elif selection_mode == "anchor_entropy":
            ordered = sorted(group_copy, key=lambda r: (-float(r["entropy"]), int(r["local_idx"])))
        elif selection_mode == "agreement":
            ordered = sorted(group_copy, key=lambda r: (float(r["js_to_anchor"]), int(r["local_idx"])))
        elif selection_mode == "disagreement":
            ordered = sorted(group_copy, key=lambda r: (-float(r["js_to_anchor"]), int(r["local_idx"])))
        else:
            selected_for_label: List[Dict] = []
            used_indices = set()
            for bin_id in js_bin_fallback_order(int(target_js_bin), int(num_js_bins)):
                bin_group = [r for r in group_copy if int(r["js_bin"]) == int(bin_id) and int(r["local_idx"]) not in used_indices]
                rng.shuffle(bin_group)
                for record in bin_group:
                    copied = dict(record)
                    copied["is_fallback"] = int(bin_id) != int(target_js_bin)
                    copied["fallback_from_bin"] = int(bin_id) if copied["is_fallback"] else None
                    copied["target_js_bin"] = int(target_js_bin)
                    selected_for_label.append(copied)
                    used_indices.add(int(record["local_idx"]))
                    if len(selected_for_label) >= shots:
                        break
                if len(selected_for_label) >= shots:
                    break
            if len(selected_for_label) < shots:
                raise ValueError(
                    f"selection_mode={selection_mode}, balance_mode={balance_mode}, "
                    f"label={label} has {len(selected_for_label)} candidates after JS-bin fallback, but shots={shots}"
                )
            for rank, record in enumerate(selected_for_label[:shots]):
                record["selection_rank"] = int(rank)
                selected_records.append(record)
            continue

        if len(ordered) < shots:
            raise ValueError(
                f"selection_mode={selection_mode}, balance_mode={balance_mode}, "
                f"label={label} has {len(ordered)} candidates after filtering, but shots={shots}"
            )
        for rank, record in enumerate(ordered[:shots]):
            copied = dict(record)
            copied["selection_rank"] = int(rank)
            copied["is_fallback"] = False
            copied["fallback_from_bin"] = None
            copied["target_js_bin"] = target_js_bin
            selected_records.append(copied)

    selected_indices = [int(record["local_idx"]) for record in selected_records]
    rng.shuffle(selected_indices)
    selected_records = sorted(selected_records, key=lambda r: (int(r[balance_key]), int(r["selection_rank"]), int(r["local_idx"])))
    return selected_indices, selected_records


def selection_statistics(selected_records: List[Dict]) -> Dict:
    fallback_count = int(sum(1 for r in selected_records if r.get("is_fallback")))
    return {
        "selected_count": len(selected_records),
        "selected_js": numeric_stats([float(r["js_to_anchor"]) for r in selected_records]),
        "selected_entropy": numeric_stats([float(r["entropy"]) for r in selected_records]),
        "selected_confidence": numeric_stats([float(r["confidence"]) for r in selected_records]),
        "selected_true_label_hist": label_hist(selected_records, "true_label"),
        "selected_pseudo_label_hist": label_hist(selected_records, "pseudo_label"),
        "selected_anchor_count": int(sum(1 for r in selected_records if r.get("is_anchor"))),
        "fallback_count": fallback_count,
        "fallback_rate": float(fallback_count / max(1, len(selected_records))),
        "selected_js_bin_hist": label_hist(selected_records, "js_bin"),
    }


def anchor_statistics(records: List[Dict], anchor_indices: List[int], anchor_ratio: float, anchor_mode: str, anchor_dist_mode: str) -> Dict:
    anchor_records = [records[i] for i in sorted(anchor_indices)]
    return {
        "anchor_ratio": float(anchor_ratio),
        "anchor_mode": anchor_mode,
        "anchor_dist_mode": anchor_dist_mode,
        "anchor_size": len(anchor_records),
        "anchor_entropy": numeric_stats([float(r["entropy"]) for r in anchor_records]),
        "anchor_confidence": numeric_stats([float(r["confidence"]) for r in anchor_records]),
        "anchor_js": numeric_stats([float(r["js_to_anchor"]) for r in anchor_records if r.get("js_to_anchor") is not None]),
        "anchor_pseudo_label_hist": label_hist(anchor_records, "pseudo_label"),
        "anchor_true_label_hist": label_hist(anchor_records, "true_label"),
        "anchor_samples": [light_record(r) for r in anchor_records],
    }


def save_light_pool_scores(records: List[Dict], path: Path) -> None:
    save_json([light_record(record) for record in records], path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-name", type=str, required=True)
    parser.add_argument("--target-root", type=str, default=None)
    parser.add_argument("--target-train", type=str, default=None)
    parser.add_argument("--target-test", type=str, default=None)
    parser.add_argument("--target-classnames", type=str, default=None)
    parser.add_argument("--split-test-ratio", type=float, default=0.5)

    parser.add_argument("--preserve", type=str, nargs="+", required=True)
    parser.add_argument("--model", type=str, default="ViT-B-32-quickgelu")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--shots-list", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--n-ctx", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-seed", type=int, default=777)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=-1)
    parser.add_argument("--cache-features", action="store_true")

    parser.add_argument("--selection-mode", type=str, default="js_q1", choices=SELECTION_MODES)
    parser.add_argument("--anchor-ratio", type=float, default=0.2)
    parser.add_argument("--anchor-per-class", type=int, default=0)
    parser.add_argument("--anchor-mode", type=str, default="global", choices=["global", "true_label", "pseudo_label"])
    parser.add_argument("--anchor-dist-mode", type=str, default="global", choices=["global", "pseudo_class"])
    parser.add_argument("--num-js-bins", type=int, default=4)
    parser.add_argument("--exclude-anchor", action="store_true")
    parser.add_argument("--balance-mode", type=str, default="true_label", choices=["true_label", "pseudo_label"])
    parser.add_argument("--save-pool-scores", action="store_true")

    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sample_seed is None:
        args.sample_seed = args.seed
    if not (0.0 < float(args.anchor_ratio) <= 1.0):
        raise ValueError("--anchor-ratio must be in (0, 1]")

    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    save_json(vars(args), output_dir / "args.json")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    device = torch.device(args.device)
    print("===== run info =====")
    print(f"device: {device}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    if device.type == "cuda":
        print(f"visible cuda device count: {torch.cuda.device_count()}")
        print(f"visible cuda device 0: {torch.cuda.get_device_name(0)}")

    print("===== load CLIP =====")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    model.float()
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
    if args.target_classnames is not None:
        target_classnames = load_classnames_file(args.target_classnames)

    target_test_loader = build_loader(
        target_test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        max_samples=args.max_eval_samples,
    )

    print("===== target info =====")
    print(f"target_name: {args.target_name}")
    print(f"split_mode: {split_mode}")
    print(f"target classes: {len(target_classnames)}")
    print(f"target train pool images: {len(target_train_pool)}")
    print(f"target test images: {len(target_test_loader.dataset)}")

    preserve_loaders = {}
    preserve_classnames = {}
    preserve_info = {}
    for spec in args.preserve:
        name, path, classnames_file = parse_dataset_spec(spec)
        dataset, classnames = build_preserve_dataset(path, transform=preprocess)
        if classnames_file is not None:
            classnames = load_classnames_file(classnames_file)
        else:
            classnames = maybe_replace_imagenet_wnid_with_torchvision_classes(name, classnames)
        loader = build_loader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            max_samples=args.max_eval_samples,
        )
        preserve_loaders[name] = loader
        preserve_classnames[name] = classnames
        preserve_info[name] = {"path": path, "num_classes": len(classnames), "num_images": len(loader.dataset)}

    print("===== preserve info =====")
    for name, info in preserve_info.items():
        print(name, info)

    print("===== original CLIP zero-shot baseline =====")
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

    curves_path = output_dir / "curves.jsonl"
    if curves_path.exists():
        print(f"[warning] remove existing curves.jsonl: {curves_path}")
        curves_path.unlink()

    print("===== score target train pool =====")
    records = score_train_pool_with_initial_clip(
        model=model,
        target_train_pool=target_train_pool,
        target_classnames=target_classnames,
        zero_text_features=target_zero_text,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    anchor_indices = select_entropy_anchor_set(
        records=records,
        anchor_ratio=args.anchor_ratio,
        anchor_per_class=args.anchor_per_class,
        anchor_mode=args.anchor_mode,
    )
    anchor_set = set(anchor_indices)
    for record in records:
        record["is_anchor"] = int(record["local_idx"]) in anchor_set
    anchor_distribution = build_anchor_distribution(records, anchor_indices, args.anchor_dist_mode)
    compute_js_to_anchor(records, anchor_distribution)
    js_bin_meta = assign_js_quantile_bins(records, args.num_js_bins, args.exclude_anchor)

    anchor_meta = anchor_statistics(records, anchor_indices, args.anchor_ratio, args.anchor_mode, args.anchor_dist_mode)
    js_bins_meta = {
        **js_bin_meta,
        **js_bin_statistics(records, args.num_js_bins, args.exclude_anchor),
    }
    save_json(anchor_meta, output_dir / "anchor_set.json")
    save_json(anchor_distribution, output_dir / "anchor_distribution.json")
    save_json(js_bins_meta, output_dir / "js_bins.json")
    if args.save_pool_scores:
        save_light_pool_scores(records, output_dir / "train_pool_scores.json")

    score_overview = {
        "num_records": len(records),
        "entropy": numeric_stats([float(r["entropy"]) for r in records]),
        "confidence": numeric_stats([float(r["confidence"]) for r in records]),
        "js_to_anchor": numeric_stats([float(r["js_to_anchor"]) for r in records]),
        "js_bin_hist": label_hist(records, "js_bin"),
        "true_label_hist": label_hist(records, "true_label"),
        "pseudo_label_hist": label_hist(records, "pseudo_label"),
    }
    save_json(score_overview, output_dir / "train_pool_score_overview.json")

    target_cached = None
    preserve_cached = None
    if args.cache_features:
        print("===== cache eval image features =====")
        target_cached = precompute_image_features(model, target_test_loader, device, desc=f"cache target={args.target_name}")
        preserve_cached = {}
        for preserve_name, loader in preserve_loaders.items():
            preserve_cached[preserve_name] = precompute_image_features(
                model, loader, device, desc=f"cache preserve={preserve_name}"
            )

    print("===== shared CoOp prompt initialization =====")
    print(f"prompt_seed: {args.prompt_seed}")
    torch.manual_seed(args.prompt_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.prompt_seed)
    base_prompt_learner = CoOpPromptLearner(
        clip_model=model,
        tokenizer=tokenizer,
        n_ctx=args.n_ctx,
        init_std=0.02,
    ).to(device)
    base_ctx = base_prompt_learner.ctx.detach().clone()
    save_json(
        {
            "prompt_seed": args.prompt_seed,
            "n_ctx": args.n_ctx,
            "ctx_shape": list(base_ctx.shape),
            "init_std": 0.02,
            "note": "All shots start from this same initial CoOp context.",
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
        "epoch_1_and_after": "CoOp prompt after selected target few-shot adaptation",
    }

    all_results = {
        "args": vars(args),
        "target_name": args.target_name,
        "split_mode": split_mode,
        "preserve_info": preserve_info,
        "zero_shot_baselines": baselines,
        "anchor_meta": {k: v for k, v in anchor_meta.items() if k != "anchor_samples"},
        "js_bins": js_bins_meta,
        "prompt_init": prompt_init_meta,
        "shots": [],
    }

    for shot in args.shots_list:
        print(f"===== shot={shot} Student CoOp training =====")
        shot_dir = ensure_dir(output_dir / f"shot_{shot}")
        selected_indices, selected_records = choose_class_balanced_records(
            records=records,
            shots=shot,
            selection_mode=args.selection_mode,
            exclude_anchor=args.exclude_anchor,
            balance_mode=args.balance_mode,
            sample_seed=args.sample_seed,
            num_js_bins=args.num_js_bins,
        )
        fewshot_subset = Subset(target_train_pool, selected_indices)
        fewshot_loader = DataLoader(
            fewshot_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        selection_stats = selection_statistics(selected_records)
        selection_meta = {
            "selection_mode": args.selection_mode,
            "shot": int(shot),
            "sample_seed": int(args.sample_seed),
            "balance_mode": args.balance_mode,
            "exclude_anchor": bool(args.exclude_anchor),
            "anchor_ratio": float(args.anchor_ratio),
            "anchor_mode": args.anchor_mode,
            "anchor_dist_mode": args.anchor_dist_mode,
            "num_js_bins": int(args.num_js_bins),
            "selected_count": len(selected_indices),
            "selected_indices": selected_indices,
            "selected_samples": [light_record(record) for record in selected_records],
            "stats": selection_stats,
            "note": "true_label balance is label-aware diagnostic selection.",
        }
        save_json(selection_meta, shot_dir / "selection.json")
        save_json(selection_stats, shot_dir / "score_stats.json")

        prompt_learner = CoOpPromptLearner(model, tokenizer, n_ctx=args.n_ctx, init_std=0.02).to(device)
        with torch.no_grad():
            prompt_learner.ctx.copy_(base_ctx)
        optimizer = torch.optim.SGD([prompt_learner.ctx], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

        shot_history = []
        row = make_zero_shot_epoch0_row(
            target_name=args.target_name,
            preserve_loaders=preserve_loaders,
            baselines=baselines,
            shot=shot,
            optimizer=optimizer,
        )
        row["selection"] = {
            "selection_mode": args.selection_mode,
            "sample_seed": args.sample_seed,
            "selected_count": len(selected_indices),
            "balance_mode": args.balance_mode,
            "exclude_anchor": bool(args.exclude_anchor),
            "num_js_bins": int(args.num_js_bins),
        }
        row["sample_seed"] = args.sample_seed
        row["split_seed"] = args.seed
        row["prompt_seed"] = args.prompt_seed
        shot_history.append(row)
        append_jsonl(row, curves_path)
        save_json(shot_history, shot_dir / "history.json")

        for epoch in range(1, args.epochs + 1):
            train_metric = train_one_epoch(
                model, prompt_learner, fewshot_loader, target_classnames, optimizer, device, epoch, args.epochs
            )
            scheduler.step()

            do_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
            if not do_eval:
                continue

            row = evaluate_current_prompt(
                model,
                prompt_learner,
                args.target_name,
                target_test_loader,
                target_classnames,
                preserve_loaders,
                preserve_classnames,
                baselines,
                device,
                shot,
                epoch,
                target_cached=target_cached,
                preserve_cached=preserve_cached,
            )
            row["selection"] = {
                "selection_mode": args.selection_mode,
                "sample_seed": args.sample_seed,
                "selected_count": len(selected_indices),
                "balance_mode": args.balance_mode,
                "exclude_anchor": bool(args.exclude_anchor),
                "num_js_bins": int(args.num_js_bins),
            }
            row["sample_seed"] = args.sample_seed
            row["split_seed"] = args.seed
            row["prompt_seed"] = args.prompt_seed
            row["train"] = {
                "train_loss": train_metric["train_loss"],
                "train_acc": train_metric["train_acc"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            row["prompt_init"] = prompt_init_meta
            shot_history.append(row)
            append_jsonl(row, curves_path)
            save_json(shot_history, shot_dir / "history.json")

        torch.save(
            {
                "sample_seed": args.sample_seed,
                "split_seed": args.seed,
                "selection_mode": args.selection_mode,
                "shot": shot,
                "ctx": prompt_learner.ctx.detach().cpu(),
                "n_ctx": args.n_ctx,
                "target_classnames": target_classnames,
                "history": shot_history,
                "args": vars(args),
                "prompt_seed": args.prompt_seed,
                "same_initial_context_across_shots": True,
                "base_ctx_at_init": base_ctx.cpu(),
                "selection_path": str(shot_dir / "selection.json"),
            },
            shot_dir / "coop_prompt_final.pt",
        )

        all_results["shots"].append(
            {
                "sample_seed": args.sample_seed,
                "selection_mode": args.selection_mode,
                "shot": shot,
                "fewshot_train_images": len(fewshot_subset),
                "final": shot_history[-1],
                "history_path": str(shot_dir / "history.json"),
                "selection_path": str(shot_dir / "selection.json"),
                "checkpoint_path": str(shot_dir / "coop_prompt_final.pt"),
            }
        )
        save_json(all_results, output_dir / "results.json")

    print("===== done =====")
    print(f"output_dir: {output_dir}")
    print(f"results: {output_dir / 'results.json'}")
    print(f"curves: {curves_path}")


if __name__ == "__main__":
    main()
