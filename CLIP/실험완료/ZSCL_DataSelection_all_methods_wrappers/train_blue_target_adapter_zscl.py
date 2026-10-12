# -*- coding: utf-8 -*-
"""
Blue target ZSCL-aware CLIP-Adapter adaptation 실험 실행 파일

[중요 수정 - 단일 파일 패치]
이전 버그: adapter 입력 직전에 F.normalize를 걸어서 unit sphere(magnitude≈1) feature를
adapter에 넣고 있었음. CLIP raw feature magnitude는 5~15라 잔차 결합이 raw 스케일에서
이루어져야 adapter 기여가 의미를 가짐.

수정 4곳:
  1) train_one_epoch_adapter: model.encode_image 직후 F.normalize 제거
  2) evaluate_adapter (uncached): model.encode_image 직후 F.normalize 제거
  3) main에서 캐시는 eval.py의 (정규화된) precompute_image_features 대신,
     이 파일 안의 _precompute_raw_image_features 사용 (raw feature 캐시)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
sys.path.append(str(SRC_DIR))

from coop_model import (
    CLIPAdapter,
    build_zero_shot_text_features,
    freeze_clip_model,
    infer_visual_output_dim,
)
from sample_selection import (
    _compute_teacher_scores,
    build_fewshot_subset_zscl,
)
from data import (
    build_loader,
    build_preserve_dataset,
    build_target_datasets,
    maybe_replace_imagenet_wnid_with_torchvision_classes,
)
from utils import append_jsonl, ensure_dir, load_classnames_file, save_json, seed_everything


# ---------------------------------------------------------------------------
# raw feature 캐시 (이 파일 전용, eval.py의 정규화 캐시 대체)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _precompute_raw_image_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """캐시는 *unnormalized* raw feature로 저장. CLIP-Adapter는 raw 스케일에서 잔차 결합."""
    model.eval()
    feats_list = []
    tgts_list = []
    for images, targets in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        f = model.encode_image(images)
        feats_list.append(f.detach().cpu())
        tgts_list.append(targets.detach().cpu())
    return torch.cat(feats_list, dim=0), torch.cat(tgts_list, dim=0)


# ---------------------------------------------------------------------------
# 보조 함수
# ---------------------------------------------------------------------------

def parse_dataset_spec(spec: str) -> Tuple[str, str, Optional[str]]:
    parts = spec.split(":", 2)
    if len(parts) == 2:
        name, path = parts
        return name, path, None
    if len(parts) == 3:
        name, path, classnames_file = parts
        return name, path, classnames_file
    raise ValueError(f"dataset spec 형식이 잘못됨: {spec}")


def get_templates(dataset_name: str) -> List[str]:
    name = dataset_name.lower()
    if "texture" in name or "dtd" in name:
        return ["{} texture.", "a photo of a {} texture.", "a close-up photo of {} texture."]
    if "eurosat" in name or "resisc" in name:
        return ["a satellite photo of {}.", "a centered satellite photo of {}."]
    if "gtsrb" in name:
        return ["a photo of a {} traffic sign.", "a centered photo of a {} traffic sign."]
    if "flower" in name:
        return ["a photo of a {}, a type of flower.", "a flower photo of a {}."]
    if "aircraft" in name or "fgvc" in name:
        return ["a photo of a {}, a type of aircraft.", "a photo of the aircraft {}."]
    return ["a photo of a {}.", "a blurry photo of a {}.", "a small photo of a {}.", "a photo of the {}."]


@torch.no_grad()
def evaluate_zero_shot(model, loader, text_features, device, desc):
    model.eval()
    logit_scale = model.logit_scale.exp()
    text_features = text_features.to(device)
    correct = 0
    total = 0
    for images, targets in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        img_feat = model.encode_image(images)
        img_feat = F.normalize(img_feat, dim=-1)
        logits = logit_scale * img_feat @ text_features.T
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.numel()
    return {
        "accuracy": float(correct / total if total > 0 else 0.0),
        "correct": int(correct),
        "total": int(total),
    }


def train_one_epoch_adapter(model, visual_adapter, train_loader, raw_text_features,
                            optimizer, device, epoch, total_epochs):
    model.eval()
    visual_adapter.train()
    logit_scale = model.logit_scale.exp().detach()
    raw_text_features = raw_text_features.to(device)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    pbar = tqdm(train_loader, desc=f"train Visual Adapter epoch {epoch}/{total_epochs}")
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # [수정] 저자 코드와 동일하게 raw image feature를 그대로 adapter에 전달.
        with torch.no_grad():
            image_features = model.encode_image(images)

        adapted_image = visual_adapter(image_features)
        logits = logit_scale * adapted_image @ raw_text_features.T
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct = (preds == targets).sum().item()

        total_loss += loss.item() * targets.numel()
        total_correct += correct
        total_seen += targets.numel()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / targets.numel():.4f}",
            alpha=f"{visual_adapter.alpha:.3f}",
        )

    return {
        "train_loss": float(total_loss / max(1, total_seen)),
        "train_acc": float(total_correct / max(1, total_seen)),
        "alpha_visual": visual_adapter.alpha,
    }


@torch.no_grad()
def evaluate_adapter(model, visual_adapter, loader, raw_text_features, device, desc,
                     cached_features=None, cached_targets=None):
    model.eval()
    visual_adapter.eval()
    logit_scale = model.logit_scale.exp()
    raw_text_features = raw_text_features.to(device)

    chunk = 4096
    correct = 0
    total = 0

    if cached_features is not None and cached_targets is not None:
        feats = cached_features.to(device)
        tgts = cached_targets.to(device)
        for start in range(0, feats.shape[0], chunk):
            end = min(start + chunk, feats.shape[0])
            adapted_img = visual_adapter(feats[start:end])
            logits = logit_scale * adapted_img @ raw_text_features.T
            correct += (logits.argmax(1) == tgts[start:end]).sum().item()
            total += end - start
    else:
        for images, targets in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # [수정] adapter 입력은 raw feature.
            img_feat = model.encode_image(images)
            adapted_img = visual_adapter(img_feat)
            logits = logit_scale * adapted_img @ raw_text_features.T
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.numel()

    return {
        "accuracy": float(correct / total if total > 0 else 0.0),
        "correct": int(correct),
        "total": int(total),
    }


def evaluate_current_adapter(model, visual_adapter, target_name, target_loader, raw_target_text,
                             preserve_loaders, raw_preserve_text, baselines, device, shot, epoch,
                             target_cached=None, preserve_cached=None) -> Dict:
    row = {"target_name": target_name, "shot": shot, "epoch": epoch, "target": {}, "preserve": {}}
    target_metric = evaluate_adapter(
        model=model, visual_adapter=visual_adapter, loader=target_loader,
        raw_text_features=raw_target_text, device=device,
        desc=f"eval Adapter target={target_name} shot={shot} epoch={epoch}",
        cached_features=target_cached[0] if target_cached is not None else None,
        cached_targets=target_cached[1] if target_cached is not None else None,
    )
    target_base = baselines["target"]["accuracy"]
    row["target"] = {
        "accuracy": target_metric["accuracy"],
        "correct": target_metric["correct"],
        "total": target_metric["total"],
        "zero_shot_baseline": target_base,
        "delta_from_zero_shot": target_metric["accuracy"] - target_base,
    }
    for preserve_name, loader in preserve_loaders.items():
        cached = preserve_cached.get(preserve_name) if preserve_cached is not None else None
        metric = evaluate_adapter(
            model=model, visual_adapter=visual_adapter, loader=loader,
            raw_text_features=raw_preserve_text[preserve_name], device=device,
            desc=f"eval Adapter preserve={preserve_name} shot={shot} epoch={epoch}",
            cached_features=cached[0] if cached is not None else None,
            cached_targets=cached[1] if cached is not None else None,
        )
        base = baselines["preserve"][preserve_name]["accuracy"]
        row["preserve"][preserve_name] = {
            "accuracy": metric["accuracy"],
            "correct": metric["correct"],
            "total": metric["total"],
            "zero_shot_baseline": base,
            "delta_from_zero_shot": metric["accuracy"] - base,
        }
    return row


def make_zero_shot_epoch0_row(target_name, preserve_loaders, baselines, shot, optimizer, visual_adapter):
    row = {"target_name": target_name, "shot": shot, "epoch": 0, "target": {}, "preserve": {}}
    target_base = baselines["target"]["accuracy"]
    row["target"] = {
        "accuracy": target_base,
        "correct": baselines["target"]["correct"],
        "total": baselines["target"]["total"],
        "zero_shot_baseline": target_base,
        "delta_from_zero_shot": 0.0,
    }
    for preserve_name in preserve_loaders.keys():
        bm = baselines["preserve"][preserve_name]
        row["preserve"][preserve_name] = {
            "accuracy": bm["accuracy"],
            "correct": bm["correct"],
            "total": bm["total"],
            "zero_shot_baseline": bm["accuracy"],
            "delta_from_zero_shot": 0.0,
        }
    row["train"] = {
        "train_loss": None,
        "train_acc": None,
        "lr": optimizer.param_groups[0]["lr"],
        "alpha_visual": visual_adapter.alpha,
    }
    row["adapter_init"] = {
        "epoch_0_meaning": "original CLIP zero-shot baseline before CLIP-Adapter adaptation",
        "note": "epoch 0은 adapter 적용 전 원본 CLIP zero-shot 결과임",
    }
    return row


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
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

    parser.add_argument("--adapter-hidden-ratio", type=float, default=0.25)
    parser.add_argument("--adapter-alpha-init", type=float, default=0.6)
    parser.add_argument("--learnable-alpha", action="store_true")
    parser.add_argument("--adapter-seed", type=int, default=777)

    parser.add_argument("--selection-method", type=str, default="teacher_confident",
                        choices=["random", "teacher_confident", "teacher_margin",
                                 "prototype", "teacher_uncertain"])
    parser.add_argument("--selection-candidate-factor", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=-1)
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()
    if args.sample_seed is None:
        args.sample_seed = args.seed

    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    save_json(vars(args), output_dir / "args.json")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda 요청했지만 torch.cuda.is_available()이 False임")

    device = torch.device(args.device)
    print("===== 실행 정보 =====")
    print(f"device: {device}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    if device.type == "cuda":
        print(f"visible cuda device count: {torch.cuda.device_count()}")
        print(f"visible cuda device 0: {torch.cuda.get_device_name(0)}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device,
    )
    model.float()
    tokenizer = open_clip.get_tokenizer(args.model)
    freeze_clip_model(model)
    model.eval()

    feature_dim = infer_visual_output_dim(model)
    print(f"CLIP feature_dim: {feature_dim}")

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

    print("===== target 정보 =====")
    print(f"target_name: {args.target_name}, split_mode: {split_mode}")
    print(f"target classes: {len(target_classnames)}, "
          f"train pool: {len(target_train_pool)}, test: {len(target_test_loader.dataset)}")

    preserve_loaders: Dict = {}
    preserve_classnames: Dict = {}
    preserve_info: Dict = {}

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
        preserve_info[name] = {"path": path, "num_classes": len(classnames),
                                "num_images": len(loader.dataset)}

    print("===== preserve 정보 =====")
    for name, info in preserve_info.items():
        print(name, info)

    print("===== raw text feature 계산 =====")
    raw_target_text = build_zero_shot_text_features(
        model=model, tokenizer=tokenizer, classnames=target_classnames,
        templates=get_templates(args.target_name), device=device,
    )
    raw_preserve_text: Dict[str, torch.Tensor] = {}
    for preserve_name in preserve_loaders.keys():
        raw_preserve_text[preserve_name] = build_zero_shot_text_features(
            model=model, tokenizer=tokenizer,
            classnames=preserve_classnames[preserve_name],
            templates=get_templates(preserve_name), device=device,
        )

    print("===== 원본 CLIP zero-shot baseline 측정 =====")
    baselines: Dict = {"target": {}, "preserve": {}}
    baselines["target"] = evaluate_zero_shot(
        model=model, loader=target_test_loader, text_features=raw_target_text,
        device=device, desc=f"zero-shot target={args.target_name}",
    )
    for preserve_name, loader in preserve_loaders.items():
        baselines["preserve"][preserve_name] = evaluate_zero_shot(
            model=model, loader=loader, text_features=raw_preserve_text[preserve_name],
            device=device, desc=f"zero-shot preserve={preserve_name}",
        )

    save_json(baselines, output_dir / "zero_shot_baselines.json")
    curves_path = output_dir / "curves.jsonl"
    curves_path.unlink(missing_ok=True)

    print("===== zero-shot baseline 요약 =====")
    print(f"target {args.target_name}: {baselines['target']['accuracy'] * 100:.2f}%")
    for preserve_name, m in baselines["preserve"].items():
        print(f"preserve {preserve_name}: {m['accuracy'] * 100:.2f}%")
        if "imagenet" in preserve_name.lower() and m["accuracy"] < 0.30:
            print(f"[경고] ImageNet zero-shot baseline이 너무 낮음 ({m['accuracy'] * 100:.2f}%). "
                  f"class name 매핑 확인 필요.")

    # ----- ZSCL teacher score 사전계산 (random이 아닐 때만) -----
    teacher_data = None
    if args.selection_method.lower() != "random":
        print(f"===== ZSCL teacher score 계산 (method={args.selection_method}) =====")
        teacher_data = _compute_teacher_scores(
            dataset=target_train_pool,
            classnames=target_classnames,
            model=model,
            tokenizer=tokenizer,
            templates=get_templates(args.target_name),
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            desc="compute teacher scores",
        )
        save_json(
            {
                "selection_method": args.selection_method,
                "selection_candidate_factor": args.selection_candidate_factor,
                "num_images": int(teacher_data["labels"].shape[0]),
                "mean_confidence": float(teacher_data["confidence"].mean().item()),
                "mean_margin": float(teacher_data["margin"].mean().item()),
                "note": "Teacher CLIP zero-shot 결과 기반 support sample 선택.",
            },
            output_dir / "selection_meta.json",
        )

    # ----- image feature 캐시 (raw) -----
    target_cached = None
    preserve_cached = None
    if args.cache_features:
        print("===== image feature 캐시 생성 (raw, unnormalized) =====")
        target_cached = _precompute_raw_image_features(
            model, target_test_loader, device, desc=f"cache target={args.target_name}",
        )
        preserve_cached = {}
        for preserve_name, loader in preserve_loaders.items():
            preserve_cached[preserve_name] = _precompute_raw_image_features(
                model, loader, device, desc=f"cache preserve={preserve_name}",
            )

    # ----- 공통 adapter 초기값 -----
    print("===== 공통 visual adapter 초기값 생성 =====")
    print(f"adapter_seed: {args.adapter_seed}, alpha_init: {args.adapter_alpha_init}, "
          f"learnable_alpha: {args.learnable_alpha}")

    torch.manual_seed(args.adapter_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.adapter_seed)

    base_visual = CLIPAdapter(
        feature_dim=feature_dim,
        hidden_ratio=args.adapter_hidden_ratio,
        alpha_init=args.adapter_alpha_init,
        alpha_learnable=args.learnable_alpha,
    ).to(device)
    base_visual_state = {k: v.detach().cpu().clone() for k, v in base_visual.state_dict().items()}

    adapter_init_meta = {
        "trainer": "CLIPAdapterVisualOnly",
        "adapter_seed": args.adapter_seed,
        "feature_dim": feature_dim,
        "hidden_ratio": args.adapter_hidden_ratio,
        "alpha_init": args.adapter_alpha_init,
        "alpha_learnable": args.learnable_alpha,
        "branches": ["visual"],
        "text_adapter": False,
        "same_initial_adapter_across_shots": True,
        "epoch_0_meaning": "original CLIP zero-shot baseline before CLIP-Adapter adaptation",
        "epoch_1_and_after": "visual-only CLIP-Adapter adapted result",
    }
    save_json(adapter_init_meta, output_dir / "adapter_initialization.json")
    torch.save(
        {"base_visual_state_dict": base_visual_state, "adapter_seed": args.adapter_seed,
         "trainer": "CLIPAdapterVisualOnly"},
        output_dir / "adapter_initialization.pt",
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    all_results = {
        "args": vars(args),
        "target_name": args.target_name,
        "split_mode": split_mode,
        "preserve_info": preserve_info,
        "zero_shot_baselines": baselines,
        "adapter_init": adapter_init_meta,
        "trainer": "CLIPAdapterVisualOnly",
        "selection_method": args.selection_method,
        "selection_candidate_factor": args.selection_candidate_factor,
        "shots": [],
    }

    # ----- shot별 학습 루프 -----
    for shot in args.shots_list:
        print(f"===== sample_seed={args.sample_seed} shot={shot} visual-only CLIP-Adapter 학습 시작 =====")
        shot_dir = ensure_dir(output_dir / f"shot_{shot}")

        fewshot_subset = build_fewshot_subset_zscl(
            target_train_pool, shot, args.sample_seed,
            method=args.selection_method,
            teacher_data=teacher_data,
            candidate_factor=args.selection_candidate_factor,
            max_shot_in_experiment=max(args.shots_list),
        )
        fewshot_loader = DataLoader(
            fewshot_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        visual_adapter = CLIPAdapter(
            feature_dim=feature_dim,
            hidden_ratio=args.adapter_hidden_ratio,
            alpha_init=args.adapter_alpha_init,
            alpha_learnable=args.learnable_alpha,
        ).to(device)
        visual_adapter.load_state_dict(base_visual_state, strict=True)

        optimizer = torch.optim.SGD(
            visual_adapter.parameters(), lr=args.lr,
            momentum=0.9, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs),
        )

        shot_history = []
        row = make_zero_shot_epoch0_row(
            target_name=args.target_name,
            preserve_loaders=preserve_loaders,
            baselines=baselines, shot=shot,
            optimizer=optimizer, visual_adapter=visual_adapter,
        )
        row["sample_seed"] = args.sample_seed
        row["split_seed"] = args.seed
        row["adapter_seed"] = args.adapter_seed
        row["trainer"] = "CLIPAdapterVisualOnly"
        row["selection_method"] = args.selection_method
        row["selection_candidate_factor"] = args.selection_candidate_factor
        shot_history.append(row)
        append_jsonl(row, curves_path)

        for epoch in range(1, args.epochs + 1):
            train_metric = train_one_epoch_adapter(
                model=model, visual_adapter=visual_adapter,
                train_loader=fewshot_loader,
                raw_text_features=raw_target_text,
                optimizer=optimizer, device=device,
                epoch=epoch, total_epochs=args.epochs,
            )
            scheduler.step()

            do_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
            if do_eval:
                row = evaluate_current_adapter(
                    model=model, visual_adapter=visual_adapter,
                    target_name=args.target_name,
                    target_loader=target_test_loader,
                    raw_target_text=raw_target_text,
                    preserve_loaders=preserve_loaders,
                    raw_preserve_text=raw_preserve_text,
                    baselines=baselines, device=device,
                    shot=shot, epoch=epoch,
                    target_cached=target_cached,
                    preserve_cached=preserve_cached,
                )
                row["sample_seed"] = args.sample_seed
                row["split_seed"] = args.seed
                row["adapter_seed"] = args.adapter_seed
                row["trainer"] = "CLIPAdapterVisualOnly"
                row["selection_method"] = args.selection_method
                row["selection_candidate_factor"] = args.selection_candidate_factor
                row["train"] = {
                    "train_loss": train_metric["train_loss"],
                    "train_acc": train_metric["train_acc"],
                    "lr": optimizer.param_groups[0]["lr"],
                    "alpha_visual": train_metric["alpha_visual"],
                }
                row["adapter_init"] = adapter_init_meta
                shot_history.append(row)
                append_jsonl(row, curves_path)
                save_json(shot_history, shot_dir / "history.json")

        torch.save(
            {
                "trainer": "CLIPAdapterVisualOnly",
                "sample_seed": args.sample_seed,
                "split_seed": args.seed,
                "shot": shot,
                "visual_adapter_state_dict": {
                    k: v.detach().cpu() for k, v in visual_adapter.state_dict().items()
                },
                "feature_dim": feature_dim,
                "adapter_hidden_ratio": args.adapter_hidden_ratio,
                "alpha_init": args.adapter_alpha_init,
                "alpha_learnable": args.learnable_alpha,
                "alpha_visual_final": visual_adapter.alpha,
                "history": shot_history,
                "args": vars(args),
                "adapter_seed": args.adapter_seed,
                "base_visual_state_at_init": base_visual_state,
            },
            shot_dir / "adapter_final.pt",
        )

        all_results["shots"].append(
            {
                "trainer": "CLIPAdapterVisualOnly",
                "sample_seed": args.sample_seed,
                "split_seed": args.seed,
                "shot": shot,
                "fewshot_train_images": len(fewshot_subset),
                "selection_method": args.selection_method,
                "selection_candidate_factor": args.selection_candidate_factor,
                "final": shot_history[-1],
                "history_path": str(shot_dir / "history.json"),
                "checkpoint_path": str(shot_dir / "adapter_final.pt"),
            }
        )
        save_json(all_results, output_dir / "results.json")

    print("===== visual-only CLIP-Adapter 실험 종료 =====")
    print(f"output_dir: {output_dir}")
    print(f"results: {output_dir / 'results.json'}")
    print(f"curves: {curves_path}")


if __name__ == "__main__":
    main()