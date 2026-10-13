# -*- coding: utf-8 -*-
"""
Blue target CoCoOp adaptation 실험 실행 파일

목표:
- 기존 CoOp 실험과 동일한 데이터 로딩 / few-shot sampling / 결과 저장 포맷 사용
- 모델만 CoCoOp으로 변경
- target accuracy 변화 측정
- ImageNet/CIFAR100 등 preserve dataset의 zero-shot-style accuracy 변화 측정

주의:
- CoCoOp은 image-conditioned prompt를 사용하므로 이미지마다 text classifier가 달라짐.
- 따라서 CoOp처럼 static text feature를 한 번 만든 뒤 재사용할 수 없음.
- ImageNet preserve evaluation은 CoOp보다 훨씬 느림.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
sys.path.append(str(SRC_DIR))

from coop_model import (
    CoCoOpPromptLearner,
    build_cocoop_logits,
    build_zero_shot_text_features,
    freeze_clip_model,
)
from data import (
    build_fewshot_subset,
    build_loader,
    build_preserve_dataset,
    build_target_datasets,
    maybe_replace_imagenet_wnid_with_torchvision_classes,
)
from eval import evaluate_classifier, precompute_image_features
from utils import append_jsonl, ensure_dir, load_classnames_file, save_json, seed_everything


def parse_dataset_spec(spec: str) -> Tuple[str, str, Optional[str]]:
    """dataset spec을 파싱함. 형식: name:path 또는 name:path:classnames_txt"""
    parts = spec.split(":", 2)
    if len(parts) == 2:
        name, path = parts
        return name, path, None
    if len(parts) == 3:
        name, path, classnames_file = parts
        return name, path, classnames_file
    raise ValueError(f"dataset spec 형식이 잘못됨: {spec}")


def get_templates(dataset_name: str) -> List[str]:
    """dataset별 hand-crafted zero-shot prompt template"""
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


def train_one_epoch_cocoop(
    model,
    prompt_learner,
    train_loader,
    classnames,
    optimizer,
    device,
    epoch,
    total_epochs,
    class_chunk_size: int,
) -> Dict[str, float]:
    """CoCoOp prompt context + meta-net을 1 epoch 학습함"""
    model.eval()
    prompt_learner.train()
    # NOTE: clip_model은 object.__setattr__로 서브모듈 등록 없이 저장되어 있으므로
    # prompt_learner.train()이 CLIP을 training mode로 바꾸지 않음. (coop_model.py 참조)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    pbar = tqdm(train_loader, desc=f"train CoCoOp epoch {epoch}/{total_epochs}")

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

        logits = build_cocoop_logits(
            model=model,
            prompt_learner=prompt_learner,
            image_features=image_features,
            classnames=classnames,
            device=device,
            class_chunk_size=class_chunk_size,
        )
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


@torch.no_grad()
def evaluate_cocoop_classifier(
    model,
    prompt_learner,
    loader,
    classnames,
    device,
    desc: str,
    class_chunk_size: int,
    eval_batch_size: int,
    cached_features=None,
    cached_targets=None,
) -> Dict[str, float]:
    """CoCoOp classifier 평가.

    CoCoOp은 이미지마다 prompt가 달라지므로 static text_features를 받지 않음.
    cached_features가 있으면 image encoder를 다시 돌리지 않고 image feature만 사용함.
    """
    model.eval()
    was_training = prompt_learner.training
    prompt_learner.eval()

    correct = 0
    total = 0
    eval_batch_size = max(1, int(eval_batch_size))

    if cached_features is not None and cached_targets is not None:
        feats = cached_features.to(device, non_blocking=True)
        tgts = cached_targets.to(device, non_blocking=True)

        iterator = range(0, feats.shape[0], eval_batch_size)
        for start in tqdm(iterator, desc=desc):
            end = min(start + eval_batch_size, feats.shape[0])
            image_features = feats[start:end]
            targets = tgts[start:end]

            logits = build_cocoop_logits(
                model=model,
                prompt_learner=prompt_learner,
                image_features=image_features,
                classnames=classnames,
                device=device,
                class_chunk_size=class_chunk_size,
            )
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    else:
        for images, targets in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            for start in range(0, image_features.shape[0], eval_batch_size):
                end = min(start + eval_batch_size, image_features.shape[0])
                logits = build_cocoop_logits(
                    model=model,
                    prompt_learner=prompt_learner,
                    image_features=image_features[start:end],
                    classnames=classnames,
                    device=device,
                    class_chunk_size=class_chunk_size,
                )
                preds = logits.argmax(dim=1)
                correct += (preds == targets[start:end]).sum().item()
                total += end - start

    if was_training:
        prompt_learner.train()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "accuracy": float(correct / total if total > 0 else 0.0),
        "correct": int(correct),
        "total": int(total),
    }


def evaluate_current_prompt_cocoop(
    model,
    prompt_learner,
    target_name,
    target_loader,
    target_classnames,
    preserve_loaders,
    preserve_classnames,
    baselines,
    device,
    shot,
    epoch,
    class_chunk_size: int,
    eval_batch_size: int,
    target_cached=None,
    preserve_cached=None,
) -> Dict:
    """현재 CoCoOp prompt 상태로 target과 preserve dataset을 모두 평가함"""
    row = {"target_name": target_name, "shot": shot, "epoch": epoch, "target": {}, "preserve": {}}

    target_metric = evaluate_cocoop_classifier(
        model=model,
        prompt_learner=prompt_learner,
        loader=target_loader,
        classnames=target_classnames,
        device=device,
        desc=f"eval CoCoOp target={target_name} shot={shot} epoch={epoch}",
        class_chunk_size=class_chunk_size,
        eval_batch_size=eval_batch_size,
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
        metric = evaluate_cocoop_classifier(
            model=model,
            prompt_learner=prompt_learner,
            loader=loader,
            classnames=preserve_classnames[preserve_name],
            device=device,
            desc=f"eval CoCoOp preserve={preserve_name} shot={shot} epoch={epoch}",
            class_chunk_size=class_chunk_size,
            eval_batch_size=eval_batch_size,
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


def make_zero_shot_epoch0_row(target_name, preserve_loaders, baselines, shot, optimizer) -> Dict:
    """epoch 0을 original CLIP zero-shot baseline으로 기록함"""
    row = {
        "target_name": target_name,
        "shot": shot,
        "epoch": 0,
        "target": {},
        "preserve": {},
    }

    target_base = baselines["target"]["accuracy"]
    row["target"] = {
        "accuracy": target_base,
        "correct": baselines["target"]["correct"],
        "total": baselines["target"]["total"],
        "zero_shot_baseline": target_base,
        "delta_from_zero_shot": 0.0,
    }

    for preserve_name in preserve_loaders.keys():
        base_metric = baselines["preserve"][preserve_name]
        base_acc = base_metric["accuracy"]
        row["preserve"][preserve_name] = {
            "accuracy": base_acc,
            "correct": base_metric["correct"],
            "total": base_metric["total"],
            "zero_shot_baseline": base_acc,
            "delta_from_zero_shot": 0.0,
        }

    row["train"] = {
        "train_loss": None,
        "train_acc": None,
        "lr": optimizer.param_groups[0]["lr"],
    }

    row["prompt_init"] = {
        "epoch_0_meaning": "original CLIP zero-shot baseline before CoCoOp adaptation",
        "note": "epoch 0 is copied from zero_shot_baselines.json, not evaluated with CoCoOp initialized prompt",
    }

    return row


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
    parser.add_argument("--n-ctx", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-seed",
        type=int,
        default=777,
        help="모든 shot에서 동일한 CoCoOp prompt/meta-net 초기값을 만들기 위한 seed (--seed와 분리)",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="few-shot sample 선택에만 사용하는 seed. None이면 --seed를 사용함",
    )
    parser.add_argument("--max-eval-samples", type=int, default=-1)
    parser.add_argument("--cache-features", action="store_true", help="image encoder feature를 캐시함")
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument(
        "--cocoop-class-chunk-size",
        type=int,
        default=128,
        help="CoCoOp text encoder에 한 번에 넣을 class 수. ImageNet OOM 방지용",
    )
    parser.add_argument(
        "--cocoop-eval-batch-size",
        type=int,
        default=1,
        help="CoCoOp evaluation 시 한 번에 처리할 image feature 수. 클수록 빠르지만 메모리 증가",
    )
    parser.add_argument(
        "--cocoop-meta-hidden-divisor",
        type=int,
        default=16,
        help="CoCoOp meta-net hidden dim = visual_dim // divisor",
    )

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

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    model.float()
    tokenizer = open_clip.get_tokenizer(args.model)
    freeze_clip_model(model)
    model.eval()

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

    print("===== preserve 정보 =====")
    for name, info in preserve_info.items():
        print(name, info)

    print("===== 원본 CLIP zero-shot baseline 측정 =====")
    baselines = {"target": {}, "preserve": {}}

    target_zero_text = build_zero_shot_text_features(
        model=model,
        tokenizer=tokenizer,
        classnames=target_classnames,
        templates=get_templates(args.target_name),
        device=device,
    )
    target_zero = evaluate_classifier(
        model=model,
        loader=target_test_loader,
        text_features=target_zero_text,
        device=device,
        desc=f"zero-shot target={args.target_name}",
    )
    baselines["target"] = target_zero

    for preserve_name, loader in preserve_loaders.items():
        preserve_zero_text = build_zero_shot_text_features(
            model=model,
            tokenizer=tokenizer,
            classnames=preserve_classnames[preserve_name],
            templates=get_templates(preserve_name),
            device=device,
        )
        metric = evaluate_classifier(
            model=model,
            loader=loader,
            text_features=preserve_zero_text,
            device=device,
            desc=f"zero-shot preserve={preserve_name}",
        )
        baselines["preserve"][preserve_name] = metric

    save_json(baselines, output_dir / "zero_shot_baselines.json")
    curves_path = output_dir / "curves.jsonl"
    curves_path.unlink(missing_ok=True)

    print("===== zero-shot baseline 요약 =====")
    print(f"target {args.target_name}: {baselines['target']['accuracy']*100:.2f}%")
    for preserve_name, m in baselines["preserve"].items():
        print(f"preserve {preserve_name}: {m['accuracy']*100:.2f}%")
        if "imagenet" in preserve_name.lower() and m["accuracy"] < 0.30:
            print(f"[경고] ImageNet zero-shot baseline이 너무 낮음 ({m['accuracy']*100:.2f}%). class name 매핑이 잘못되었을 가능성 있음.")
            print("[경고] 예상 범위: ViT-B-32 OpenAI CLIP은 ~58~63% 부근.")

    target_cached = None
    preserve_cached = None
    if args.cache_features:
        print("===== image feature 캐시 생성 =====")
        target_cached = precompute_image_features(
            model, target_test_loader, device, desc=f"cache target={args.target_name}"
        )
        preserve_cached = {}
        for preserve_name, loader in preserve_loaders.items():
            preserve_cached[preserve_name] = precompute_image_features(
                model, loader, device, desc=f"cache preserve={preserve_name}"
            )

    print("===== 공통 CoCoOp prompt/meta-net 초기값 생성 =====")
    print(f"prompt_seed: {args.prompt_seed}")
    torch.manual_seed(args.prompt_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.prompt_seed)

    base_prompt_learner = CoCoOpPromptLearner(
        clip_model=model,
        tokenizer=tokenizer,
        n_ctx=args.n_ctx,
        init_std=0.02,
        meta_hidden_divisor=args.cocoop_meta_hidden_divisor,
    ).to(device)
    base_state = {k: v.detach().cpu().clone() for k, v in base_prompt_learner.state_dict().items()}

    save_json(
        {
            "prompt_seed": args.prompt_seed,
            "n_ctx": args.n_ctx,
            "ctx_shape": list(base_prompt_learner.ctx.shape),
            "init_std": 0.02,
            "meta_hidden_divisor": args.cocoop_meta_hidden_divisor,
            "note": "모든 shot 실험은 이 base CoCoOp prompt/meta-net state에서 시작함",
        },
        output_dir / "prompt_initialization.json",
    )
    torch.save(
        {
            "base_state_dict": base_state,
            "prompt_seed": args.prompt_seed,
            "n_ctx": args.n_ctx,
            "trainer": "CoCoOp",
        },
        output_dir / "prompt_initialization.pt",
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompt_init_meta = {
        "trainer": "CoCoOp",
        "prompt_seed": args.prompt_seed,
        "same_initial_context_and_meta_net_across_shots": True,
        "epoch_0_meaning": "original CLIP zero-shot baseline before CoCoOp adaptation",
        "epoch_1_and_after": "CoCoOp prompt/meta-net after target few-shot adaptation",
    }

    all_results = {
        "args": vars(args),
        "target_name": args.target_name,
        "split_mode": split_mode,
        "preserve_info": preserve_info,
        "zero_shot_baselines": baselines,
        "prompt_init": prompt_init_meta,
        "trainer": "CoCoOp",
        "shots": [],
    }

    for shot in args.shots_list:
        print(f"===== sample_seed={args.sample_seed} shot={shot} CoCoOp 학습 시작 =====")
        shot_dir = ensure_dir(output_dir / f"shot_{shot}")
        fewshot_subset = build_fewshot_subset(target_train_pool, shot, args.sample_seed)
        fewshot_loader = DataLoader(
            fewshot_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        prompt_learner = CoCoOpPromptLearner(
            model,
            tokenizer,
            n_ctx=args.n_ctx,
            init_std=0.02,
            meta_hidden_divisor=args.cocoop_meta_hidden_divisor,
        ).to(device)
        prompt_learner.load_state_dict(base_state, strict=True)
        prompt_learner.to(device)

        optimizer = torch.optim.SGD(
            # prompt_learner.parameters()를 그대로 넘기면 안 됨.
            # clip_model이 서브모듈로 등록된 경우 frozen CLIP params까지 포함되어
            # optimizer 메모리 낭비 및 momentum buffer 비대화 발생.
            # requires_grad=True인 파라미터(ctx + meta_net)만 명시적으로 전달함.
            [p for p in prompt_learner.parameters() if p.requires_grad],
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

        shot_history = []
        row = make_zero_shot_epoch0_row(
            target_name=args.target_name,
            preserve_loaders=preserve_loaders,
            baselines=baselines,
            shot=shot,
            optimizer=optimizer,
        )
        row["sample_seed"] = args.sample_seed
        row["split_seed"] = args.seed
        row["prompt_seed"] = args.prompt_seed
        row["trainer"] = "CoCoOp"
        shot_history.append(row)
        append_jsonl(row, curves_path)

        for epoch in range(1, args.epochs + 1):
            train_metric = train_one_epoch_cocoop(
                model=model,
                prompt_learner=prompt_learner,
                train_loader=fewshot_loader,
                classnames=target_classnames,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                total_epochs=args.epochs,
                class_chunk_size=args.cocoop_class_chunk_size,
            )
            scheduler.step()

            do_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
            if do_eval:
                row = evaluate_current_prompt_cocoop(
                    model=model,
                    prompt_learner=prompt_learner,
                    target_name=args.target_name,
                    target_loader=target_test_loader,
                    target_classnames=target_classnames,
                    preserve_loaders=preserve_loaders,
                    preserve_classnames=preserve_classnames,
                    baselines=baselines,
                    device=device,
                    shot=shot,
                    epoch=epoch,
                    class_chunk_size=args.cocoop_class_chunk_size,
                    eval_batch_size=args.cocoop_eval_batch_size,
                    target_cached=target_cached,
                    preserve_cached=preserve_cached,
                )
                row["sample_seed"] = args.sample_seed
                row["split_seed"] = args.seed
                row["prompt_seed"] = args.prompt_seed
                row["trainer"] = "CoCoOp"
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
                "trainer": "CoCoOp",
                "sample_seed": args.sample_seed,
                "split_seed": args.seed,
                "shot": shot,
                "prompt_learner_state_dict": {k: v.detach().cpu() for k, v in prompt_learner.state_dict().items()},
                "n_ctx": args.n_ctx,
                "target_classnames": target_classnames,
                "history": shot_history,
                "args": vars(args),
                "prompt_seed": args.prompt_seed,
                "same_initial_context_and_meta_net_across_shots": True,
                "base_state_at_init": base_state,
            },
            shot_dir / "cocoop_prompt_final.pt",
        )

        all_results["shots"].append(
            {
                "trainer": "CoCoOp",
                "sample_seed": args.sample_seed,
                "split_seed": args.seed,
                "shot": shot,
                "fewshot_train_images": len(fewshot_subset),
                "final": shot_history[-1],
                "history_path": str(shot_dir / "history.json"),
                "checkpoint_path": str(shot_dir / "cocoop_prompt_final.pt"),
            }
        )
        save_json(all_results, output_dir / "results.json")

    print("===== CoCoOp 실험 종료 =====")
    print(f"output_dir: {output_dir}")
    print(f"results: {output_dir / 'results.json'}")
    print(f"curves: {output_dir / 'curves.jsonl'}")


if __name__ == "__main__":
    main()
