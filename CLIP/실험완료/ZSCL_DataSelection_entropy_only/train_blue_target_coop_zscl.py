# -*- coding: utf-8 -*-

"""
Blue target CoOp adaptation 실험 실행 파일

목표:
- CLIP이 zero-shot에서 약한 target dataset에 CoOp few-shot adaptation 수행
- target accuracy 변화 측정
- ImageNet/CIFAR100/SUN 등 preserve dataset의 zero-shot-style accuracy 변화 측정
- 그래프용 json/jsonl 저장
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
    CoOpPromptLearner,
    build_coop_text_features,
    build_zero_shot_text_features,
    freeze_clip_model,
)
from sample_selection import (
    _compute_teacher_scores,
    build_fewshot_subset_zscl,
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


def train_one_epoch(model, prompt_learner, train_loader, classnames, optimizer, device, epoch, total_epochs) -> Dict[str, float]:
    """CoOp prompt context를 1 epoch 학습함"""
    model.eval()
    prompt_learner.train()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{total_epochs}")

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

        text_features = build_coop_text_features(model, prompt_learner, classnames, device)
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


def evaluate_current_prompt(
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
    target_cached=None,
    preserve_cached=None,
) -> Dict:
    """현재 prompt 상태로 target과 preserve dataset을 모두 평가함"""
    row = {"target_name": target_name, "shot": shot, "epoch": epoch, "target": {}, "preserve": {}}

    target_text = build_coop_text_features(model, prompt_learner, target_classnames, device)
    if target_cached is not None:
        target_metric = evaluate_classifier(
            model=model,
            loader=None,
            text_features=target_text,
            device=device,
            desc=f"eval target={target_name} shot={shot} epoch={epoch}",
            cached_features=target_cached[0],
            cached_targets=target_cached[1],
        )
    else:
        target_metric = evaluate_classifier(
            model=model,
            loader=target_loader,
            text_features=target_text,
            device=device,
            desc=f"eval target={target_name} shot={shot} epoch={epoch}",
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
        preserve_text = build_coop_text_features(model, prompt_learner, preserve_classnames[preserve_name], device)
        if preserve_cached is not None and preserve_name in preserve_cached:
            metric = evaluate_classifier(
                model=model,
                loader=None,
                text_features=preserve_text,
                device=device,
                desc=f"eval preserve={preserve_name} shot={shot} epoch={epoch}",
                cached_features=preserve_cached[preserve_name][0],
                cached_targets=preserve_cached[preserve_name][1],
            )
        else:
            metric = evaluate_classifier(
                model=model,
                loader=loader,
                text_features=preserve_text,
                device=device,
                desc=f"eval preserve={preserve_name} shot={shot} epoch={epoch}",
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

def make_zero_shot_epoch0_row(
    target_name,
    preserve_loaders,
    baselines,
    shot,
    optimizer,
):
    """
    epoch 0을 original CLIP zero-shot baseline으로 기록함.

    의미:
    - epoch 0 = adaptation 전 original CLIP zero-shot
    - epoch 1 이후 = CoOp prompt 학습 후 성능
    """
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
        "epoch_0_meaning": "original CLIP zero-shot baseline before CoOp adaptation",
        "note": "epoch 0 is copied from zero_shot_baselines.json, not evaluated with CoOp initialized prompt",
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-seed", type=int, default=777,
                        help="모든 shot에서 동일한 CoOp prompt 초기값을 만들기 위한 seed (--seed와 분리)")
    parser.add_argument(
    "--sample-seed",
    type=int,
    default=None,
    help="few-shot sample 선택에만 사용하는 seed. None이면 --seed를 사용함",
    )
    parser.add_argument("--max-eval-samples", type=int, default=-1)
    parser.add_argument("--cache-features", action="store_true",
                        help="image encoder가 freeze이므로 image feature를 한 번만 계산해서 재사용함")
    parser.add_argument("--output-dir", type=str, required=True)

    # ----- entropy-only data selection 옵션 -----
    parser.add_argument(
        "--selection-method",
        type=str,
        default="teacher_entropy",
        choices=["teacher_entropy"],
        help=(
            "support sample 선택 방법. 이 코드팩은 entropy-only 버전이므로 "
            "teacher_entropy만 지원함."
        ),
    )
    parser.add_argument(
        "--selection-candidate-factor",
        type=int,
        default=2,
        help=(
            "클래스당 score top-K 후보 풀 크기 = max_shot * factor. "
            "작을수록 selection이 deterministic하고 seed간 다양성 줄어듦."
        ),
    )
    args = parser.parse_args()

    # sample_seed는 few-shot subset 선택에만 사용함.
    # args.seed는 train/test split, DataLoader shuffle, 기타 random state 고정용으로 유지함.
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
    if curves_path.exists():
        print(f"[경고] 기존 curves.jsonl 삭제: {curves_path}")
        curves_path.unlink()

    print("===== zero-shot baseline 요약 =====")
    print(f"target {args.target_name}: {baselines['target']['accuracy']*100:.2f}%")
    for preserve_name, m in baselines["preserve"].items():
        print(f"preserve {preserve_name}: {m['accuracy']*100:.2f}%")
        if "imagenet" in preserve_name.lower() and m["accuracy"] < 0.30:
            print(f"[경고] ImageNet zero-shot baseline이 너무 낮음 ({m['accuracy']*100:.2f}%). class name 매핑이 잘못되었을 가능성 있음.")
            print(f"[경고] 예상 범위: ViT-B-32 OpenAI CLIP은 ~58~63% 부근.")

    # image encoder가 freeze 상태이므로 평가 시 image feature를 캐싱해두면
    # epoch별 평가가 image encoder 통과를 안 하므로 매우 빨라짐.
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

    # ===== entropy-only sample selection: teacher entropy 사전 계산 =====
    # 모든 shot에서 동일한 teacher_data를 사용하므로 한 번만 계산하면 됨.
    teacher_data = None
    print(f"===== Teacher CLIP entropy 계산 (selection_method={args.selection_method}) =====")
    teacher_data = _compute_teacher_scores(
        dataset=target_train_pool,
        classnames=target_classnames,
        model=model,
        tokenizer=tokenizer,
        templates=get_templates(args.target_name),
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        desc=f"teacher entropy on {args.target_name} train pool",
    )
    save_json(
        {
            "selection_method": args.selection_method,
            "candidate_factor": args.selection_candidate_factor,
            "n_pool_images": int(teacher_data["labels"].numel()),
            "mean_entropy": float(teacher_data["entropy"].mean().item()),
            "mean_normalized_entropy": float(teacher_data["normalized_entropy"].mean().item()),
            "mean_gt_confidence": float(teacher_data["confidence"].mean().item()),
            "note": (
                "Teacher CLIP zero-shot softmax distribution의 entropy가 높은 target support sample을 선택함. "
                "기존 teacher_uncertain=-confidence 방식이 아니라 entropy=-sum p log p 기준임."
            ),
        },
        output_dir / "selection_meta.json",
    )

    # ===== 모든 shot 실험의 공통 CoOp prompt 초기값 생성 =====
    # shot=1, 2, 4, 8, 16에서 prompt 초기 ctx가 서로 다르면
    # epoch 0 결과가 shot마다 달라져 "shot 차이"와 "초기값 차이"가 섞이게 됨.
    # 따라서 prompt_seed를 별도로 두고, 모든 shot이 같은 base_ctx에서 출발하게 함.
    print("===== 공통 CoOp prompt 초기값 생성 =====")
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
            "note": "모든 shot 실험은 이 base_ctx에서 시작함",
        },
        output_dir / "prompt_initialization.json",
    )
    # base_ctx 자체도 .pt로 저장해 두면 추후 검증/재현이 쉬움
    torch.save({"base_ctx": base_ctx.cpu(), "prompt_seed": args.prompt_seed, "n_ctx": args.n_ctx},
               output_dir / "prompt_initialization.pt")
    # prompt_seed로 RNG가 advance된 상태이므로, 이후 random 연산(DataLoader shuffle 등)이
    # 기존 args.seed 기반 결정성을 유지하도록 state를 args.seed로 복원함.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompt_init_meta = {
    "prompt_seed": args.prompt_seed,
    "same_initial_context_across_shots": True,
    "epoch_0_meaning": "original CLIP zero-shot baseline before CoOp adaptation",
    "epoch_1_and_after": "CoOp prompt after target few-shot adaptation",
    }

    all_results = {
        "args": vars(args),
        "target_name": args.target_name,
        "split_mode": split_mode,
        "preserve_info": preserve_info,
        "zero_shot_baselines": baselines,
        "prompt_init": prompt_init_meta,
        "shots": [],
    }

    for shot in args.shots_list:
        print(f"===== shot={shot} CoOp 학습 시작 =====")
        shot_dir = ensure_dir(output_dir / f"shot_{shot}")
        fewshot_subset = build_fewshot_subset_zscl(
            target_train_pool,
            shot,
            args.sample_seed,
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

        prompt_learner = CoOpPromptLearner(model, tokenizer, n_ctx=args.n_ctx, init_std=0.02).to(device)
        # 모든 shot이 동일한 base_ctx에서 시작하도록 강제 복사.
        # 이 복사를 빼면 shot별 epoch 0 결과가 서로 다른 random init 때문에 달라짐.
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
        row["sample_seed"] = args.sample_seed
        row["split_seed"] = args.seed
        row["prompt_seed"] = args.prompt_seed

        shot_history.append(row)
        append_jsonl(row, curves_path)


        for epoch in range(1, args.epochs + 1):
            train_metric = train_one_epoch(
                model, prompt_learner, fewshot_loader, target_classnames, optimizer, device, epoch, args.epochs
            )
            scheduler.step()

            do_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
            if do_eval:
                row = evaluate_current_prompt(
                    model, prompt_learner, args.target_name, target_test_loader, target_classnames,
                    preserve_loaders, preserve_classnames, baselines, device, shot, epoch,
                    target_cached=target_cached, preserve_cached=preserve_cached,
                )
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
            {   "sample_seed": args.sample_seed,
                "split_seed": args.seed,    
                "shot": shot,
                "ctx": prompt_learner.ctx.detach().cpu(),
                "n_ctx": args.n_ctx,
                "target_classnames": target_classnames,
                "history": shot_history,
                "args": vars(args),
                "prompt_seed": args.prompt_seed,
                "same_initial_context_across_shots": True,
                "base_ctx_at_init": base_ctx.cpu(),
            },
            shot_dir / "coop_prompt_final.pt",
        )

        all_results["shots"].append(
            {   
                "sample_seed": args.sample_seed,
                "shot": shot,
                "fewshot_train_images": len(fewshot_subset),
                "final": shot_history[-1],
                "history_path": str(shot_dir / "history.json"),
                "checkpoint_path": str(shot_dir / "coop_prompt_final.pt"),
            }
        )
        save_json(all_results, output_dir / "results.json")

    print("===== 실험 종료 =====")
    print(f"output_dir: {output_dir}")
    print(f"results: {output_dir / 'results.json'}")
    print(f"curves: {output_dir / 'curves.jsonl'}")

if __name__ == "__main__":
    main()
