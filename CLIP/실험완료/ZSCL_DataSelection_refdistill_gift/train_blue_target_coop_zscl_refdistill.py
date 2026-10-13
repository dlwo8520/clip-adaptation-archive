# -*- coding: utf-8 -*-

"""
Reference dataset 기반 distillation을 추가한 CoOp 학습 스크립트.

기존 train_blue_target_coop_zscl.py 에 다음을 추가함:
- 학습 중 매 step마다 reference dataset에서 sampling한 image/class에 대해
  원본 CLIP teacher와 현재 CoOp student의 similarity 분포를 맞추는
  distillation loss (Zheng et al. ZSCL, CVPR 2023, Eq.(3)(4) 변형).
- 옵션으로 weight consolidation (ZSCL Eq.(5)) loss를 ctx에 대해 적용.

학습 loss:
    L = L_ce(target support set)
      + lambda_distill * (L_lwf_img + L_lwf_txt)   # reference data로 계산
      + lambda_wc * ||ctx - ctx_init||^2           # 옵션 (--lambda-wc > 0일 때)

기존 sample selection (teacher_confident / teacher_margin / teacher_entropy) 은
그대로 사용 가능.
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
    build_loader,
    build_preserve_dataset,
    build_target_datasets,
    maybe_replace_imagenet_wnid_with_torchvision_classes,
)
from eval import evaluate_classifier, precompute_image_features
from utils import append_jsonl, ensure_dir, load_classnames_file, save_json, seed_everything
from reference_distill import CoOpReferenceDistiller, compute_weight_consolidation_loss


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


def get_reference_templates(reference_name: str) -> List[str]:
    """reference dataset 용 zero-shot template. 일반적인 자연 이미지 가정."""
    name = reference_name.lower()
    if "imagenet" in name:
        # CLIP 논문 부록의 ImageNet 80-template subset 중 가벼운 4개 사용
        return [
            "a photo of a {}.",
            "a blurry photo of a {}.",
            "a small photo of a {}.",
            "a photo of the {}.",
        ]
    if "cifar" in name:
        return ["a photo of a {}.", "a low resolution photo of a {}."]
    return ["a photo of a {}."]


def train_one_epoch(
    model,
    prompt_learner,
    train_loader,
    classnames,
    optimizer,
    device,
    epoch,
    total_epochs,
    distiller: Optional[CoOpReferenceDistiller] = None,
    lambda_distill: float = 0.0,
    base_ctx: Optional[torch.Tensor] = None,
    lambda_wc: float = 0.0,
    distill_num_image_samples: int = 64,
    distill_num_class_samples: int = 64,
) -> Dict[str, float]:
    """
    CoOp ctx를 1 epoch 학습함.

    Loss 구성:
        L = L_ce (target support set)
          + lambda_distill * L_distill (reference dataset에 대한 KL distillation)
          + lambda_wc      * L_wc       (ctx의 L2 anchoring on base_ctx)
    """
    model.eval()
    prompt_learner.train()

    total_loss = 0.0
    total_ce = 0.0
    total_dist = 0.0
    total_wc = 0.0
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
        loss_ce = F.cross_entropy(logits, targets)

        loss = loss_ce

        # ----- reference distillation -----
        loss_dist_val = 0.0
        if distiller is not None and lambda_distill > 0.0:
            loss_dist = distiller.compute_distill_loss(
                prompt_learner=prompt_learner,
                build_coop_text_features_fn=build_coop_text_features,
                num_image_samples=distill_num_image_samples,
                num_class_samples=distill_num_class_samples,
            )
            loss = loss + lambda_distill * loss_dist
            loss_dist_val = float(loss_dist.detach().item())

        # ----- weight consolidation -----
        loss_wc_val = 0.0
        if base_ctx is not None and lambda_wc > 0.0:
            loss_wc = compute_weight_consolidation_loss(prompt_learner, base_ctx)
            loss = loss + lambda_wc * loss_wc
            loss_wc_val = float(loss_wc.detach().item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct = (preds == targets).sum().item()
        bs = targets.numel()
        total_loss += loss.item() * bs
        total_ce += loss_ce.item() * bs
        total_dist += loss_dist_val * bs
        total_wc += loss_wc_val * bs
        total_correct += correct
        total_seen += bs

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ce=f"{loss_ce.item():.4f}",
            dist=f"{loss_dist_val:.4f}",
            wc=f"{loss_wc_val:.4f}",
            acc=f"{correct / bs:.4f}",
        )

    return {
        "train_loss": float(total_loss / max(1, total_seen)),
        "train_loss_ce": float(total_ce / max(1, total_seen)),
        "train_loss_distill_raw": float(total_dist / max(1, total_seen)),
        "train_loss_wc_raw": float(total_wc / max(1, total_seen)),
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
    """epoch 0을 original CLIP zero-shot baseline으로 기록함."""
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
        "train_loss_ce": None,
        "train_loss_distill_raw": None,
        "train_loss_wc_raw": None,
        "train_acc": None,
        "lr": optimizer.param_groups[0]["lr"],
    }
    row["prompt_init"] = {
        "epoch_0_meaning": "original CLIP zero-shot baseline before CoOp adaptation",
        "note": "epoch 0 is copied from zero_shot_baselines.json",
    }
    return row


def main():
    parser = argparse.ArgumentParser()

    # ----- target / preserve -----
    parser.add_argument("--target-name", type=str, required=True)
    parser.add_argument("--target-root", type=str, default=None)
    parser.add_argument("--target-train", type=str, default=None)
    parser.add_argument("--target-test", type=str, default=None)
    parser.add_argument("--target-classnames", type=str, default=None)
    parser.add_argument("--split-test-ratio", type=float, default=0.5)
    parser.add_argument("--preserve", type=str, nargs="+", required=True)

    # ----- CLIP backbone -----
    parser.add_argument("--model", type=str, default="ViT-B-32-quickgelu")
    parser.add_argument("--pretrained", type=str, default="openai")

    # ----- CoOp / 학습 옵션 -----
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
                        help="모든 shot이 같은 CoOp ctx 초기값에서 출발하도록 하는 seed")
    parser.add_argument("--sample-seed", type=int, default=None,
                        help="few-shot sample 선택에만 사용. None이면 --seed 사용.")
    parser.add_argument("--max-eval-samples", type=int, default=-1)
    parser.add_argument("--cache-features", action="store_true",
                        help="image encoder freeze이므로 target/preserve image feature를 캐싱")
    parser.add_argument("--output-dir", type=str, required=True)

    # ----- support sample selection (기존 옵션 유지) -----
    parser.add_argument(
        "--selection-method",
        type=str,
        default="teacher_confident",
        choices=["teacher_confident", "teacher_margin", "teacher_entropy"],
        help="support sample 선택 방법. 본 실험에서는 margin/entropy/confident만 사용.",
    )
    parser.add_argument("--selection-candidate-factor", type=int, default=2)

    # ----- reference distillation (신규) -----
    parser.add_argument("--reference-name", type=str, default="imagenet",
                        help="reference dataset 이름 (template 선택에만 사용)")
    parser.add_argument("--reference-root", type=str, default=None,
                        help="reference image dataset (ImageFolder 구조). "
                             "None이면 distillation 비활성. lambda-distill > 0이면 필수.")
    parser.add_argument("--reference-classnames", type=str, default=None,
                        help="reference classnames txt 경로. None이면 ImageFolder의 폴더명 사용.")
    parser.add_argument("--max-reference-images", type=int, default=10000,
                        help="reference image pool 최대 크기. 캐싱 메모리 절약용.")
    parser.add_argument("--reference-precompute-seed", type=int, default=0,
                        help="reference image subsampling용 seed. 모든 sample-seed 간 고정 권장.")
    parser.add_argument("--reference-store-cpu", action="store_true",
                        help="reference image feature를 GPU 대신 CPU에 저장 (메모리 절약)")
    parser.add_argument("--lambda-distill", type=float, default=1.0,
                        help="reference distillation loss 계수. 0이면 distill 비활성.")
    parser.add_argument("--distill-num-image-samples", type=int, default=64,
                        help="매 step에 sampling할 reference image 수")
    parser.add_argument("--distill-num-class-samples", type=int, default=64,
                        help="매 step에 sampling할 reference class 수 "
                             "(student text feature를 매 step 그 만큼 새로 계산함)")

    # ----- weight consolidation (신규) -----
    parser.add_argument("--lambda-wc", type=float, default=0.0,
                        help="ctx의 L2 anchoring 계수. 0이면 WC 비활성.")

    args = parser.parse_args()

    # sample-seed 분리
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

    # ===== CLIP 로딩 및 freeze =====
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device,
    )
    model.float()
    tokenizer = open_clip.get_tokenizer(args.model)
    freeze_clip_model(model)
    model.eval()

    # ===== target dataset =====
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
        target_test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, max_samples=args.max_eval_samples,
    )

    print("===== target 정보 =====")
    print(f"target_name: {args.target_name}, split_mode: {split_mode}")
    print(f"target classes: {len(target_classnames)}, train pool: {len(target_train_pool)}, "
          f"test: {len(target_test_loader.dataset)}")

    # ===== preserve datasets =====
    preserve_loaders, preserve_classnames, preserve_info = {}, {}, {}
    for spec in args.preserve:
        name, path, classnames_file = parse_dataset_spec(spec)
        dataset, classnames = build_preserve_dataset(path, transform=preprocess)
        if classnames_file is not None:
            classnames = load_classnames_file(classnames_file)
        else:
            classnames = maybe_replace_imagenet_wnid_with_torchvision_classes(name, classnames)
        loader = build_loader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=False, max_samples=args.max_eval_samples,
        )
        preserve_loaders[name] = loader
        preserve_classnames[name] = classnames
        preserve_info[name] = {"path": path, "num_classes": len(classnames), "num_images": len(loader.dataset)}

    print("===== preserve 정보 =====")
    for name, info in preserve_info.items():
        print(name, info)

    # ===== original CLIP zero-shot baseline =====
    print("===== 원본 CLIP zero-shot baseline 측정 =====")
    baselines = {"target": {}, "preserve": {}}
    target_zero_text = build_zero_shot_text_features(
        model=model, tokenizer=tokenizer, classnames=target_classnames,
        templates=get_templates(args.target_name), device=device,
    )
    baselines["target"] = evaluate_classifier(
        model=model, loader=target_test_loader, text_features=target_zero_text,
        device=device, desc=f"zero-shot target={args.target_name}",
    )
    for preserve_name, loader in preserve_loaders.items():
        preserve_zero_text = build_zero_shot_text_features(
            model=model, tokenizer=tokenizer, classnames=preserve_classnames[preserve_name],
            templates=get_templates(preserve_name), device=device,
        )
        baselines["preserve"][preserve_name] = evaluate_classifier(
            model=model, loader=loader, text_features=preserve_zero_text,
            device=device, desc=f"zero-shot preserve={preserve_name}",
        )
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
            print(f"[경고] ImageNet zero-shot baseline이 너무 낮음 ({m['accuracy']*100:.2f}%). "
                  f"class name 매핑 점검 필요.")

    # ===== image feature 캐싱 =====
    target_cached, preserve_cached = None, None
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

    # ===== teacher score 사전 계산 (sample selection 용) =====
    print(f"===== Teacher CLIP score 계산 (selection_method={args.selection_method}) =====")
    teacher_data = _compute_teacher_scores(
        dataset=target_train_pool, classnames=target_classnames,
        model=model, tokenizer=tokenizer, templates=get_templates(args.target_name),
        device=device, batch_size=args.batch_size, num_workers=args.num_workers,
        desc=f"teacher score on {args.target_name} train pool",
    )
    save_json(
        {
            "selection_method": args.selection_method,
            "candidate_factor": args.selection_candidate_factor,
            "n_pool_images": int(teacher_data["labels"].numel()),
            "mean_confidence": float(teacher_data["confidence"].mean().item()),
            "mean_margin": float(teacher_data["margin"].mean().item()),
            "mean_entropy": float(teacher_data["entropy"].mean().item()),
        },
        output_dir / "selection_meta.json",
    )

    # ===== reference distiller 준비 =====
    distiller: Optional[CoOpReferenceDistiller] = None
    if args.lambda_distill > 0.0:
        if args.reference_root is None:
            raise ValueError(
                "lambda-distill > 0 인데 --reference-root가 주어지지 않음. "
                "ImageFolder 형식의 reference dataset 경로를 넘기세요."
            )
        print("===== reference dataset 로딩 =====")
        ref_dataset, ref_classnames_folder = build_preserve_dataset(
            args.reference_root, transform=preprocess
        )
        # classnames: 우선순위 = (파일) > (폴더 자동 치환) > (폴더 그대로)
        if args.reference_classnames is not None:
            ref_classnames = load_classnames_file(args.reference_classnames)
            print(f"[reference] classnames 파일 사용: {args.reference_classnames} "
                  f"(n={len(ref_classnames)})")
        else:
            ref_classnames = maybe_replace_imagenet_wnid_with_torchvision_classes(
                args.reference_name, ref_classnames_folder
            )
            print(f"[reference] folder 기반 classnames 사용 (n={len(ref_classnames)})")

        ref_templates = get_reference_templates(args.reference_name)

        # leakage 점검 (preserve와 reference 경로가 같으면 경고)
        for pname, pinfo in preserve_info.items():
            if Path(pinfo["path"]).resolve() == Path(args.reference_root).resolve():
                print(
                    f"[경고] reference-root와 preserve='{pname}' 경로가 동일함 "
                    f"({pinfo['path']}). 동일한 이미지를 distillation과 평가에 모두 쓰면 "
                    f"preserve 성능이 직접적으로 supervised됨 (leakage). "
                    f"가능하면 다른 split/디렉토리 사용 권장."
                )

        distiller = CoOpReferenceDistiller(
            model=model,
            tokenizer=tokenizer,
            reference_dataset=ref_dataset,
            reference_classnames=ref_classnames,
            reference_templates=ref_templates,
            device=device,
            precompute_batch_size=max(64, args.batch_size),
            precompute_num_workers=args.num_workers,
            max_reference_images=args.max_reference_images,
            precompute_seed=args.reference_precompute_seed,
            store_image_features_on_cpu=args.reference_store_cpu,
        )
        save_json(
            {
                "reference_name": args.reference_name,
                "reference_root": args.reference_root,
                "reference_classnames_file": args.reference_classnames,
                "num_classes": len(ref_classnames),
                "num_images_after_subsample": distiller.num_images,
                "max_reference_images": args.max_reference_images,
                "templates": ref_templates,
                "lambda_distill": args.lambda_distill,
                "distill_num_image_samples": args.distill_num_image_samples,
                "distill_num_class_samples": args.distill_num_class_samples,
                "precompute_seed": args.reference_precompute_seed,
                "store_image_features_on_cpu": bool(args.reference_store_cpu),
            },
            output_dir / "reference_distill_meta.json",
        )
    else:
        print("===== distillation 비활성 (lambda_distill = 0) =====")

    # ===== 공통 CoOp prompt 초기값 (모든 shot이 동일한 base_ctx에서 출발) =====
    print(f"===== 공통 CoOp prompt 초기값 (prompt_seed={args.prompt_seed}) =====")
    torch.manual_seed(args.prompt_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.prompt_seed)
    base_prompt_learner = CoOpPromptLearner(
        clip_model=model, tokenizer=tokenizer, n_ctx=args.n_ctx, init_std=0.02,
    ).to(device)
    base_ctx = base_prompt_learner.ctx.detach().clone()
    save_json(
        {
            "prompt_seed": args.prompt_seed,
            "n_ctx": args.n_ctx,
            "ctx_shape": list(base_ctx.shape),
            "init_std": 0.02,
        },
        output_dir / "prompt_initialization.json",
    )
    torch.save(
        {"base_ctx": base_ctx.cpu(), "prompt_seed": args.prompt_seed, "n_ctx": args.n_ctx},
        output_dir / "prompt_initialization.pt",
    )
    # RNG 복원 (DataLoader shuffle 등의 결정성 유지를 위해 args.seed로)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompt_init_meta = {
        "prompt_seed": args.prompt_seed,
        "same_initial_context_across_shots": True,
        "epoch_0_meaning": "original CLIP zero-shot baseline before CoOp adaptation",
        "epoch_1_and_after": "CoOp prompt after target few-shot adaptation",
        "lambda_distill": args.lambda_distill,
        "lambda_wc": args.lambda_wc,
        "uses_reference_distill": distiller is not None,
        "uses_weight_consolidation": (args.lambda_wc > 0.0),
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

    # ===== shot별 학습 루프 =====
    for shot in args.shots_list:
        print(f"===== shot={shot} CoOp 학습 시작 =====")
        shot_dir = ensure_dir(output_dir / f"shot_{shot}")
        fewshot_subset = build_fewshot_subset_zscl(
            target_train_pool, shot, args.sample_seed,
            method=args.selection_method, teacher_data=teacher_data,
            candidate_factor=args.selection_candidate_factor,
            max_shot_in_experiment=max(args.shots_list),
        )
        fewshot_loader = DataLoader(
            fewshot_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        prompt_learner = CoOpPromptLearner(model, tokenizer, n_ctx=args.n_ctx, init_std=0.02).to(device)
        with torch.no_grad():
            prompt_learner.ctx.copy_(base_ctx)
        optimizer = torch.optim.SGD(
            [prompt_learner.ctx], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

        shot_history = []
        row = make_zero_shot_epoch0_row(
            target_name=args.target_name, preserve_loaders=preserve_loaders,
            baselines=baselines, shot=shot, optimizer=optimizer,
        )
        row["sample_seed"] = args.sample_seed
        row["split_seed"] = args.seed
        row["prompt_seed"] = args.prompt_seed
        shot_history.append(row)
        append_jsonl(row, curves_path)

        for epoch in range(1, args.epochs + 1):
            train_metric = train_one_epoch(
                model=model,
                prompt_learner=prompt_learner,
                train_loader=fewshot_loader,
                classnames=target_classnames,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                total_epochs=args.epochs,
                distiller=distiller,
                lambda_distill=args.lambda_distill,
                base_ctx=base_ctx if args.lambda_wc > 0.0 else None,
                lambda_wc=args.lambda_wc,
                distill_num_image_samples=args.distill_num_image_samples,
                distill_num_class_samples=args.distill_num_class_samples,
            )
            scheduler.step()

            do_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
            if do_eval:
                row = evaluate_current_prompt(
                    model, prompt_learner, args.target_name, target_test_loader,
                    target_classnames, preserve_loaders, preserve_classnames,
                    baselines, device, shot, epoch,
                    target_cached=target_cached, preserve_cached=preserve_cached,
                )
                row["sample_seed"] = args.sample_seed
                row["split_seed"] = args.seed
                row["prompt_seed"] = args.prompt_seed
                row["train"] = {
                    "train_loss": train_metric["train_loss"],
                    "train_loss_ce": train_metric["train_loss_ce"],
                    "train_loss_distill_raw": train_metric["train_loss_distill_raw"],
                    "train_loss_wc_raw": train_metric["train_loss_wc_raw"],
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
                "shot": shot,
                "ctx": prompt_learner.ctx.detach().cpu(),
                "n_ctx": args.n_ctx,
                "target_classnames": target_classnames,
                "history": shot_history,
                "args": vars(args),
                "prompt_seed": args.prompt_seed,
                "same_initial_context_across_shots": True,
                "base_ctx_at_init": base_ctx.cpu(),
                "lambda_distill": args.lambda_distill,
                "lambda_wc": args.lambda_wc,
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
