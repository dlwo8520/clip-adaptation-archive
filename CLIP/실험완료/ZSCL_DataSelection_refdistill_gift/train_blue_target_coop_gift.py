# -*- coding: utf-8 -*-

"""
GIFT-inspired synthetic-pair distillation + CoOp few-shot adaptation training script.

기존 train_blue_target_coop_zscl_refdistill.py 의 기능을 모두 유지하면서
새 옵션 --preserve-method 로 두 가지 보존 (preservation) 방식을 선택할 수 있게 함.

옵션:
  --preserve-method refdistill
      기존 ZSCL-style reference distillation (실제 reference 이미지 + classname 풀 사용).
      train_blue_target_coop_zscl_refdistill.py 와 동일 거동.

  --preserve-method gift_synthetic
      GIFT-inspired synthetic-pair distillation.
      매칭된 (synthetic image, prompt) pair에 대해
        L_GIFT = α · L_CD + β · L_ITA   (+ γ · L_AWC if --gift-use-awc)
      를 계산하여 L_CE에 더함.

학습 loss (gift_synthetic):
    L_total = L_CE + α · L_CD + β · L_ITA + γ · L_AWC

기존 sample selection (teacher_confident / teacher_margin / teacher_entropy) 은
그대로 사용 가능. 평가 코드도 그대로.

기존 train_blue_target_coop_zscl_refdistill.py 는 삭제하지 않았으므로 기존 실험은
그대로 재현 가능함.
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

# --- refdistill (기존) ---
from reference_distill import CoOpReferenceDistiller, compute_weight_consolidation_loss

# --- gift_synthetic (신규) ---
from gift_dataset import SyntheticImageTextDataset
from gift_distill import (
    GIFTSyntheticDistiller,
    compute_gift_step_losses,
)


# ============================================================ helpers


def parse_dataset_spec(spec: str) -> Tuple[str, str, Optional[str]]:
    """dataset spec을 파싱. 형식: name:path 또는 name:path:classnames_txt"""
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
    """reference dataset 용 template (refdistill에서만 사용)"""
    name = reference_name.lower()
    if "imagenet" in name:
        return [
            "a photo of a {}.",
            "a blurry photo of a {}.",
            "a small photo of a {}.",
            "a photo of the {}.",
        ]
    if "cifar" in name:
        return ["a photo of a {}.", "a low resolution photo of a {}."]
    return ["a photo of a {}."]


# ============================================================ training step


def train_one_epoch(
    model,
    prompt_learner,
    train_loader,
    classnames,
    optimizer,
    device,
    epoch,
    total_epochs,
    *,
    preserve_method: str,
    # refdistill 전용
    ref_distiller: Optional[CoOpReferenceDistiller] = None,
    lambda_distill: float = 0.0,
    distill_num_image_samples: int = 64,
    distill_num_class_samples: int = 64,
    # gift_synthetic 전용
    gift_distiller: Optional[GIFTSyntheticDistiller] = None,
    gift_alpha: float = 1.0,
    gift_beta: float = 0.25,
    gift_batch_size: int = 64,
    gift_use_awc: bool = False,
    gift_lambda_awc: float = 0.0,
    gift_awc_eps: float = 0.0,
    # 공통 WC (refdistill 경로용)
    base_ctx: Optional[torch.Tensor] = None,
    lambda_wc: float = 0.0,
) -> Dict[str, float]:
    """1 epoch 동안 CoOp ctx를 학습함.

    preserve_method == 'refdistill':
        L = L_CE + λ_distill · (L_lwf_img + L_lwf_txt)  (+ λ_wc · ||ctx - ctx_init||² if lambda_wc>0)

    preserve_method == 'gift_synthetic':
        L = L_CE + α · L_CD + β · L_ITA  (+ γ · L_AWC if gift_use_awc)
    """
    model.eval()
    prompt_learner.train()

    n_seen = 0
    sums = {
        "loss": 0.0,
        "ce": 0.0,
        # refdistill 측 logging keys (호환성 유지)
        "distill_raw": 0.0,
        "wc_raw": 0.0,
        # gift 측 logging keys
        "cd_raw": 0.0,
        "ita_raw": 0.0,
        "awc_raw": 0.0,
        "fisher_l1": 0.0,
        "fisher_max": 0.0,
        "correct": 0,
    }

    pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{total_epochs} [{preserve_method}]")

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

        # per-step logging values
        v_dist_raw = 0.0
        v_wc_raw = 0.0
        v_cd_raw = 0.0
        v_ita_raw = 0.0
        v_awc_raw = 0.0
        v_fisher_l1 = 0.0
        v_fisher_max = 0.0

        if preserve_method == "refdistill":
            # ---- 기존 ZSCL reference distillation 경로 ----
            if ref_distiller is not None and lambda_distill > 0.0:
                loss_dist = ref_distiller.compute_distill_loss(
                    prompt_learner=prompt_learner,
                    build_coop_text_features_fn=build_coop_text_features,
                    num_image_samples=distill_num_image_samples,
                    num_class_samples=distill_num_class_samples,
                )
                loss = loss + lambda_distill * loss_dist
                v_dist_raw = float(loss_dist.detach().item())

            if base_ctx is not None and lambda_wc > 0.0:
                loss_wc = compute_weight_consolidation_loss(prompt_learner, base_ctx)
                loss = loss + lambda_wc * loss_wc
                v_wc_raw = float(loss_wc.detach().item())

        elif preserve_method == "gift_synthetic":
            # ---- 신규 GIFT-inspired synthetic 경로 ----
            if gift_distiller is not None and (gift_alpha > 0.0 or gift_beta > 0.0):
                gift_losses = compute_gift_step_losses(
                    distiller=gift_distiller,
                    prompt_learner=prompt_learner,
                    build_coop_text_features_fn=build_coop_text_features,
                    batch_size=gift_batch_size,
                    alpha=gift_alpha,
                    beta=gift_beta,
                    use_awc=gift_use_awc,
                    base_ctx=base_ctx,
                    lambda_awc=gift_lambda_awc,
                    awc_eps=gift_awc_eps,
                )
                # preserve loss (alpha와 beta는 안에서 이미 곱해진 합)
                loss = loss + gift_losses["loss_preserve"]
                # awc는 안에서 lambda_awc까지 곱해진 값
                if gift_use_awc and gift_lambda_awc > 0.0:
                    loss = loss + gift_losses["loss_awc"]

                v_cd_raw = float(gift_losses["loss_cd"].detach().item())
                v_ita_raw = float(gift_losses["loss_ita"].detach().item())
                v_awc_raw = float(gift_losses["loss_awc"].detach().item()) / max(gift_lambda_awc, 1e-12) \
                    if (gift_use_awc and gift_lambda_awc > 0.0) else 0.0
                v_fisher_l1 = float(gift_losses["fisher_l1"].detach().item())
                v_fisher_max = float(gift_losses["fisher_max"].detach().item())

            # GIFT 경로에서도 기존 ZSCL Eq.(5) 그대로의 unweighted WC를 같이 쓸 수 있도록 유지
            # (gift_use_awc와 직교; 사용자가 굳이 둘 다 켜고 싶다면 가능)
            if base_ctx is not None and lambda_wc > 0.0:
                loss_wc = compute_weight_consolidation_loss(prompt_learner, base_ctx)
                loss = loss + lambda_wc * loss_wc
                v_wc_raw = float(loss_wc.detach().item())

        else:
            raise ValueError(f"알 수 없는 preserve_method: {preserve_method}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct = (preds == targets).sum().item()
        bs = targets.numel()
        n_seen += bs
        sums["loss"] += loss.item() * bs
        sums["ce"] += loss_ce.item() * bs
        sums["distill_raw"] += v_dist_raw * bs
        sums["wc_raw"] += v_wc_raw * bs
        sums["cd_raw"] += v_cd_raw * bs
        sums["ita_raw"] += v_ita_raw * bs
        sums["awc_raw"] += v_awc_raw * bs
        sums["fisher_l1"] += v_fisher_l1 * bs
        sums["fisher_max"] = max(sums["fisher_max"], v_fisher_max)  # 누적이 아닌 max
        sums["correct"] += correct

        if preserve_method == "refdistill":
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ce=f"{loss_ce.item():.4f}",
                dist=f"{v_dist_raw:.4f}",
                wc=f"{v_wc_raw:.4f}",
                acc=f"{correct / bs:.4f}",
            )
        else:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ce=f"{loss_ce.item():.4f}",
                cd=f"{v_cd_raw:.4f}",
                ita=f"{v_ita_raw:.4f}",
                awc=f"{v_awc_raw:.4f}",
                acc=f"{correct / bs:.4f}",
            )

    denom = max(1, n_seen)
    return {
        "train_loss": float(sums["loss"] / denom),
        "train_loss_ce": float(sums["ce"] / denom),
        # 기존 호환 logging (refdistill에서만 의미 있음. gift에서는 0)
        "train_loss_distill_raw": float(sums["distill_raw"] / denom),
        "train_loss_wc_raw": float(sums["wc_raw"] / denom),
        # gift 전용 logging (refdistill에서는 0)
        "train_loss_cd_raw": float(sums["cd_raw"] / denom),
        "train_loss_ita_raw": float(sums["ita_raw"] / denom),
        "train_loss_awc_raw": float(sums["awc_raw"] / denom),
        "train_fisher_l1": float(sums["fisher_l1"] / denom),
        "train_fisher_max": float(sums["fisher_max"]),
        "train_acc": float(sums["correct"] / denom),
    }


# ============================================================ evaluation row


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
    """현재 prompt 상태로 target과 preserve dataset 평가."""
    row = {"target_name": target_name, "shot": shot, "epoch": epoch, "target": {}, "preserve": {}}

    target_text = build_coop_text_features(model, prompt_learner, target_classnames, device)
    if target_cached is not None:
        target_metric = evaluate_classifier(
            model=model, loader=None, text_features=target_text, device=device,
            desc=f"eval target={target_name} shot={shot} epoch={epoch}",
            cached_features=target_cached[0], cached_targets=target_cached[1],
        )
    else:
        target_metric = evaluate_classifier(
            model=model, loader=target_loader, text_features=target_text, device=device,
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
                model=model, loader=None, text_features=preserve_text, device=device,
                desc=f"eval preserve={preserve_name} shot={shot} epoch={epoch}",
                cached_features=preserve_cached[preserve_name][0],
                cached_targets=preserve_cached[preserve_name][1],
            )
        else:
            metric = evaluate_classifier(
                model=model, loader=loader, text_features=preserve_text, device=device,
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


def make_zero_shot_epoch0_row(target_name, preserve_loaders, baselines, shot, optimizer):
    """epoch 0을 original CLIP zero-shot baseline으로 기록."""
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
        "train_loss": None, "train_loss_ce": None,
        "train_loss_distill_raw": None, "train_loss_wc_raw": None,
        "train_loss_cd_raw": None, "train_loss_ita_raw": None,
        "train_loss_awc_raw": None,
        "train_fisher_l1": None, "train_fisher_max": None,
        "train_acc": None,
        "lr": optimizer.param_groups[0]["lr"],
    }
    row["prompt_init"] = {
        "epoch_0_meaning": "original CLIP zero-shot baseline before CoOp adaptation",
        "note": "epoch 0 is copied from zero_shot_baselines.json",
    }
    return row


# ============================================================ main


def main():
    parser = argparse.ArgumentParser()

    # ---------- preserve method (신규) ----------
    parser.add_argument(
        "--preserve-method",
        type=str,
        default="refdistill",
        choices=["refdistill", "gift_synthetic"],
        help="보존(preservation) 방식 선택. refdistill: 기존 ZSCL-style real reference distillation. "
             "gift_synthetic: GIFT-inspired synthetic matched (image, prompt) pair distillation.",
    )

    # ---------- target / preserve ----------
    parser.add_argument("--target-name", type=str, required=True)
    parser.add_argument("--target-root", type=str, default=None)
    parser.add_argument("--target-train", type=str, default=None)
    parser.add_argument("--target-test", type=str, default=None)
    parser.add_argument("--target-classnames", type=str, default=None)
    parser.add_argument("--split-test-ratio", type=float, default=0.5)
    parser.add_argument("--preserve", type=str, nargs="+", required=True)

    # ---------- CLIP backbone ----------
    parser.add_argument("--model", type=str, default="ViT-B-32-quickgelu")
    parser.add_argument("--pretrained", type=str, default="openai")

    # ---------- CoOp / 학습 옵션 ----------
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

    # ---------- support sample selection (기존, 변경 없음) ----------
    parser.add_argument(
        "--selection-method",
        type=str,
        default="teacher_confident",
        choices=["teacher_confident", "teacher_margin", "teacher_entropy"],
    )
    parser.add_argument("--selection-candidate-factor", type=int, default=2)

    # ---------- refdistill 전용 (기존 호환) ----------
    parser.add_argument("--reference-name", type=str, default="imagenet")
    parser.add_argument("--reference-root", type=str, default=None,
                        help="ImageFolder 구조 reference image dataset. refdistill에서 필수.")
    parser.add_argument("--reference-classnames", type=str, default=None)
    parser.add_argument("--max-reference-images", type=int, default=10000)
    parser.add_argument("--reference-precompute-seed", type=int, default=0)
    parser.add_argument("--reference-store-cpu", action="store_true")
    parser.add_argument("--lambda-distill", type=float, default=1.0,
                        help="(refdistill 전용) reference distillation 계수.")
    parser.add_argument("--distill-num-image-samples", type=int, default=64)
    parser.add_argument("--distill-num-class-samples", type=int, default=64)

    # ---------- 공통 WC (ZSCL Eq.(5) raw) ----------
    parser.add_argument("--lambda-wc", type=float, default=0.0,
                        help="(공통) ctx의 unweighted L2 anchoring 계수. "
                             "gift_synthetic에서도 함께 사용 가능 (AWC와 별개).")

    # ---------- gift_synthetic 전용 (신규) ----------
    parser.add_argument("--gift-synthetic-root", type=str, default=None,
                        help="(gift_synthetic 전용) synthetic 이미지 폴더 루트. "
                             "scripts/generate_synthetic_dataset.py 등으로 생성된 디렉토리.")
    parser.add_argument("--gift-metadata", type=str, default=None,
                        help="(gift_synthetic 전용) metadata.jsonl 경로. "
                             "None이면 <gift-synthetic-root>/metadata.jsonl.")
    parser.add_argument("--gift-alpha", type=float, default=1.0,
                        help="(gift_synthetic) L_CD (contrastive distillation) 계수. GIFT 논문 기본 1.0.")
    parser.add_argument("--gift-beta", type=float, default=0.25,
                        help="(gift_synthetic) L_ITA (image-text alignment) 계수. GIFT 논문 기본 0.25.")
    parser.add_argument("--gift-batch-size", type=int, default=64,
                        help="(gift_synthetic) 매 step에 sampling할 synthetic pair 수 B.")
    parser.add_argument("--gift-num-workers", type=int, default=4,
                        help="(gift_synthetic) feature 캐싱 시 DataLoader num_workers.")
    parser.add_argument("--gift-max-images", type=int, default=0,
                        help="(gift_synthetic) cache할 synthetic image 수 상한. "
                             "0 이하이면 metadata.jsonl 전부 사용.")
    parser.add_argument("--gift-precompute-seed", type=int, default=0,
                        help="(gift_synthetic) subsampling용 seed. sample-seed간 풀 고정 권장.")
    parser.add_argument("--gift-store-cpu", action="store_true",
                        help="(gift_synthetic) feature를 GPU 대신 CPU에 캐싱 (메모리 절약).")
    parser.add_argument("--gift-cache-image-features", action="store_true",
                        help="(gift_synthetic) image feature 캐싱 강제 (현재 기본값으로 항상 캐싱하므로 no-op).")
    parser.add_argument("--gift-use-awc", action="store_true",
                        help="(gift_synthetic) Adaptive Weight Consolidation on/off.")
    parser.add_argument("--gift-lambda-awc", type=float, default=0.0,
                        help="(gift_synthetic) AWC loss 계수. gift-use-awc=True일 때만 적용.")
    parser.add_argument("--gift-awc-eps", type=float, default=0.0,
                        help="(gift_synthetic) Fisher에 더할 floor (vanilla WC 백업).")
    parser.add_argument("--gift-verify-paths", action="store_true",
                        help="(gift_synthetic) init 시 모든 image 경로 존재 검증.")

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
    print(f"preserve_method: {args.preserve_method}")
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

    # ===== distiller 준비 (preserve_method에 따라 분기) =====
    ref_distiller: Optional[CoOpReferenceDistiller] = None
    gift_distiller: Optional[GIFTSyntheticDistiller] = None

    if args.preserve_method == "refdistill":
        if args.lambda_distill > 0.0:
            if args.reference_root is None:
                raise ValueError(
                    "preserve_method=refdistill 이고 lambda-distill > 0 인데 "
                    "--reference-root가 주어지지 않음."
                )
            print("===== reference dataset 로딩 (refdistill) =====")
            ref_dataset, ref_classnames_folder = build_preserve_dataset(
                args.reference_root, transform=preprocess
            )
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

            # leakage 점검
            for pname, pinfo in preserve_info.items():
                if Path(pinfo["path"]).resolve() == Path(args.reference_root).resolve():
                    print(
                        f"[경고] reference-root와 preserve='{pname}' 경로가 동일함 "
                        f"({pinfo['path']}). leakage 가능성. 다른 split 권장."
                    )

            ref_distiller = CoOpReferenceDistiller(
                model=model, tokenizer=tokenizer, reference_dataset=ref_dataset,
                reference_classnames=ref_classnames, reference_templates=ref_templates,
                device=device, precompute_batch_size=max(64, args.batch_size),
                precompute_num_workers=args.num_workers,
                max_reference_images=args.max_reference_images,
                precompute_seed=args.reference_precompute_seed,
                store_image_features_on_cpu=args.reference_store_cpu,
            )
            save_json(
                {
                    "preserve_method": "refdistill",
                    "reference_name": args.reference_name,
                    "reference_root": args.reference_root,
                    "reference_classnames_file": args.reference_classnames,
                    "num_classes": len(ref_classnames),
                    "num_images_after_subsample": ref_distiller.num_images,
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
            print("===== refdistill 비활성 (lambda_distill = 0) =====")

    elif args.preserve_method == "gift_synthetic":
        if args.gift_synthetic_root is None:
            raise ValueError(
                "preserve_method=gift_synthetic 인데 --gift-synthetic-root가 주어지지 않음. "
                "synthetic image 폴더 경로를 넘기세요."
            )
        synth_metadata_path = args.gift_metadata
        if synth_metadata_path is None:
            synth_metadata_path = str(Path(args.gift_synthetic_root) / "metadata.jsonl")

        print("===== synthetic dataset 로딩 (gift_synthetic) =====")
        print(f"  synthetic_root: {args.gift_synthetic_root}")
        print(f"  metadata     : {synth_metadata_path}")

        synth_dataset = SyntheticImageTextDataset(
            root=args.gift_synthetic_root,
            metadata_path=synth_metadata_path,
            transform=preprocess,
            verify_paths=bool(args.gift_verify_paths),
        )
        print(f"  loaded {synth_dataset.num_images} synthetic pairs, "
              f"{synth_dataset.num_unique_classes} unique classes, "
              f"sources={synth_dataset.source_counts()}")

        gift_distiller = GIFTSyntheticDistiller(
            model=model, tokenizer=tokenizer, synthetic_dataset=synth_dataset,
            device=device, precompute_batch_size=max(64, args.batch_size),
            precompute_num_workers=args.gift_num_workers,
            max_synthetic_images=args.gift_max_images if args.gift_max_images > 0 else None,
            precompute_seed=args.gift_precompute_seed,
            store_features_on_cpu=args.gift_store_cpu,
        )

        # base concepts vs target concepts 카운트
        source_counts_full = synth_dataset.source_counts()
        n_base = sum(v for k, v in source_counts_full.items() if k == "base")
        n_target = sum(v for k, v in source_counts_full.items() if k.startswith("target"))
        n_other = sum(v for k, v in source_counts_full.items() if k not in {"base"} and not k.startswith("target"))

        save_json(
            {
                "preserve_method": "gift_synthetic",
                "synthetic_root": args.gift_synthetic_root,
                "metadata_path": synth_metadata_path,
                "num_synthetic_total": synth_dataset.num_images,
                "num_active_after_subsample": gift_distiller.num_active_images,
                "num_unique_classes_total": synth_dataset.num_unique_classes,
                "num_unique_classes_active": gift_distiller.num_unique_classes_active,
                "source_counts_full": source_counts_full,
                "source_counts_active": gift_distiller.source_counts_active(),
                "num_base_concepts": n_base,
                "num_target_concepts": n_target,
                "num_other_sources": n_other,
                "gift_alpha": args.gift_alpha,
                "gift_beta": args.gift_beta,
                "gift_batch_size": args.gift_batch_size,
                "gift_use_awc": bool(args.gift_use_awc),
                "gift_lambda_awc": args.gift_lambda_awc,
                "gift_awc_eps": args.gift_awc_eps,
                "gift_precompute_seed": args.gift_precompute_seed,
                "gift_store_cpu": bool(args.gift_store_cpu),
            },
            output_dir / "gift_distill_meta.json",
        )

        # leakage 점검: synthetic root가 어떤 preserve set과 같은 경로면 경고
        for pname, pinfo in preserve_info.items():
            if Path(pinfo["path"]).resolve() == Path(args.gift_synthetic_root).resolve():
                print(
                    f"[경고] gift-synthetic-root와 preserve='{pname}' 경로가 동일함 "
                    f"({pinfo['path']}). 합성 이미지와 평가 이미지가 같으면 leakage."
                )
    else:
        raise ValueError(f"알 수 없는 preserve_method: {args.preserve_method}")

    # ===== 공통 CoOp prompt 초기값 =====
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
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompt_init_meta = {
        "preserve_method": args.preserve_method,
        "prompt_seed": args.prompt_seed,
        "same_initial_context_across_shots": True,
        "epoch_0_meaning": "original CLIP zero-shot baseline before CoOp adaptation",
        "epoch_1_and_after": "CoOp prompt after target few-shot adaptation",
        # refdistill 관련
        "lambda_distill": args.lambda_distill,
        "uses_reference_distill": ref_distiller is not None,
        # gift 관련
        "gift_alpha": args.gift_alpha,
        "gift_beta": args.gift_beta,
        "gift_use_awc": bool(args.gift_use_awc),
        "gift_lambda_awc": args.gift_lambda_awc,
        # 공통
        "lambda_wc": args.lambda_wc,
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
        print(f"===== shot={shot} CoOp 학습 시작 (preserve_method={args.preserve_method}) =====")
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
                preserve_method=args.preserve_method,
                # refdistill
                ref_distiller=ref_distiller,
                lambda_distill=args.lambda_distill,
                distill_num_image_samples=args.distill_num_image_samples,
                distill_num_class_samples=args.distill_num_class_samples,
                # gift
                gift_distiller=gift_distiller,
                gift_alpha=args.gift_alpha,
                gift_beta=args.gift_beta,
                gift_batch_size=args.gift_batch_size,
                gift_use_awc=bool(args.gift_use_awc),
                gift_lambda_awc=args.gift_lambda_awc,
                gift_awc_eps=args.gift_awc_eps,
                # WC (공통)
                base_ctx=base_ctx if args.lambda_wc > 0.0 else None,
                lambda_wc=args.lambda_wc,
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
                    # 기존 호환 logging
                    "train_loss_distill_raw": train_metric["train_loss_distill_raw"],
                    "train_loss_wc_raw": train_metric["train_loss_wc_raw"],
                    # gift 전용 logging
                    "train_loss_cd_raw": train_metric["train_loss_cd_raw"],
                    "train_loss_ita_raw": train_metric["train_loss_ita_raw"],
                    "train_loss_awc_raw": train_metric["train_loss_awc_raw"],
                    "train_fisher_l1": train_metric["train_fisher_l1"],
                    "train_fisher_max": train_metric["train_fisher_max"],
                    "train_acc": train_metric["train_acc"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
                row["prompt_init"] = prompt_init_meta
                shot_history.append(row)
                append_jsonl(row, curves_path)
                save_json(shot_history, shot_dir / "history.json")

        torch.save(
            {
                "preserve_method": args.preserve_method,
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
