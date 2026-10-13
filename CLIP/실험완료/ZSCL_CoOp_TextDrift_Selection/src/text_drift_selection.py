# -*- coding: utf-8 -*-
"""
Text-side drift based data selection for CoOp.

배경:
- ZSCL/PDS 류의 selection은 J(theta)를 무엇으로 두느냐가 핵심 (PDS 논문 Table 10).
- 본 모듈은 CoOp few-shot adaptation 전용 selection으로,
  J(theta) ≈ "원본 CLIP 일반 zero-shot 능력의 보존" 으로 간접 정의함.
- 그 정의를 다음과 같이 구체화:
    * Vocabulary V (예: ImageNet-1K 1000 classnames)에 대해
    * 원본 base_ctx에서의 CoOp text feature φ_V(base_ctx)와
    * 후보 sample x_n으로 1-step gradient update한 ctx_pert에서의
      φ_V(ctx_pert) 사이의 거리(drift)를 측정.
- drift가 작은 샘플 = ctx를 흔들지 않는 샘플
              = adaptation 후에도 일반 zero-shot text representation이 보존됨.

왜 CoOp에 적합한가:
- CoOp이 학습하는 것은 ctx(prompt context vector) 단 하나.
- 같은 ctx가 모든 클래스의 text feature를 만들어내므로,
  V 전체에서의 drift가 곧 "이 샘플이 일반 prompt를 망가뜨리는 정도".
- 평가용 image dataset이 필요 없음. text-only reference로 충분.
- Vocabulary는 ImageNet-1K 같은 broad한 명사 집합이면 됨.
  CLIP 학습 분포(WebImageText)에 직접 접근 불가능한 상황의 우회.

연결되는 PDS 수식 (Theorem 2.1 Eq.6):
    score(x_n) ∝ λ_{t+1}^T ∇l(x_n, θ_t)
- 본 모듈은 한-스텝 근사로 λ_{t+1} ≈ -∇J(theta_0)에 해당하며,
  J(θ) = sum_v ||φ(v;ctx) - φ(v;ctx_0)||^2 (보존 loss).
- 즉, "이 샘플의 gradient가 보존 loss 감소 방향과 얼마나 정렬되나" 를
  측정하는 것과 같음.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from coop_model import (
    clean_class_name,
    encode_text_from_embeddings,
)


# 단일 템플릿은 CoOp 학습 시점의 prompt 형식과 가장 일치함.
# 멀티 템플릿은 surface-form 변동에 강건한 drift 측정용.
DEFAULT_VOCAB_TEMPLATES = ["{}."]
ROBUST_VOCAB_TEMPLATES = [
    "{}.",
    "a photo of a {}.",
    "a photo of the {}.",
    "an image of a {}.",
]


def load_default_vocabulary(path: Optional[str] = None) -> List[str]:
    """vocabulary 명사 목록을 로드함.

    --text-drift-vocab-file 인자가 주어지면 해당 파일 사용.
    아니면 torchvision의 ImageNet-1K 1000 classnames를 fallback으로 사용함.
    """
    if path is not None:
        from utils import load_classnames_file
        names = load_classnames_file(path)
        if len(names) == 0:
            raise ValueError(f"vocabulary file at {path} is empty")
        print(f"[text_drift] vocabulary loaded from file: {path} ({len(names)} names)")
        return names

    try:
        from torchvision.models import ResNet50_Weights
        names = list(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])
        print(f"[text_drift] vocabulary fallback to torchvision ImageNet-1K ({len(names)} names)")
        return names
    except Exception as e:
        raise RuntimeError(
            f"vocabulary 로드 실패: {e}. --text-drift-vocab-file을 직접 지정하세요."
        )


def _build_prompt_pieces_for_vocab(
    clip_model: nn.Module,
    tokenizer,
    classnames: List[str],
    templates: List[str],
    n_ctx: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """vocab × templates 모든 조합에 대해 ctx-독립 부분(prefix/suffix/tokenized)을 미리 계산함.

    CoOp prompt 구조: [SOS][ctx_1...ctx_M][TEMPLATE_FILLED_WITH_NAME][EOS]
    - prefix = [SOS]
    - ctx 부분은 forward 시점에 ctx_param로 교체됨
    - suffix = [TEMPLATE_FILLED_WITH_NAME][EOS]

    Returns:
        prefix:    [N_total, 1, D]
        suffix:    [N_total, L_suffix, D]
        tokenized: [N_total, L]
        class_groups: 각 클래스가 N_total 안에서 차지하는 [start, end) 인덱스.
                      한 클래스마다 len(templates)개의 entry가 들어 있음.
                      (멀티 템플릿이면 평균 후 정규화하여 클래스 1개의 feature로 만듦)
    """
    dummy = " ".join(["X"] * n_ctx)
    prompt_texts: List[str] = []
    class_groups: List[Tuple[int, int]] = []

    cursor = 0
    for cname in classnames:
        clean = clean_class_name(cname)
        start = cursor
        for tmpl in templates:
            filled = tmpl.format(clean)
            prompt_texts.append(f"{dummy} {filled}")
            cursor += 1
        end = cursor
        class_groups.append((start, end))

    tokenized = tokenizer(prompt_texts).to(device)
    with torch.no_grad():
        embedding = clip_model.token_embedding(tokenized)  # [N_total, L, D]
    prefix = embedding[:, :1, :].detach()
    suffix = embedding[:, 1 + n_ctx:, :].detach()
    return prefix, suffix, tokenized, class_groups


def _text_features_with_ctx(
    clip_model: nn.Module,
    ctx: torch.Tensor,
    prefix: torch.Tensor,
    suffix: torch.Tensor,
    tokenized: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """ctx를 prefix와 suffix 사이에 끼워 넣어 prompt embedding을 만들고
    text encoder를 통해 각 prompt의 text feature를 계산함.

    Returns: [N_total, d_text], L2 정규화됨.
    """
    N = prefix.shape[0]
    feats: List[torch.Tensor] = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        pref = prefix[start:end]
        suf = suffix[start:end]
        tok = tokenized[start:end]
        bsz = pref.shape[0]
        ctx_expanded = ctx.unsqueeze(0).expand(bsz, -1, -1)
        prompt_embed = torch.cat([pref, ctx_expanded, suf], dim=1)
        feats.append(encode_text_from_embeddings(clip_model, prompt_embed, tok))
    feats_cat = torch.cat(feats, dim=0)
    return F.normalize(feats_cat, dim=-1)


def _aggregate_classwise(
    features: torch.Tensor,
    class_groups: List[Tuple[int, int]],
) -> torch.Tensor:
    """클래스별 group을 평균낸 뒤 다시 L2 정규화함.

    Input:  [N_total, D]
    Output: [n_classes, D]
    """
    out: List[torch.Tensor] = []
    for start, end in class_groups:
        cls_feat = features[start:end].mean(dim=0)
        cls_feat = F.normalize(cls_feat, dim=0)
        out.append(cls_feat)
    return torch.stack(out, dim=0)


@torch.no_grad()
def _precompute_image_features_normalized(
    clip_model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    desc: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """train pool 전체 이미지의 정규화된 image feature를 한 번만 계산함.

    image encoder가 freeze이므로 candidate별로 매번 forward할 필요 없음.
    """
    clip_model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    feats_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    for images, targets in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        f = clip_model.encode_image(images)
        f = F.normalize(f, dim=-1)
        feats_list.append(f.detach().cpu())
        labels_list.append(targets.detach().cpu())
    return torch.cat(feats_list, dim=0), torch.cat(labels_list, dim=0)


def compute_text_drift_scores(
    clip_model: nn.Module,
    tokenizer,
    target_train_pool: Dataset,
    target_classnames: List[str],
    vocabulary: List[str],
    base_ctx: torch.Tensor,
    n_ctx: int,
    step_size: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    vocab_templates: Optional[List[str]] = None,
    target_templates: Optional[List[str]] = None,
    text_encoder_chunk_size: int = 1024,
) -> Dict[str, torch.Tensor]:
    """target_train_pool의 모든 candidate에 대해 text drift score를 계산함.

    Args:
        clip_model: freeze된 CLIP 모델 (외부에서 freeze_clip_model 호출 가정)
        base_ctx: [n_ctx, D], 모든 shot 실험의 공통 prompt 초기값
        step_size: ctx_pert = base_ctx - step_size * g_n 의 step.
                   ranking에는 영향 없으나 작을수록 linearization이 정확함.
        vocab_templates: V에 적용할 템플릿. 기본 ["{}."]로 CoOp 학습 형식과 일치.
        target_templates: target에 적용할 템플릿. 기본 ["{}."]로 CoOp 학습 형식과 일치.

    Returns:
        dict with:
            "labels":      [N], target_train_pool의 라벨
            "drift_value": [N], 측정된 raw drift (||phi_pert - phi_base||^2 평균)
            "drift_score": [N], = -drift_value (높을수록 보존이 잘 됨 → 우선 선택)

    높은 score = 작은 drift = adaptation 후에도 V의 text feature가 적게 변함
                = 일반 zero-shot 능력이 잘 보존되는 sample.
    """
    if vocab_templates is None:
        vocab_templates = DEFAULT_VOCAB_TEMPLATES
    if target_templates is None:
        target_templates = DEFAULT_VOCAB_TEMPLATES

    clip_model.eval()
    base_ctx = base_ctx.detach().to(device)
    logit_scale = clip_model.logit_scale.exp().detach()

    # ----- 1) Vocabulary V의 prompt pieces (ctx-독립) -----
    print(f"[text_drift] preparing vocabulary prompt pieces "
          f"|V|={len(vocabulary)}, templates={len(vocab_templates)}, "
          f"total prompts={len(vocabulary) * len(vocab_templates)}")
    vocab_prefix, vocab_suffix, vocab_tokenized, vocab_groups = _build_prompt_pieces_for_vocab(
        clip_model, tokenizer, vocabulary, vocab_templates, n_ctx, device,
    )

    # ----- 2) base_ctx에서의 reference φ_V (drift 비교 기준) -----
    print("[text_drift] computing reference vocabulary text features at base_ctx")
    with torch.no_grad():
        vocab_feats_base_raw = _text_features_with_ctx(
            clip_model, base_ctx, vocab_prefix, vocab_suffix, vocab_tokenized,
            chunk_size=text_encoder_chunk_size,
        )
        phi_V_base = _aggregate_classwise(vocab_feats_base_raw, vocab_groups)  # [|V|, d]

    # ----- 3) target classes의 prompt pieces (gradient 계산용) -----
    target_prefix, target_suffix, target_tokenized, target_groups = _build_prompt_pieces_for_vocab(
        clip_model, tokenizer, target_classnames, target_templates, n_ctx, device,
    )

    # ----- 4) train pool image feature 캐싱 -----
    image_features, all_labels = _precompute_image_features_normalized(
        clip_model, target_train_pool, batch_size, num_workers, device,
        desc="[text_drift] cache target train pool image features",
    )
    image_features = image_features.to(device)
    all_labels_dev = all_labels.to(device)

    # ----- 5) per-sample drift 계산 -----
    N = image_features.shape[0]
    drift_values = torch.empty(N, dtype=torch.float32)

    print(f"[text_drift] computing per-sample drift for N={N} candidates "
          f"(step_size={step_size})")

    for n in tqdm(range(N), desc="[text_drift] per-sample drift"):
        # (a) ctx_param: leaf tensor with grad
        ctx_param = base_ctx.clone().detach().requires_grad_(True)

        # (b) target text features at ctx_param + per-sample loss + grad
        with torch.enable_grad():
            target_feats_raw = _text_features_with_ctx(
                clip_model, ctx_param, target_prefix, target_suffix, target_tokenized,
                chunk_size=text_encoder_chunk_size,
            )
            phi_target = _aggregate_classwise(target_feats_raw, target_groups)  # [C_target, d]

            img_feat_n = image_features[n].unsqueeze(0)  # [1, d]
            logits_n = logit_scale * img_feat_n @ phi_target.T  # [1, C_target]
            label_n = all_labels_dev[n].unsqueeze(0)
            loss_n = F.cross_entropy(logits_n, label_n)

            (g_n,) = torch.autograd.grad(loss_n, ctx_param, create_graph=False, retain_graph=False)

        # (c) ctx_pert = base_ctx - step_size * g_n, drift 계산
        with torch.no_grad():
            ctx_pert = base_ctx - step_size * g_n
            vocab_feats_pert_raw = _text_features_with_ctx(
                clip_model, ctx_pert, vocab_prefix, vocab_suffix, vocab_tokenized,
                chunk_size=text_encoder_chunk_size,
            )
            phi_V_pert = _aggregate_classwise(vocab_feats_pert_raw, vocab_groups)  # [|V|, d]
            # 정규화된 vector이므로 ||a-b||^2 = 2 - 2 cos(a,b)
            drift = (phi_V_pert - phi_V_base).pow(2).sum(dim=-1).mean()
            drift_values[n] = float(drift.item())

    drift_score = -drift_values  # 높을수록 보존 우수

    return {
        "labels": all_labels.detach().cpu(),
        "drift_value": drift_values,
        "drift_score": drift_score,
    }
