# -*- coding: utf-8 -*-
"""평가 함수"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def precompute_image_features(
    model: nn.Module, loader: DataLoader, device: torch.device, desc: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    image encoder가 freeze되어 있으므로 image feature를 한 번만 계산해서 재사용함.
    return: (features [N, D] L2 normalized, targets [N])
    """
    model.eval()
    feats_list = []
    tgts_list = []
    for images, targets in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        f = model.encode_image(images)
        f = F.normalize(f, dim=-1)
        feats_list.append(f.detach().cpu())
        tgts_list.append(targets.detach().cpu())
    return torch.cat(feats_list, dim=0), torch.cat(tgts_list, dim=0)


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: Optional[DataLoader],
    text_features: torch.Tensor,
    device: torch.device,
    desc: str,
    cached_features: Optional[torch.Tensor] = None,
    cached_targets: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    image feature와 text feature를 비교하여 accuracy를 계산함.
    cached_features/cached_targets가 주어지면 image encoder를 다시 돌리지 않음.
    """
    model.eval()
    logit_scale = model.logit_scale.exp()

    if cached_features is not None and cached_targets is not None:
        feats = cached_features.to(device, non_blocking=True)
        tgts = cached_targets.to(device, non_blocking=True)
        # 한 번에 처리하기엔 클 수 있으므로 chunk
        chunk = 4096
        correct = 0
        total = 0
        for start in range(0, feats.shape[0], chunk):
            end = min(start + chunk, feats.shape[0])
            logits = logit_scale * feats[start:end] @ text_features.t()
            preds = logits.argmax(dim=1)
            correct += (preds == tgts[start:end]).sum().item()
            total += end - start
        return {"accuracy": float(correct / total if total > 0 else 0.0), "correct": int(correct), "total": int(total)}

    correct = 0
    total = 0
    for images, targets in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        logits = logit_scale * image_features @ text_features.t()
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.numel()

    return {"accuracy": float(correct / total if total > 0 else 0.0), "correct": int(correct), "total": int(total)}
