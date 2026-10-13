# -*- coding: utf-8 -*-
"""
ZSCL 논문(Zheng et al., CVPR 2023)의 정신을 따른 few-shot sample selection.

핵심 아이디어:
- ZSCL은 "원본 CLIP의 zero-shot transfer ability를 보존"하는 것을 목표로 함.
- ZSCL은 distillation loss로 이를 달성하지만, 본 모듈은 동일한 목표를
  "few-shot adaptation에 사용할 support set 선택" 단계에서 달성함.
- Teacher CLIP이 잘 인식하는 (prototypical) 샘플을 선택하면,
  adaptation 시 원본 representation을 덜 망가뜨려 zero-shot 성능이 보존됨.
- 반대로 Teacher가 헷갈리는 (atypical, hard) 샘플을 선택하면 큰 gradient가 발생하여
  representation을 크게 바꾸므로 preserve dataset 성능이 더 떨어짐.

선택 방법:
- random            : 기존 무작위 선택 (baseline 비교용)
- teacher_confident : Teacher CLIP zero-shot에서 정답 클래스 confidence가 높은 샘플
- teacher_margin    : 정답 vs 차상위 클래스 margin이 큰 샘플 (decision boundary에서 먼)
- prototype         : Teacher CLIP image feature의 클래스 prototype에 가장 가까운 medoid
- teacher_uncertain : (대조군) Teacher가 헷갈리는 샘플 — preserve 성능이 더 떨어질 것으로 예상

Nested 보장:
- 1-shot ⊂ 2-shot ⊂ 4-shot ⊂ 8-shot ⊂ 16-shot 구조를 유지함.
- 클래스마다 max_shot * candidate_factor개의 후보를 score 기준으로 선정한 뒤,
  sample_seed로 후보 내 순서를 셔플하여 1/2/4/8/16-shot을 잘라냄.
- 같은 sample_seed에서 모든 shot이 일관된 nested subset을 가짐.
- 다른 sample_seed에서는 후보 풀 안에서 다른 조합을 뽑아 100 seed의 다양성을 확보함.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


@torch.no_grad()
def _compute_teacher_scores(
    dataset: Dataset,
    classnames: List[str],
    model,
    tokenizer,
    templates: List[str],
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 4,
    desc: str = "compute teacher scores",
) -> Dict[str, torch.Tensor]:
    """원본 CLIP teacher로 dataset 내 모든 이미지의 score를 계산함.

    Returns:
        labels:           [N] long, 정답 라벨
        image_features:   [N, D] float, L2 정규화된 image feature
        text_features:    [C, D] float, L2 정규화된 text feature (templates 평균)
        confidence:       [N] float, 정답 클래스의 softmax probability
        margin:           [N] float, (정답 prob) - (차상위 prob)
        gt_similarity:    [N] float, 정답 클래스 cosine similarity (raw)
    """
    # 1) text feature 준비 (templates 평균)
    model.eval()
    text_feats = []
    for cname in classnames:
        clean = cname.replace("_", " ").replace("-", " ")
        prompts = [t.format(clean) for t in templates]
        tokenized = tokenizer(prompts).to(device)
        emb = model.encode_text(tokenized)
        emb = F.normalize(emb, dim=-1)
        emb = emb.mean(dim=0)
        emb = F.normalize(emb, dim=-1)
        text_feats.append(emb)
    text_features = torch.stack(text_feats, dim=0)  # [C, D]

    # 2) image feature 및 score 일괄 계산
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    logit_scale = model.logit_scale.exp()

    all_image_feats = []
    all_labels = []
    all_conf = []
    all_margin = []
    all_gt_sim = []

    for images, targets in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        img_feat = F.normalize(model.encode_image(images), dim=-1)
        sim = img_feat @ text_features.T  # [B, C], cosine similarity
        logits = logit_scale * sim
        probs = logits.softmax(dim=-1)  # [B, C]

        # 정답 클래스 confidence
        gt_prob = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
        gt_sim = sim.gather(1, targets.unsqueeze(1)).squeeze(1)     # [B]

        # margin: 정답 prob - top-1 (정답 제외) prob
        probs_masked = probs.clone()
        probs_masked.scatter_(1, targets.unsqueeze(1), -1.0)
        runner_up = probs_masked.max(dim=1).values  # [B]
        margin = gt_prob - runner_up

        all_image_feats.append(img_feat.cpu())
        all_labels.append(targets.cpu())
        all_conf.append(gt_prob.cpu())
        all_margin.append(margin.cpu())
        all_gt_sim.append(gt_sim.cpu())

    return {
        "labels": torch.cat(all_labels, dim=0),
        "image_features": torch.cat(all_image_feats, dim=0),
        "text_features": text_features.cpu(),
        "confidence": torch.cat(all_conf, dim=0),
        "margin": torch.cat(all_margin, dim=0),
        "gt_similarity": torch.cat(all_gt_sim, dim=0),
    }


def _scores_for_method(
    method: str,
    teacher_data: Dict[str, torch.Tensor],
    class_indices: List[int],
) -> torch.Tensor:
    """주어진 method에 대해 한 클래스 내 샘플들의 점수를 반환함.

    점수가 높을수록 '선택 우선순위가 높은' 샘플임.
    """
    method = method.lower()

    if method == "teacher_confident":
        # Teacher가 정답 클래스에 높은 확신을 가지는 샘플 → prototypical
        return teacher_data["confidence"][class_indices]

    if method == "teacher_margin":
        # Teacher 정답 vs 차상위 클래스 margin 큼 → decision boundary에서 멀고 안정적
        return teacher_data["margin"][class_indices]

    if method == "teacher_uncertain":
        # 대조군: Teacher가 헷갈리는 샘플 → preserve가 더 망가질 것으로 예상
        return -teacher_data["confidence"][class_indices]

    if method == "prototype":
        # 클래스 image feature의 평균(prototype)에 가까운 medoid
        feats = teacher_data["image_features"][class_indices]  # [n_c, D]
        proto = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)  # [1, D]
        cos_to_proto = (feats @ proto.T).squeeze(1)  # [n_c]
        return cos_to_proto

    raise ValueError(f"알 수 없는 selection method: {method}")


def build_fewshot_subset_zscl(
    dataset: Dataset,
    shots: int,
    seed: int,
    method: str = "teacher_confident",
    teacher_data: Optional[Dict[str, torch.Tensor]] = None,
    candidate_factor: int = 2,
    max_shot_in_experiment: int = 16,
) -> Subset:
    """ZSCL-aware few-shot subset 구성.

    Args:
        dataset: target train pool
        shots: 클래스당 선택할 이미지 수
        seed: nested 안에서 후보를 셔플할 sample_seed
        method: 'random' | 'teacher_confident' | 'teacher_margin'
                | 'prototype' | 'teacher_uncertain'
        teacher_data: _compute_teacher_scores() 결과. method='random'이면 None 가능.
        candidate_factor: 클래스당 score top-K 후보 풀 크기 = max_shot * factor.
                          작을수록 seed간 다양성 줄고 selection이 더 deterministic.
        max_shot_in_experiment: nested 구조의 최대 shot 값 (보통 16).

    Returns:
        torch.utils.data.Subset
    """
    if shots <= 0:
        raise ValueError("shots는 1 이상이어야 함")

    # 빠른 random 경로 (teacher_data 불필요)
    if method.lower() == "random":
        return _random_subset(dataset, shots, seed)

    if teacher_data is None:
        raise ValueError(
            f"method='{method}' 에는 teacher_data가 필요함. "
            "_compute_teacher_scores()를 먼저 호출해 dict를 넘기세요."
        )

    labels = teacher_data["labels"]

    # 클래스별 인덱스 그룹화
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels.tolist()):
        label_to_indices[lbl].append(idx)

    rng = random.Random(seed)
    selected: List[int] = []

    for lbl in sorted(label_to_indices.keys()):
        class_idx = label_to_indices[lbl]
        if len(class_idx) < shots:
            raise ValueError(
                f"label={lbl} 이미지 수가 shots={shots}보다 적음. 현재 {len(class_idx)}장"
            )

        scores = _scores_for_method(method, teacher_data, class_idx)
        # score 내림차순으로 정렬
        sorted_idx_in_class = sorted(
            range(len(class_idx)),
            key=lambda i: float(scores[i].item()),
            reverse=True,
        )

        # top-K 후보 풀: score가 높은 K개
        K = max(max_shot_in_experiment, max_shot_in_experiment * candidate_factor)
        K = min(K, len(class_idx))
        candidate_local = sorted_idx_in_class[:K]

        # seed로 후보 풀 안에서 셔플 → nested 보장: shots개를 앞에서 잘라 사용
        # 같은 seed면 같은 셔플 결과 → 1-shot ⊂ 2-shot ⊂ 4-shot ... 자동
        rng_class = random.Random((seed * 1000003) ^ (int(lbl) + 1))
        rng_class.shuffle(candidate_local)

        chosen_local = candidate_local[:shots]
        selected.extend(class_idx[i] for i in chosen_local)

    rng.shuffle(selected)
    return Subset(dataset, selected)


def _random_subset(dataset: Dataset, shots: int, seed: int) -> Subset:
    """method='random' 경로. 기존 build_fewshot_subset과 동일한 동작."""
    # 라벨 추출 (Dataset 객체 형식에 따라 다름)
    if hasattr(dataset, "samples"):
        labels = [s[1] for s in dataset.samples]
    elif hasattr(dataset, "targets"):
        labels = list(dataset.targets)
    else:
        labels = [int(dataset[i][1]) for i in range(len(dataset))]

    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        label_to_indices[int(lbl)].append(idx)

    rng = random.Random(seed)
    selected = []
    for lbl in sorted(label_to_indices.keys()):
        copied = label_to_indices[lbl][:]
        rng.shuffle(copied)
        if len(copied) < shots:
            raise ValueError(f"label={lbl} 이미지 수가 shots={shots}보다 적음")
        selected.extend(copied[:shots])
    rng.shuffle(selected)
    return Subset(dataset, selected)
