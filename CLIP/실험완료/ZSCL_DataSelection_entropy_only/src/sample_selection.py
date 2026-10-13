# -*- coding: utf-8 -*-
"""
Entropy-only support sample selection for CoOp few-shot adaptation.

목적:
- 기존 teacher_uncertain = -confidence 방식 대신 entropy 기준으로 불확실한 샘플을 선택함.
- margin, prototype, confidence 기반 selection은 제거함.
- 같은 sample_seed에서 1-shot ⊂ 2-shot ⊂ 4-shot ⊂ 8-shot ⊂ 16-shot nested 구조를 유지함.

선택 방법:
- teacher_entropy: Teacher CLIP zero-shot softmax 분포의 entropy가 큰 샘플 선택

Entropy 의미:
- softmax 분포가 한 클래스에 몰리면 entropy가 낮음.
- 여러 클래스에 넓게 퍼져 있으면 entropy가 높음.
- 따라서 teacher_entropy는 CLIP teacher가 더 헷갈리는 샘플을 우선 후보로 둠.
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
    desc: str = "compute teacher entropy scores",
) -> Dict[str, torch.Tensor]:
    """원본 CLIP teacher로 dataset 내 모든 이미지의 entropy score를 계산함.

    Returns:
        labels:             [N] long, 정답 라벨
        entropy:            [N] float, teacher softmax 분포 entropy
        normalized_entropy: [N] float, entropy / log(num_classes)
        confidence:         [N] float, 정답 클래스 softmax probability. 분석용으로만 저장함.
    """
    model.eval()

    # 1) text feature 준비: template별 text feature를 평균냄.
    text_feats = []
    for cname in classnames:
        clean = cname.replace("_", " ").replace("-", " ")
        prompts = [template.format(clean) for template in templates]
        tokenized = tokenizer(prompts).to(device)

        text_emb = model.encode_text(tokenized)
        text_emb = F.normalize(text_emb, dim=-1)
        text_emb = text_emb.mean(dim=0)
        text_emb = F.normalize(text_emb, dim=-1)
        text_feats.append(text_emb)

    text_features = torch.stack(text_feats, dim=0)  # [C, D]
    num_classes = text_features.shape[0]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logit_scale = model.logit_scale.exp()

    all_labels = []
    all_entropy = []
    all_normalized_entropy = []
    all_confidence = []

    for images, targets in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        sim = image_features @ text_features.T
        logits = logit_scale * sim
        probs = logits.softmax(dim=-1)

        # entropy = -sum_c p(c|x) log p(c|x)
        # clamp는 log(0) 방지용. 확률값을 실험적으로 바꾸려는 의도는 아님.
        safe_probs = probs.clamp_min(1e-12)
        entropy = -(safe_probs * safe_probs.log()).sum(dim=-1)
        normalized_entropy = entropy / torch.log(
            torch.tensor(float(num_classes), device=device)
        )

        gt_confidence = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        all_labels.append(targets.cpu())
        all_entropy.append(entropy.cpu())
        all_normalized_entropy.append(normalized_entropy.cpu())
        all_confidence.append(gt_confidence.cpu())

    return {
        "labels": torch.cat(all_labels, dim=0),
        "entropy": torch.cat(all_entropy, dim=0),
        "normalized_entropy": torch.cat(all_normalized_entropy, dim=0),
        "confidence": torch.cat(all_confidence, dim=0),
    }


def _scores_for_method(
    method: str,
    teacher_data: Dict[str, torch.Tensor],
    class_indices: List[int],
) -> torch.Tensor:
    """주어진 method에 대해 한 클래스 내 샘플들의 점수를 반환함.

    점수가 높을수록 선택 우선순위가 높음.
    """
    method = method.lower()

    if method == "teacher_entropy":
        return teacher_data["entropy"][class_indices]

    raise ValueError(
        f"알 수 없는 selection method: {method}. "
        "이 코드팩은 entropy-only 버전이므로 --selection-method teacher_entropy만 사용하세요."
    )


def build_fewshot_subset_zscl(
    dataset: Dataset,
    shots: int,
    seed: int,
    method: str = "teacher_entropy",
    teacher_data: Optional[Dict[str, torch.Tensor]] = None,
    candidate_factor: int = 2,
    max_shot_in_experiment: int = 16,
) -> Subset:
    """Entropy-only few-shot subset 구성.

    Args:
        dataset: target train pool
        shots: 클래스당 선택할 이미지 수
        seed: 후보 풀 내부 shuffle에 사용하는 sample_seed
        method: teacher_entropy만 지원
        teacher_data: _compute_teacher_scores() 결과
        candidate_factor: 클래스당 score top-K 후보 풀 크기 = max_shot * factor
        max_shot_in_experiment: nested 구조의 최대 shot 값. 보통 16.

    Returns:
        torch.utils.data.Subset
    """
    if shots <= 0:
        raise ValueError("shots는 1 이상이어야 함")

    if method.lower() != "teacher_entropy":
        raise ValueError(
            f"method={method}는 지원하지 않음. "
            "entropy-only 실험에서는 --selection-method teacher_entropy만 사용하세요."
        )

    if teacher_data is None:
        raise ValueError("teacher_entropy에는 teacher_data가 필요함")

    labels = teacher_data["labels"]

    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        label_to_indices[int(label)].append(idx)

    selected: List[int] = []

    for label in sorted(label_to_indices.keys()):
        class_indices = label_to_indices[label]
        if len(class_indices) < shots:
            raise ValueError(
                f"label={label} 이미지 수가 shots={shots}보다 적음. 현재 {len(class_indices)}장"
            )

        scores = _scores_for_method("teacher_entropy", teacher_data, class_indices)

        # entropy 내림차순. entropy가 큰 샘플일수록 teacher가 더 헷갈리는 샘플임.
        sorted_local = sorted(
            range(len(class_indices)),
            key=lambda i: float(scores[i].item()),
            reverse=True,
        )

        # 후보 풀을 너무 작게 잡으면 seed별 다양성이 거의 사라짐.
        # 기존 코드와 맞추기 위해 max_shot * candidate_factor 구조 유지.
        candidate_k = max(max_shot_in_experiment, max_shot_in_experiment * candidate_factor)
        candidate_k = min(candidate_k, len(class_indices))
        candidate_local = sorted_local[:candidate_k]

        # 같은 sample_seed에서는 shot별 nested 구조가 유지됨.
        rng_class = random.Random((seed * 1000003) ^ (int(label) + 1))
        rng_class.shuffle(candidate_local)

        chosen_local = candidate_local[:shots]
        selected.extend(class_indices[i] for i in chosen_local)

    rng = random.Random(seed)
    rng.shuffle(selected)
    return Subset(dataset, selected)
