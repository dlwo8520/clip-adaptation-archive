# -*- coding: utf-8 -*-
"""
데이터셋 로딩, split, few-shot subset 관련 함수

지원 구조:
1. root/train/class/image.png + root/test/class/image.png
2. root/class/image.png 형태만 있는 경우 자동으로 train pool/test split 생성
"""
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

def has_train_test_split(root: str | Path) -> bool:
    """root 아래 train/test 폴더가 있는지 확인함"""
    root = Path(root)
    return (root / "train").is_dir() and (root / "test").is_dir()

def make_classwise_split_indices(dataset: ImageFolder, test_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """root/class/image.png 구조에서 class별로 train/test index를 나눔"""
    rng = random.Random(seed)
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(idx)

    train_indices, test_indices = [], []
    for label, indices in sorted(label_to_indices.items()):
        copied = indices[:]
        rng.shuffle(copied)
        n_test = max(1, int(len(copied) * test_ratio))
        test_part = copied[:n_test]
        train_part = copied[n_test:]
        if len(train_part) == 0:
            train_part = test_part[:1]
            test_part = test_part[1:]
        train_indices.extend(train_part)
        test_indices.extend(test_part)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices

def build_target_datasets(
    target_root: Optional[str],
    target_train: Optional[str],
    target_test: Optional[str],
    transform,
    split_test_ratio: float,
    seed: int,
):
    """target dataset의 train pool과 test dataset을 생성함"""
    if target_train is not None and target_test is not None:
        train_dataset = ImageFolder(target_train, transform=transform)
        test_dataset = ImageFolder(target_test, transform=transform)
        if train_dataset.classes != test_dataset.classes:
            raise ValueError("target train/test class 목록이 다름")
        return train_dataset, test_dataset, train_dataset.classes, "explicit_train_test"

    if target_root is None:
        raise ValueError("--target-root 또는 --target-train/--target-test 중 하나는 필요함")

    root = Path(target_root)
    if has_train_test_split(root):
        train_dataset = ImageFolder(root / "train", transform=transform)
        test_dataset = ImageFolder(root / "test", transform=transform)
        if train_dataset.classes != test_dataset.classes:
            raise ValueError("target root의 train/test class 목록이 다름")
        return train_dataset, test_dataset, train_dataset.classes, "root_train_test"

    full_dataset = ImageFolder(root, transform=transform)
    train_indices, test_indices = make_classwise_split_indices(full_dataset, split_test_ratio, seed)
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    return train_dataset, test_dataset, full_dataset.classes, "auto_classwise_split"

def _fast_label_extract(dataset: Dataset) -> List[int]:
    """
    이미지 디코딩 없이 라벨 리스트를 빠르게 얻음.
    ImageFolder의 .samples (path, label) 또는 .targets를 활용.
    """
    if isinstance(dataset, Subset):
        underlying = dataset.dataset
        if hasattr(underlying, "samples"):
            return [int(underlying.samples[idx][1]) for idx in dataset.indices]
        if hasattr(underlying, "targets"):
            return [int(underlying.targets[idx]) for idx in dataset.indices]
    if hasattr(dataset, "samples"):
        return [int(s[1]) for s in dataset.samples]
    if hasattr(dataset, "targets"):
        return [int(t) for t in dataset.targets]
    # 마지막 수단: 느린 경로 (이미지 디코딩 발생)
    print("[경고] dataset에 .samples/.targets가 없어 느린 경로로 라벨 추출 중")
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(int(label))
    return labels


def build_fewshot_subset(dataset: Dataset, shots: int, seed: int) -> Subset:
    """train pool dataset에서 class마다 shots개씩 선택함"""
    if shots <= 0:
        raise ValueError("shots는 1 이상이어야 함")
    rng = random.Random(seed)
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    labels = _fast_label_extract(dataset)
    for local_idx, label in enumerate(labels):
        label_to_indices[label].append(local_idx)

    selected = []
    for label, indices in sorted(label_to_indices.items()):
        copied = indices[:]
        rng.shuffle(copied)
        if len(copied) < shots:
            raise ValueError(f"label={label} 이미지 수가 shots={shots}보다 적음. 현재 {len(copied)}장")
        selected.extend(copied[:shots])
    rng.shuffle(selected)
    return Subset(dataset, selected)

def build_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, max_samples: int = -1) -> DataLoader:
    """Dataset에서 DataLoader를 생성함"""
    if max_samples > 0:
        indices = list(range(min(max_samples, len(dataset))))
        dataset = Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

def build_preserve_dataset(path: str, transform):
    """preserve 평가용 ImageFolder dataset을 생성함"""
    dataset = ImageFolder(path, transform=transform)
    return dataset, dataset.classes

def maybe_replace_imagenet_wnid_with_torchvision_classes(dataset_name: str, classnames: List[str]) -> List[str]:
    """
    ImageNet 폴더명이 n01440764 같은 WordNet ID이면 prompt로 부적절함.
    torchvision ResNet weight metadata의 1000 class name으로 교체함.
    """
    lower_name = dataset_name.lower()
    looks_like_imagenet = "imagenet" in lower_name
    looks_like_wnid = len(classnames) == 1000 and all(c.startswith("n") for c in classnames[:10])
    if not (looks_like_imagenet and looks_like_wnid):
        return classnames
    try:
        from torchvision.models import ResNet50_Weights
        categories = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
        if len(categories) == 1000:
            print("[경고] ImageNet class folder가 WordNet ID처럼 보여 torchvision class name으로 교체함")
            print("[경고] class index 매핑이 서버 폴더 순서와 다르면 ImageNet accuracy가 왜곡될 수 있음")
            return list(categories)
    except Exception as e:
        print(f"[경고] torchvision ImageNet class name 로드 실패: {e}")
    return classnames
