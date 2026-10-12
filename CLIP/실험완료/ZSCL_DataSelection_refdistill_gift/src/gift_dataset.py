# -*- coding: utf-8 -*-
"""
GIFT-inspired synthetic image-text pair dataset loader.

이 모듈은 다음 디렉터리 구조를 가정함:

    <synthetic_root>/
    ├── images/
    │   ├── imagenet_goldfish_0000.png
    │   ├── ...
    │   └── dtd_banded_0000.png
    └── metadata.jsonl

`metadata.jsonl`는 줄당 한 개의 JSON 객체. 필수 필드:
    - "path": <synthetic_root>로부터의 상대 경로 또는 절대 경로
    - "class_name": prompt 친화적인 영문 class name
    - "prompt": 이미지 생성에 사용된 raw text prompt (e.g. "a photo of a goldfish.")

선택 필드:
    - "source": "base" / "target_<dataset_name>" 등 (logging 용)
    - "seed": 이미지 생성 seed (logging/재현성 용)

학습 코드는 prompt와 class_name이 매칭된 (image, text) pair를 사용하므로
metadata.jsonl이 반드시 일관되어야 함.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


_REQUIRED_FIELDS = ("path", "class_name", "prompt")


def _resolve_image_path(root: Path, rel_or_abs: str) -> Path:
    """metadata의 path를 root 기준으로 해석함."""
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return root / p


def load_metadata(metadata_path: str | Path) -> List[Dict]:
    """metadata.jsonl을 읽어 dict 리스트로 반환.

    오류 케이스:
    - 파일 자체가 없으면 FileNotFoundError
    - 한 줄이라도 JSON parse 실패 / 필수 필드 누락이면
      구체적인 줄번호와 함께 ValueError.

    선택 필드(source, seed)는 없어도 무시되며, 기본값(None)으로 채워짐.
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"synthetic metadata file을 찾을 수 없음: {metadata_path}\n"
            f"  먼저 scripts/generate_synthetic_dataset.py 등으로 metadata.jsonl을 생성하세요."
        )

    items: List[Dict] = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{metadata_path}:{lineno} JSON parse 실패: {e}\n  line='{line[:200]}'"
                ) from e

            missing = [k for k in _REQUIRED_FIELDS if k not in row]
            if missing:
                raise ValueError(
                    f"{metadata_path}:{lineno} 필수 필드 누락 {missing}. "
                    f"필요 필드={list(_REQUIRED_FIELDS)}, 받은 키={list(row.keys())}"
                )

            row.setdefault("source", None)
            row.setdefault("seed", None)
            items.append(row)

    if not items:
        raise ValueError(f"{metadata_path}에서 유효한 metadata 항목을 하나도 읽지 못함.")

    return items


class SyntheticImageTextDataset(Dataset):
    """Synthetic image-text pair dataset.

    각 sample은 다음을 반환:
        (image_tensor, class_name, prompt, source, idx)

    PyTorch DataLoader는 string을 기본 collate로 처리할 수 있으므로
    GIFTSyntheticDistiller가 직접 이 Dataset을 인덱싱하는 것을 권장 (Distiller가 미리
    이미지 feature를 캐싱하고, 학습 step에서는 인덱스만 사용함).

    Args:
        root: synthetic dataset 루트 디렉토리 (images/ 와 metadata.jsonl 포함).
        metadata_path: metadata.jsonl 경로 (None이면 root/metadata.jsonl).
        transform: CLIP preprocess.
        verify_paths: True면 init 시점에 모든 이미지 경로 존재 검증 (느릴 수 있음).
    """

    def __init__(
        self,
        root: str | Path,
        metadata_path: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
        verify_paths: bool = False,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"synthetic dataset root를 찾을 수 없음: {self.root}\n"
                f"  scripts/generate_synthetic_dataset.py 등으로 생성하세요."
            )

        if metadata_path is None:
            metadata_path = self.root / "metadata.jsonl"
        self.metadata_path = Path(metadata_path)

        self.items = load_metadata(self.metadata_path)
        self.transform = transform

        # 경로 사전 해석 (절대 경로화)
        self._resolved_paths: List[Path] = []
        for row in self.items:
            self._resolved_paths.append(_resolve_image_path(self.root, row["path"]))

        if verify_paths:
            missing = [str(p) for p in self._resolved_paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)}개의 synthetic image 경로가 실제로 존재하지 않음. "
                    f"예시: {missing[:3]}"
                )

        # 통계: distinct class / prompt / source
        self._unique_classes = sorted({row["class_name"] for row in self.items})
        self._unique_sources = sorted({str(row.get("source")) for row in self.items})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, str, Optional[str], int]:
        row = self.items[index]
        img_path = self._resolved_paths[index]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"synthetic image 로드 실패 (idx={index}, path={img_path}): {e}"
            ) from e

        if self.transform is not None:
            img = self.transform(img)

        return img, row["class_name"], row["prompt"], row.get("source"), index

    # -------- introspection helpers (logging 용) --------
    @property
    def num_images(self) -> int:
        return len(self.items)

    @property
    def num_unique_classes(self) -> int:
        return len(self._unique_classes)

    @property
    def unique_class_names(self) -> List[str]:
        return list(self._unique_classes)

    @property
    def unique_sources(self) -> List[str]:
        return list(self._unique_sources)

    def source_counts(self) -> Dict[str, int]:
        """source별 sample 수 집계 (None은 'unknown'으로 표기)."""
        counts: Dict[str, int] = {}
        for row in self.items:
            key = row.get("source")
            key = "unknown" if key is None else str(key)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def get_all_class_names(self) -> List[str]:
        """sample i와 인덱스가 동일한 class_name 리스트 (길이 N)."""
        return [row["class_name"] for row in self.items]

    def get_all_prompts(self) -> List[str]:
        """sample i와 인덱스가 동일한 prompt 리스트 (길이 N)."""
        return [row["prompt"] for row in self.items]


def synthetic_collate_skip_strings(batch):
    """
    학습 단계에서 GIFT 모듈은 이미 feature를 캐싱했기 때문에
    실제 DataLoader가 이 Dataset을 yield하는 경우는 거의 없지만, debugging 시 필요할 수 있음.

    PyTorch 기본 collate는 string list를 (default_collate가) 잘 처리하지만,
    혹시 collate 단계에서 텐서/문자열을 일관되게 다루고 싶을 때 사용.
    """
    images = torch.stack([b[0] for b in batch], dim=0)
    class_names = [b[1] for b in batch]
    prompts = [b[2] for b in batch]
    sources = [b[3] for b in batch]
    indices = torch.tensor([b[4] for b in batch], dtype=torch.long)
    return images, class_names, prompts, sources, indices
