# -*- coding: utf-8 -*-
"""
공통 보조 함수 모음
"""
import json
import random
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch

def seed_everything(seed: int) -> None:
    """실험 재현성을 위해 random seed를 고정함"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    """폴더가 없으면 생성함"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(obj: Any, path: str | Path) -> None:
    """객체를 json 파일로 저장함"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def append_jsonl(obj: Dict[str, Any], path: str | Path) -> None:
    """jsonl 파일에 한 줄 추가함"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_classnames_file(path: str | Path):
    """한 줄에 하나씩 들어간 class name 파일을 읽음"""
    path = Path(path)
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(line)
    return names
