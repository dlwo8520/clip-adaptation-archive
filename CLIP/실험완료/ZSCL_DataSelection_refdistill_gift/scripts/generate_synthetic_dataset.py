# -*- coding: utf-8 -*-
"""
GIFT-style synthetic image-text pair offline 생성 스크립트.

핵심 의도:
- **학습 코드는 Stable Diffusion에 의존하지 않음**.
- 이 스크립트로 미리 한 번 이미지를 생성/저장해 두고, 학습 시에는 그 폴더만 읽음.
- 또한 SD가 설치되지 않은 환경에서도 smoke test가 돌 수 있도록
  placeholder 모드 (deterministic 합성 이미지)를 기본으로 제공.

생성 디렉터리 구조:

    <out_dir>/
    ├── images/
    │   ├── base_goldfish_0000.png
    │   ├── base_goldfish_0001.png
    │   ├── target_texture_banded_0000.png
    │   └── ...
    └── metadata.jsonl

metadata.jsonl 의 한 줄:
    {"path": "images/base_goldfish_0000.png",
     "class_name": "goldfish",
     "prompt": "a photo of a goldfish.",
     "source": "base",
     "seed": 123}

사용 예 (placeholder 모드, SD 불필요):
    python scripts/generate_synthetic_dataset.py \
        --out-dir runs/synthetic_gift_smoke \
        --base-classnames-file gift_classnames/imagenet_classes_subset.txt \
        --target-classnames-file gift_classnames/dtd_classes.txt \
        --target-source-tag target_dtd \
        --num-per-class 4 \
        --mode placeholder

사용 예 (Stable Diffusion 모드, diffusers 필요):
    python scripts/generate_synthetic_dataset.py \
        --out-dir runs/synthetic_gift_full \
        --base-classnames-file gift_classnames/imagenet_classes_full.txt \
        --target-classnames-file gift_classnames/dtd_classes.txt \
        --target-source-tag target_dtd \
        --num-per-class 10 \
        --mode stable_diffusion \
        --sd-model runwayml/stable-diffusion-v1-5
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# -------------------------------------------------------------------- helpers


def safe_filename_part(name: str) -> str:
    """class name을 파일명에 쓸 수 있게 정리."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    cleaned = cleaned.strip("_")
    return cleaned[:80] if cleaned else "class"


def load_classnames_file(path: str | Path) -> List[str]:
    """한 줄 하나 class name 파일."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"classnames 파일을 찾을 수 없음: {p}")
    names: List[str] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    if not names:
        raise ValueError(f"classnames 파일이 비어 있음: {p}")
    return names


def build_prompt(class_name: str, template: str) -> str:
    """template에 class name을 끼워 넣어 prompt 완성."""
    return template.format(class_name)


def derive_seed(class_name: str, idx: int, base_seed: int) -> int:
    """class name + index + base_seed로 결정적 seed 생성 (재현성)."""
    key = f"{class_name}|{idx}|{base_seed}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16)


# --------------------------------------------------------------- placeholder


def generate_placeholder_image(
    width: int,
    height: int,
    seed: int,
    class_name: str,
) -> "np.ndarray":
    """SD 없이도 학습 파이프라인이 돌 수 있도록 만든 deterministic synthetic 이미지.

    - class name으로부터 base color tone을 결정 (sha256 → 3 byte → RGB)
    - seed로부터 perlin-like gradient noise 패턴을 추가
    - 결과: [H, W, 3] uint8

    학습/평가 성능을 의도하지 않음. 코드 경로 (loader/distiller)가 깨지지 않는지
    smoke test 용. 본격 실험을 위해서는 --mode stable_diffusion 권장.
    """
    rng = np.random.default_rng(seed)

    # class color (deterministic from class name)
    h = hashlib.sha256(class_name.encode("utf-8")).digest()
    base_color = np.array([h[0], h[1], h[2]], dtype=np.float32)

    # gradient noise: 두 개의 random sinusoid를 합성
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = xx / max(1, width)
    ny = yy / max(1, height)

    freq1 = float(rng.uniform(2.0, 8.0))
    freq2 = float(rng.uniform(2.0, 8.0))
    phase1 = float(rng.uniform(0.0, 2 * np.pi))
    phase2 = float(rng.uniform(0.0, 2 * np.pi))

    pattern_r = 0.5 + 0.5 * np.sin(2 * np.pi * freq1 * nx + phase1)
    pattern_g = 0.5 + 0.5 * np.sin(2 * np.pi * freq2 * ny + phase2)
    pattern_b = 0.5 + 0.5 * np.sin(2 * np.pi * (freq1 + freq2) * (nx + ny) * 0.5 + phase1 + phase2)

    # 살짝 noise 추가
    noise = rng.normal(loc=0.0, scale=0.05, size=(height, width, 3)).astype(np.float32)

    base = np.stack([pattern_r, pattern_g, pattern_b], axis=-1)  # [H, W, 3], 0~1
    img = base_color[None, None, :] * 0.5 + base * 255.0 * 0.5
    img = np.clip(img + noise * 255.0, 0, 255).astype(np.uint8)
    return img


def save_image(arr: "np.ndarray", path: Path) -> None:
    from PIL import Image
    Image.fromarray(arr).save(path, format="PNG")


# --------------------------------------------------------------- Stable Diffusion


def try_load_sd_pipeline(sd_model: str, device: str = "cuda"):
    """diffusers Stable Diffusion pipeline 로딩. 실패 시 RuntimeError로 변환."""
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except Exception as e:
        raise RuntimeError(
            f"diffusers / torch import 실패: {e}\n"
            f"  --mode stable_diffusion 을 쓰려면 pip install diffusers transformers accelerate safetensors 필요."
        ) from e

    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    try:
        pipe = StableDiffusionPipeline.from_pretrained(sd_model, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
        pipe.safety_checker = None  # warn-only
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        return pipe
    except Exception as e:
        raise RuntimeError(
            f"Stable Diffusion pipeline 로딩 실패 (model='{sd_model}', device='{device}'): {e}"
        ) from e


def generate_sd_image(
    pipe,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
):
    """SD pipeline으로 이미지 한 장 생성. PIL.Image 반환."""
    import torch

    device_str = str(pipe.device)
    if device_str.startswith("cuda"):
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
    else:
        generator = torch.Generator().manual_seed(int(seed))

    with torch.no_grad():
        result = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
    return result.images[0]


# --------------------------------------------------------------- main loop


def iter_class_specs(
    base_classnames: List[str],
    target_classnames: List[str],
    target_source_tag: str,
    base_template: str,
    target_template: str,
    num_per_class_base: int,
    num_per_class_target: int,
) -> List[Tuple[str, str, str, str, int]]:
    """(class_name, prompt, source, file_prefix, num_per_class) 를 나열함."""
    specs: List[Tuple[str, str, str, str, int]] = []
    for cname in base_classnames:
        prompt = build_prompt(cname, base_template)
        prefix = f"base_{safe_filename_part(cname)}"
        specs.append((cname, prompt, "base", prefix, num_per_class_base))
    for cname in target_classnames:
        prompt = build_prompt(cname, target_template)
        tag = safe_filename_part(target_source_tag)
        prefix = f"{tag}_{safe_filename_part(cname)}"
        specs.append((cname, prompt, target_source_tag, prefix, num_per_class_target))
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True,
                        help="생성된 이미지/metadata를 저장할 루트.")
    parser.add_argument("--base-classnames-file", type=str, required=True,
                        help="pre-training knowledge approx. 용 class name 파일 "
                             "(예: ImageNet 1000-class 또는 그 subset).")
    parser.add_argument("--target-classnames-file", type=str, default=None,
                        help="downstream target class name 파일 (DTD 47-class 등). "
                             "None이면 target 합성을 생략하고 base만 만듦.")
    parser.add_argument("--target-source-tag", type=str, default="target",
                        help="metadata.jsonl 의 'source' 필드 값. 보통 target_<target_name>.")
    parser.add_argument("--base-template", type=str, default="a photo of a {}.")
    parser.add_argument("--target-template", type=str, default="a photo of a {}.")
    parser.add_argument("--num-per-class", type=int, default=4,
                        help="(default) base/target class 마다 합성할 이미지 수.")
    parser.add_argument("--num-per-class-base", type=int, default=None,
                        help="base만 따로 지정 (None이면 --num-per-class 값 사용).")
    parser.add_argument("--num-per-class-target", type=int, default=None,
                        help="target만 따로 지정 (None이면 --num-per-class 값 사용).")
    parser.add_argument("--base-seed", type=int, default=2024,
                        help="seed 베이스 값. 실제 image seed = sha256(class|idx|base_seed)으로 결정.")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--mode", type=str, default="placeholder",
                        choices=["placeholder", "stable_diffusion"],
                        help="placeholder: SD 불필요 (deterministic 합성). "
                             "stable_diffusion: diffusers 필요.")
    # SD 옵션
    parser.add_argument("--sd-model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--sd-device", type=str, default="cuda")
    parser.add_argument("--sd-steps", type=int, default=30)
    parser.add_argument("--sd-guidance", type=float, default=7.5)

    parser.add_argument("--overwrite", action="store_true",
                        help="기존 metadata.jsonl을 지우고 새로 만듦. 기본은 append 안전 모드(append 안 함; 존재하면 종료).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    metadata_path = out_dir / "metadata.jsonl"
    images_dir.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists() and not args.overwrite:
        print(f"[error] metadata.jsonl이 이미 존재: {metadata_path}. "
              f"--overwrite 로 덮어쓰거나 다른 --out-dir 사용.", file=sys.stderr)
        sys.exit(2)
    if metadata_path.exists() and args.overwrite:
        metadata_path.unlink()

    base_classnames = load_classnames_file(args.base_classnames_file)
    if args.target_classnames_file is not None:
        target_classnames = load_classnames_file(args.target_classnames_file)
    else:
        target_classnames = []

    n_base = args.num_per_class_base if args.num_per_class_base is not None else args.num_per_class
    n_target = args.num_per_class_target if args.num_per_class_target is not None else args.num_per_class

    specs = iter_class_specs(
        base_classnames=base_classnames,
        target_classnames=target_classnames,
        target_source_tag=args.target_source_tag,
        base_template=args.base_template,
        target_template=args.target_template,
        num_per_class_base=n_base,
        num_per_class_target=n_target,
    )

    total = sum(num for _, _, _, _, num in specs)
    print(f"===== synthetic dataset 생성 시작 =====")
    print(f"  mode             : {args.mode}")
    print(f"  out_dir          : {out_dir}")
    print(f"  base classes     : {len(base_classnames)} × {n_base} = {len(base_classnames) * n_base}")
    print(f"  target classes   : {len(target_classnames)} × {n_target} = {len(target_classnames) * n_target}")
    print(f"  total images     : {total}")
    print(f"  size             : {args.width}x{args.height}")

    # SD pipeline 로딩 (placeholder면 None)
    sd_pipe = None
    if args.mode == "stable_diffusion":
        print(f"  loading SD pipeline: {args.sd_model} (device={args.sd_device})")
        sd_pipe = try_load_sd_pipeline(args.sd_model, device=args.sd_device)

    # 생성 루프
    written = 0
    with open(metadata_path, "w", encoding="utf-8") as meta_f:
        for (class_name, prompt, source, prefix, num_per) in specs:
            for i in range(num_per):
                seed = derive_seed(class_name, i, args.base_seed)
                img_filename = f"{prefix}_{i:04d}.png"
                img_path = images_dir / img_filename

                if args.mode == "placeholder":
                    arr = generate_placeholder_image(
                        width=args.width, height=args.height, seed=seed, class_name=class_name
                    )
                    save_image(arr, img_path)
                else:
                    img = generate_sd_image(
                        sd_pipe,
                        prompt=prompt,
                        seed=seed,
                        width=args.width,
                        height=args.height,
                        num_inference_steps=args.sd_steps,
                        guidance_scale=args.sd_guidance,
                    )
                    img.save(img_path, format="PNG")

                record = {
                    "path": f"images/{img_filename}",
                    "class_name": class_name,
                    "prompt": prompt,
                    "source": source,
                    "seed": int(seed),
                }
                meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

                if written % max(1, total // 20) == 0 or written == total:
                    print(f"  [{written:6d}/{total}] last={img_path.name}")

    print(f"===== 완료 =====")
    print(f"  written {written} images")
    print(f"  metadata: {metadata_path}")
    print(f"  images  : {images_dir}")


if __name__ == "__main__":
    main()
