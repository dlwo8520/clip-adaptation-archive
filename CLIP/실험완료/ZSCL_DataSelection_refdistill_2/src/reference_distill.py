# -*- coding: utf-8 -*-
"""
Reference dataset 기반 distillation loss for CoOp.

이론적 근거 (Zheng et al., CVPR 2023, "Preventing Zero-Shot Transfer Degradation
in Continual Learning of Vision-Language Models", ZSCL):

- ZSCL은 CLIP을 downstream task로 학습할 때, "원본 CLIP zero-shot transfer 능력"이
  catastrophic forgetting 형태로 망가지는 것을 방지하기 위한 distillation 기법을 제안함.
- Reference dataset (label/text 매칭 불필요, semantic diversity만 충분하면 됨)에서
  원본 CLIP teacher와 현재 학습 중인 student의 (image, text) similarity 분포를
  맞추는 것이 핵심 (논문 Fig. 2, Eq. (3), (4)).

본 모듈에서의 적용:

- CoOp 셋팅에서 image encoder는 freeze이고 ctx만 학습됨.
- Teacher: 원본 CLIP에 hand-crafted template ("a photo of a {c}.") 적용한 zero-shot.
- Student: 동일한 CLIP backbone에 학습 중인 CoOp ctx를 적용한 prompt
  ("[V1]...[VM] {c}.")로 만든 text feature.
- Image feature는 두 모델에서 동일하므로 한 번만 캐싱.
- 매 step마다 reference image/class를 sampling해서 두 모델의 logit 분포 KL을 맞춤
  (image-perspective + text-perspective).

본 모듈은 학습용이며, 평가에는 사용하지 않음.
"""

import random
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


def _clean(name: str) -> str:
    """class name을 prompt 친화적으로 정리함."""
    return name.replace("_", " ").replace("-", " ")


class CoOpReferenceDistiller:
    """
    Reference (image, classname) 풀에서 매 step subset을 뽑아
    teacher (zero-shot CLIP)와 student (CoOp) 의 similarity 분포를 맞추는 distiller.

    사용 흐름:
        distiller = CoOpReferenceDistiller(
            model=clip_model,
            tokenizer=tokenizer,
            reference_dataset=ImageFolder(...),
            reference_classnames=[...],
            reference_templates=["a photo of a {}."],
            device=device,
            max_reference_images=10000,
        )

        # 매 step:
        loss_distill = distiller.compute_distill_loss(
            prompt_learner=prompt_learner,
            build_coop_text_features_fn=build_coop_text_features,
            num_image_samples=64,
            num_class_samples=64,
        )
        loss_total = loss_ce + lambda_distill * loss_distill

    설계 메모:
    - image feature는 image encoder가 freeze이므로 (teacher == student) 미리 캐싱.
    - teacher text feature도 학습 중 변하지 않으므로 한 번만 캐싱.
    - student text feature만 매 step 새로 계산 (이 부분에 gradient가 흐름).
    - reference image/class 수 N, C가 크면 OOM 위험이 있어 매 step subset만 사용.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        reference_dataset: Dataset,
        reference_classnames: List[str],
        reference_templates: List[str],
        device: torch.device,
        precompute_batch_size: int = 256,
        precompute_num_workers: int = 4,
        max_reference_images: Optional[int] = None,
        precompute_seed: int = 0,
        store_image_features_on_cpu: bool = False,
    ):
        if len(reference_classnames) == 0:
            raise ValueError("reference_classnames가 비어 있음")
        if len(reference_templates) == 0:
            raise ValueError("reference_templates가 비어 있음")

        self.model = model
        self.tokenizer = tokenizer
        self.reference_classnames = list(reference_classnames)
        self.reference_templates = list(reference_templates)
        self.device = device
        self.store_image_features_on_cpu = store_image_features_on_cpu

        # ===== reference image pool subsampling =====
        if (
            max_reference_images is not None
            and max_reference_images > 0
            and max_reference_images < len(reference_dataset)
        ):
            rng = random.Random(precompute_seed)
            all_idx = list(range(len(reference_dataset)))
            rng.shuffle(all_idx)
            reference_dataset = Subset(reference_dataset, all_idx[:max_reference_images])
            print(
                f"[reference distiller] subsampled {max_reference_images}/{len(all_idx)} "
                f"reference images with seed={precompute_seed}"
            )

        self.num_images = len(reference_dataset)
        self.num_classes = len(self.reference_classnames)

        if self.num_images == 0:
            raise ValueError("reference dataset이 비어 있음")

        # ===== teacher image features 캐싱 (frozen이므로 한 번만) =====
        print(f"[reference distiller] image feature 캐싱 시작 (N={self.num_images})")
        self._image_features = self._precompute_image_features(
            reference_dataset,
            batch_size=precompute_batch_size,
            num_workers=precompute_num_workers,
        )
        print(
            f"[reference distiller] image feature shape={tuple(self._image_features.shape)}, "
            f"on_cpu={self.store_image_features_on_cpu}"
        )

        # ===== teacher text features 캐싱 (frozen CLIP, template 평균) =====
        print(f"[reference distiller] teacher text feature 캐싱 시작 (C={self.num_classes})")
        self._teacher_text_features = self._precompute_teacher_text_features()
        print(
            f"[reference distiller] teacher text feature shape="
            f"{tuple(self._teacher_text_features.shape)}"
        )

        # logit scale (CoOp에서도 freeze 상태이므로 한 번만 가져옴)
        self._logit_scale_value = float(model.logit_scale.exp().detach().cpu().item())

    @torch.no_grad()
    def _precompute_image_features(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
    ) -> torch.Tensor:
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        feats = []
        for images, _ in tqdm(loader, desc="reference image feats"):
            images = images.to(self.device, non_blocking=True)
            f = self.model.encode_image(images)
            f = F.normalize(f, dim=-1)
            if self.store_image_features_on_cpu:
                feats.append(f.detach().cpu())
            else:
                feats.append(f.detach())
        out = torch.cat(feats, dim=0)
        return out

    @torch.no_grad()
    def _precompute_teacher_text_features(self) -> torch.Tensor:
        """원본 CLIP에 template를 적용해서 reference classname별 text feature를 만듦."""
        self.model.eval()
        all_feats = []
        for cname in self.reference_classnames:
            clean = _clean(cname)
            prompts = [t.format(clean) for t in self.reference_templates]
            tokens = self.tokenizer(prompts).to(self.device)
            emb = self.model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
            class_feat = emb.mean(dim=0)
            class_feat = F.normalize(class_feat, dim=0)
            all_feats.append(class_feat)
        out = torch.stack(all_feats, dim=0).detach()  # [C, D]
        return out

    def _sample_indices(self, num_image_samples: int, num_class_samples: int):
        """매 step에 사용할 reference image/class index를 sampling함."""
        n_img = min(int(num_image_samples), self.num_images)
        n_cls = min(int(num_class_samples), self.num_classes)
        img_idx = torch.randint(0, self.num_images, (n_img,), device=self.device)
        cls_idx = torch.randint(0, self.num_classes, (n_cls,), device=self.device)
        return img_idx, cls_idx

    def compute_distill_loss(
        self,
        prompt_learner,
        build_coop_text_features_fn: Callable,
        num_image_samples: int = 64,
        num_class_samples: int = 64,
    ) -> torch.Tensor:
        """
        ZSCL Eq. (3), (4)에 대응하는 image-perspective + text-perspective distillation.

        - L_img: 각 reference image i에 대해
                 p_teacher(c | i) vs p_student(c | i) 분포 매칭.
        - L_txt: 각 reference class c에 대해
                 p_teacher(i | c) vs p_student(i | c) 분포 매칭.

        Returns:
            scalar tensor (= L_img + L_txt). 외부에서 lambda_distill로 곱해서 더하면 됨.
        """
        img_idx, cls_idx = self._sample_indices(num_image_samples, num_class_samples)

        # image features: 캐싱된 것을 index_select. CPU에 있다면 device로 옮김.
        img_feats_src = self._image_features.index_select(0, img_idx.to(self._image_features.device))
        img_feats = img_feats_src.to(self.device, non_blocking=True)  # [B, D]

        # teacher text feature (frozen): cls_idx로 select
        teacher_txt = self._teacher_text_features.index_select(
            0, cls_idx.to(self._teacher_text_features.device)
        ).to(self.device, non_blocking=True)  # [C', D]

        # student text feature: 현재 CoOp ctx를 사용해서 같은 classnames에 대해 build
        sampled_classnames = [self.reference_classnames[int(i.item())] for i in cls_idx]
        student_txt = build_coop_text_features_fn(
            self.model, prompt_learner, sampled_classnames, self.device
        )  # [C', D]

        scale = self._logit_scale_value

        # similarity matrices
        # teacher 쪽은 gradient 흐름 차단
        teacher_logits = (scale * img_feats @ teacher_txt.t()).detach()  # [B, C']
        student_logits = scale * img_feats @ student_txt.t()             # [B, C']

        # ===== L_img: image-perspective =====
        # ZSCL Eq.(3): CE(p, log p_bar) = -sum_j p_j log p_bar_j
        # 여기서 p = teacher prob (target), log p_bar = student log_softmax.
        teacher_p_img = F.softmax(teacher_logits, dim=-1)            # [B, C']
        student_logp_img = F.log_softmax(student_logits, dim=-1)     # [B, C']
        loss_img = -(teacher_p_img * student_logp_img).sum(dim=-1).mean()

        # ===== L_txt: text-perspective =====
        teacher_logits_t = teacher_logits.t()                        # [C', B]
        student_logits_t = student_logits.t()                        # [C', B]
        teacher_p_txt = F.softmax(teacher_logits_t, dim=-1)
        student_logp_txt = F.log_softmax(student_logits_t, dim=-1)
        loss_txt = -(teacher_p_txt * student_logp_txt).sum(dim=-1).mean()

        return loss_img + loss_txt


def compute_weight_consolidation_loss(
    prompt_learner,
    base_ctx: torch.Tensor,
) -> torch.Tensor:
    """
    ZSCL Eq.(5) Weight Consolidation을 CoOp ctx에 적용함.

    L_WC = sum_i (ctx_i - base_ctx_i)^2

    CoOp은 ctx만 학습 가능하므로, base_ctx (학습 시작 시점의 ctx 복사본)와의 L2 거리.
    이 값에 외부에서 lambda_wc를 곱해 total loss에 더하면 됨.

    주의:
    - base_ctx는 학습 시작 직후 detach + clone해서 보관해야 함.
    - base_ctx가 ctx와 같은 device가 아니면 자동으로 옮김.
    """
    diff = prompt_learner.ctx - base_ctx.to(
        device=prompt_learner.ctx.device, dtype=prompt_learner.ctx.dtype
    )
    return (diff ** 2).sum()
