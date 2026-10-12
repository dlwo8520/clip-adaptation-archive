# -*- coding: utf-8 -*-
"""
GIFT-inspired synthetic-pair contrastive distillation for CoOp continual learning.

근거 논문 (이 모듈은 reproduction이 아니라 inspired 구현):
    Wu, Shi, Wang, Ye, "Synthetic Data is an Elegant GIFT for Continual
    Vision-Language Models", CVPR 2025.

원본 GIFT vs 본 구현의 차이:
- 원본은 CLIP 전체를 fine-tune하고, teacher = θ^{t-1} (이전 task 모델).
- 본 구현은 CoOp prompt tuning이므로 CLIP 자체는 freeze이고
  학습 파라미터는 ctx (n_ctx × ctx_dim ≈ 16×512)뿐.
  Teacher는 원본 CLIP + raw prompt template, Student는 동일 CLIP backbone +
  현재 학습 중인 CoOp ctx 적용된 prompt.

핵심 loss 구성 (한 step 기준, batch B = synthetic matched pair 수):

    M_teacher[i,j] = teacher_img_feat[i] · teacher_txt_feat[j]   (B×B, frozen)
    M_student[i,j] = student_img_feat[i] · student_txt_feat[j]   (B×B, ctx로부터 grad)

    L_KD_image = -mean_i Σ_j P_teacher_row[i,j] · log P_student_row[i,j]
    L_KD_text  = -mean_j Σ_i P_teacher_col[i,j] · log P_student_col[i,j]
    L_CD = L_KD_image + L_KD_text

    L_align_img = CE(M_student rows,  labels=arange(B))   # 대각이 positive
    L_align_txt = CE(M_student cols,  labels=arange(B))
    L_ITA = L_align_img + L_align_txt

    L_GIFT = α · L_CD + β · L_ITA

선택 사항:
    L_AWC = Σ_i Fisher_i · (ctx_i - ctx_ref_i)^2
    Fisher_i ≈ (∂L_GIFT / ∂ctx_i)^2  (detach 후 weighting으로만 사용)

설계 메모:
- image encoder가 freeze이므로 image feature는 한 번만 캐싱.
  teacher_img_feat == student_img_feat 이고 양쪽 모두 grad 없음.
- teacher text feature도 frozen CLIP + raw prompt이므로 한 번만 캐싱.
- student text feature만 매 step CoOp ctx로부터 새로 계산 (gradient 흐름).
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def _clean(name: str) -> str:
    """class name을 prompt 친화적으로 정리함 (e.g. 'lawn_chair' -> 'lawn chair')."""
    return name.replace("_", " ").replace("-", " ")


class GIFTSyntheticDistiller:
    """
    Synthetic matched (image, prompt) pair pool에서 매 step batch B를 sampling해서
    teacher (원본 CLIP, raw prompt)와 student (CoOp ctx 적용된 prompt)의
    유사도 행렬 분포를 맞추는 distiller.

    사용 흐름:
        distiller = GIFTSyntheticDistiller(
            model=clip_model,
            tokenizer=tokenizer,
            synthetic_dataset=SyntheticImageTextDataset(...),
            device=device,
            precompute_batch_size=128,
        )

        # 매 step:
        losses = distiller.compute_gift_losses(
            prompt_learner=prompt_learner,
            build_coop_text_features_fn=build_coop_text_features,
            batch_size=64,
            return_individual=True,
        )
        loss = loss_ce + alpha * losses["loss_cd"] + beta * losses["loss_ita"]

    Args:
        model: CLIP model (freeze된 상태로 사용).
        tokenizer: open_clip tokenizer.
        synthetic_dataset: SyntheticImageTextDataset 인스턴스.
        device: torch device.
        precompute_batch_size / precompute_num_workers: feature 캐싱용 DataLoader 설정.
        max_synthetic_images: cache할 image 수 상한 (None/<=0이면 전부 사용).
        precompute_seed: subsampling용 seed (다른 sample_seed 간 reference 풀 고정 권장).
        store_features_on_cpu: image/text feature를 GPU 대신 CPU에 저장 (메모리 절약).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        synthetic_dataset,  # SyntheticImageTextDataset
        device: torch.device,
        precompute_batch_size: int = 128,
        precompute_num_workers: int = 4,
        max_synthetic_images: Optional[int] = None,
        precompute_seed: int = 0,
        store_features_on_cpu: bool = False,
    ):
        if len(synthetic_dataset) == 0:
            raise ValueError("synthetic_dataset이 비어 있음")

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.store_features_on_cpu = bool(store_features_on_cpu)

        # ----- subsampling (옵션) -----
        all_indices = list(range(len(synthetic_dataset)))
        if max_synthetic_images is not None and 0 < max_synthetic_images < len(all_indices):
            import random as _random
            rng = _random.Random(precompute_seed)
            rng.shuffle(all_indices)
            all_indices = all_indices[:max_synthetic_images]
            print(
                f"[gift distiller] subsampled {max_synthetic_images}/{len(synthetic_dataset)} "
                f"synthetic images with precompute_seed={precompute_seed}"
            )

        self._dataset = synthetic_dataset
        self._active_indices = all_indices  # 활성 sample의 원본 dataset 인덱스
        self.num_images = len(all_indices)

        # 활성 sample에 매칭되는 class_name / prompt / source 캐싱
        self._class_names: List[str] = [
            synthetic_dataset.items[i]["class_name"] for i in all_indices
        ]
        self._prompts: List[str] = [
            synthetic_dataset.items[i]["prompt"] for i in all_indices
        ]
        self._sources: List[Optional[str]] = [
            synthetic_dataset.items[i].get("source") for i in all_indices
        ]
        # unique 통계
        self._unique_classes_active = sorted(set(self._class_names))

        # logit scale (CoOp frozen CLIP이므로 한 번만 가져옴)
        self._logit_scale_value = float(model.logit_scale.exp().detach().cpu().item())

        # ----- teacher image / text feature 캐싱 -----
        print(
            f"[gift distiller] image+text feature 캐싱 시작 "
            f"(N_active={self.num_images}, store_on_cpu={self.store_features_on_cpu})"
        )
        self._image_features, self._text_features = self._precompute_features(
            batch_size=precompute_batch_size,
            num_workers=precompute_num_workers,
        )
        print(
            f"[gift distiller] image feature {tuple(self._image_features.shape)}, "
            f"text feature {tuple(self._text_features.shape)} 캐싱 완료"
        )

    # ----------------------------------------------------------- feature caching

    @torch.no_grad()
    def _precompute_features(
        self,
        batch_size: int,
        num_workers: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """teacher image feature와 teacher text feature를 한 번에 계산해서 캐싱.

        - image feature: CLIP image encoder (frozen) 사용. L2 normalized.
        - text feature: prompt를 그대로 tokenizer/encoder에 통과 (template 평균 없음).
          (raw prompt가 이미 'a photo of a {c}.' 형태로 만들어져 있다고 가정.)
        """
        self.model.eval()

        # active 인덱스만 모은 sub-dataset wrapper
        # PyTorch Subset 대신 직접 sampling 루프로 처리해도 되지만,
        # DataLoader를 쓰면 num_workers로 IO 가속이 가능해서 편리.
        from torch.utils.data import Subset

        sub_dataset = Subset(self._dataset, self._active_indices)
        loader = DataLoader(
            sub_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        img_feats: List[torch.Tensor] = []
        txt_feats: List[torch.Tensor] = []

        for batch in tqdm(loader, desc="gift teacher feats"):
            images, _class_names, prompts, _sources, _indices = batch
            images = images.to(self.device, non_blocking=True)

            # ----- image features -----
            f_img = self.model.encode_image(images)
            f_img = F.normalize(f_img, dim=-1)

            # ----- text features (prompts as-is) -----
            tokens = self.tokenizer(list(prompts)).to(self.device)
            f_txt = self.model.encode_text(tokens)
            f_txt = F.normalize(f_txt, dim=-1)

            if self.store_features_on_cpu:
                img_feats.append(f_img.detach().cpu())
                txt_feats.append(f_txt.detach().cpu())
            else:
                img_feats.append(f_img.detach())
                txt_feats.append(f_txt.detach())

        img_cat = torch.cat(img_feats, dim=0)
        txt_cat = torch.cat(txt_feats, dim=0)
        return img_cat, txt_cat

    # ----------------------------------------------------------- per-step ops

    def _sample_indices(self, batch_size: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """매 step에 사용할 batch indices를 sampling (replacement 허용)."""
        n_b = min(int(batch_size), self.num_images)
        if generator is not None:
            idx = torch.randint(0, self.num_images, (n_b,), generator=generator)
        else:
            idx = torch.randint(0, self.num_images, (n_b,))
        return idx

    def _gather_features(self, idx_cpu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """주어진 batch 인덱스에 대해 (teacher_img, teacher_txt, class_names)를 반환.

        feature가 CPU에 저장돼 있다면 device로 옮김.
        """
        img = self._image_features.index_select(0, idx_cpu.to(self._image_features.device))
        txt = self._text_features.index_select(0, idx_cpu.to(self._text_features.device))
        img = img.to(self.device, non_blocking=True)
        txt = txt.to(self.device, non_blocking=True)
        class_names = [self._class_names[int(i.item())] for i in idx_cpu]
        return img, txt, class_names

    def compute_gift_losses(
        self,
        prompt_learner,
        build_coop_text_features_fn: Callable,
        batch_size: int = 64,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """한 batch에 대한 GIFT-inspired contrastive distillation + ITA 손실 계산.

        Returns:
            dict with keys:
                'loss_cd':  scalar tensor, contrastive distillation loss (KD_image + KD_text)
                'loss_ita': scalar tensor, image-text alignment loss
                'loss_kd_image': scalar tensor
                'loss_kd_text':  scalar tensor
                'loss_align_image': scalar tensor
                'loss_align_text':  scalar tensor

        외부에서 alpha/beta 로 weighting해서 더하면 됨:
            L = L_CE + alpha * losses['loss_cd'] + beta * losses['loss_ita']
        """
        idx_cpu = self._sample_indices(batch_size, generator=generator)
        teacher_img, teacher_txt, class_names = self._gather_features(idx_cpu)
        # CoOp에서 image encoder는 freeze이므로 student_img = teacher_img.
        # 따라서 M_teacher와 M_student의 유일한 차이는 text feature.

        # student text feature (gradient 흐름)
        student_txt = build_coop_text_features_fn(
            self.model, prompt_learner, class_names, self.device
        )  # [B, D]

        scale = self._logit_scale_value

        # similarity 행렬
        # M_teacher는 detach: gradient가 teacher 쪽으로 흐르지 않음
        M_teacher = (scale * teacher_img @ teacher_txt.t()).detach()  # [B, B]
        M_student = scale * teacher_img @ student_txt.t()             # [B, B], grad flows via student_txt

        # ---------- L_CD : contrastive distillation (KL via cross-entropy form) ----------
        # rows: 각 image i가 batch 내 text들에 대한 분포
        teacher_p_row = F.softmax(M_teacher, dim=-1)
        student_logp_row = F.log_softmax(M_student, dim=-1)
        L_KD_image = -(teacher_p_row * student_logp_row).sum(dim=-1).mean()

        # cols: 각 text j가 batch 내 image들에 대한 분포
        teacher_p_col = F.softmax(M_teacher.t(), dim=-1)
        student_logp_col = F.log_softmax(M_student.t(), dim=-1)
        L_KD_text = -(teacher_p_col * student_logp_col).sum(dim=-1).mean()

        loss_cd = L_KD_image + L_KD_text

        # ---------- L_ITA : matched diagonal as positive ----------
        B = M_student.size(0)
        labels = torch.arange(B, device=self.device)
        L_align_image = F.cross_entropy(M_student, labels)
        L_align_text = F.cross_entropy(M_student.t(), labels)
        loss_ita = L_align_image + L_align_text

        return {
            "loss_cd": loss_cd,
            "loss_ita": loss_ita,
            "loss_kd_image": L_KD_image,
            "loss_kd_text": L_KD_text,
            "loss_align_image": L_align_image,
            "loss_align_text": L_align_text,
        }

    # ----------------------------------------------------------- introspection

    @property
    def num_active_images(self) -> int:
        return self.num_images

    @property
    def num_unique_classes_active(self) -> int:
        return len(self._unique_classes_active)

    @property
    def unique_class_names_active(self) -> List[str]:
        return list(self._unique_classes_active)

    def source_counts_active(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in self._sources:
            key = "unknown" if s is None else str(s)
            counts[key] = counts.get(key, 0) + 1
        return counts


# ================================================================ AWC helpers


def compute_awc_fisher(
    distill_loss: torch.Tensor,
    prompt_learner,
    retain_graph: bool = True,
) -> torch.Tensor:
    """
    GIFT-inspired Adaptive Weight Consolidation의 Fisher 정보 추정 (safe minimal 버전).

    Fisher_i ≈ (∂L_distill / ∂ctx_i)^2

    설계 메모:
    - autograd.grad(create_graph=False)로 그래디언트를 한 번 뽑은 뒤 .pow(2).detach()해
      weighting으로만 사용하므로 second-order 그래프를 만들지 않음 (메모리 안전).
    - retain_graph=True를 디폴트로 두는 이유: 이 함수 호출 후 보통 동일 graph로부터
      main backward (loss_total.backward())가 이어지기 때문.
    - allow_unused=True로 두면 prompt_learner.ctx에 그래디언트가 흐르지 않을 때
      None을 반환하므로, distill_loss가 ctx에 의존하지 않는 (즉 distillation이
      비활성인) 상황은 호출 측에서 분기해야 함.

    Returns:
        fisher: [n_ctx, ctx_dim] tensor, detached. distill_loss가 ctx에 의존하지
        않으면 모든 값이 0인 텐서.
    """
    ctx = prompt_learner.ctx
    grads = torch.autograd.grad(
        outputs=distill_loss,
        inputs=[ctx],
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )[0]
    if grads is None:
        # distill_loss가 ctx에 의존하지 않음 (lambda_distill=0 등). 0 tensor 반환.
        return torch.zeros_like(ctx).detach()
    return grads.detach().pow(2)


def compute_awc_loss(
    prompt_learner,
    base_ctx: torch.Tensor,
    fisher: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor:
    """
    Adaptive Weight Consolidation loss (CoOp ctx에 한정).

    L_AWC = Σ_i (Fisher_i + eps) · (ctx_i - base_ctx_i)^2

    - Fisher는 detached tensor이므로 gradient는 (ctx - base_ctx)^2 쪽으로만 흐름.
    - 원래 GIFT 논문의 EWC 변형과 같은 형태이지만, 여기서는 CoOp ctx로 한정.
    - eps는 Fisher가 모두 0인 극단적 케이스에서도 약한 anchoring을 유지하려고 할 때 사용.
      (기본 0.0이면 vanilla weight consolidation을 Fisher가 0인 dim에서는 비활성화)

    주의:
    - base_ctx는 학습 시작 직후 detach + clone한 tensor.
    - device/dtype은 자동으로 ctx에 맞춤.
    """
    ctx = prompt_learner.ctx
    base = base_ctx.to(device=ctx.device, dtype=ctx.dtype)
    weight = fisher.detach().to(device=ctx.device, dtype=ctx.dtype)
    if eps > 0:
        weight = weight + float(eps)
    diff_sq = (ctx - base) ** 2
    return (weight * diff_sq).sum()


def compute_gift_step_losses(
    distiller: GIFTSyntheticDistiller,
    prompt_learner,
    build_coop_text_features_fn: Callable,
    batch_size: int,
    alpha: float,
    beta: float,
    use_awc: bool,
    base_ctx: Optional[torch.Tensor],
    lambda_awc: float,
    awc_eps: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    한 step의 GIFT 손실들을 종합 계산 (학습 loop에서 호출되는 편의 함수).

    L_preserve = α · L_CD + β · L_ITA      (이 값이 ctx에 대한 distillation loss)
    Fisher_i  ≈ (∂L_preserve / ∂ctx_i)^2   (use_awc=True일 때만)
    L_AWC     = λ_awc · Σ Fisher_i · (ctx_i - base_ctx_i)^2

    Returns:
        dict with:
            'loss_cd'        : scalar tensor (no scaling)
            'loss_ita'       : scalar tensor (no scaling)
            'loss_preserve'  : scalar tensor, α * L_CD + β * L_ITA
            'loss_awc'       : scalar tensor (이미 λ_awc 곱한 값) or 0-tensor when disabled
            'fisher_l1'      : scalar tensor (logging 용; L1 norm of fisher)
            'fisher_max'     : scalar tensor (logging 용; max abs of fisher)
    """
    losses = distiller.compute_gift_losses(
        prompt_learner=prompt_learner,
        build_coop_text_features_fn=build_coop_text_features_fn,
        batch_size=batch_size,
    )
    loss_cd = losses["loss_cd"]
    loss_ita = losses["loss_ita"]

    loss_preserve = alpha * loss_cd + beta * loss_ita

    if use_awc and base_ctx is not None and lambda_awc > 0.0:
        fisher = compute_awc_fisher(
            distill_loss=loss_preserve,
            prompt_learner=prompt_learner,
            retain_graph=True,
        )
        loss_awc_raw = compute_awc_loss(prompt_learner, base_ctx, fisher, eps=awc_eps)
        loss_awc = lambda_awc * loss_awc_raw
        fisher_l1 = fisher.detach().abs().sum()
        fisher_max = fisher.detach().abs().max()
    else:
        # disabled case: 0 tensor를 만들어서 외부 backward()가 깨지지 않게 함.
        # device를 prompt_learner.ctx와 맞춤.
        device = prompt_learner.ctx.device
        loss_awc = torch.zeros((), device=device)
        fisher_l1 = torch.zeros((), device=device)
        fisher_max = torch.zeros((), device=device)

    return {
        "loss_cd": loss_cd,
        "loss_ita": loss_ita,
        "loss_preserve": loss_preserve,
        "loss_awc": loss_awc,
        "fisher_l1": fisher_l1,
        "fisher_max": fisher_max,
    }
