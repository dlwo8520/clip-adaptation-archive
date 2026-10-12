# -*- coding: utf-8 -*-
"""
CoOp prompt learner 구현

Prompt 형태:
[SOS] [V]1 [V]2 ... [V]M [CLASS] . [EOS]

학습되는 것은 [V] context vector뿐임.
"""
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def clean_class_name(class_name: str) -> str:
    """class name을 prompt에 넣기 좋게 정리함"""
    return class_name.replace("_", " ").replace("-", " ")

def freeze_clip_model(model: nn.Module) -> None:
    """CLIP 전체 parameter를 freeze함"""
    for param in model.parameters():
        param.requires_grad = False

class CoOpPromptLearner(nn.Module):
    """unified context CoOp 구현"""
    def __init__(self, clip_model: nn.Module, tokenizer, n_ctx: int = 16, init_std: float = 0.02):
        super().__init__()
        # object.__setattr__를 사용해 clip_model을 nn.Module 서브모듈로 등록하지 않음.
        # self.clip_model = clip_model로 하면 CLIP 전체가 서브모듈로 등록되어:
        #   - prompt_learner.train() 호출 시 CLIP도 training mode가 됨 (dropout 활성화 위험)
        #   - prompt_learner.parameters()에 CLIP frozen params 포함 → optimizer 낭비
        #   - state_dict()에 CLIP 수백 MB 포함 → 체크포인트 비대화
        object.__setattr__(self, '_clip_model', clip_model)
        self.tokenizer = tokenizer
        self.n_ctx = n_ctx
        ctx_dim = clip_model.token_embedding.weight.shape[1]
        self.ctx = nn.Parameter(torch.empty(n_ctx, ctx_dim))
        nn.init.normal_(self.ctx, std=init_std)

    def forward(self, classnames: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """class별 prompt embedding을 생성함"""
        dummy_context = " ".join(["X"] * self.n_ctx)
        prompt_texts = [f"{dummy_context} {clean_class_name(name)}." for name in classnames]
        tokenized_prompts = self.tokenizer(prompt_texts).to(device)

        with torch.no_grad():
            embedding = self._clip_model.token_embedding(tokenized_prompts)

        token_prefix = embedding[:, :1, :]
        token_suffix = embedding[:, 1 + self.n_ctx :, :]
        n_cls = len(classnames)
        ctx = self.ctx.unsqueeze(0).expand(n_cls, -1, -1)
        prompt_embeddings = torch.cat([token_prefix, ctx, token_suffix], dim=1)
        return prompt_embeddings, tokenized_prompts

def encode_text_from_embeddings(model: nn.Module, prompt_embeddings: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
    """직접 만든 prompt embedding을 CLIP text encoder에 넣음"""
    if hasattr(model.transformer, "get_cast_dtype"):
        cast_dtype = model.transformer.get_cast_dtype()
    else:
        cast_dtype = model.token_embedding.weight.dtype

    x = prompt_embeddings.to(dtype=cast_dtype)
    pos = model.positional_embedding[: x.shape[1], :].to(device=x.device, dtype=cast_dtype)
    x = x + pos

    # open_clip 버전에 따라 transformer가 batch_first일 수도 있고 아닐 수도 있음.
    # 명시적으로 분기해서 silent하게 잘못된 attention이 나오는 것을 막음.
    is_batch_first = bool(getattr(model.transformer, "batch_first", False))
    attn_mask = getattr(model, "attn_mask", None)

    if is_batch_first:
        # [B, L, D] 그대로
        try:
            x_out = model.transformer(x, attn_mask=attn_mask)
        except TypeError:
            x_out = model.transformer(x)
    else:
        # [B, L, D] -> [L, B, D] -> transformer -> [L, B, D] -> [B, L, D]
        x_seq = x.permute(1, 0, 2)
        try:
            x_seq = model.transformer(x_seq, attn_mask=attn_mask)
        except TypeError:
            x_seq = model.transformer(x_seq)
        x_out = x_seq.permute(1, 0, 2)

    x = model.ln_final(x_out)
    eot_indices = tokenized_prompts.argmax(dim=-1)
    x = x[torch.arange(x.shape[0], device=x.device), eot_indices]

    if getattr(model, "text_projection", None) is not None:
        if isinstance(model.text_projection, nn.Linear):
            x = model.text_projection(x)
        else:
            x = x @ model.text_projection
    return x

@torch.no_grad()
def build_zero_shot_text_features(model: nn.Module, tokenizer, classnames: List[str], templates: List[str], device: torch.device) -> torch.Tensor:
    """hand-crafted prompt template 기반 원본 CLIP zero-shot classifier 생성"""
    model.eval()
    all_features = []
    for class_name in classnames:
        class_name = clean_class_name(class_name)
        prompts = [template.format(class_name) for template in templates]
        tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
        class_feature = text_features.mean(dim=0)
        class_feature = F.normalize(class_feature, dim=0)
        all_features.append(class_feature)
    return torch.stack(all_features, dim=0)

def build_coop_text_features(model: nn.Module, prompt_learner: CoOpPromptLearner, classnames: List[str], device: torch.device) -> torch.Tensor:
    """학습된 CoOp context 기반 classifier text feature 생성"""
    prompt_embeddings, tokenized_prompts = prompt_learner(classnames, device)
    text_features = encode_text_from_embeddings(model, prompt_embeddings, tokenized_prompts)
    return F.normalize(text_features, dim=-1)


class CoCoOpPromptLearner(nn.Module):
    """CoCoOp prompt learner 구현.

    CoOp과 달리 이미지 feature를 meta-net에 넣어 instance-conditioned prompt를 생성함.
    CLIP image/text encoder는 freeze하고, 학습되는 것은 ctx와 meta_net뿐임.

    Prompt 형태:
    [SOS] [V + bias(image)]1 ... [V + bias(image)]M [CLASS] . [EOS]
    """

    def __init__(
        self,
        clip_model: nn.Module,
        tokenizer,
        n_ctx: int = 16,
        init_std: float = 0.02,
        meta_hidden_divisor: int = 16,
    ):
        super().__init__()
        # object.__setattr__로 clip_model을 nn.Module 서브모듈로 등록하지 않음.
        # 등록하면 prompt_learner.train() 시 CLIP도 train mode가 되고,
        # parameters()/state_dict()에 CLIP 전체가 포함되어 optimizer/checkpoint가 비대해짐.
        object.__setattr__(self, '_clip_model', clip_model)
        self.tokenizer = tokenizer
        self.n_ctx = n_ctx

        ctx_dim = clip_model.token_embedding.weight.shape[1]
        vis_dim = self._infer_visual_output_dim(clip_model)
        hidden_dim = max(1, vis_dim // meta_hidden_divisor)

        self.ctx = nn.Parameter(torch.empty(n_ctx, ctx_dim))
        nn.init.normal_(self.ctx, std=init_std)

        self.meta_net = nn.Sequential(
            nn.Linear(vis_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, ctx_dim),
        )

    @staticmethod
    def _infer_visual_output_dim(clip_model: nn.Module) -> int:
        if hasattr(clip_model.visual, "output_dim"):
            return int(clip_model.visual.output_dim)

        # open_clip ViT 계열에서 proj가 [width, output_dim] 형태인 경우가 많음.
        visual_proj = getattr(clip_model.visual, "proj", None)
        if visual_proj is not None:
            if hasattr(visual_proj, "shape") and len(visual_proj.shape) >= 2:
                return int(visual_proj.shape[-1])

        text_projection = getattr(clip_model, "text_projection", None)
        if text_projection is not None:
            if isinstance(text_projection, nn.Linear):
                return int(text_projection.out_features)
            if hasattr(text_projection, "shape") and len(text_projection.shape) >= 2:
                return int(text_projection.shape[-1])

        raise ValueError("visual output dimension을 추론하지 못함")

    def _build_class_token_parts(self, classnames: List[str], device: torch.device):
        dummy_context = " ".join(["X"] * self.n_ctx)
        prompt_texts = [f"{dummy_context} {clean_class_name(name)}." for name in classnames]
        tokenized_prompts = self.tokenizer(prompt_texts).to(device)

        with torch.no_grad():
            embedding = self._clip_model.token_embedding(tokenized_prompts)

        token_prefix = embedding[:, :1, :]
        token_suffix = embedding[:, 1 + self.n_ctx :, :]
        return token_prefix, token_suffix, tokenized_prompts

    def forward(self, classnames: List[str], image_features: torch.Tensor, device: torch.device):
        """이미지별 조건부 prompt embedding을 생성함.

        Args:
            classnames: class name list
            image_features: [B, image_feature_dim], L2 normalized image features
        Returns:
            prompt_embeddings: [B, C, L, D]
            tokenized_prompts: [C, L]
        """
        token_prefix, token_suffix, tokenized_prompts = self._build_class_token_parts(classnames, device)

        # image-conditioned bias: [B, D]
        bias = self.meta_net(image_features)
        ctx_shifted = self.ctx.unsqueeze(0) + bias.unsqueeze(1)  # [B, n_ctx, D]

        bsz = image_features.shape[0]
        n_cls = len(classnames)

        prefix = token_prefix.unsqueeze(0).expand(bsz, n_cls, -1, -1)
        suffix = token_suffix.unsqueeze(0).expand(bsz, n_cls, -1, -1)
        ctx = ctx_shifted.unsqueeze(1).expand(bsz, n_cls, -1, -1)

        prompt_embeddings = torch.cat([prefix, ctx, suffix], dim=2)
        return prompt_embeddings, tokenized_prompts


def build_cocoop_logits(
    model: nn.Module,
    prompt_learner: CoCoOpPromptLearner,
    image_features: torch.Tensor,
    classnames: List[str],
    device: torch.device,
    class_chunk_size: int = 128,
) -> torch.Tensor:
    """CoCoOp instance-conditioned prompt로 logits를 계산함.

    image_features는 이미 L2 normalize되어 있다고 가정함.
    CoCoOp은 이미지마다 text classifier가 달라지므로 static text feature를 만들 수 없음.
    class_chunk_size로 class를 나누어 ImageNet 1000-class 평가 시 메모리 사용량을 줄임.
    """
    if class_chunk_size <= 0:
        class_chunk_size = len(classnames)

    logits_chunks = []
    logit_scale = model.logit_scale.exp()

    for start in range(0, len(classnames), class_chunk_size):
        end = min(start + class_chunk_size, len(classnames))
        cls_chunk = classnames[start:end]

        prompt_embeddings, tokenized_prompts = prompt_learner(cls_chunk, image_features, device)
        bsz, n_cls, seq_len, dim = prompt_embeddings.shape

        flat_prompts = prompt_embeddings.reshape(bsz * n_cls, seq_len, dim)
        flat_tokens = tokenized_prompts.unsqueeze(0).expand(bsz, n_cls, -1).reshape(bsz * n_cls, -1)

        text_features = encode_text_from_embeddings(model, flat_prompts, flat_tokens)
        text_features = F.normalize(text_features, dim=-1).reshape(bsz, n_cls, -1)

        logits = logit_scale * torch.einsum("bd,bcd->bc", image_features, text_features)
        logits_chunks.append(logits)

    return torch.cat(logits_chunks, dim=1)


def infer_visual_output_dim(model: nn.Module) -> int:
    """CLIP image encoder의 output dimension을 추론함 (CLIPAdapter 초기화에 사용)"""
    if hasattr(model.visual, "output_dim"):
        return int(model.visual.output_dim)
    visual_proj = getattr(model.visual, "proj", None)
    if visual_proj is not None and hasattr(visual_proj, "shape") and len(visual_proj.shape) >= 2:
        return int(visual_proj.shape[-1])
    text_projection = getattr(model, "text_projection", None)
    if text_projection is not None:
        if isinstance(text_projection, nn.Linear):
            return int(text_projection.out_features)
        if hasattr(text_projection, "shape") and len(text_projection.shape) >= 2:
            return int(text_projection.shape[-1])
    raise ValueError("CLIP visual output dim을 추론하지 못함")


class CLIPAdapter(nn.Module):
    """CLIP-Adapter: visual feature adapter.

    논문 비교용으로 수정한 점:
    - bottleneck MLP는 Linear -> ReLU -> Linear 구조만 사용함.
      원래 업로드 코드에 있던 마지막 ReLU는 제거함.
    - alpha는 기본적으로 고정 residual ratio로 사용함.
      DTD 비교용 기본값은 0.6임.
    - 필요하면 alpha_learnable=True로 learnable alpha도 사용할 수 있음.

    입력 x는 L2 normalize된 CLIP feature라고 가정함.
    출력도 L2 normalize함.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_ratio: float = 0.25,
        alpha_init: float = 0.6,
        alpha_learnable: bool = False,
    ):
        super().__init__()
        import math

        if not (0.0 < alpha_init < 1.0):
            raise ValueError(f"alpha_init은 0과 1 사이여야 함. 현재 값: {alpha_init}")

        hidden_dim = max(1, int(feature_dim * hidden_ratio))

        # 논문 식에 맞게 마지막 ReLU를 넣지 않음.
        # Adapter(x) = Linear2(ReLU(Linear1(x)))
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim, bias=False),
        )

        self.alpha_learnable = bool(alpha_learnable)
        if self.alpha_learnable:
            alpha_logit_init = math.log(alpha_init / (1.0 - alpha_init))
            self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_logit_init)))
        else:
            self.register_buffer("alpha_value", torch.tensor(float(alpha_init)))

    @property
    def alpha(self) -> float:
        """현재 alpha 값을 float으로 반환함."""
        if self.alpha_learnable:
            return float(torch.sigmoid(self.alpha_logit).item())
        return float(self.alpha_value.item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., D], L2 normalized features
        Returns:
            [..., D], L2 normalized adapted features
        """
        if self.alpha_learnable:
            alpha = torch.sigmoid(self.alpha_logit)
        else:
            alpha = self.alpha_value.to(device=x.device, dtype=x.dtype)

        adapted = self.adapter(x)
        out = alpha * adapted + (1.0 - alpha) * x
        return F.normalize(out, dim=-1)
