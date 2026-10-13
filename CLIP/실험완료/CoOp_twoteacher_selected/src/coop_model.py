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
        self.clip_model = clip_model
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
            embedding = self.clip_model.token_embedding(tokenized_prompts)

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
