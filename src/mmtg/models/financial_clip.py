"""Financial-CLIP: dual encoder over financial charts and macro text.

A compact CLIP architecture trained from scratch with the standard symmetric
InfoNCE objective on (chart image, structural break caption) pairs. At demo
scale the towers are small; the classes take their widths from ClipConfig so
a local run can scale them up, or initialize from an open CLIP checkpoint and
fine tune, which is the blueprint's original setting.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ClipConfig
from .layers import TransformerStack, masked_mean


class VisionEncoder(nn.Module):
    """ViT style patch transformer over chart images."""

    def __init__(self, cfg: ClipConfig, img_size: int):
        super().__init__()
        if img_size % cfg.patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        n_patches = (img_size // cfg.patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, cfg.vision_dim, cfg.patch_size,
                                     stride=cfg.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.vision_dim))
        self.pos_embed = nn.Parameter(
            torch.empty(1, n_patches + 1, cfg.vision_dim).normal_(std=0.02))
        self.encoder = TransformerStack(cfg.vision_dim, cfg.vision_depth,
                                        cfg.vision_heads, dropout=cfg.dropout)
        self.proj = nn.Linear(cfg.vision_dim, cfg.embed_dim, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.proj(x[:, 0])


class TextEncoder(nn.Module):
    """Small transformer over tokenized financial text, masked mean pooled."""

    def __init__(self, cfg: ClipConfig, vocab_size: int):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, cfg.text_dim)
        self.pos_embed = nn.Parameter(
            torch.empty(1, cfg.max_text_len, cfg.text_dim).normal_(std=0.02))
        self.encoder = TransformerStack(cfg.text_dim, cfg.text_depth,
                                        cfg.text_heads, dropout=cfg.dropout)
        self.proj = nn.Linear(cfg.text_dim, cfg.embed_dim, bias=False)
        self.max_len = cfg.max_text_len

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if ids.shape[1] > self.max_len:
            ids, mask = ids[:, : self.max_len], mask[:, : self.max_len]
        x = self.token_embed(ids) + self.pos_embed[:, : ids.shape[1]]
        x = self.encoder(x, key_padding_mask=(mask == 0))
        return self.proj(masked_mean(x, mask))


class FinancialCLIP(nn.Module):
    def __init__(self, cfg: ClipConfig, vocab_size: int, img_size: int):
        super().__init__()
        self.cfg = cfg
        self.visual = VisionEncoder(cfg, img_size)
        self.textual = TextEncoder(cfg, vocab_size)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(cfg.logit_scale_init)))

    def encode_image(self, images: torch.Tensor,
                     normalize: bool = True) -> torch.Tensor:
        emb = self.visual(images)
        return F.normalize(emb, dim=-1) if normalize else emb

    def encode_text(self, ids: torch.Tensor, mask: torch.Tensor,
                    normalize: bool = True) -> torch.Tensor:
        emb = self.textual(ids, mask)
        return F.normalize(emb, dim=-1) if normalize else emb

    def forward(self, images: torch.Tensor, ids: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.encode_image(images)
        txt = self.encode_text(ids, mask)
        scale = self.logit_scale.exp().clamp(max=self.cfg.logit_scale_max)
        return scale * img @ txt.t(), scale * txt @ img.t()


def clip_infonce(logits_per_image: torch.Tensor,
                 logits_per_text: torch.Tensor,
                 key_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Symmetric InfoNCE over a batch of aligned pairs.

    Overlapping chart windows produce textually identical captions, and two
    identical captions must not act as negatives for each other. When key_ids
    (a stable id per unique caption text) is given, off diagonal entries whose
    captions collide are masked out of both softmaxes.
    """
    n = logits_per_image.shape[0]
    device = logits_per_image.device
    targets = torch.arange(n, device=device)
    if key_ids is not None:
        collide = key_ids.unsqueeze(0) == key_ids.unsqueeze(1)
        collide &= ~torch.eye(n, dtype=torch.bool, device=device)
        logits_per_image = logits_per_image.masked_fill(collide, -1e9)
        logits_per_text = logits_per_text.masked_fill(collide, -1e9)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return 0.5 * (loss_i + loss_t)


@torch.no_grad()
def retrieval_metrics(model: FinancialCLIP, images: torch.Tensor,
                      ids: torch.Tensor, mask: torch.Tensor,
                      key_ids: torch.Tensor | None = None) -> dict:
    """Image to text retrieval accuracy over one evaluation pool.

    With key_ids, a retrieval counts as correct when the retrieved caption
    TEXT matches the ground truth text, the right criterion when the pool
    holds duplicate captions from overlapping windows.
    """
    img = model.encode_image(images)
    txt = model.encode_text(ids, mask)
    sims = img @ txt.t()
    ranks = sims.argsort(dim=1, descending=True)
    n = sims.shape[0]
    targets = torch.arange(n, device=sims.device)
    if key_ids is None:
        hit = ranks == targets.unsqueeze(1)
    else:
        hit = key_ids[ranks] == key_ids.unsqueeze(1)
    pos = hit.float().argmax(dim=1)
    return {
        "top1": round(float((pos == 0).float().mean()), 4),
        "top5": round(float((pos < 5).float().mean()), 4),
        "mean_rank": round(float(pos.float().mean() + 1.0), 2),
        "pool_size": n,
        "criterion": "caption_text_match" if key_ids is not None else "index",
    }
