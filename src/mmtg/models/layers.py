"""Shared transformer building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block with GELU MLP."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                                need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerStack(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int,
                 mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the sequence dimension, ignoring padding.

    x: (B, L, D), mask: (B, L) with 1 for real tokens.
    """
    mask = mask.unsqueeze(-1).to(x.dtype)
    total = (x * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1.0)
    return total / count
