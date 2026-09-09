"""VLA action head: multimodal fusion to portfolio weights.

The sequence to sequence policy of the blueprint, at research scale. The
frozen Financial-CLIP vision tower supplies the chart token, the retrieved
DKG subgraph supplies a graph context token, the port CNN and the hawk dove
scorer supply a scalar feature token, and the raw statement tokens flow in
as a sequence. A small transformer fuses everything and an action readout
emits one weight per tradable asset, normalized to a gross exposure cap and
serialized as the JSON action.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ASSETS, VlaConfig
from .layers import TransformerStack


class VlaActionHead(nn.Module):
    def __init__(self, cfg: VlaConfig, vocab_size: int, clip_dim: int,
                 n_assets: int = len(ASSETS)):
        super().__init__()
        self.cfg = cfg
        self.n_assets = n_assets

        self.token_embed = nn.Embedding(vocab_size, cfg.dim)
        self.pos_embed = nn.Parameter(
            torch.empty(1, cfg.max_statement_len, cfg.dim).normal_(std=0.02))

        self.act_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.img_proj = nn.Linear(clip_dim, cfg.dim)
        self.ctx_proj = nn.Linear(clip_dim, cfg.dim)
        self.feat_proj = nn.Sequential(
            nn.Linear(2, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, cfg.dim))
        self.type_embed = nn.Embedding(4, cfg.dim)

        self.encoder = TransformerStack(cfg.dim, cfg.depth, cfg.heads,
                                        dropout=cfg.dropout)
        self.readout = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim, n_assets),
        )

    def forward(self, chart_emb: torch.Tensor, ids: torch.Tensor,
                mask: torch.Tensor, ctx_emb: torch.Tensor,
                feats: torch.Tensor) -> torch.Tensor:
        """Returns normalized weights (B, n_assets), gross capped."""
        B = ids.shape[0]
        L = min(ids.shape[1], self.cfg.max_statement_len)
        ids, mask = ids[:, :L], mask[:, :L]

        specials = torch.stack([
            self.act_token.expand(B, 1, -1).squeeze(1),
            self.img_proj(chart_emb),
            self.ctx_proj(ctx_emb),
            self.feat_proj(feats),
        ], dim=1)
        specials = specials + self.type_embed.weight.unsqueeze(0)

        text = self.token_embed(ids) + self.pos_embed[:, :L]
        x = torch.cat([specials, text], dim=1)

        pad = torch.zeros(B, 4, dtype=torch.bool, device=ids.device)
        key_padding = torch.cat([pad, mask == 0], dim=1)
        x = self.encoder(x, key_padding_mask=key_padding)

        raw = torch.tanh(self.readout(x[:, 0]))
        return normalize_gross(raw, self.cfg.gross_cap)


def normalize_gross(raw: torch.Tensor, gross_cap: float) -> torch.Tensor:
    """Scale rows so that sum(|w|) equals gross_cap exactly.

    Scaling to the cap rather than merely clipping to it is deliberate. Under
    a squared error objective a head that is free to shrink will do exactly
    that, since predicting near zero is the risk minimizing response to a
    noisy label; the result is a portfolio with no view. Fixing the gross
    exposure removes shrinkage as an option and makes the head commit to the
    relative ranking across assets, which is what a tilt actually expresses.
    """
    gross = raw.abs().sum(dim=-1, keepdim=True)
    return raw * (gross_cap / gross.clamp(min=1e-6))


def vla_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Direction first: cosine distance carries the loss, MSE regularizes.

    The deliverable is a set of relative tilts, so agreement in direction is
    what matters and magnitude error is secondary. Weighting the two the
    other way round (plain MSE) measurably degraded held out sign agreement
    during development.
    """
    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    mse = F.mse_loss(pred, target)
    return cos + 0.3 * mse


def to_action_json(weights: torch.Tensor, rationale: list[str] | None = None,
                   month: str | None = None,
                   evidence: list[str] | None = None) -> dict:
    """Serialize one weight vector into the blueprint's JSON action format.

    `rationale` holds the causal chains that argue for this action, already
    filtered to the stance the source text expresses. `evidence` holds the
    raw retrieved edges with their observation counts. They are kept apart
    because an edge the graph happens to know is not the same thing as a
    reason for this trade, and merging them lets a contradictory relation
    read as supporting argument.
    """
    w = weights.detach().cpu().numpy().ravel()
    action = {ticker: round(float(w[i]), 4) for i, ticker in enumerate(ASSETS)}
    out: dict = {"action": action,
                 "gross_exposure": round(float(abs(w).sum()), 4),
                 "net_exposure": round(float(w.sum()), 4)}
    if month is not None:
        out["month"] = month
    if rationale is not None:
        out["rationale"] = rationale
    if evidence is not None:
        out["evidence"] = evidence
    return out


@torch.no_grad()
def action_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Diagnostics: direction agreement and magnitude error."""
    sign_ok = ((pred.sign() == target.sign()) & (target.abs() > 1e-4)).float()
    denom = (target.abs() > 1e-4).float().sum().clamp(min=1.0)
    return {
        "sign_agreement": float(sign_ok.sum() / denom),
        "mae": float((pred - target).abs().mean()),
        "gross_mean": float(pred.abs().sum(dim=-1).mean()),
    }
