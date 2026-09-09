"""Container density CNN over port imagery.

Regresses the fraction of quay area stacked with containers from a satellite
style scene. The estimate acts as a trade volume and supply chain congestion
proxy feature for the action head, the blueprint's "count container density
as a proxy for trade volume" idea in miniature.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import PortCnnConfig


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=2, padding=1),
        nn.BatchNorm2d(cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.GELU(),
    )


class PortDensityCNN(nn.Module):
    def __init__(self, cfg: PortCnnConfig | None = None):
        super().__init__()
        cfg = cfg or PortCnnConfig()
        chans = [3] + list(cfg.channels)
        self.features = nn.Sequential(
            *[_block(chans[i], chans[i + 1]) for i in range(len(chans) - 1)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(chans[-1], 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Raw density estimates, shape (B,). Targets live in [0, 1]; the
        head is linear because a sigmoid here sits on a long flat gradient
        plateau before the count signal breaks through. Use predict() for
        clamped inference values."""
        return self.head(self.features(images)).squeeze(-1)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Density estimates clamped to the valid [0, 1] range."""
        return self.forward(images).clamp(0.0, 1.0)
