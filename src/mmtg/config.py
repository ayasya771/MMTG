"""Central configuration for every stage of the MMTG pipeline.

All defaults are demo scale so that the full three phase pipeline trains in
minutes on CPU. For a serious run, raise the model widths, depths, data volume
and epochs from the command line flags exposed by the training scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def repo_root() -> str:
    """Resolve the repository root from this file's location."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class Paths:
    root: str = field(default_factory=repo_root)
    data_name: str = "generated"

    @property
    def data_dir(self) -> str:
        return os.path.join(self.root, "data", self.data_name)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "data", "raw")

    @property
    def charts_dir(self) -> str:
        return os.path.join(self.data_dir, "charts")

    @property
    def ports_dir(self) -> str:
        return os.path.join(self.data_dir, "ports")

    @property
    def runs_dir(self) -> str:
        return os.path.join(self.root, "runs")

    @property
    def docs_dir(self) -> str:
        return os.path.join(self.root, "docs")

    def ensure(self) -> "Paths":
        for d in (self.data_dir, self.raw_dir, self.charts_dir,
                  self.ports_dir, self.runs_dir):
            os.makedirs(d, exist_ok=True)
        return self


REGIMES = ["goldilocks", "overheating", "stagflation", "recession", "recovery"]

ASSETS = ["SPY", "QQQ", "TLT", "GLD", "DXY", "USO"]

ASSET_ENTITY = {
    "SPY": "US_Equities",
    "QQQ": "Tech_Equities",
    "TLT": "Long_Treasuries",
    "GLD": "Gold",
    "DXY": "US_Dollar",
    "USO": "Crude_Oil",
}


@dataclass
class SimConfig:
    months: int = 720
    start_year: int = 1966
    start_month: int = 1
    seed: int = 7
    persistence: float = 0.90


@dataclass
class ChartConfig:
    img_size: int = 128
    window: int = 24
    stride: int = 2
    types: tuple = ("yield_curve", "cpi_gdp", "policy_path")
    dpi: int = 64
    seed: int = 11


@dataclass
class PortConfig:
    img_size: int = 96
    per_month: int = 3
    seed: int = 13


@dataclass
class TokenizerConfig:
    vocab_size: int = 4096
    min_freq: int = 1
    caption_len: int = 64
    statement_len: int = 96


@dataclass
class ClipConfig:
    embed_dim: int = 128
    patch_size: int = 16
    vision_dim: int = 192
    vision_depth: int = 4
    vision_heads: int = 4
    text_dim: int = 192
    text_depth: int = 4
    text_heads: int = 4
    max_text_len: int = 64
    dropout: float = 0.1
    logit_scale_init: float = 14.285
    logit_scale_max: float = 100.0


@dataclass
class PortCnnConfig:
    channels: tuple = (32, 64, 128)
    dropout: float = 0.1


@dataclass
class DkgConfig:
    decay_lambda: float = 0.03
    min_confidence: float = 0.5
    top_k_nodes: int = 6
    hops: int = 1
    max_context_edges: int = 12


@dataclass
class VlaConfig:
    dim: int = 128
    depth: int = 2
    heads: int = 4
    dropout: float = 0.15
    horizon: int = 3
    gross_cap: float = 1.0
    max_statement_len: int = 96


@dataclass
class TrainClipConfig:
    epochs: int = 26
    batch_size: int = 96
    lr: float = 5.0e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.1
    seed: int = 17
    train_end_month: int = 600
    embargo_months: int = 8


@dataclass
class TrainPortConfig:
    epochs: int = 12
    batch_size: int = 64
    lr: float = 2.0e-3
    seed: int = 19


@dataclass
class TrainVlaConfig:
    epochs: int = 45
    batch_size: int = 64
    lr: float = 3.0e-4
    weight_decay: float = 0.05
    seed: int = 23
    train_end_month: int = 600
    embargo_months: int = 8
    swa_start: int = 30
