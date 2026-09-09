"""End to end inference: multimodal inputs to a JSON portfolio action.

Loads the three trained components plus the knowledge graph from a run
directory and exposes generate(), which takes a chart image, a policy
statement and optionally a port image, retrieves graph context as of the
query month, and returns the action JSON with its rationale.
"""

from __future__ import annotations

import os
from dataclasses import asdict

import numpy as np
import torch

from .config import ClipConfig, DkgConfig, PortCnnConfig, VlaConfig
from .data.datasets import load_image_tensor
from .data.statements import hawk_dove_score
from .dkg.graph import DynamicKnowledgeGraph
from .dkg.retrieve import GraphRetriever
from .models.financial_clip import FinancialCLIP
from .models.port_cnn import PortDensityCNN
from .models.vla_head import VlaActionHead, to_action_json
from .tokenizer import FinancialTokenizer
from .utils import load_json, save_json


def save_checkpoint(model: torch.nn.Module, cfg, path: str, **meta) -> None:
    torch.save({"state_dict": model.state_dict(),
                "cfg": asdict(cfg), "meta": meta}, path)


def hawk_dove(text: str) -> float:
    """Lexicon hawk dove score of arbitrary text in [-1, 1]."""
    return hawk_dove_score(text)


class MacroTradePipeline:
    def __init__(self, clip: FinancialCLIP, port_cnn: PortDensityCNN,
                 vla: VlaActionHead, dkg: DynamicKnowledgeGraph,
                 tokenizer: FinancialTokenizer, manifest: dict,
                 device: str = "cpu"):
        self.clip = clip.to(device).eval()
        self.port_cnn = port_cnn.to(device).eval()
        self.vla = vla.to(device).eval()
        self.dkg = dkg
        self.retriever = GraphRetriever(dkg, DkgConfig(**manifest["dkg_cfg"]))
        self.tok = tokenizer
        self.manifest = manifest
        self.device = device

    @classmethod
    def load(cls, run_dir: str, device: str = "cpu") -> "MacroTradePipeline":
        manifest = load_json(os.path.join(run_dir, "manifest.json"))
        tok = FinancialTokenizer.load(os.path.join(run_dir, manifest["vocab"]))

        clip_ckpt = torch.load(os.path.join(run_dir, manifest["clip"]),
                               map_location="cpu", weights_only=False)
        clip = FinancialCLIP(ClipConfig(**clip_ckpt["cfg"]),
                             vocab_size=clip_ckpt["meta"]["vocab_size"],
                             img_size=clip_ckpt["meta"]["img_size"])
        clip.load_state_dict(clip_ckpt["state_dict"])

        port_ckpt = torch.load(os.path.join(run_dir, manifest["port"]),
                               map_location="cpu", weights_only=False)
        port_cnn = PortDensityCNN(PortCnnConfig(**{
            **port_ckpt["cfg"],
            "channels": tuple(port_ckpt["cfg"]["channels"])}))
        port_cnn.load_state_dict(port_ckpt["state_dict"])

        vla_ckpt = torch.load(os.path.join(run_dir, manifest["vla"]),
                              map_location="cpu", weights_only=False)
        vla = VlaActionHead(VlaConfig(**vla_ckpt["cfg"]),
                            vocab_size=vla_ckpt["meta"]["vocab_size"],
                            clip_dim=vla_ckpt["meta"]["clip_dim"])
        vla.load_state_dict(vla_ckpt["state_dict"])

        dkg = DynamicKnowledgeGraph.load(os.path.join(run_dir, manifest["dkg"]))
        return cls(clip, port_cnn, vla, dkg, tok, manifest, device)

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        ids = torch.tensor([self.tok.encode(text, self.clip.textual.max_len)],
                           dtype=torch.long, device=self.device)
        mask = (ids != self.tok.pad_id).long()
        return self.clip.encode_text(ids, mask)[0].cpu().numpy()

    @torch.no_grad()
    def embed_chart(self, png_path: str) -> np.ndarray:
        img = load_image_tensor(png_path, self.manifest["img_size"])
        return self.clip.encode_image(img.unsqueeze(0).to(self.device))[0].cpu().numpy()

    @torch.no_grad()
    def port_density(self, png_path: str) -> float:
        img = load_image_tensor(png_path, self.manifest["port_img_size"])
        return float(self.port_cnn.predict(img.unsqueeze(0).to(self.device))[0])

    @torch.no_grad()
    def generate(self, chart_paths: list[str] | str, statement: str,
                 as_of: str, port_path: str | None = None) -> dict:
        if isinstance(chart_paths, str):
            chart_paths = [chart_paths]
        chart_emb = np.mean([self.embed_chart(p) for p in chart_paths], axis=0)
        chart_emb /= np.linalg.norm(chart_emb) + 1e-9

        hd = hawk_dove(statement)
        query = self.embed_text(statement)
        ctx = self.retriever.retrieve(query, as_of, stance=hd)
        ctx_emb = (ctx.context_embedding if ctx.context_embedding is not None
                   else np.zeros(self.manifest["clip_dim"], dtype=np.float32))

        density = self.port_density(port_path) if port_path else 0.5

        ids = torch.tensor([self.tok.encode(statement,
                                            self.vla.cfg.max_statement_len)],
                           dtype=torch.long, device=self.device)
        mask = (ids != self.tok.pad_id).long()
        weights = self.vla(
            torch.tensor(chart_emb, dtype=torch.float32,
                         device=self.device).unsqueeze(0),
            ids, mask,
            torch.tensor(ctx_emb, dtype=torch.float32,
                         device=self.device).unsqueeze(0),
            torch.tensor([[density, hd]], dtype=torch.float32,
                         device=self.device),
        )[0]

        action = to_action_json(weights, rationale=ctx.chains, month=as_of,
                                evidence=ctx.context_lines[:4])
        action["signals"] = {
            "hawk_dove": round(hd, 3),
            "port_density": round(density, 3),
            "retrieved_nodes": ctx.nodes,
            "asset_hints": ctx.asset_hints,
        }
        return action

    def save_action(self, action: dict, path: str) -> None:
        save_json(action, path)
