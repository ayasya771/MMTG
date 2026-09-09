"""Integration: save a micro run to disk, reload it through the pipeline,
and generate an action from a rendered chart plus a statement."""

import os

import numpy as np
import torch

from mmtg.config import (ClipConfig, PortCnnConfig, VlaConfig)
from mmtg.dkg.extract import RuleBasedExtractor
from mmtg.dkg.graph import DynamicKnowledgeGraph
from mmtg.models.financial_clip import FinancialCLIP
from mmtg.models.port_cnn import PortDensityCNN
from mmtg.models.vla_head import VlaActionHead
from mmtg.pipeline import MacroTradePipeline, save_checkpoint
from mmtg.utils import save_json


def _build_run(tmp_path, tokenizer, corpus):
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(0)

    clip_cfg = ClipConfig(vision_dim=64, vision_depth=1, text_dim=64,
                          text_depth=1, embed_dim=32, vision_heads=2,
                          text_heads=2, max_text_len=32)
    clip = FinancialCLIP(clip_cfg, vocab_size=len(tokenizer), img_size=128)
    save_checkpoint(clip, clip_cfg, os.path.join(run_dir, "clip.pt"),
                    vocab_size=len(tokenizer), img_size=128)

    port_cfg = PortCnnConfig(channels=(8, 16, 32))
    port = PortDensityCNN(port_cfg)
    save_checkpoint(port, port_cfg, os.path.join(run_dir, "port_cnn.pt"))

    vla_cfg = VlaConfig(dim=32, depth=1, heads=2, max_statement_len=48)
    vla = VlaActionHead(vla_cfg, vocab_size=len(tokenizer), clip_dim=32)
    save_checkpoint(vla, vla_cfg, os.path.join(run_dir, "vla.pt"),
                    vocab_size=len(tokenizer), clip_dim=32)

    dkg = DynamicKnowledgeGraph()
    dkg.ingest(RuleBasedExtractor().extract_corpus(corpus))
    rng = np.random.default_rng(1)
    dkg.set_embeddings({n: rng.normal(size=32).astype(np.float32)
                        for n in dkg.g.nodes})
    dkg.save(os.path.join(run_dir, "dkg.json"))

    tokenizer.save(os.path.join(run_dir, "vocab.json"))
    save_json({"vocab": "vocab.json", "clip": "clip.pt", "port": "port_cnn.pt",
               "vla": "vla.pt", "dkg": "dkg.json", "img_size": 128,
               "port_img_size": 96, "clip_dim": 32,
               "dkg_cfg": {"decay_lambda": 0.03, "min_confidence": 0.5,
                           "top_k_nodes": 4, "hops": 1,
                           "max_context_edges": 8}},
              os.path.join(run_dir, "manifest.json"))
    return run_dir


def test_pipeline_generate(tmp_path, tokenizer, corpus, chart_data, port_data):
    chart_rows, charts_dir = chart_data
    port_rows, ports_dir = port_data
    run_dir = _build_run(tmp_path, tokenizer, corpus)

    pipe = MacroTradePipeline.load(run_dir)
    chart = os.path.join(charts_dir, chart_rows[-1]["file"])
    port = os.path.join(ports_dir, port_rows[-1]["file"])
    month = corpus[-1]["month"]

    action = pipe.generate([chart], corpus[-1]["statement"], month,
                           port_path=port)
    assert set(action["action"]) == {"SPY", "QQQ", "TLT", "GLD", "DXY", "USO"}
    assert action["gross_exposure"] <= 1.0 + 1e-6
    assert action["month"] == month
    assert 0.0 <= action["signals"]["port_density"] <= 1.0
    assert -1.0 <= action["signals"]["hawk_dove"] <= 1.0
    assert isinstance(action["rationale"], list)
    assert isinstance(action["evidence"], list)
    assert all("[seen" not in c for c in action["rationale"])
    assert action["signals"]["retrieved_nodes"], "graph retrieval came back empty"
    assert all(abs(v) <= 1.0 for v in action["signals"]["asset_hints"].values())


def test_pipeline_without_port_image(tmp_path, tokenizer, corpus, chart_data):
    chart_rows, charts_dir = chart_data
    run_dir = _build_run(tmp_path, tokenizer, corpus)
    pipe = MacroTradePipeline.load(run_dir)
    chart = os.path.join(charts_dir, chart_rows[0]["file"])
    action = pipe.generate(chart, corpus[30]["statement"], corpus[30]["month"])
    assert action["signals"]["port_density"] == 0.5
