"""Export a trained run into browser-runnable assets for the GitHub Pages demo.

This is what makes the published page *live* rather than a static snapshot.
It produces, under ``docs/web-assets/``:

- ``text_enc.onnx``    Financial-CLIP text tower (ids, mask -> normalized embedding)
- ``image_enc.onnx``   Financial-CLIP vision tower (chart PNG tensor -> normalized embedding)
- ``port_cnn.onnx``    container density CNN (port PNG tensor -> [0, 1] density)
- ``vla.onnx``         the VLA action head (fusion inputs -> 6 asset weights)
- ``bundle.json``      tokenizer vocab, knowledge graph (temporal edges + node
                       embeddings), relation directions, hawk/dove lexicons,
                       image normalization constants and Python reference
                       outputs used by the page's on-load parity self-check
- ``explorer.json``    per held-out-month content (statement, chart embeddings,
                       retrieved subgraph, the pipeline's generated action) so
                       the demo's explorer section is instant, plus a few
                       months of base64 chart / port imagery

It then injects the run's metrics, tilts and backtest into the demo-data
JSON block of ``docs/index.html``.

Usage:
    python scripts/export_web_model.py [--run runs/main] [--showcase 12]
"""

import argparse
import base64
import datetime as dt
import json
import os
import re

import _bootstrap

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmtg.config import ASSETS, ChartConfig, DkgConfig, Paths, PortConfig
from mmtg.data.datasets import load_image_tensor
from mmtg.dkg.graph import DynamicKnowledgeGraph
from mmtg.dkg.schema import DOVISH_TERMS, HAWKISH_TERMS, RELATIONS, ENTITIES
from mmtg.pipeline import MacroTradePipeline, hawk_dove
from mmtg.utils import load_json, load_jsonl, log

_DOCS_ASSETS = "docs/web-assets"
_DATA_RE = re.compile(
    r'(<script id="demo-data" type="application/json">).*?(</script>)',
    re.DOTALL)

OPSET = 17


class _TextTower(nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.textual = clip.textual

    def forward(self, ids, mask):
        return F.normalize(self.textual(ids, mask), dim=-1)


class _VisionTower(nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.visual = clip.visual

    def forward(self, image):
        return F.normalize(self.visual(image), dim=-1)


class _PortWrapper(nn.Module):
    """Mirror of PortDensityCNN.predict: clamp density to [0, 1]."""

    def __init__(self, port_cnn):
        super().__init__()
        self.features = port_cnn.features
        self.head = port_cnn.head

    def forward(self, image):
        return self.head(self.features(image)).squeeze(-1).clamp(0.0, 1.0)


def _export(model, sample_inputs, names, path: str) -> None:
    torch.onnx.export(
        model, sample_inputs, path,
        input_names=names["inputs"], output_names=names["outputs"],
        dynamic_axes={n: {0: "B"} for n in names["inputs"]},
        opset_version=OPSET, do_constant_folding=True,
    )
    _inline_weights(path)
    log(f"exported {path}")


def _inline_weights(path: str) -> None:
    """Fold external .onnx.data weights back into the single .onnx file.

    onnxruntime-web cannot fetch sidecar external-data files over HTTP; it
    fails with "Module.MountedFiles is not available". These models are a few
    MB each, so embedding the weights keeps the browser demo loadable and
    removes an entire class of deployment breakage.
    """
    import onnx

    model = onnx.load(path, load_external_data=True)
    onnx.save(model, path, save_as_external_data=False)
    sidecar = path + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)


def export_onnx(pipe: MacroTradePipeline, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    B = 2

    _export(_TextTower(pipe.clip),
            (torch.zeros(B, pipe.clip.textual.max_len, dtype=torch.long),
             torch.ones(B, pipe.clip.textual.max_len, dtype=torch.long)),
            {"inputs": ["ids", "mask"], "outputs": ["text_emb"]},
            os.path.join(out_dir, "text_enc.onnx"))

    img_size = pipe.manifest["img_size"]
    _export(_VisionTower(pipe.clip),
            (torch.zeros(B, 3, img_size, img_size),),
            {"inputs": ["image"], "outputs": ["image_emb"]},
            os.path.join(out_dir, "image_enc.onnx"))

    port_size = pipe.manifest["port_img_size"]
    _export(_PortWrapper(pipe.port_cnn),
            (torch.zeros(B, 3, port_size, port_size),),
            {"inputs": ["port_image"], "outputs": ["density"]},
            os.path.join(out_dir, "port_cnn.onnx"))

    clip_dim = pipe.manifest["clip_dim"]
    vla_cfg = pipe.vla.cfg
    _export(pipe.vla,
            (torch.zeros(B, clip_dim),
             torch.zeros(B, vla_cfg.max_statement_len, dtype=torch.long),
             torch.ones(B, vla_cfg.max_statement_len, dtype=torch.long),
             torch.zeros(B, clip_dim),
             torch.zeros(B, 2)),
            {"inputs": ["chart_emb", "ids", "mask", "ctx_emb", "feats"],
             "outputs": ["weights"]},
            os.path.join(out_dir, "vla.onnx"))


def onnx_text(pipe: MacroTradePipeline, text: str) -> np.ndarray:
    """Normalized embedding of `text` evaluated through the exported ONNX."""
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(_DOCS_ASSETS, "text_enc.onnx"),
                                providers=["CPUExecutionProvider"])
    ids = torch.tensor([pipe.tok.encode(text, pipe.clip.textual.max_len)])
    mask = (ids != pipe.tok.pad_id).long()
    (out,) = sess.run(None, {"ids": ids.numpy(), "mask": mask.numpy()})
    return out[0]


def onnx_image(pipe: MacroTradePipeline, png_path: str) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(_DOCS_ASSETS, "image_enc.onnx"),
                                providers=["CPUExecutionProvider"])
    img = load_image_tensor(png_path, pipe.manifest["img_size"]).unsqueeze(0)
    (out,) = sess.run(None, {"image": img.numpy()})
    return out[0]


def onnx_port(pipe: MacroTradePipeline, png_path: str) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(_DOCS_ASSETS, "port_cnn.onnx"),
                                providers=["CPUExecutionProvider"])
    img = load_image_tensor(png_path, pipe.manifest["port_img_size"]).unsqueeze(0)
    (out,) = sess.run(None, {"port_image": img.numpy()})
    return np.asarray(out).reshape(-1)


def onnx_vla(pipe: MacroTradePipeline, chart_emb: np.ndarray, text: str,
             ctx_emb: np.ndarray, feats: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(_DOCS_ASSETS, "vla.onnx"),
                                providers=["CPUExecutionProvider"])
    ids = torch.tensor([pipe.tok.encode(text, pipe.vla.cfg.max_statement_len)])
    mask = (ids != pipe.tok.pad_id).long()
    (out,) = sess.run(None, {
        "chart_emb": chart_emb.astype(np.float32)[None],
        "ids": ids.numpy(), "mask": mask.numpy(),
        "ctx_emb": ctx_emb.astype(np.float32)[None],
        "feats": feats.astype(np.float32)[None]})
    return out[0]


def sanitize(obj):
    """Recursively convert numpy / torch values to JSON primitives."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def b64_png(path: str) -> str:
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def verify_onnx_parity(pipe: MacroTradePipeline, sample: str,
                       chart_path: str, port_path: str) -> dict:
    """Assert exported ONNX reproduces the torch evaluation, then store the
    ONNX outputs as the page's byte-reference for its on-load self-check."""
    with torch.no_grad():
        q = torch.tensor([pipe.tok.encode(sample, pipe.clip.textual.max_len)])
        qm = (q != pipe.tok.pad_id).long()
        torch_text = pipe.clip.encode_text(q, qm)[0].cpu().numpy()
        img = load_image_tensor(chart_path, pipe.manifest["img_size"]).unsqueeze(0)
        torch_image = pipe.clip.encode_image(img)[0].cpu().numpy()
        pimg = load_image_tensor(port_path, pipe.manifest["port_img_size"]).unsqueeze(0)
        torch_density = float(pipe.port_cnn.predict(pimg)[0])

    hd = hawk_dove(sample)
    ids96 = torch.tensor([pipe.tok.encode(sample, pipe.vla.cfg.max_statement_len)])
    mask96 = (ids96 != pipe.tok.pad_id).long()
    ctx = pipe.retriever.retrieve(torch_text, "2025-12", stance=hd)
    ctx_emb = (ctx.context_embedding if ctx.context_embedding is not None
               else np.zeros(pipe.manifest["clip_dim"], np.float32))
    chart_emb = onnx_image(pipe, chart_path)
    feats = np.asarray([torch_density, hd], dtype=np.float32)

    text_out = onnx_text(pipe, sample)
    image_out = onnx_image(pipe, chart_path)
    port_out = onnx_port(pipe, port_path)
    vla_out = onnx_vla(pipe, chart_emb, sample, ctx_emb, feats)

    checks = {
        "text": float(np.abs(text_out - torch_text).max()),
        "image": float(np.abs(image_out - torch_image).max()),
        "port": float(abs(port_out[0] - torch_density)),
    }
    ref = {
        "text": {"text": sample, "ids": q.numpy()[0].tolist(),
                 "mask": qm.numpy()[0].tolist(),
                 "out": text_out.tolist()},
        "image": {"png": b64_png(chart_path), "out": image_out.tolist()},
        "port": {"png": b64_png(port_path), "out": [float(port_out[0])]},
        "vla": {"text": sample, "chart_emb": chart_emb.tolist(),
                "ids": ids96.numpy()[0].tolist(), "mask": mask96.numpy()[0].tolist(),
                "ctx_emb": ctx_emb.tolist(), "feats": feats.tolist(),
                "out": vla_out.tolist()},
    }
    for k, v in checks.items():
        assert v < 1e-4, f"ONNX parity broken for {k}: {v}"
    log(f"ONNX vs torch parity verified {checks}")
    return ref


def trimmed_dkg(dkg: DynamicKnowledgeGraph, as_of: str) -> dict:
    nodes = [{"id": n, "type": a.get("type"), "ticker": a.get("ticker"),
              "mentions": a.get("mentions", 0)}
             for n, a in dkg.g.nodes(data=True)]
    edges = [{"h": h, "t": t, "r": r, "count": a["count"],
              "first_seen": a["first_seen"], "last_seen": a["last_seen"],
              "confidence": a["confidence"]}
             for h, t, r, a in dkg.g.edges(keys=True, data=True)]
    return {"nodes": nodes, "edges": edges,
            "embeddings": {k: sanitize(v) for k, v in dkg.embeddings.items()},
            "decay_lambda": dkg.decay_lambda}


def build_bundle(pipe, dkg, tok, refs: dict, out_dir: str,
                 caption_pool: list[str], caption_embs: list[list[float]]) -> None:
    bundle = {
        "generated": dt.date.today().isoformat(),
        "constants": {
            "assets": ASSETS,
            "asset_entity": {k: v for k, v in [
                ("SPY", "US_Equities"), ("QQQ", "Tech_Equities"),
                ("TLT", "Long_Treasuries"), ("GLD", "Gold"),
                ("DXY", "US_Dollar"), ("USO", "Crude_Oil")]},
            "img_mean": 0.5, "img_std": 0.27,
            "img_size": pipe.manifest["img_size"],
            "port_size": pipe.manifest["port_img_size"],
            "clip_dim": pipe.manifest["clip_dim"],
            "max_text_len": pipe.clip.textual.max_len,
            "max_statement_len": pipe.vla.cfg.max_statement_len,
            "pad_id": tok.pad_id, "unk_id": tok.unk_id,
            "cls_id": tok.cls_id, "sep_id": tok.sep_id,
        },
        "tokenizer": tok.vocab,
        "hawk_terms": HAWKISH_TERMS,
        "dove_terms": DOVISH_TERMS,
        "relations": {name: spec["direction"]
                      for name, spec in RELATIONS.items()},
        "entity_tickers": {name: spec.get("ticker")
                           for name, spec in ENTITIES.items()},
        "dkg": trimmed_dkg(dkg, "2025-12"),
        "caption_pool": caption_pool,
        "caption_embs": caption_embs,
        "refs": refs,
    }
    path = os.path.join(out_dir, "bundle.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    log(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")


def build_explorer(pipe: MacroTradePipeline, paths: Paths, chart_rows, port_rows,
                   corpus_map: dict, tilts_by_month: dict,
                   showcase: int, run_dir: str) -> dict:
    months = sorted(m for m in tilts_by_month)
    step = max(1, len(months) // showcase)
    showcase_months = set(months[::step]) if step > 1 else set(months[:2])

    charts_by_month: dict[str, list[dict]] = {}
    for r in chart_rows:
        charts_by_month.setdefault(r["month"], []).append(r)
    ports_by_month: dict[str, list[dict]] = {}
    for r in port_rows:
        ports_by_month.setdefault(r["month"], []).append(r)

    out = []
    for month in months:
        doc = corpus_map[month]
        cs = charts_by_month.get(month, [])
        charts = [os.path.join(paths.charts_dir, r["file"]) for r in cs]
        port_row = (ports_by_month.get(month) or [None])[0]
        port_path = (os.path.join(paths.ports_dir, port_row["file"])
                     if port_row else None)

        embs = [pipe.embed_chart(p) for p in charts]
        chart_emb = np.mean(embs, axis=0)
        chart_emb /= np.linalg.norm(chart_emb) + 1e-9

        hd = float(doc.get("hawk_dove", 0.0))
        ctx = pipe.retriever.retrieve(pipe.embed_text(doc["statement"]),
                                      month, stance=hd)
        density = (pipe.port_density(port_path) if port_path else 0.5)

        entry = {
            "month": month,
            "regime": doc.get("regime"),
            "statement": doc["statement"],
            "hd": round(hd, 3),
            "density": round(float(density), 3),
            "chart_emb": chart_emb.tolist(),
            "chart_captions": [r["caption"] for r in cs],
            "ctx": {
                "nodes": ctx.nodes,
                "edges": [[e["h"], e["t"], e["r"], round(float(e["weight"]), 3)]
                          for e in ctx.edges],
                "chains": ctx.chains,
                "evidence": ctx.context_lines[:4],
                "asset_hints": ctx.asset_hints,
            },
        }
        if month in showcase_months:
            entry["images"] = {
                "charts": [b64_png(p) for p in charts],
                "port": b64_png(port_path) if port_path else None,
            }
        out.append(entry)

    payload = {"generated": dt.date.today().isoformat(), "months": out}
    path = os.path.join(_DOCS_ASSETS, "explorer.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    log(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB, "
        f"{len(out)} months, {len(showcase_months)} showcase)")


def demo_payload(paths: Paths, run_dir: str, tilts: list[dict]) -> dict:
    m1 = load_json(os.path.join(run_dir, "metrics_phase1.json"))
    m2 = load_json(os.path.join(run_dir, "metrics_phase2.json"))
    m3 = load_json(os.path.join(run_dir, "metrics_phase3.json"))
    backtest = load_json(os.path.join(run_dir, "backtest.json"))
    dkg = DynamicKnowledgeGraph.load(os.path.join(run_dir, "dkg.json"))
    return {
        "generated": dt.date.today().isoformat(),
        "metrics": {
            "clip": m1.get("clip_retrieval"),
            "port_cnn": m1.get("port_cnn"),
            "extraction": m2.get("extraction"),
            "graph": m2.get("graph"),
            "vla": {k: m3[k] for k in ("sign_agreement", "mae") if k in m3},
            "backtest": m3.get("backtest"),
        },
        "dkg": trimmed_dkg(dkg, "2025-12"),
        "tilts": [t for t in tilts],
        "backtest": {"rows": backtest["rows"]},
        "example_action": load_json(os.path.join(run_dir, "example_action.json")),
    }


def inject_demo_data(payload: dict) -> None:
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    index_path = os.path.join(_DOCS_ASSETS, "..", "index.html")
    with open(index_path, encoding="utf-8") as fh:
        html = fh.read()
    html, n = _DATA_RE.subn(lambda m: m.group(1) + blob + m.group(2), html)
    if n != 1:
        raise SystemExit("demo-data markers not found in docs/index.html")
    with open(index_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    log(f"injected {len(blob) / 1024:.0f} KB demo data into docs/index.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-name", default="generated")
    parser.add_argument("--run", default="runs/main")
    parser.add_argument("--showcase", type=int, default=12,
                        help="how many explorer months get inline imagery")
    args = parser.parse_args()

    paths = Paths(data_name=args.data_name)
    run_dir = os.path.join(paths.root, args.run)
    out_dir = os.path.join(paths.root, _DOCS_ASSETS)
    os.makedirs(out_dir, exist_ok=True)

    pipe = MacroTradePipeline.load(run_dir)
    dkg = DynamicKnowledgeGraph.load(os.path.join(run_dir, "dkg.json"))
    tilts = load_json(os.path.join(run_dir, "tilts_history.json"))
    val_months = {t["month"] for t in tilts if t["split"] == "val"}

    tokenizer = pipe.tok
    corpus = load_jsonl(f"{paths.data_dir}/corpus.jsonl")
    corpus_map = {d["month"]: d for d in corpus}
    chart_rows = load_jsonl(f"{paths.data_dir}/chart_manifest.jsonl")
    port_rows = load_jsonl(f"{paths.data_dir}/port_manifest.jsonl")

    sample_stmt = next(c["statement"] for c in corpus
                       if c["month"] == sorted(val_months)[len(val_months) // 2])
    chart_file = next(r["file"] for r in chart_rows
                      if r["month"] in val_months)
    port_file = next(r["file"] for r in port_rows if r["month"] in val_months)

    export_onnx(pipe, out_dir)
    refs = verify_onnx_parity(
        pipe, sample_stmt,
        os.path.join(paths.charts_dir, chart_file),
        os.path.join(paths.ports_dir, port_file))

    pool = list(dict.fromkeys(
        r["caption"] for r in chart_rows if r["month"] in val_months))[:256]
    with torch.no_grad():
        caption_embs = []
        for text in pool:
            ids = torch.tensor([tokenizer.encode(text, pipe.clip.textual.max_len)])
            mask = (ids != tokenizer.pad_id).long()
            caption_embs.append(
                pipe.clip.encode_text(ids, mask)[0].cpu().numpy().tolist())
    build_bundle(pipe, dkg, tokenizer, refs, out_dir, pool, caption_embs)

    tilts_by_month = {t["month"]: t for t in tilts if t["split"] == "val"}
    build_explorer(pipe, paths, chart_rows, port_rows, corpus_map,
                   tilts_by_month, args.showcase, run_dir)

    inject_demo_data(demo_payload(paths, run_dir, tilts))
    log("export complete")


if __name__ == "__main__":
    main()
