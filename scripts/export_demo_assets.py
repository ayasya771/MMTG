"""Export run artifacts into the GitHub Pages demo.

Collects metrics, the knowledge graph, tilts, the backtest, sample images
(base64 inlined) and one fully generated action from a trained run, and
injects the bundle into docs/index.html between the demo-data script tags.
The page is fully self contained afterwards: no fetches, works from file://
and on GitHub Pages alike.

Usage:
    python scripts/export_demo_assets.py [--run runs/main]
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

from mmtg.config import Paths, TrainClipConfig
from mmtg.data.datasets import chronological_split
from mmtg.dkg.graph import DynamicKnowledgeGraph
from mmtg.pipeline import MacroTradePipeline
from mmtg.utils import load_json, load_jsonl, log

_DATA_RE = re.compile(
    r'(<script id="demo-data" type="application/json">).*?(</script>)',
    re.DOTALL)


def b64_png(path: str) -> str:
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def trimmed_dkg(dkg: DynamicKnowledgeGraph, as_of: str) -> dict:
    nodes = [{"id": n, "type": a.get("type"), "ticker": a.get("ticker"),
              "mentions": a.get("mentions", 0)}
             for n, a in dkg.g.nodes(data=True)]
    edges = [{"h": h, "t": t, "r": r, "count": a["count"],
              "first_seen": a["first_seen"], "last_seen": a["last_seen"],
              "weight": round(dkg.edge_weight(a, as_of), 3)}
             for h, t, r, a in dkg.g.edges(keys=True, data=True)]
    edges.sort(key=lambda e: e["weight"], reverse=True)
    return {"nodes": nodes, "edges": edges}


@torch.no_grad()
def clip_retrieval_sample(pipe: MacroTradePipeline, val_rows: list[dict],
                          charts_dir: str, row: dict,
                          pool: int = 256) -> tuple[list[dict], int]:
    """Top 3 captions CLIP retrieves for one chart from the held out pool.

    The pool is drawn from validation charts only, and the query's own
    caption is guaranteed to be in it, so a miss is a genuine ranking miss
    rather than an absent target.
    """
    texts = list(dict.fromkeys(r["caption"] for r in val_rows))[:pool]
    if row["caption"] not in texts:
        texts = [row["caption"]] + texts[: pool - 1]
    embs = np.stack([pipe.embed_text(t) for t in texts])
    q = pipe.embed_chart(os.path.join(charts_dir, row["file"]))
    order = np.argsort(-(embs @ q))[:3]
    return ([{"caption": texts[i], "match": texts[i] == row["caption"]}
             for i in order], len(texts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-name", default="generated",
                        help="dataset subdirectory under data/")
    parser.add_argument("--run", default="runs/main")
    args = parser.parse_args()

    paths = Paths(data_name=args.data_name)
    run_dir = os.path.join(paths.root, args.run)

    m1 = load_json(os.path.join(run_dir, "metrics_phase1.json"))
    m2 = load_json(os.path.join(run_dir, "metrics_phase2.json"))
    m3 = load_json(os.path.join(run_dir, "metrics_phase3.json"))
    backtest = load_json(os.path.join(run_dir, "backtest.json"))
    tilts = load_json(os.path.join(run_dir, "tilts_history.json"))
    chart_rows = load_jsonl(f"{paths.data_dir}/chart_manifest.jsonl")
    port_rows = load_jsonl(f"{paths.data_dir}/port_manifest.jsonl")
    corpus = load_jsonl(f"{paths.data_dir}/corpus.jsonl")
    dkg = DynamicKnowledgeGraph.load(os.path.join(run_dir, "dkg.json"))
    pipe = MacroTradePipeline.load(run_dir)

    last_val_month = next(t["month"] for t in reversed(tilts)
                          if t["split"] == "val")

    charts = [os.path.join(paths.charts_dir, r["file"])
              for r in chart_rows if r["month"] == last_val_month]
    statement = next(d["statement"] for d in corpus
                     if d["month"] == last_val_month)
    port = next((os.path.join(paths.ports_dir, r["file"])
                 for r in port_rows if r["month"] == last_val_month), None)
    action = pipe.generate(charts, statement, last_val_month, port_path=port)

    clip_cfg = TrainClipConfig()
    train_rows, val_rows_split = chronological_split(
        chart_rows, clip_cfg.train_end_month, clip_cfg.embargo_months)

    samples_charts = []
    retrieval_pool = 0
    for row in [r for r in chart_rows if r["month"] == last_val_month]:
        entry = {"chart_type": row["chart_type"], "month": row["month"],
                 "caption": row["caption"],
                 "png": b64_png(os.path.join(paths.charts_dir, row["file"]))}
        if not samples_charts:
            entry["retrieved"], retrieval_pool = clip_retrieval_sample(
                pipe, val_rows_split, paths.charts_dir, row)
        samples_charts.append(entry)

    ports_sorted = sorted(port_rows, key=lambda r: r["density"])
    picks = [ports_sorted[2], ports_sorted[-3]]
    samples_ports = []
    for p in picks:
        fpath = os.path.join(paths.ports_dir, p["file"])
        samples_ports.append({"density": p["density"],
                              "pred": pipe.port_density(fpath),
                              "png": b64_png(fpath)})

    statement_doc = next(d for d in corpus if d["month"] == last_val_month)
    sentences = statement_doc["statement"].split(". ")
    excerpt = ". ".join(s.rstrip(".") for s in sentences[:2]) + "."

    data = {
        "generated": dt.date.today().isoformat(),
        "metrics": {
            "clip": m1.get("clip_retrieval"),
            "port_cnn": m1.get("port_cnn"),
            "extraction": m2.get("extraction"),
            "graph": m2.get("graph"),
            "vla": {k: m3[k] for k in ("sign_agreement", "mae") if k in m3},
            "backtest": m3.get("backtest"),
        },
        "counts": {"clip_train": len(train_rows), "clip_val": len(val_rows_split),
                   "retrieval_pool": retrieval_pool},
        "dkg": trimmed_dkg(dkg, last_val_month),
        "tilts": [t for t in tilts if t["split"] != "train"][-72:],
        "backtest": {"rows": backtest["rows"]},
        "example_action": action,
        "samples": {"charts": samples_charts, "ports": samples_ports,
                    "statement": {"month": last_val_month, "text": excerpt,
                                  "hawk_dove": statement_doc["hawk_dove"]}},
    }

    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    index_path = os.path.join(paths.docs_dir, "index.html")
    with open(index_path, encoding="utf-8") as fh:
        html = fh.read()
    html, n = _DATA_RE.subn(lambda m: m.group(1) + payload + m.group(2), html)
    if n != 1:
        raise SystemExit("demo-data markers not found in docs/index.html")
    with open(index_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    log(f"injected {len(payload) / 1024:.0f} KB of demo data into docs/index.html")


if __name__ == "__main__":
    main()
