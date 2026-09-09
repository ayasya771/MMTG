"""Phase 2: construct the Dynamic Knowledge Graph from the text corpus.

Runs information extraction over every statement and news sentence, ingests
the triplets into the temporal graph, embeds every node with the Phase 1
text tower, scores extraction against the corpus ground truth, and saves
the graph as JSON.

Usage:
    python scripts/build_dkg.py [--run runs/main] [--llm] [--llm-model NAME]
"""

import argparse
import os

import _bootstrap
import numpy as np
import torch

from mmtg.config import ClipConfig, DkgConfig, Paths
from mmtg.dkg.extract import RuleBasedExtractor, llm_extract
from mmtg.dkg.graph import DynamicKnowledgeGraph
from mmtg.dkg.schema import ENTITIES
from mmtg.models.financial_clip import FinancialCLIP
from mmtg.tokenizer import FinancialTokenizer
from mmtg.utils import load_jsonl, log, save_json


def extraction_scores(pred: list, docs: list[dict]) -> dict:
    """Micro precision and recall of (h, r, t, month) sets versus ground truth."""
    truth = {(t["h"], t["r"], t["t"], d["month"])
             for d in docs for t in d["triplets"]}
    got = {(p.h, p.r, p.t, p.month) for p in pred}
    tp = len(truth & got)
    precision = tp / len(got) if got else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "n_pred": len(got), "n_truth": len(truth)}


@torch.no_grad()
def embed_nodes(run_dir: str, device: str = "cpu") -> dict[str, np.ndarray]:
    """Embed every canonical entity with the trained CLIP text tower."""
    tok = FinancialTokenizer.load(os.path.join(run_dir, "vocab.json"))
    ckpt = torch.load(os.path.join(run_dir, "clip.pt"), map_location="cpu",
                      weights_only=False)
    clip = FinancialCLIP(ClipConfig(**ckpt["cfg"]),
                         vocab_size=ckpt["meta"]["vocab_size"],
                         img_size=ckpt["meta"]["img_size"]).to(device).eval()
    clip.load_state_dict(ckpt["state_dict"])

    table: dict[str, np.ndarray] = {}
    for name, spec in ENTITIES.items():
        text = f"{name.replace('_', ' ').lower()}. {spec['gloss']}"
        ids = torch.tensor([tok.encode(text, clip.textual.max_len)],
                           dtype=torch.long, device=device)
        mask = (ids != tok.pad_id).long()
        table[name] = clip.encode_text(ids, mask)[0].cpu().numpy()
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-name", default="generated",
                        help="dataset subdirectory under data/")
    parser.add_argument("--run", default="runs/main")
    parser.add_argument("--llm", action="store_true",
                        help="use the LLM extraction engine (needs transformers "
                             "and local weights); default is the rule engine")
    parser.add_argument("--llm-model",
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    paths = Paths(data_name=args.data_name)
    run_dir = os.path.join(paths.root, args.run)
    docs = load_jsonl(f"{paths.data_dir}/corpus.jsonl")

    if args.llm:
        triplets = llm_extract(docs, model_name=args.llm_model)
        log(f"llm extraction produced {len(triplets)} triplets")
    else:
        triplets = RuleBasedExtractor().extract_corpus(docs)
        log(f"rule based extraction produced {len(triplets)} triplets")

    scores = extraction_scores(triplets, docs)
    log(f"extraction vs ground truth: {scores}")

    dkg = DynamicKnowledgeGraph(DkgConfig().decay_lambda)
    dkg.ingest(triplets)
    dkg.set_embeddings(embed_nodes(run_dir, args.device))
    dkg.save(os.path.join(run_dir, "dkg.json"))

    stats = dkg.stats()
    log(f"graph: {stats['n_nodes']} nodes, {stats['n_edges']} edges, "
        f"{stats['total_observations']} observations")

    last_month = docs[-1]["month"]
    chains = dkg.causal_chains("Federal_Reserve", "Tech_Equities", last_month)
    example = ["; ".join(f"{h} {r} {t}" for h, r, t in c) for c in chains[:3]]
    for line in example:
        log(f"chain: {line}")

    save_json({"extraction": scores, "graph": stats,
               "engine": "llm" if args.llm else "rule_based",
               "example_chains": example,
               "dkg_cfg": {"decay_lambda": DkgConfig().decay_lambda}},
              os.path.join(run_dir, "metrics_phase2.json"))


if __name__ == "__main__":
    main()
