"""Retrieval augmented graph context (the RAG stage).

Given a query embedding (the statement text embedded by the Financial-CLIP
text tower), retrieve the most similar graph nodes, expand to their local
neighborhood as of the query month, and serialize the strongest edges into
a compact context: text lines for rationale, one pooled context embedding
for the action head, and per asset directional hints derived from relation
directions terminating on asset nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import ASSETS, DkgConfig
from .graph import DynamicKnowledgeGraph
from .schema import RELATIONS, entity_ticker

_STANCE_DEADBAND = 0.15


@dataclass
class SubgraphContext:
    nodes: list[str] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    context_lines: list[str] = field(default_factory=list)
    context_embedding: np.ndarray | None = None
    asset_hints: dict[str, float] = field(default_factory=dict)
    chains: list[str] = field(default_factory=list)


def _edge_line(h: str, r: str, t: str, attrs: dict, weight: float) -> str:
    return (f"{h} --{r}--> {t} "
            f"[seen {attrs['count']}x, latest {attrs['last_seen']}, "
            f"w={weight:.2f}]")


def verbalize_chain(hops: list[tuple]) -> str:
    return "; ".join(f"{h} {r} {t}" for h, r, t in hops)


class GraphRetriever:
    def __init__(self, dkg: DynamicKnowledgeGraph, cfg: DkgConfig | None = None):
        self.dkg = dkg
        self.cfg = cfg or DkgConfig()
        names = sorted(dkg.embeddings)
        self._names = names
        if names:
            mat = np.stack([dkg.embeddings[n] for n in names]).astype(np.float32)
            mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
            self._matrix = mat
        else:
            self._matrix = np.zeros((0, 1), dtype=np.float32)

    def top_nodes(self, query_emb: np.ndarray, as_of: str, k: int) -> list[str]:
        if self._matrix.shape[0] == 0:
            return []
        q = np.asarray(query_emb, dtype=np.float32).ravel()
        q /= np.linalg.norm(q) + 1e-9
        sims = self._matrix @ q
        active = {n for edge in self.dkg.active_edges(as_of) for n in edge[:2]}
        order = np.argsort(-sims)
        picked = [self._names[i] for i in order if self._names[i] in active]
        return picked[:k]

    def retrieve(self, query_emb: np.ndarray, as_of: str,
                 stance: float = 0.0) -> SubgraphContext:
        """Retrieve graph context for one query.

        `stance` is the hawk/dove score of the source text. The graph holds
        both tightening and easing relations because both occurred over the
        history, so without a stance filter a hawkish statement can be
        rationalized by an easing chain and a dovish one by a tightening
        chain. Passing the stance suppresses chains routed through the
        opposing policy theme, which is the difference between a rationale
        and a list of everything the graph knows.
        """
        cfg = self.cfg
        seeds = self.top_nodes(query_emb, as_of, cfg.top_k_nodes)
        ctx = SubgraphContext()
        if not seeds:
            return ctx

        keep = set(seeds)
        edges = self.dkg.active_edges(as_of)
        for _hop in range(cfg.hops):
            frontier = set()
            for h, t, _r, _a, _w in edges:
                if h in keep:
                    frontier.add(t)
                if t in keep:
                    frontier.add(h)
            keep |= frontier

        sub = [(h, t, r, a, w) for h, t, r, a, w in edges
               if h in keep and t in keep and a["confidence"] >= cfg.min_confidence]
        sub.sort(key=lambda e: e[4], reverse=True)
        sub = sub[: cfg.max_context_edges]

        ctx.nodes = sorted({n for h, t, *_ in sub for n in (h, t)} | set(seeds))
        ctx.edges = [{"h": h, "t": t, "r": r, "weight": round(w, 3),
                      "count": a["count"], "last_seen": a["last_seen"]}
                     for h, t, r, a, w in sub]
        ctx.context_lines = [_edge_line(h, r, t, a, w) for h, t, r, a, w in sub]

        node_w: dict[str, float] = {n: 0.1 for n in seeds}
        for h, t, _r, _a, w in sub:
            node_w[h] = node_w.get(h, 0.0) + w
            node_w[t] = node_w.get(t, 0.0) + w
        vecs, weights = [], []
        for n, w in node_w.items():
            if n in self.dkg.embeddings:
                vecs.append(self.dkg.embeddings[n])
                weights.append(w)
        if vecs:
            pooled = np.average(np.stack(vecs), axis=0, weights=np.asarray(weights))
            pooled = pooled / (np.linalg.norm(pooled) + 1e-9)
            ctx.context_embedding = pooled.astype(np.float32)

        raw_hints: dict[str, float] = {}
        for _h, t, r, _a, w in sub:
            direction = RELATIONS.get(r, {}).get("direction", "flat")
            sign = {"up": 1.0, "down": -1.0}.get(direction, 0.0)
            ticker = entity_ticker(t)
            if ticker in ASSETS and sign != 0.0:
                raw_hints[ticker] = raw_hints.get(ticker, 0.0) + sign * w
        scale = max((abs(v) for v in raw_hints.values()), default=0.0)
        if scale > 0:
            ctx.asset_hints = {k: round(v / scale, 3)
                               for k, v in sorted(raw_hints.items(),
                                                  key=lambda kv: -abs(kv[1]))}

        blocked = set()
        if stance > _STANCE_DEADBAND:
            blocked = {"Policy_Easing"}
        elif stance < -_STANCE_DEADBAND:
            blocked = {"Policy_Tightening"}
        sources = [s for s in ("Federal_Reserve", "Policy_Tightening",
                               "Policy_Easing", "Recession_Risk")
                   if s not in blocked]
        raw_chains: list[list[tuple]] = []
        for asset_entity in [n for n in ctx.nodes if entity_ticker(n)]:
            for chains_from in sources:
                if chains_from not in ctx.nodes:
                    continue
                for hops in self.dkg.causal_chains(
                        chains_from, asset_entity, as_of, max_hops=4):
                    if any(node in blocked for h, _r, t in hops
                           for node in (h, t)):
                        continue
                    raw_chains.append(hops)
                    break
        ctx.chains = _dedupe_chains(raw_chains)[:4]
        return ctx


def _dedupe_chains(chains: list[list[tuple]]) -> list[str]:
    """Drop chains fully contained in a longer one, then dedupe by text.

    Retrieval routinely surfaces both "Fed signals Tightening; Tightening
    lifts Yields; Yields weigh on Bonds" and its own tail. Showing both as
    separate reasons overstates the evidence, so only the maximal chain
    survives.
    """
    ordered = sorted(chains, key=len, reverse=True)
    kept: list[list[tuple]] = []
    for chain in ordered:
        if not any(_is_subsequence(chain, longer) for longer in kept):
            kept.append(chain)
    seen: set[str] = set()
    out: list[str] = []
    for chain in kept:
        text = verbalize_chain(chain)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _is_subsequence(short: list[tuple], long: list[tuple]) -> bool:
    """True when `short` appears as a contiguous run of hops inside `long`."""
    n, m = len(short), len(long)
    if n > m:
        return False
    return any(long[i: i + n] == short for i in range(m - n + 1))
