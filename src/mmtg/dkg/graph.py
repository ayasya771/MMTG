"""The Dynamic Knowledge Graph.

A NetworkX MultiDiGraph in which nodes are canonical macro entities and each
(head, relation, tail) pair is one edge that accumulates observations over
time: first_seen, last_seen, observation count, confidence and example
sources. Edges are weighted by recency decayed counts, and every query takes
an as_of month so retrieval never sees relations extracted from the future.
The structure serializes to plain JSON; a Neo4j export is a straight mapping
of the same node and edge dictionaries (see docs in README).
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from ..utils import load_json, month_diff, save_json
from .extract import Triplet
from .schema import ENTITIES

MAX_SOURCES_PER_EDGE = 3


class DynamicKnowledgeGraph:
    def __init__(self, decay_lambda: float = 0.03):
        self.g = nx.MultiDiGraph()
        self.decay_lambda = decay_lambda
        self.embeddings: dict[str, np.ndarray] = {}

    def _ensure_node(self, name: str) -> None:
        if not self.g.has_node(name):
            spec = ENTITIES.get(name, {})
            self.g.add_node(name, type=spec.get("type", "entity"),
                            gloss=spec.get("gloss", name.replace("_", " ").lower()),
                            ticker=spec.get("ticker"), mentions=0)

    def add_observation(self, h: str, r: str, t: str, month: str,
                        sentence: str = "", confidence: float = 0.9) -> None:
        self._ensure_node(h)
        self._ensure_node(t)
        self.g.nodes[h]["mentions"] += 1
        self.g.nodes[t]["mentions"] += 1
        if self.g.has_edge(h, t, key=r):
            e = self.g.edges[h, t, r]
            e["count"] += 1
            e["first_seen"] = min(e["first_seen"], month)
            e["last_seen"] = max(e["last_seen"], month)
            e["confidence"] = max(e["confidence"], confidence)
            if sentence and len(e["sources"]) < MAX_SOURCES_PER_EDGE \
                    and sentence not in e["sources"]:
                e["sources"].append(sentence)
        else:
            self.g.add_edge(h, t, key=r, relation=r, count=1,
                            first_seen=month, last_seen=month,
                            confidence=confidence,
                            sources=[sentence] if sentence else [])

    def ingest(self, triplets: list[Triplet]) -> None:
        for trip in sorted(triplets, key=lambda x: x.month):
            self.add_observation(trip.h, trip.r, trip.t, trip.month,
                                 trip.sentence, trip.confidence)

    def edge_weight(self, attrs: dict, as_of: str) -> float:
        """Recency decayed evidence weight of one edge at a given month."""
        if attrs["first_seen"] > as_of:
            return 0.0
        last = min(attrs["last_seen"], as_of)
        age = max(0, month_diff(as_of, last))
        return attrs["count"] * attrs["confidence"] * math.exp(-self.decay_lambda * age)

    def active_edges(self, as_of: str, min_weight: float = 1e-6) -> list[tuple]:
        """(h, t, relation, attrs, weight) for edges known by as_of."""
        out = []
        for h, t, r, attrs in self.g.edges(keys=True, data=True):
            w = self.edge_weight(attrs, as_of)
            if w > min_weight:
                out.append((h, t, r, attrs, w))
        return out

    def causal_chains(self, source: str, target: str, as_of: str,
                      max_hops: int = 3, min_weight: float = 0.05) -> list[list[tuple]]:
        """Directed paths source -> target through active edges.

        Returns lists of (h, relation, t) hops, strongest chains first,
        where a chain's strength is the weakest edge along it.
        """
        edges = self.active_edges(as_of, min_weight)
        flat = nx.DiGraph()
        for h, t, r, _attrs, w in edges:
            if not flat.has_edge(h, t) or flat.edges[h, t]["weight"] < w:
                flat.add_edge(h, t, weight=w, relation=r)
        if source not in flat or target not in flat:
            return []
        chains = []
        try:
            for path in nx.all_simple_paths(flat, source, target, cutoff=max_hops):
                hops = [(path[i], flat.edges[path[i], path[i + 1]]["relation"],
                         path[i + 1]) for i in range(len(path) - 1)]
                strength = min(flat.edges[path[i], path[i + 1]]["weight"]
                               for i in range(len(path) - 1))
                chains.append((strength, hops))
        except nx.NodeNotFound:
            return []
        chains.sort(key=lambda c: c[0], reverse=True)
        return [hops for _s, hops in chains]

    def set_embeddings(self, table: dict[str, np.ndarray]) -> None:
        self.embeddings = {k: np.asarray(v, dtype=np.float32) for k, v in table.items()}

    def to_dict(self) -> dict:
        nodes = [{"id": n, **{k: v for k, v in attrs.items()}}
                 for n, attrs in self.g.nodes(data=True)]
        edges = [{"h": h, "t": t, "r": r, **{k: v for k, v in attrs.items()
                                             if k != "relation"}}
                 for h, t, r, attrs in self.g.edges(keys=True, data=True)]
        return {
            "decay_lambda": self.decay_lambda,
            "nodes": nodes,
            "edges": edges,
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
        }

    def save(self, path: str) -> None:
        save_json(self.to_dict(), path)

    @classmethod
    def from_dict(cls, d: dict) -> "DynamicKnowledgeGraph":
        dkg = cls(decay_lambda=d.get("decay_lambda", 0.03))
        for node in d["nodes"]:
            attrs = {k: v for k, v in node.items() if k != "id"}
            dkg.g.add_node(node["id"], **attrs)
        for edge in d["edges"]:
            attrs = {k: v for k, v in edge.items() if k not in ("h", "t", "r")}
            dkg.g.add_edge(edge["h"], edge["t"], key=edge["r"],
                           relation=edge["r"], **attrs)
        dkg.embeddings = {k: np.asarray(v, dtype=np.float32)
                          for k, v in d.get("embeddings", {}).items()}
        return dkg

    @classmethod
    def load(cls, path: str) -> "DynamicKnowledgeGraph":
        return cls.from_dict(load_json(path))

    def stats(self) -> dict:
        rel_counts: dict[str, int] = {}
        for _h, _t, r, attrs in self.g.edges(keys=True, data=True):
            rel_counts[r] = rel_counts.get(r, 0) + attrs["count"]
        return {
            "n_nodes": self.g.number_of_nodes(),
            "n_edges": self.g.number_of_edges(),
            "total_observations": sum(
                a["count"] for *_ignored, a in self.g.edges(keys=True, data=True)),
            "relations": dict(sorted(rel_counts.items(),
                                     key=lambda kv: kv[1], reverse=True)),
        }
