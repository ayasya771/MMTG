import numpy as np

from mmtg.data.statements import CAUSAL_BANK
from mmtg.config import DkgConfig
from mmtg.dkg.extract import RuleBasedExtractor, split_sentences
from mmtg.dkg.graph import DynamicKnowledgeGraph
from mmtg.dkg.retrieve import GraphRetriever
from mmtg.dkg.schema import ENTITIES, RELATIONS


def test_schema_consistency():
    for name, spec in RELATIONS.items():
        assert spec["surfaces"], name
    surface_owner = {}
    for name, spec in ENTITIES.items():
        for s in spec["surfaces"]:
            assert s == s.lower()
            assert surface_owner.setdefault(s, name) == name, \
                f"surface '{s}' claimed by {surface_owner[s]} and {name}"


def test_every_causal_template_extracts():
    ex = RuleBasedExtractor()
    for pool in CAUSAL_BANK.values():
        for sentence, truth in pool:
            got = {t.key() for t in ex.extract_sentence(sentence, "2020-01")}
            assert truth in got, f"failed on: {sentence} -> {got}"


def test_extractor_ignores_plain_text():
    ex = RuleBasedExtractor()
    assert ex.extract_sentence("Unemployment stood at 4.2%.", "2020-01") == []


def test_corpus_extraction_perfect_on_schema_text(corpus):
    ex = RuleBasedExtractor()
    pred = ex.extract_corpus(corpus)
    truth = {(t["h"], t["r"], t["t"], d["month"])
             for d in corpus for t in d["triplets"]}
    got = {(p.h, p.r, p.t, p.month) for p in pred}
    assert truth == got


def test_split_sentences():
    parts = split_sentences("Rates rose. Inflation cooled! Curve inverted?")
    assert len(parts) == 3


def _small_graph():
    dkg = DynamicKnowledgeGraph(decay_lambda=0.05)
    dkg.add_observation("Federal_Reserve", "raises", "Policy_Rate",
                        "2022-03", "s1", 0.9)
    dkg.add_observation("Federal_Reserve", "raises", "Policy_Rate",
                        "2022-06", "s2", 0.9)
    dkg.add_observation("Policy_Rate", "lifts", "Real_Yields", "2022-06",
                        "s3", 0.9)
    dkg.add_observation("Real_Yields", "compresses", "Tech_Multiples",
                        "2022-07", "s4", 0.9)
    dkg.add_observation("Tech_Multiples", "weighs_on", "Tech_Equities",
                        "2022-08", "s5", 0.9)
    return dkg


def test_graph_merges_repeat_observations():
    dkg = _small_graph()
    edge = dkg.g.edges["Federal_Reserve", "Policy_Rate", "raises"]
    assert edge["count"] == 2
    assert edge["first_seen"] == "2022-03"
    assert edge["last_seen"] == "2022-06"


def test_edge_weight_temporal_masking():
    dkg = _small_graph()
    attrs = dkg.g.edges["Real_Yields", "Tech_Multiples", "compresses"]
    assert dkg.edge_weight(attrs, "2022-01") == 0.0
    fresh = dkg.edge_weight(attrs, "2022-07")
    stale = dkg.edge_weight(attrs, "2024-07")
    assert fresh > stale > 0.0


def test_causal_chain_found():
    dkg = _small_graph()
    chains = dkg.causal_chains("Federal_Reserve", "Tech_Equities",
                               "2022-12", max_hops=4)
    assert chains
    hops = chains[0]
    assert hops[0][0] == "Federal_Reserve"
    assert hops[-1][2] == "Tech_Equities"
    assert dkg.causal_chains("Federal_Reserve", "Tech_Equities",
                             "2022-05", max_hops=4) == []


def test_graph_roundtrip(tmp_path):
    dkg = _small_graph()
    dkg.set_embeddings({"Federal_Reserve": np.ones(8, dtype=np.float32)})
    path = str(tmp_path / "dkg.json")
    dkg.save(path)
    loaded = DynamicKnowledgeGraph.load(path)
    assert loaded.stats() == dkg.stats()
    assert np.allclose(loaded.embeddings["Federal_Reserve"],
                       dkg.embeddings["Federal_Reserve"])


def _retriever():
    dkg = _small_graph()
    rng = np.random.default_rng(0)
    names = list(dkg.g.nodes)
    base = {n: rng.normal(size=16).astype(np.float32) for n in names}
    dkg.set_embeddings(base)
    return dkg, GraphRetriever(dkg, DkgConfig(top_k_nodes=2, hops=1))


def test_retrieval_returns_seeded_subgraph():
    dkg, retriever = _retriever()
    query = dkg.embeddings["Federal_Reserve"]
    ctx = retriever.retrieve(query, "2022-12")
    assert "Federal_Reserve" in ctx.nodes
    assert ctx.edges
    assert ctx.context_embedding is not None
    assert abs(np.linalg.norm(ctx.context_embedding) - 1.0) < 1e-5


def test_retrieval_respects_as_of():
    dkg, retriever = _retriever()
    query = dkg.embeddings["Tech_Multiples"]
    past = retriever.retrieve(query, "2022-01")
    assert past.edges == []
    later = retriever.retrieve(query, "2022-12")
    assert later.edges


def test_dedupe_chains_drops_contained_tails():
    from mmtg.dkg.retrieve import _dedupe_chains

    full = [("Federal_Reserve", "signals", "Policy_Tightening"),
            ("Policy_Tightening", "lifts", "Long_End_Yields"),
            ("Long_End_Yields", "weighs_on", "Long_Treasuries")]
    tail = full[1:]
    other = [("CPI_Inflation", "boosts", "Gold")]
    out = _dedupe_chains([full, tail, other, list(full)])
    assert len(out) == 2
    assert any("Federal_Reserve signals" in c for c in out)
    assert any("CPI_Inflation boosts Gold" in c for c in out)
    assert not any(c.startswith("Policy_Tightening lifts") for c in out)


def test_dedupe_chains_keeps_disjoint_same_length():
    from mmtg.dkg.retrieve import _dedupe_chains

    a = [("Policy_Easing", "boosts", "US_Equities")]
    b = [("Recession_Risk", "boosts", "Long_Treasuries")]
    assert len(_dedupe_chains([a, b])) == 2


def test_retrieval_chains_have_no_nested_duplicates():
    dkg, retriever = _retriever()
    ctx = retriever.retrieve(dkg.embeddings["Federal_Reserve"], "2022-12")
    for i, outer in enumerate(ctx.chains):
        for j, inner in enumerate(ctx.chains):
            if i != j:
                assert inner not in outer


def test_asset_hints_direction_and_scale():
    dkg, retriever = _retriever()
    ctx = retriever.retrieve(dkg.embeddings["Tech_Equities"], "2022-12")
    if "QQQ" in ctx.asset_hints:
        assert ctx.asset_hints["QQQ"] < 0
    assert all(abs(v) <= 1.0 + 1e-9 for v in ctx.asset_hints.values())
    if ctx.asset_hints:
        assert max(abs(v) for v in ctx.asset_hints.values()) == 1.0


def _stance_graph():
    dkg = DynamicKnowledgeGraph(decay_lambda=0.01)
    for month in ("2020-01", "2021-01"):
        dkg.add_observation("Federal_Reserve", "signals", "Policy_Tightening",
                            month, "hawkish", 0.9)
        dkg.add_observation("Policy_Tightening", "boosts", "US_Dollar",
                            month, "hawkish", 0.9)
        dkg.add_observation("Federal_Reserve", "signals", "Policy_Easing",
                            month, "dovish", 0.9)
        dkg.add_observation("Policy_Easing", "boosts", "Tech_Equities",
                            month, "dovish", 0.9)
    rng = np.random.default_rng(3)
    dkg.set_embeddings({n: rng.normal(size=16).astype(np.float32)
                        for n in dkg.g.nodes})
    return dkg, GraphRetriever(dkg, DkgConfig(top_k_nodes=6, hops=1,
                                              max_context_edges=12))


def test_stance_filter_suppresses_contradictory_chains():
    dkg, retriever = _stance_graph()
    q = dkg.embeddings["Federal_Reserve"]

    hawkish = retriever.retrieve(q, "2022-01", stance=0.8).chains
    assert hawkish, "a hawkish query should still produce a rationale"
    assert not any("Policy_Easing" in c for c in hawkish)

    dovish = retriever.retrieve(q, "2022-01", stance=-0.8).chains
    assert dovish
    assert not any("Policy_Tightening" in c for c in dovish)

    neutral = retriever.retrieve(q, "2022-01", stance=0.0).chains
    assert any("Policy_Easing" in c for c in neutral)
    assert any("Policy_Tightening" in c for c in neutral)
