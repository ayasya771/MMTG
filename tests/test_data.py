import os

import numpy as np
import pytest

from mmtg.config import ASSETS, REGIMES, SimConfig
from mmtg.data.charts import caption_window
from mmtg.data.macro_sim import MacroPanel, simulate, target_weights
from mmtg.data.ports import render_port_image
from mmtg.data.statements import hawk_dove_score
from mmtg.tokenizer import FinancialTokenizer, tokenize_text


def test_panel_shapes(panel):
    n = panel.n_months
    assert n == 90
    assert panel.returns.shape == (n, len(ASSETS))
    assert len(panel.months) == n
    assert panel.months[1] == "1966-02"


def test_panel_ranges(panel):
    assert np.all(panel.policy_rate >= 0.0)
    assert np.all((panel.congestion >= 0.0) & (panel.congestion <= 1.0))
    assert np.all(np.isfinite(panel.returns))
    assert set(np.unique(panel.regime)).issubset(set(range(len(REGIMES))))


def test_simulation_deterministic():
    a = simulate(SimConfig(months=40, seed=11))
    b = simulate(SimConfig(months=40, seed=11))
    assert np.allclose(a.returns, b.returns)
    assert a.months == b.months


def test_panel_roundtrip(panel):
    clone = MacroPanel.from_dict(panel.to_dict())
    assert np.allclose(clone.returns, panel.returns)
    assert clone.months == panel.months


def test_target_weights_gross_cap(panel):
    w = target_weights(panel, 40, horizon=3, gross_cap=1.0)
    assert w.shape == (len(ASSETS),)
    assert abs(np.abs(w).sum() - 1.0) < 1e-9


def test_target_weights_needs_future(panel):
    with pytest.raises(IndexError):
        target_weights(panel, panel.n_months - 1)


def _flat_panel(months=40, y2=5.0, y10=4.0):
    """Constant panel with a controllable curve for caption assertions."""
    n = months
    return MacroPanel(
        months=[f"2000-{(i % 12) + 1:02d}" for i in range(n)],
        regime=np.zeros(n, dtype=np.int64),
        gdp_growth=np.full(n, 2.0), cpi_yoy=np.full(n, 3.0),
        unemployment=np.full(n, 5.0), policy_rate=np.full(n, 5.0),
        y2=np.full(n, y2), y10=np.full(n, y10),
        congestion=np.full(n, 0.5),
        returns=np.zeros((n, len(ASSETS))),
    )


def test_caption_reports_inversion():
    cap = caption_window(_flat_panel(y2=5.0, y10=4.0), 30, 24, "yield_curve")
    assert "inverted" in cap
    assert "100 basis points" in cap


def test_caption_reports_positive_slope():
    cap = caption_window(_flat_panel(y2=3.0, y10=4.5), 30, 24, "yield_curve")
    assert "positive slope" in cap


def test_caption_unknown_type_raises(panel):
    with pytest.raises(ValueError):
        caption_window(panel, 40, 24, "volume_profile")


def test_chart_files_exist(chart_data):
    rows, out = chart_data
    assert rows, "no charts rendered"
    for row in rows[:5]:
        assert os.path.exists(os.path.join(out, row["file"]))
        assert row["caption"].startswith(("yield curve chart", "macro chart",
                                          "policy rate chart"))


def test_port_density_visible_in_pixels():
    rng = np.random.default_rng(0)
    sparse = [render_port_image(0.05, rng, 96).astype(float).std()
              for _ in range(6)]
    rng = np.random.default_rng(0)
    dense = [render_port_image(0.95, rng, 96).astype(float).std()
             for _ in range(6)]
    assert np.mean(dense) > np.mean(sparse)


def test_port_manifest_labels(port_data):
    rows, out = port_data
    for row in rows[:5]:
        assert 0.0 <= row["density"] <= 1.0
        assert os.path.exists(os.path.join(out, row["file"]))


def test_corpus_carries_ground_truth(corpus):
    assert len(corpus) == 90
    total = sum(len(d["triplets"]) for d in corpus)
    assert total > len(corpus), "every month should carry at least one triplet"
    for doc in corpus:
        assert doc["statement"]
        assert -1.0 <= doc["hawk_dove"] <= 1.0


def test_hawk_dove_direction():
    hawk = "Inflation remains unacceptably elevated. The committee raised rates."
    dove = "The committee lowered rates and remains patient as cooling continued."
    assert hawk_dove_score(hawk) > hawk_dove_score(dove)


def test_tokenize_numbers_bucketed():
    toks = tokenize_text("the curve is inverted by 47 basis points at -1.4%")
    assert "<n50>" in toks
    assert "<neg1%>" in toks


def test_tokenizer_roundtrip(tokenizer):
    ids = tokenizer.encode("the yield curve inverted by 50 basis points", 32)
    assert len(ids) == 32
    assert ids[0] == tokenizer.cls_id
    decoded = tokenizer.decode(ids)
    assert "yield" in decoded and "curve" in decoded


def test_tokenizer_unknown_words(tokenizer):
    ids = tokenizer.encode("qzxv blorptastic figments", 16)
    assert tokenizer.unk_id in ids


def test_tokenizer_save_load(tokenizer, tmp_path):
    path = str(tmp_path / "vocab.json")
    tokenizer.save(path)
    loaded = FinancialTokenizer.load(path)
    text = "inflation accelerating at 6.0%"
    assert loaded.encode(text, 24) == tokenizer.encode(text, 24)


def test_resolve_train_end_caps_short_runs():
    from mmtg.data.datasets import chronological_split, resolve_train_end

    assert resolve_train_end(100, 600) == 82
    assert resolve_train_end(1000, 600) == 600
    rows = [{"month_index": i} for i in range(100)]
    train, val = chronological_split(rows, 600, 8)
    assert train and val, "a short dataset must still yield a validation split"
    assert max(r["month_index"] for r in train) < min(r["month_index"] for r in val)


def test_chronological_split_purge_gap():
    from mmtg.data.datasets import chronological_split

    rows = [{"month_index": i} for i in range(1000)]
    train, val = chronological_split(rows, 600, 8)
    gap = min(r["month_index"] for r in val) - max(r["month_index"] for r in train)
    assert gap > 8


def test_attention_mask(tokenizer):
    ids = tokenizer.encode("gold", 12)
    mask = tokenizer.attention_mask(ids)
    assert mask[0] == 1 and mask[-1] == 0
    assert sum(mask) == 3
