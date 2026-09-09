import torch

from mmtg.config import ASSETS, ClipConfig, PortCnnConfig, VlaConfig
from mmtg.models.financial_clip import (FinancialCLIP, clip_infonce,
                                        retrieval_metrics)
from mmtg.models.port_cnn import PortDensityCNN
from mmtg.models.vla_head import (VlaActionHead, action_metrics,
                                  normalize_gross, to_action_json, vla_loss)

_CLIP_CFG = ClipConfig(vision_dim=64, vision_depth=1, text_dim=64,
                       text_depth=1, embed_dim=32, vision_heads=2,
                       text_heads=2, max_text_len=16)


def _clip():
    torch.manual_seed(0)
    return FinancialCLIP(_CLIP_CFG, vocab_size=50, img_size=64)


def _text_batch(n, low=4):
    ids = torch.randint(low, 50, (n, 16))
    ids[:, 0] = 2
    mask = torch.ones_like(ids)
    mask[:, -3:] = 0
    return ids, mask


def test_clip_shapes():
    model = _clip()
    images = torch.randn(4, 3, 64, 64)
    ids, mask = _text_batch(4)
    li, lt = model(images, ids, mask)
    assert li.shape == (4, 4) and lt.shape == (4, 4)
    assert torch.allclose(li, lt.t(), atol=1e-5)
    emb = model.encode_image(images)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_infonce_prefers_aligned():
    aligned = torch.eye(6) * 10.0
    shuffled = aligned[torch.randperm(6)]
    assert clip_infonce(aligned, aligned.t()) < clip_infonce(shuffled,
                                                             shuffled.t())


def test_infonce_duplicate_masking():
    logits = torch.eye(4) * 5.0
    logits[0, 1] = 5.0
    logits[1, 0] = 5.0
    keys = torch.tensor([7, 7, 8, 9])
    unmasked = clip_infonce(logits, logits.t())
    masked = clip_infonce(logits, logits.t(), key_ids=keys)
    assert masked < unmasked
    assert torch.isfinite(masked)


def test_retrieval_metrics_text_match():
    model = _clip()
    images = torch.randn(6, 3, 64, 64)
    ids, mask = _text_batch(6)
    keys = torch.tensor([0, 0, 1, 2, 3, 4])
    m = retrieval_metrics(model, images, ids, mask, key_ids=keys)
    assert m["criterion"] == "caption_text_match"
    assert 0.0 <= m["top1"] <= m["top5"] <= 1.0
    assert m["pool_size"] == 6


def test_port_cnn_shapes_and_clamp():
    model = PortDensityCNN(PortCnnConfig())
    out = model(torch.randn(3, 3, 96, 96))
    assert out.shape == (3,)
    pred = model.predict(torch.randn(3, 3, 96, 96) * 50)
    assert torch.all((pred >= 0) & (pred <= 1))


def test_normalize_gross_hits_cap_exactly():
    raw = torch.tensor([[0.8, -0.8, 0.4, 0.0, 0.0, 0.0]])
    w = normalize_gross(raw, 1.0)
    assert abs(w.abs().sum().item() - 1.0) < 1e-6
    small = torch.tensor([[0.01, -0.02, 0.0, 0.0, 0.0, 0.0]])
    w2 = normalize_gross(small, 1.0)
    assert abs(w2.abs().sum().item() - 1.0) < 1e-6
    assert torch.all(w2.sign() == small.sign())


def test_normalize_gross_handles_zero_row():
    w = normalize_gross(torch.zeros(1, 6), 1.0)
    assert torch.all(torch.isfinite(w))


def test_vla_forward_and_cap():
    torch.manual_seed(0)
    cfg = VlaConfig(dim=32, depth=1, heads=2, max_statement_len=24)
    model = VlaActionHead(cfg, vocab_size=50, clip_dim=16)
    B = 5
    ids, mask = _text_batch(B)
    ids, mask = ids[:, :24], mask[:, :24]
    w = model(torch.randn(B, 16), ids, mask, torch.randn(B, 16),
              torch.randn(B, 2))
    assert w.shape == (B, len(ASSETS))
    assert torch.all(w.abs().sum(dim=-1) <= cfg.gross_cap + 1e-5)


def test_vla_loss_zero_at_perfect():
    target = torch.tensor([[0.2, -0.3, 0.1, 0.0, 0.25, -0.15]])
    assert vla_loss(target.clone(), target).item() < 1e-6


def test_action_json_schema():
    w = torch.tensor([0.25, -0.25, 0.2, -0.1, 0.1, -0.1])
    out = to_action_json(w, rationale=["Federal_Reserve raises Policy_Rate"],
                         month="2025-06",
                         evidence=["Federal_Reserve --raises--> Policy_Rate [seen 3x]"])
    assert set(out["action"]) == set(ASSETS)
    assert out["month"] == "2025-06"
    assert abs(out["gross_exposure"] - 1.0) < 1e-6
    assert abs(out["net_exposure"] - 0.1) < 1e-6
    assert out["rationale"] == ["Federal_Reserve raises Policy_Rate"]
    assert out["evidence"] and "[" in out["evidence"][0]


def test_action_json_omits_absent_optionals():
    out = to_action_json(torch.zeros(6))
    assert "rationale" not in out and "evidence" not in out and "month" not in out


def test_action_metrics():
    pred = torch.tensor([[0.5, -0.5, 0.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([[0.3, 0.4, 0.0, 0.0, 0.0, 0.0]])
    m = action_metrics(pred, target)
    assert abs(m["sign_agreement"] - 0.5) < 1e-6


def test_vla_gradients_flow():
    torch.manual_seed(1)
    cfg = VlaConfig(dim=32, depth=1, heads=2, max_statement_len=24)
    model = VlaActionHead(cfg, vocab_size=50, clip_dim=16)
    ids, mask = _text_batch(2)
    w = model(torch.randn(2, 16), ids[:, :24], mask[:, :24],
              torch.randn(2, 16), torch.randn(2, 2))
    loss = vla_loss(w, torch.randn(2, len(ASSETS)))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
