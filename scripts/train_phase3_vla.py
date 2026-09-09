"""Phase 3: VLA alignment. Freeze perception, train the action head.

Precomputes frozen features for every eligible month (chart embeddings from
the frozen Financial-CLIP vision tower, port density from the frozen CNN,
retrieved graph context as of that month), then trains the action head to
map them plus the raw statement tokens onto forward looking target weights.
Evaluates on a purged chronological holdout and runs a monthly rebalance
backtest of the predicted tilts against an equal weight benchmark.

Usage:
    python scripts/train_phase3_vla.py [--run runs/main] [--epochs 40]
                                       [--device cpu]
"""

import argparse
import os
from collections import defaultdict

import _bootstrap
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from mmtg.config import (ASSETS, ChartConfig, ClipConfig, DkgConfig, Paths,
                         PortCnnConfig, PortConfig, TrainVlaConfig, VlaConfig)
from mmtg.data.datasets import load_image_tensor
from mmtg.data.macro_sim import MacroPanel, target_weights
from mmtg.dkg.graph import DynamicKnowledgeGraph
from mmtg.dkg.retrieve import GraphRetriever
from mmtg.models.financial_clip import FinancialCLIP
from mmtg.models.port_cnn import PortDensityCNN
from mmtg.models.vla_head import (VlaActionHead, action_metrics, to_action_json,
                                  vla_loss)
from mmtg.pipeline import save_checkpoint
from mmtg.tokenizer import FinancialTokenizer
from mmtg.utils import Timer, load_json, load_jsonl, log, save_json, set_seed


@torch.no_grad()
def precompute(paths: Paths, run_dir: str, device: str,
               vla_cfg: VlaConfig) -> dict:
    """Frozen features per eligible month, ordered by month index."""
    panel = MacroPanel.from_dict(load_json(f"{paths.data_dir}/panel.json"))
    corpus = {d["month"]: d for d in load_jsonl(f"{paths.data_dir}/corpus.jsonl")}
    chart_rows = load_jsonl(f"{paths.data_dir}/chart_manifest.jsonl")
    port_rows = load_jsonl(f"{paths.data_dir}/port_manifest.jsonl")

    tok = FinancialTokenizer.load(os.path.join(run_dir, "vocab.json"))
    clip_ckpt = torch.load(os.path.join(run_dir, "clip.pt"),
                           map_location="cpu", weights_only=False)
    clip = FinancialCLIP(ClipConfig(**clip_ckpt["cfg"]),
                         vocab_size=clip_ckpt["meta"]["vocab_size"],
                         img_size=clip_ckpt["meta"]["img_size"]).to(device).eval()
    clip.load_state_dict(clip_ckpt["state_dict"])
    port_ckpt = torch.load(os.path.join(run_dir, "port_cnn.pt"),
                           map_location="cpu", weights_only=False)
    port_cnn = PortDensityCNN(PortCnnConfig(**{
        **port_ckpt["cfg"], "channels": tuple(port_ckpt["cfg"]["channels"])}))
    port_cnn.load_state_dict(port_ckpt["state_dict"])
    port_cnn = port_cnn.to(device).eval()
    dkg = DynamicKnowledgeGraph.load(os.path.join(run_dir, "dkg.json"))
    retriever = GraphRetriever(dkg, DkgConfig())

    charts_by_month: dict[int, list[str]] = defaultdict(list)
    for row in chart_rows:
        charts_by_month[row["month_index"]].append(
            os.path.join(paths.charts_dir, row["file"]))
    ports_by_month: dict[int, list[str]] = defaultdict(list)
    for row in port_rows:
        ports_by_month[row["month_index"]].append(
            os.path.join(paths.ports_dir, row["file"]))

    img_size = clip_ckpt["meta"]["img_size"]
    port_size = PortConfig().img_size
    eligible = [t for t in range(ChartConfig().window,
                                 panel.n_months - vla_cfg.horizon)
                if t in charts_by_month and panel.months[t] in corpus]

    all_files = [f for t in eligible for f in charts_by_month[t]]
    embs = []
    for i in range(0, len(all_files), 64):
        batch = torch.stack([load_image_tensor(f, img_size)
                             for f in all_files[i: i + 64]]).to(device)
        embs.append(clip.encode_image(batch).cpu())
    embs = torch.cat(embs) if embs else torch.zeros(0, clip.cfg.embed_dim)

    chart_emb: dict[int, torch.Tensor] = {}
    cursor = 0
    for t in eligible:
        k = len(charts_by_month[t])
        mean = embs[cursor: cursor + k].mean(dim=0)
        chart_emb[t] = mean / (mean.norm() + 1e-9)
        cursor += k

    rows = {"month_index": [], "months": [], "chart": [], "ids": [],
            "mask": [], "ctx": [], "feats": [], "target": [], "regime": []}
    for t in eligible:
        month = panel.months[t]
        doc = corpus[month]
        ids = tok.encode(doc["statement"], vla_cfg.max_statement_len)
        mask = tok.attention_mask(ids)

        q_ids = torch.tensor([tok.encode(doc["statement"],
                                         clip.textual.max_len)],
                             dtype=torch.long, device=device)
        q_mask = (q_ids != tok.pad_id).long()
        query = clip.encode_text(q_ids, q_mask)[0].cpu().numpy()
        ctx = retriever.retrieve(query, month, stance=doc["hawk_dove"])
        ctx_emb = (ctx.context_embedding if ctx.context_embedding is not None
                   else np.zeros(clip.cfg.embed_dim, dtype=np.float32))

        pfiles = ports_by_month.get(t, [])
        if pfiles:
            pbatch = torch.stack([load_image_tensor(f, port_size)
                                  for f in pfiles]).to(device)
            density = float(port_cnn.predict(pbatch).mean())
        else:
            density = 0.5

        rows["month_index"].append(t)
        rows["months"].append(month)
        rows["chart"].append(chart_emb[t])
        rows["ids"].append(torch.tensor(ids, dtype=torch.long))
        rows["mask"].append(torch.tensor(mask, dtype=torch.long))
        rows["ctx"].append(torch.tensor(ctx_emb, dtype=torch.float32))
        rows["feats"].append(torch.tensor([density, doc["hawk_dove"]],
                                          dtype=torch.float32))
        rows["target"].append(torch.tensor(
            target_weights(panel, t, vla_cfg.horizon, vla_cfg.gross_cap),
            dtype=torch.float32))
        rows["regime"].append(doc["regime"])

    return {
        "panel": panel,
        "tok": tok,
        "clip_dim": clip.cfg.embed_dim,
        "month_index": np.asarray(rows["month_index"]),
        "months": rows["months"],
        "regime": rows["regime"],
        "chart": torch.stack(rows["chart"]),
        "ids": torch.stack(rows["ids"]),
        "mask": torch.stack(rows["mask"]),
        "ctx": torch.stack(rows["ctx"]),
        "feats": torch.stack(rows["feats"]),
        "target": torch.stack(rows["target"]),
    }


def backtest(panel: MacroPanel, month_index: np.ndarray, weights: np.ndarray,
             targets: np.ndarray) -> dict:
    """Hold each month's weights for the following month, monthly rebalance.

    Three curves are tracked so the result is interpretable:
    - vla: the model's predicted tilts,
    - eq: an equal weight long only basket, i.e. plain market beta,
    - oracle: the supervision targets themselves, which is the ceiling any
      model trained on this label could reach. The model's share of the
      oracle return is the honest measure of how much of the learnable
      signal was actually captured.
    """
    rows = []
    curves = {"vla": 1.0, "eq": 1.0, "oracle": 1.0}
    eq_w = np.ones(len(ASSETS)) / len(ASSETS)
    for i, t in enumerate(month_index):
        if t + 1 >= panel.n_months:
            continue
        r_next = panel.returns[t + 1]
        rets = {"vla": float(weights[i] @ r_next),
                "eq": float(eq_w @ r_next),
                "oracle": float(targets[i] @ r_next)}
        for k, v in rets.items():
            curves[k] *= 1.0 + v
        rows.append({"month": panel.months[t],
                     "vla_ret": round(rets["vla"], 5),
                     "eq_ret": round(rets["eq"], 5),
                     "oracle_ret": round(rets["oracle"], 5),
                     "vla_curve": round(curves["vla"], 5),
                     "eq_curve": round(curves["eq"], 5),
                     "oracle_curve": round(curves["oracle"], 5)})

    def sharpe(key: str) -> float:
        x = np.asarray([r[key] for r in rows])
        return float(x.mean() / (x.std() + 1e-9) * np.sqrt(12.0)) if len(x) else 0.0

    oracle_excess = curves["oracle"] - 1.0
    return {
        "rows": rows,
        "vla_total_return": round(curves["vla"] - 1.0, 4),
        "eq_total_return": round(curves["eq"] - 1.0, 4),
        "oracle_total_return": round(oracle_excess, 4),
        "vla_sharpe": round(sharpe("vla_ret"), 3),
        "eq_sharpe": round(sharpe("eq_ret"), 3),
        "oracle_sharpe": round(sharpe("oracle_ret"), 3),
        "oracle_capture": (round((curves["vla"] - 1.0) / oracle_excess, 3)
                           if abs(oracle_excess) > 1e-9 else None),
        "n_months": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-name", default="generated",
                        help="dataset subdirectory under data/")
    parser.add_argument("--run", default="runs/main")
    parser.add_argument("--epochs", type=int, default=TrainVlaConfig().epochs)
    parser.add_argument("--batch-size", type=int,
                        default=TrainVlaConfig().batch_size)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    paths = Paths(data_name=args.data_name)
    run_dir = os.path.join(paths.root, args.run)
    cfg = TrainVlaConfig(epochs=args.epochs, batch_size=args.batch_size)
    vla_cfg = VlaConfig()
    set_seed(cfg.seed)

    with Timer() as t_feat:
        feats = precompute(paths, run_dir, args.device, vla_cfg)
    log(f"precomputed features for {len(feats['months'])} months ({t_feat})")

    from mmtg.data.datasets import resolve_train_end
    train_end = resolve_train_end(int(feats["month_index"].max()),
                                  cfg.train_end_month)
    is_train = feats["month_index"] <= train_end
    is_val = feats["month_index"] > train_end + cfg.embargo_months
    log(f"vla samples: {int(is_train.sum())} train / {int(is_val.sum())} val "
        f"(boundary month index {train_end})")

    def subset(mask: np.ndarray) -> TensorDataset:
        idx = torch.from_numpy(np.where(mask)[0])
        return TensorDataset(feats["chart"][idx], feats["ids"][idx],
                             feats["mask"][idx], feats["ctx"][idx],
                             feats["feats"][idx], feats["target"][idx])

    set_seed(cfg.seed)
    train_dl = DataLoader(subset(is_train), batch_size=cfg.batch_size,
                          shuffle=True)
    model = VlaActionHead(vla_cfg, vocab_size=len(feats["tok"]),
                          clip_dim=feats["clip_dim"]).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)

    swa_sum: dict[str, torch.Tensor] | None = None
    swa_n = 0
    for epoch in range(cfg.epochs):
        model.train()
        running, count = 0.0, 0
        for chart, ids, mask, ctx, extra, target in train_dl:
            pred = model(chart.to(args.device), ids.to(args.device),
                         mask.to(args.device), ctx.to(args.device),
                         extra.to(args.device))
            loss = vla_loss(pred, target.to(args.device))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += loss.item()
            count += 1
        if epoch >= cfg.swa_start:
            state = model.state_dict()
            if swa_sum is None:
                swa_sum = {k: v.detach().float().clone() for k, v in state.items()}
            else:
                for k, v in state.items():
                    swa_sum[k] += v.detach().float()
            swa_n += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"vla epoch {epoch + 1}/{cfg.epochs} "
                f"loss {running / max(count, 1):.4f}")

    if swa_sum is not None and swa_n > 1:
        ref = model.state_dict()
        model.load_state_dict({k: (v / swa_n).to(ref[k].dtype)
                               for k, v in swa_sum.items()})
        log(f"loaded SWA weights averaged over the last {swa_n} epochs")

    model.eval()
    with torch.no_grad():
        preds = model(feats["chart"].to(args.device),
                      feats["ids"].to(args.device),
                      feats["mask"].to(args.device),
                      feats["ctx"].to(args.device),
                      feats["feats"].to(args.device)).cpu()

    val_idx = np.where(is_val)[0]
    metrics = action_metrics(preds[val_idx], feats["target"][val_idx])
    log(f"vla validation: {metrics}")

    bt = backtest(feats["panel"], feats["month_index"][val_idx],
                  preds[val_idx].numpy(), feats["target"][val_idx].numpy())
    log(f"backtest over {bt['n_months']} held out months: "
        f"vla {bt['vla_total_return']:+.1%} (sharpe {bt['vla_sharpe']}) · "
        f"equal weight {bt['eq_total_return']:+.1%} (sharpe {bt['eq_sharpe']}) · "
        f"oracle label {bt['oracle_total_return']:+.1%} "
        f"(sharpe {bt['oracle_sharpe']}) · capture {bt['oracle_capture']}")

    tilts = [{"month": feats["months"][i], "regime": feats["regime"][i],
              "weights": {a: round(float(preds[i, j]), 4)
                          for j, a in enumerate(ASSETS)},
              "target": {a: round(float(feats["target"][i, j]), 4)
                         for j, a in enumerate(ASSETS)},
              "split": "val" if bool(is_val[i]) else
                       ("train" if bool(is_train[i]) else "embargo")}
             for i in range(len(feats["months"]))]

    save_checkpoint(model, vla_cfg, os.path.join(run_dir, "vla.pt"),
                    vocab_size=len(feats["tok"]), clip_dim=feats["clip_dim"])
    save_json(metrics | {"backtest": {k: v for k, v in bt.items()
                                      if k != "rows"}},
              os.path.join(run_dir, "metrics_phase3.json"))
    save_json(bt, os.path.join(run_dir, "backtest.json"))
    save_json(tilts, os.path.join(run_dir, "tilts_history.json"))

    from dataclasses import asdict
    save_json({"vocab": "vocab.json", "clip": "clip.pt", "port": "port_cnn.pt",
               "vla": "vla.pt", "dkg": "dkg.json",
               "img_size": ChartConfig().img_size,
               "port_img_size": PortConfig().img_size,
               "clip_dim": feats["clip_dim"],
               "dkg_cfg": asdict(DkgConfig())},
              os.path.join(run_dir, "manifest.json"))

    if len(val_idx):
        i = int(val_idx[-1])
        example = to_action_json(preds[i], month=feats["months"][i])
        save_json(example, os.path.join(run_dir, "example_action.json"))
        log(f"example action {example['month']}: {example['action']}")


if __name__ == "__main__":
    main()
