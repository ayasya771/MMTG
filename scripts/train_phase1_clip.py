"""Phase 1: train Financial-CLIP with InfoNCE, plus the port density CNN.

Builds the financial tokenizer from the corpus, trains the dual encoder on
(chart, caption) pairs with the symmetric InfoNCE objective, evaluates image
to text retrieval on a purged chronological holdout, and trains the container
density CNN on the port scenes.

Usage:
    python scripts/train_phase1_clip.py [--run runs/main] [--epochs 8]
                                        [--batch-size 64] [--device cpu]
"""

import argparse
import math
import os

import _bootstrap
import torch
from torch.utils.data import DataLoader

from mmtg.config import (ChartConfig, ClipConfig, Paths, PortConfig,
                         TokenizerConfig, TrainClipConfig, TrainPortConfig)
from mmtg.data.datasets import (ChartCaptionDataset, PortDensityDataset,
                                chronological_split)
from mmtg.dkg.schema import ENTITIES
from mmtg.models.financial_clip import (FinancialCLIP, clip_infonce,
                                        retrieval_metrics)
from mmtg.models.port_cnn import PortDensityCNN
from mmtg.pipeline import save_checkpoint
from mmtg.tokenizer import FinancialTokenizer
from mmtg.utils import Timer, load_jsonl, log, save_json, set_seed


def build_tokenizer(corpus: list[dict], charts: list[dict],
                    cfg: TokenizerConfig) -> FinancialTokenizer:
    texts = [row["caption"] for row in charts]
    texts += [doc["statement"] for doc in corpus]
    for doc in corpus:
        texts.extend(doc["news"])
    texts += [f"{name}. {spec['gloss']}" for name, spec in ENTITIES.items()]
    return FinancialTokenizer.build(texts, cfg.vocab_size, cfg.min_freq)


def cosine_warmup(optimizer, total_steps: int, warmup_frac: float):
    warmup = max(1, int(total_steps * warmup_frac))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_clip(args, paths: Paths, tok: FinancialTokenizer,
               chart_rows: list[dict], run_dir: str) -> dict:
    cfg = TrainClipConfig(epochs=args.epochs, batch_size=args.batch_size)
    clip_cfg = ClipConfig()
    chart_cfg = ChartConfig()
    set_seed(cfg.seed)

    train_rows, val_rows = chronological_split(chart_rows, cfg.train_end_month,
                                               cfg.embargo_months)
    log(f"clip pairs: {len(train_rows)} train / {len(val_rows)} val (purged split)")

    train_ds = ChartCaptionDataset(train_rows, paths.charts_dir, tok,
                                   clip_cfg.max_text_len, chart_cfg.img_size)
    val_ds = ChartCaptionDataset(val_rows, paths.charts_dir, tok,
                                 clip_cfg.max_text_len, chart_cfg.img_size)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          drop_last=True, num_workers=0)

    model = FinancialCLIP(clip_cfg, vocab_size=len(tok),
                          img_size=chart_cfg.img_size).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
    sched = cosine_warmup(optim, cfg.epochs * len(train_dl), cfg.warmup_frac)

    for epoch in range(cfg.epochs):
        model.train()
        running, count = 0.0, 0
        for images, ids, mask, keys in train_dl:
            images, ids, mask, keys = (images.to(args.device),
                                       ids.to(args.device),
                                       mask.to(args.device),
                                       keys.to(args.device))
            logits_i, logits_t = model(images, ids, mask)
            loss = clip_infonce(logits_i, logits_t, key_ids=keys)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running += loss.item()
            count += 1
        log(f"clip epoch {epoch + 1}/{cfg.epochs} loss {running / max(count, 1):.4f}")

    model.eval()
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False)
    images_all, ids_all, mask_all, keys_all = [], [], [], []
    for images, ids, mask, keys in val_dl:
        images_all.append(images)
        ids_all.append(ids)
        mask_all.append(mask)
        keys_all.append(keys)
    metrics = {"top1": None, "top5": None, "mean_rank": None, "pool_size": 0}
    if images_all:
        pool = min(sum(x.shape[0] for x in images_all), 256)
        images_cat = torch.cat(images_all)[:pool].to(args.device)
        ids_cat = torch.cat(ids_all)[:pool].to(args.device)
        mask_cat = torch.cat(mask_all)[:pool].to(args.device)
        keys_cat = torch.cat(keys_all)[:pool].to(args.device)
        metrics = retrieval_metrics(model, images_cat, ids_cat, mask_cat,
                                    key_ids=keys_cat)
    log(f"clip retrieval: {metrics}")

    save_checkpoint(model, clip_cfg, os.path.join(run_dir, "clip.pt"),
                    vocab_size=len(tok), img_size=chart_cfg.img_size)
    return metrics


def train_port(args, paths: Paths, port_rows: list[dict], run_dir: str) -> dict:
    from mmtg.config import PortCnnConfig

    cfg = TrainPortConfig()
    port_cfg = PortConfig()
    set_seed(cfg.seed)
    train_rows, val_rows = chronological_split(
        port_rows, TrainClipConfig().train_end_month,
        TrainClipConfig().embargo_months)
    train_dl = DataLoader(
        PortDensityDataset(train_rows, paths.ports_dir, port_cfg.img_size),
        batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(
        PortDensityDataset(val_rows, paths.ports_dir, port_cfg.img_size),
        batch_size=128, shuffle=False)

    model = PortDensityCNN(PortCnnConfig()).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for epoch in range(cfg.epochs):
        model.train()
        running, count = 0.0, 0
        for images, density in train_dl:
            pred = model(images.to(args.device))
            loss = torch.nn.functional.mse_loss(pred, density.to(args.device))
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item()
            count += 1
        log(f"port cnn epoch {epoch + 1}/{cfg.epochs} "
            f"mse {running / max(count, 1):.5f}")

    model.eval()
    errs = []
    with torch.no_grad():
        for images, density in val_dl:
            pred = model.predict(images.to(args.device)).cpu()
            errs.append((pred - density).abs())
    mae = float(torch.cat(errs).mean()) if errs else None
    log(f"port cnn val MAE {mae}")
    save_checkpoint(model, PortCnnConfig(), os.path.join(run_dir, "port_cnn.pt"))
    return {"val_mae": mae, "n_train": len(train_rows), "n_val": len(val_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-name", default="generated",
                        help="dataset subdirectory under data/")
    parser.add_argument("--run", default="runs/main")
    parser.add_argument("--epochs", type=int, default=TrainClipConfig().epochs)
    parser.add_argument("--batch-size", type=int,
                        default=TrainClipConfig().batch_size)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    paths = Paths(data_name=args.data_name).ensure()
    run_dir = os.path.join(paths.root, args.run)
    os.makedirs(run_dir, exist_ok=True)

    corpus = load_jsonl(f"{paths.data_dir}/corpus.jsonl")
    chart_rows = load_jsonl(f"{paths.data_dir}/chart_manifest.jsonl")
    port_rows = load_jsonl(f"{paths.data_dir}/port_manifest.jsonl")

    tok = build_tokenizer(corpus, chart_rows, TokenizerConfig())
    tok.save(os.path.join(run_dir, "vocab.json"))
    log(f"tokenizer vocabulary: {len(tok)} entries")

    with Timer() as t_clip:
        clip_metrics = train_clip(args, paths, tok, chart_rows, run_dir)
    with Timer() as t_port:
        port_metrics = train_port(args, paths, port_rows, run_dir)

    save_json({"clip_retrieval": clip_metrics, "port_cnn": port_metrics,
               "clip_seconds": t_clip.elapsed, "port_seconds": t_port.elapsed},
              os.path.join(run_dir, "metrics_phase1.json"))
    log(f"phase 1 done (clip {t_clip}, port {t_port})")


if __name__ == "__main__":
    main()
