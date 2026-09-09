# Multimodal LLM-Driven Macro Regime Trade Generation via Knowledge Graph Fusion

An autonomous global macro quantitative analyst. The system ingests central
bank transcripts (text), yield curve shifts (rendered charts) and alternative
economic indicators (satellite-style port imagery), projects them through a
from-scratch **Financial-CLIP** encoder into a temporal **Dynamic Knowledge
Graph**, and lets a **VLA action head** emit long-short global macro portfolio
tilts as JSON, together with the causal chain the graph used to argue for them.

Output from the shipped run, abridged:

```json
{
  "month": "2025-09",
  "action": { "SPY": -0.2626, "QQQ": -0.1669, "TLT": 0.0218,
              "GLD": 0.2280, "DXY": 0.2740, "USO": 0.0466 },
  "gross_exposure": 1.0,
  "net_exposure": 0.141,
  "rationale": [
    "Federal_Reserve signals Policy_Easing; Policy_Easing boosts US_Equities",
    "Recession_Risk weighs_on Crude_Oil"
  ],
  "evidence": [
    "Federal_Reserve --holds--> Policy_Rate [seen 502x, latest 2024-05, w=279.57]"
  ],
  "signals": { "hawk_dove": -0.667, "port_density": 0.235 }
}
```

`rationale` holds the causal chains that argue for the trade, filtered to the
policy stance the statement actually expresses; `evidence` holds the raw
retrieved edges. They are separate fields because a relation the graph happens
to know is not the same thing as a reason to put on a position.

---

## Architecture

| Stage | Component | What it does |
|---|---|---|
| 1 | **Multimodal ingestion** | Central bank statements and financial news, yield-curve / CPI-GDP / policy-path charts, and port scenes whose container density proxies trade volume. |
| 2 | **Financial-CLIP** (`models/financial_clip.py`) | ViT + text transformer trained from scratch with symmetric InfoNCE on (chart image, structural-break caption) pairs. |
| 3 | **Port CNN** (`models/port_cnn.py`) | Regresses container density from satellite-style imagery. |
| 4 | **Dynamic Knowledge Graph** (`dkg/`) | NetworkX `MultiDiGraph` of macro entities and typed relations, each edge carrying observation count, `first_seen`/`last_seen` and a recency-decayed weight. |
| 5 | **RAG retrieval** (`dkg/retrieve.py`) | Embeds the statement with the CLIP text tower, retrieves the nearest graph nodes, expands to their neighbourhood **as of that month**, and pools a context vector plus verbalized causal chains. |
| 6 | **VLA action head** (`models/vla_head.py`) | Fuses chart embedding, statement tokens, graph context and scalar signals; emits one weight per asset under a fixed gross-exposure budget. |

Tradable sleeve: `SPY, QQQ, TLT, GLD, DXY, USO`.

### Two design decisions worth knowing about

**The graph is temporal, and retrieval is as-of.** Every edge records when it
was first and last observed, and `edge_weight` returns `0.0` for any edge whose
`first_seen` postdates the query month. A query for 2005 cannot see a relation
the corpus only established in 2015. Without this, a knowledge graph built over
the full history leaks the future into every backtest.

**The action head is not allowed to shrink.** Under a squared-error objective a
head free to scale down its outputs will do exactly that, because predicting
near-zero is the risk-minimizing response to a noisy label, and the result is
a portfolio with no view. `normalize_gross` therefore scales every prediction
to the gross budget exactly, and the loss leads with cosine distance so the
head is graded on the relative ranking it actually expresses. Before this
change the mean gross exposure collapsed to 0.41.

---

## Quickstart

```bash
pip install -r requirements.txt

python scripts/quickstart_demo.py      # full pipeline at reduced scale into data/quickstart + runs/quickstart
python serve_demo.py                   # open the interactive demo
```

Or run the phases individually:

```bash
python scripts/make_synthetic_data.py            # simulate the economy, render all modalities
python scripts/train_phase1_clip.py              # Financial-CLIP + port CNN
python scripts/build_dkg.py                      # information extraction -> knowledge graph
python scripts/train_phase3_vla.py               # VLA alignment + purged backtest
python scripts/generate_trades.py --month 2025-09
python scripts/export_demo_assets.py             # refresh docs/index.html from the run
```

Custom inputs:

```bash
python scripts/generate_trades.py \
    --charts my_curve.png my_cpi.png \
    --statement fomc_statement.txt \
    --port longbeach.png \
    --as-of 2025-09
```

Tests: `pytest` (52 tests).

---

## Results from the shipped run

720 simulated months, chronological split at month 600 with an 8-month purge
gap, so every number below is from 121 held-out months the model never saw.

| Phase | Metric | Value |
|---|---|---|
| 1 | CLIP image→text retrieval, top-5 (caption-text match) | **32.4%** (chance ≈ 2%) |
| 1 | CLIP mean retrieval rank | **16.2** out of a 256-caption pool |
| 1 | Port CNN density MAE | **0.054** (labels span 0–1, σ ≈ 0.155) |
| 2 | Triplet extraction precision / recall | **1.00 / 1.00** on 3,018 ground-truth triplets |
| 2 | Graph size | 21 nodes, 26 relations, 3,049 observations |
| 3 | Held-out weight sign agreement | **53.7%** |
| 3 | Backtest total return | **+106.7%** (Sharpe 0.74) vs +72.5% equal-weight (Sharpe 0.72) |

### How to read those numbers honestly

- **53.7% sign agreement sounds low, and it is close to the ceiling.** The
  label is a 3-month forward risk-adjusted return, which is mostly noise. An
  oracle told the *true* hidden regime and given the exact regime-conditional
  return distribution scores about 65%; a mapping from the true regime to its
  average historical weights scores about 59%. Text alone predicts the regime
  at roughly 68% accuracy. Composing those puts the realistic ceiling near
  56–58%, so the model is recovering most of what this label admits.
- **The backtest beats its benchmark modestly, and that is the right size.**
  The equal-weight basket is long-only beta in a simulator with positive drift,
  which is a genuinely hard benchmark for a market-neutral-ish tilt.
- **The oracle column on the demo page is a look-ahead ceiling, not a target.**
  Holding the supervision labels themselves returns +4,558% at Sharpe 4.0
  because they are built from future returns. The model captures 2.3% of that.
  It is reported so nobody mistakes the label for an attainable strategy.
- **Seed variance is real.** Held-out sign agreement across five seeds spans
  0.52–0.58. Stochastic weight averaging over the final 15 epochs is enabled
  because it tightens that spread (±0.013 vs ±0.019) at equal mean.

**None of this is a claim of live alpha.** The backtest runs on held-out months
of a simulator whose regime structure the model is meant to recover. Treat it
as evidence the architecture learns the intended mechanism, not as tradable
performance. Nothing here is investment advice.

[`RUN_REPORT.md`](RUN_REPORT.md) has the full per-phase breakdown, the thirteen
bugs found and fixed during the build, and the known limitations.

---

## Why the shipped run uses a simulator

This was built on a network-restricted machine with no outbound access to
market-data endpoints, so the demo trains against a calibrated
regime-switching economy (`data/macro_sim.py`). That constraint turned out to
be worth keeping: the repository runs end to end with no API keys, no
downloads and no account anywhere. The economy is one in which every modality is
causally linked: a hidden Markov regime chain drives macro state, yields
respond to the policy path, port congestion rises in overheating and
stagflation, and asset returns are drawn from regime-conditional
distributions. Text, charts and satellite scenes are all rendered *from that
same state*, so the supervision is consistent across modalities and extraction
quality is exactly measurable against ground truth.

Every component consumes the same interfaces either way.

### Data Used

```bash
python scripts/download_data.py --start 1990-01
```

| Modality | Source | Notes |
|---|---|---|
| Central bank text | [FOMC statements](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm), Hawk/Dove dataset on Kaggle | scraped by `fetch_fomc_statements` |
| Macro series | [FRED](https://fred.stlouisfed.org/docs/api/fred/): FEDFUNDS, DGS2, DGS10, CPIAUCSL, UNRATE, INDPRO | public CSV endpoint needs no key; set `FRED_API_KEY` to use the JSON API |
| Charts | rendered from FRED windows by `data/charts.py`, or the stock-line-chart dataset on Hugging Face | captions stay exact because they are computed from the series |
| Satellite | [EuroSAT](https://github.com/phelber/EuroSAT), [SpaceNet](https://spacenet.ai/), or Sentinel-2 port chips via Google Earth Engine | one-time bulk download; keep the manifest format from `data/ports.py` |
| Prices | [Stooq](https://stooq.com/db/h/) monthly CSV | free, no key |

Then rerun the three phases against `data/raw/real_panel.json`. Two things to
supply yourself: a congestion series (there is no free live feed, so
`build_real_panel` defaults it to 0.5) and, if you want the LLM extraction path, local weights.

### LLM triplet extraction

The default extraction engine is a deterministic lexicon-and-pattern matcher,
because it is reproducible, dependency-free, auditable, and its recall is
directly measurable against the corpus ground truth. The blueprint's Phase-2
LLM engine is implemented and drops in behind a flag:

```bash
pip install transformers
python scripts/build_dkg.py --llm --llm-model meta-llama/Meta-Llama-3-8B-Instruct
```

Both engines emit the same canonical `Triplet` objects against the shared
schema in `dkg/schema.py`, so the graph, retrieval and action head are
unchanged.

---

## Live Demo Testing

The console is not limited to replaying stored months. You can:

- pick any of the 121 held-out months as a starting point,
- rewrite or paste any central bank statement into the text box,
- upload your own chart images, which are encoded by the real vision tower,
- upload your own port image, whose container density the CNN reads from the
  pixels,
- set the **as of** month to any `YYYY-MM` to time-travel the knowledge graph,
  so retrieval only sees relations observed by that date.

Everything runs client-side; nothing is sent anywhere.

## Layout

```
src/mmtg/
  config.py            all hyperparameters, one place
  tokenizer.py         word-level financial tokenizer with number bucketing
  pipeline.py          end-to-end inference, checkpoint loading
  data/
    macro_sim.py       regime-switching economy + supervision targets
    statements.py      central bank statements & news with ground-truth triplets
    charts.py          chart rendering + structural-break captioning
    ports.py           procedural satellite-style port scenes
    datasets.py        torch datasets, purged chronological splits
    downloaders.py     FRED / Stooq / FOMC fetchers for local runs
  models/
    financial_clip.py  dual encoder, InfoNCE, retrieval metrics
    port_cnn.py        container density regressor
    vla_head.py        fusion action head, JSON serialization
    layers.py          shared transformer blocks
  dkg/
    schema.py          canonical entities and relations
    extract.py         rule-based + LLM information extraction
    graph.py           temporal knowledge graph
    retrieve.py        as-of subgraph retrieval and rationale chains
serve_demo.py          one-command launcher for the interactive demo
scripts/               data generation, three training phases, inference, export
tests/                 52 tests
docs/index.html        GitHub Pages demo
```

## License

Apache 2.0. See [LICENSE](LICENSE).
