# Run report

Everything below comes from one reproducible run of the shipped defaults:

```bash
python scripts/make_synthetic_data.py
python scripts/train_phase1_clip.py
python scripts/build_dkg.py
python scripts/train_phase3_vla.py
python scripts/export_demo_assets.py
```

Environment: CPU only (2 cores), PyTorch 2.13, Python 3.11, no network access. Total wall clock
about 22 minutes, dominated by Phase 1.

## Dataset

| | |
|---|---|
| Simulated months | 720 (1966-01 → 2025-12) |
| Chart images | 2,088 across three chart types |
| Port scenes | 2,160 |
| Statements | 720, carrying 3,018 unique ground-truth triplets |
| Split | train ≤ month 600, 8-month purge gap, validation from month 609 |
| VLA samples | 564 train / 8 embargoed / 121 held out |

## Phase 1: Financial-CLIP and the port CNN

| Metric | Value |
|---|---|
| Contrastive pairs | 1,698 train / 366 validation (purged) |
| Final InfoNCE loss | 1.72 |
| Image→text top-1 (caption-text match) | 18.8% |
| Image→text top-5 | 32.4% |
| Mean retrieval rank | 16.2 of a 256-caption pool |
| Port density MAE (held out) | 0.054 |
| Wall clock | 564 s CLIP + 110 s port CNN |

Retrieval is scored by **caption text match**, not row index. Overlapping
windows produce textually identical captions, so index-matching would mark a
correct retrieval wrong. The same collision is masked out of the InfoNCE
softmax during training, since two identical captions must not act as
negatives for each other.

## Phase 2: Dynamic Knowledge Graph

| Metric | Value |
|---|---|
| Triplets extracted | 3,049 |
| Precision / recall / F1 vs ground truth | 1.000 / 1.000 / 1.000 |
| Graph | 21 nodes, 26 typed relations, 3,049 observations |
| Engine | rule-based (LLM path available via `--llm`) |

Perfect extraction is expected and is **not** a claim about real transcripts:
the synthetic corpus is written from the same canonical surface forms the
extractor matches. The number's purpose is to prove the extraction, graph
ingestion and scoring path is wired correctly end to end. On real FOMC text
this figure will drop, which is exactly why the LLM engine exists.

## Phase 3: VLA alignment

| Metric | Value |
|---|---|
| Held-out sign agreement | 53.7% |
| Held-out weight MAE | 0.219 |
| Mean gross exposure | 1.00 (fixed by construction) |
| Backtest, 121 held-out months | **+106.7%**, Sharpe 0.743 |
| Equal-weight long benchmark | +72.5%, Sharpe 0.721 |
| Perfect-foresight label ceiling | +4,557.7%, Sharpe 3.997 |
| Share of the ceiling captured | 2.3% |

## Bugs found and fixed during the build

These were all caught by measuring rather than by inspection, and each one is
recorded here because the fix is the interesting part.

1. **The port CNN would not learn (MSE stuck at the label variance, 0.0226).**
   Cause: GroupNorm normalizes each sample independently, which erases the
   global activation amplitude carrying the container count. The regressor
   then had nothing to do but predict the label mean. Switching to BatchNorm
   reached the noise floor within four epochs; MAE went 0.245 → 0.054. Worth
   knowing before anyone "modernizes" that layer back to GroupNorm.

2. **A sigmoid output head sat on a dead gradient plateau.** Removed in favour
   of a linear head with clamping at inference (`predict()`).

3. **Identical captions were fighting each other in the contrastive loss.**
   Overlapping chart windows generate the same caption text; as in-batch
   negatives they punished correct behaviour. Fixed with caption-key masking,
   which also corrected the evaluation criterion. Top-5 went 20.7% → 32.4%.

4. **The action head was shrinking to avoid risk.** Under MSE the head learned
   to output near-zero weights (mean gross exposure 0.41) because that
   minimizes squared error against a noisy label, i.e. a portfolio with no view.
   Fixed by normalizing to the gross budget exactly and leading the loss with
   cosine distance. Backtest went +22.9% → +106.7%.

5. **`--months` silently overrode the config.** The argparse default was a
   hardcoded 480 while `SimConfig` said 720, so a config change appeared to do
   nothing. Defaults now derive from the config object.

6. **The split boundary could swallow a short dataset.** A configured boundary
   of month 600 against a 300-month quickstart run left zero validation rows.
   `resolve_train_end` now caps the boundary at 82% of available history.

7. **Metrics moved when unrelated upstream stages changed.** Feature
   precompute consumed RNG before model init, so the head's weights depended
   on it. Re-seeding immediately before model construction fixed it; held-out
   sign agreement went 51.9% → 53.7% purely from removing that confound.

8. **The quickstart destroyed the main dataset.** It wrote to the same
   `data/generated` directory the full run's checkpoints were trained
   against. Every script now takes `--data-name`.

9. **The rationale contradicted itself.** The graph holds both tightening and
   easing relations because both occurred historically, so a hawkish
   statement could be handed an easing chain as its justification. Retrieval
   now takes the text's hawk/dove stance and suppresses chains through the
   opposing policy theme.

10. **Nested chains were double-counted as separate reasons.** "Fed signals
    Tightening; Tightening lifts Yields; Yields weigh on Bonds" and its own
    tail were both shown, overstating the evidence. Only maximal chains
    survive now.

11. **`asset_hints` were unnormalized observation counts** running into the
    hundreds, which made the published JSON unreadable. Rescaled to [-1, 1].

12. **Argument and evidence were merged into one list.** Split into
    `rationale` (stance-filtered causal chains) and `evidence` (raw retrieved
    edges), because a relation the graph knows is not a reason for a trade.

13. **FRED and Stooq CSV parsing was brittle.** FRED has shipped the date
    column as both `DATE` and `observation_date`; the parser now addresses
    columns positionally and fails with a readable message instead of a
    `KeyError`.

## Browser demo: bugs found and fixed

The published page runs the models client-side. Four defects made that path
dead on arrival; all were found by serving the page and driving a real browser,
not by reading the code.

14. **Only two of four models were exported.** `torch.onnx.export` needs
    `onnxscript` on current torch, and without it the export aborts. Added to
    the `web` extra so the failure cannot recur silently.

15. **The exported models could not be loaded by the browser.** The exporter
    wrote weights as sidecar `.onnx.data` files; onnxruntime-web cannot fetch
    external data and fails with "Module.MountedFiles is not available".
    `_inline_weights` now folds the tensors back into a single `.onnx` file
    per model (13 MB total, still small enough to publish).

16. **`InferenceSession.createAsync` does not exist.** The web runtime exposes
    only `create`, so every session load threw immediately.

17. **`generate()` used `ctx` without computing it,** so any live inference
    raised `ctx is not defined`. The retrieval call had been lost from the
    function; restored to mirror the Python order (hawk/dove, embed, retrieve).

18. **Chains were always empty, and images mismatched.** Two porting bugs
    against the Python reference:
    `causalChains` tested node membership against a map keyed only by edge
    heads, so any terminal asset node looked absent and the walker returned
    nothing (networkx `add_edge` registers both endpoints); and
    `decodeImageTensor` emitted interleaved HWC while the models expect planar
    CHW, which is what `transpose(2, 0, 1)` produces in Python. With both
    fixed the page's parity self-check passes 7/7: tokenizer, hawk/dove, text
    encoder, graph retrieval, VLA head, image encoder and port CNN all match
    the Python outputs to within 1e-4.

## Known limitations

- **Seed variance.** Held-out sign agreement spans 0.52–0.58 across five
  seeds. Stochastic weight averaging over the last 15 epochs is enabled
  because it tightens the spread (±0.013 vs ±0.019) at equal mean. Any single
  reported figure should be read with that band in mind.
- **The model is near its label's ceiling, and the ceiling is low.** An oracle
  with the true hidden regime scores ~65%; regime→average-weights scores ~59%;
  text predicts the regime at ~68%. The realistic composed ceiling is 56–58%.
- **Overfitting persists.** Train sign agreement runs ~0.69 against ~0.54 held
  out, on 564 training samples. More simulated history or a smaller head would
  narrow it; both were tried, and the current config is the better trade.
- **Extraction metrics do not transfer to real transcripts.** See Phase 2.
- **No transaction costs, slippage, or position limits** beyond the gross
  exposure budget. The backtest rebalances monthly at the close.
- **Congestion has no free live feed.** `build_real_panel` defaults it to 0.5;
  supply your own port-call series or a Sentinel-2 derived index.

Nothing in this repository is investment advice.
