/* MMTG in-browser inference engine: tokenizer, hawk/dove scorer, temporal DKG,
 * as-of retriever and the four ONNX models (CLIP towers, port CNN, VLA head)
 * through onnxruntime-web. Mirrors src/mmtg so the parity self-check holds.
 * In Node tests, set globalThis.ort, globalThis.document and globalThis.fetch
 * before importing; in the browser, load ort.min.js first.
 */
"use strict";

(function (global) {
  const MMTG = {};

  MMTG.STANCE_DEADBAND = 0.15;

  const TOKEN_RE = /[a-z0-9]+(?:\.[0-9]+)?%?|[+-][0-9]+(?:\.[0-9]+)?%?|bp\b/g;
  const NUM_RE = /^[+-]?[0-9]+(?:\.[0-9]+)?%?$/;

  /* ----------------------------------------------------------- tokenizer */
  function normalizeNumber(tok) {
    if (!NUM_RE.test(tok)) return tok;
    const pct = tok.endsWith("%");
    const body = pct ? tok.slice(0, -1) : tok;
    const val = Number.parseFloat(body);
    if (Number.isNaN(val)) return tok;
    const sign = val < 0 ? "neg" : "";
    const mag = Math.floor(Math.abs(val));
    if (pct) return "<" + sign + Math.min(mag, 15) + "%>";
    if (mag >= 1000) return "<bignum>";
    return "<n" + sign + (Math.round(mag / 25.0) * 25) + ">";
  }

  function tokenizeText(text) {
    const low = text.toLowerCase().replace(/&/g, " and ");
    const out = [];
    let m;
    TOKEN_RE.lastIndex = 0;
    while ((m = TOKEN_RE.exec(low)) !== null) out.push(normalizeNumber(m[0]));
    return out;
  }

  function makeTokenizer(bundle) {
    const vocab = bundle.tokenizer;
    const c = bundle.constants;
    return {
      vocab,
      padId: c.pad_id, unkId: c.unk_id, clsId: c.cls_id, sepId: c.sep_id,
      encode(text, maxLen) {
        const toks = tokenizeText(text).slice(0, maxLen - 2);
        const ids = [this.clsId];
        for (const t of toks) {
          ids.push(vocab[t] !== undefined ? vocab[t] : this.unkId);
        }
        ids.push(this.sepId);
        while (ids.length < maxLen) ids.push(this.padId);
        return ids;
      },
    };
  }

  /* --------------------------------------------------------- hawk / dove */
  function countSub(hay, needle) {
    if (needle === "") return 0;
    let n = 0, i = 0;
    while ((i = hay.indexOf(needle, i)) !== -1) { n++; i += needle.length; }
    return n;
  }

  function hawkDove(text, hawkTerms, doveTerms) {
    const low = text.toLowerCase();
    let h = 0, d = 0;
    for (const t of hawkTerms) h += countSub(low, t);
    for (const t of doveTerms) d += countSub(low, t);
    return Math.max(-1.0, Math.min(1.0, (h - d) / (h + d + 2.0)));
  }

  /* ----------------------------------------------------- temporal DKG math */
  function monthDiff(later, earlier) {
    const parts = s => { const [y, m] = s.split("-").map(Number); return y * 12 + m; };
    return parts(later) - parts(earlier);
  }

  function edgeWeight(e, asOf, decay) {
    if (e.first_seen > asOf) return 0.0;
    const last = e.last_seen < asOf ? e.last_seen : asOf;
    const age = Math.max(0, monthDiff(asOf, last));
    return e.count * e.confidence * Math.exp(-decay * age);
  }

  function activeEdges(edges, asOf, decay, minW) {
    const out = [];
    for (const e of edges) {
      const w = edgeWeight(e, asOf, decay);
      if (w > minW) out.push(Object.assign({}, e, { w }));
    }
    return out;
  }

  function l2normInPlace(v) {
    let s = 0;
    for (const x of v) s += x * x;
    s = Math.sqrt(s) + 1e-9;
    for (let i = 0; i < v.length; i++) v[i] /= s;
    return v;
  }

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }
/* networkx.all_simple_paths equivalent (directed, depth-first). */
  function allSimplePaths(adj, source, target, maxHops) {
    const result = [];
    const hops = [];
    (function dfs(node, depth) {
      if (node === target && depth > 0) { result.push(hops.slice()); return; }
      if (depth >= maxHops) return;
      const nbrs = adj.get(node);
      if (!nbrs) return;
      for (const [nxt, edge] of nbrs) {
        if (hops.some(h => h[2] === nxt)) continue;
        hops.push([node, edge.rel, nxt]);
        dfs(nxt, depth + 1);
        hops.pop();
      }
    })(source, 0);
    return result;
  }

  function causalChains(edges, asOf, decay, source, target, maxHops) {
    const flat = new Map();
    const present = new Set();
    for (const e of activeEdges(edges, asOf, decay, 0.05)) {
      let row = flat.get(e.h);
      if (!row) { row = new Map(); flat.set(e.h, row); }
      const prev = row.get(e.t);
      if (!prev || prev.w < e.w) row.set(e.t, { w: e.w, rel: e.r });
      present.add(e.h);
      present.add(e.t);
    }
    if (!present.has(source) || !present.has(target)) return [];
    return allSimplePaths(flat, source, target, maxHops);
  }

  function isSubsequence(short, long) {
    const n = short.length, m = long.length;
    if (n > m) return false;
    for (let i = 0; i <= m - n; i++) {
      let ok = true;
      for (let j = 0; j < n; j++) {
        if (short[j][0] !== long[i + j][0] || short[j][1] !== long[i + j][1] ||
            short[j][2] !== long[i + j][2]) { ok = false; break; }
      }
      if (ok) return true;
    }
    return false;
  }

  function dedupeChains(chains) {
    const ordered = chains.slice().sort((a, b) => b.length - a.length);
    const kept = [];
    for (const chain of ordered) {
      if (!kept.some(longer => isSubsequence(chain, longer))) kept.push(chain);
    }
    const seen = new Set(), out = [];
    for (const c of kept) {
      const text = c.map(h => h[0] + " " + h[1] + " " + h[2]).join("; ");
      if (!seen.has(text)) { seen.add(text); out.push(text); }
    }
    return out;
  }

  function edgeLine(e) {
    return e.h + " --" + e.r + "--> " + e.t +
      " [seen " + e.count + "x, latest " + e.last_seen +
      ", w=" + e.w.toFixed(2) + "]";
  }

  /* -------------------------------------------------------------- retriever */
  function retrieve(bundle, queryEmb, asOf, stance, cfg) {
    const c = bundle.constants;
    const dkg = bundle.dkg;
    const emb = dkg.embeddings;
    const all = activeEdges(dkg.edges, asOf, dkg.decay_lambda, 1e-6);
    const active = new Set();
    for (const e of all) { active.add(e.h); active.add(e.t); }

    const names = Object.keys(emb).sort();
    const sims = names.map(n => ({ name: n, s: dot(emb[n], queryEmb) }));
    sims.sort((a, b) => b.s - a.s);
    const seedsTop = [];
    for (const x of sims) { if (active.has(x.name)) seedsTop.push(x.name); }
    seedsTop.length = Math.min(seedsTop.length, cfg.top_k_nodes);

    let keep = new Set(seedsTop);
    for (let hop = 0; hop < cfg.hops; hop++) {
      const frontier = new Set();
      for (const e of all) {
        if (keep.has(e.h)) frontier.add(e.t);
        if (keep.has(e.t)) frontier.add(e.h);
      }
      for (const f of frontier) keep.add(f);
    }

    let sub = all.filter(e => keep.has(e.h) && keep.has(e.t) &&
                              e.confidence >= cfg.min_confidence);
    sub.sort((a, b) => b.w - a.w);
    sub = sub.slice(0, cfg.max_context_edges);

    const nodeSet = new Set(seedsTop);
    for (const e of sub) { nodeSet.add(e.h); nodeSet.add(e.t); }
    const nodes = Array.from(nodeSet).sort();

    const contextLines = sub.map(edgeLine);

    const nodeW = new Map();
    for (const s of seedsTop) nodeW.set(s, 0.1);
    for (const e of sub) {
      nodeW.set(e.h, (nodeW.get(e.h) || 0) + e.w);
      nodeW.set(e.t, (nodeW.get(e.t) || 0) + e.w);
    }
    const vecs = [], ws = [];
    for (const [n, w] of nodeW) {
      if (emb[n]) { vecs.push(emb[n]); ws.push(w); }
    }
    let contextEmbedding;
    if (vecs.length) {
      contextEmbedding = new Array(c.clip_dim).fill(0);
      for (let i = 0; i < vecs.length; i++) {
        for (let j = 0; j < c.clip_dim; j++) {
          contextEmbedding[j] += vecs[i][j] * ws[i];
        }
      }
      l2normInPlace(contextEmbedding);
    } else {
      contextEmbedding = new Array(c.clip_dim).fill(0);
    }

    const rawHints = new Map();
    for (const e of sub) {
      const dir = bundle.relations[e.r] || "flat";
      const sign = dir === "up" ? 1.0 : dir === "down" ? -1.0 : 0.0;
      const ticker = bundle.entity_tickers[e.t];
      if (ticker && c.assets.includes(ticker) && sign !== 0.0) {
        rawHints.set(ticker, (rawHints.get(ticker) || 0) + sign * e.w);
      }
    }
    let scale = 0;
    for (const v of rawHints.values()) scale = Math.max(scale, Math.abs(v));
    const assetHints = {};
    if (scale > 0) {
      const arr = Array.from(rawHints.entries())
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
      for (const [k, v] of arr) assetHints[k] = Math.round(v / scale * 1000) / 1000;
    }

    let blocked = new Set();
    if (stance > MMTG.STANCE_DEADBAND) blocked = new Set(["Policy_Easing"]);
    else if (stance < -MMTG.STANCE_DEADBAND) blocked = new Set(["Policy_Tightening"]);
    const sources = ["Federal_Reserve", "Policy_Tightening", "Policy_Easing",
                     "Recession_Risk"].filter(s => !blocked.has(s));

    const inNodes = new Set(nodes);
    const rawChains = [];
    for (const assetNode of nodes) {
      if (!bundle.entity_tickers[assetNode]) continue;
      for (const src of sources) {
        if (!inNodes.has(src)) continue;
        const chains = causalChains(dkg.edges, asOf, dkg.decay_lambda,
                                    src, assetNode, 4);
        for (const hops of chains) {
          const tainted = hops.some(h => blocked.has(h[0]) || blocked.has(h[2]));
          if (!tainted) { rawChains.push(hops); break; }
        }
      }
    }
    const chains = dedupeChains(rawChains).slice(0, 4);

    return {
      nodes, edges: sub.map(e => ({ h: e.h, t: e.t, r: e.r, w: e.w,
                                    count: e.count, last_seen: e.last_seen })),
      contextLines, contextEmbedding, assetHints, chains, seeds: seedsTop,
    };
  }

  /* ------------------------------------------------------- image decoding */
  function decodeImageTensor(dataUrl, size, mean, std) {
    return new Promise((resolve, reject) => {
      if (!global.document) { reject(new Error("image decode needs a DOM")); return; }
      const img = global.document.createElement("img");
      img.onload = () => {
        try {
          const cv = global.document.createElement("canvas");
          cv.width = size; cv.height = size;
          const ctx = cv.getContext("2d");
          ctx.drawImage(img, 0, 0, size, size);
          const px = ctx.getImageData(0, 0, size, size);
          const data = px.data;
          const n = size * size;
          const out = new Float32Array(n * 3);
          for (let i = 0; i < n; i++) {
            for (let ch = 0; ch < 3; ch++) {
              out[ch * n + i] = (data[i * 4 + ch] / 255 - mean) / std;
            }
          }
          resolve(out);
        } catch (err) { reject(err); }
      };
      img.onerror = reject;
      img.src = dataUrl;
    });
  }

  /* ------------------------------------------------------------- ONNX I/O */
  let ortMod = null;
  MMTG.setOrt = mod => { ortMod = mod; };
  MMTG.enginesReady = () => ortMod !== null;

  async function loadSession(name, baseUrl) {
    if (!ortMod) throw new Error("onnxruntime not loaded");
    return ortMod.InferenceSession.create(baseUrl + name,
      { executionProviders: ["wasm"], logLevel: 3 });
  }

  /* --------------------------------------------------------------- pipeline */
  MMTG.RETRIEVE_CFG = { top_k_nodes: 6, hops: 1, min_confidence: 0.5,
                        max_context_edges: 12 };

  async function Pipeline(bundle, baseUrl) {
    const c = bundle.constants;
    const tok = makeTokenizer(bundle);
    const sess = {
      text: await loadSession("text_enc.onnx", baseUrl),
      image: await loadSession("image_enc.onnx", baseUrl),
      port: await loadSession("port_cnn.onnx", baseUrl),
      vla: await loadSession("vla.onnx", baseUrl),
    };

    function tf(type, data, dims) { return new ortMod.Tensor(type, data, dims); }
    function toArr(t) { return Array.from(t.data); }

    async function embedText(text) {
      const ids = tok.encode(text, c.max_text_len);
      const mask = ids.map(x => x === tok.padId ? 0 : 1);
      const res = await sess.text.run(
        { ids: tf("int64", ids, [1, c.max_text_len]),
          mask: tf("int64", mask, [1, c.max_text_len]) }, ["text_emb"]);
      return toArr(res.text_emb);
    }

    async function embedChart(chwTensor) {
      const res = await sess.image.run(
        { image: tf("float32", chwTensor, [1, 3, c.img_size, c.img_size]) },
        ["image_emb"]);
      return toArr(res.image_emb);
    }

    async function portDensity(chwTensor) {
      const res = await sess.port.run(
        { port_image: tf("float32", chwTensor, [1, 3, c.port_size, c.port_size]) },
        ["density"]);
      return Math.max(0, Math.min(1, res.density.data[0]));
    }

    function retrieveCtx(queryEmb, asOf, stance) {
      return retrieve(bundle, queryEmb, asOf, stance, MMTG.RETRIEVE_CFG);
    }

    async function generate(stmt, asOf, opts) {
      const o = opts || {};
      const hd = o.hd !== undefined ? o.hd : hawkDove(stmt, bundle.hawk_terms,
                                                       bundle.dove_terms);
      const queryEmb = o.queryEmb !== undefined ? o.queryEmb
                     : await embedText(stmt);

      let chartEmb = o.chartEmb;
      if (!chartEmb) {
        let embs = o.chartEmbs;
        if (!embs && o.chartTensors) {
          embs = [];
          for (const t of o.chartTensors) embs.push(await embedChart(t));
        }
        if (embs) {
          chartEmb = new Array(c.clip_dim).fill(0);
          for (const e of embs)
            for (let i = 0; i < c.clip_dim; i++) chartEmb[i] += e[i] / embs.length;
          l2normInPlace(chartEmb);
        }
      }
      if (!chartEmb) throw new Error("generate needs chartEmb or chartTensors");

      const ctx = retrieveCtx(queryEmb, asOf, hd);

      const density = o.density !== undefined ? o.density
                    : o.portTensor ? await portDensity(o.portTensor) : 0.5;

      const ids = tok.encode(stmt, c.max_statement_len);
      const mask = ids.map(x => x === tok.padId ? 0 : 1);
      const res = await sess.vla.run({
        chart_emb: tf("float32", chartEmb, [1, c.clip_dim]),
        ids: tf("int64", ids, [1, c.max_statement_len]),
        mask: tf("int64", mask, [1, c.max_statement_len]),
        ctx_emb: tf("float32", ctx.contextEmbedding, [1, c.clip_dim]),
        feats: tf("float32", [density, hd], [1, 2]),
      }, ["weights"]);
      const weights = toArr(res.weights);

      const action = { month: asOf };
      const wmap = {};
      let gross = 0, net = 0;
      for (let i = 0; i < c.assets.length; i++) {
        const v = weights[i];
        wmap[c.assets[i]] = Math.round(v * 10000) / 10000;
        gross += Math.abs(v); net += v;
      }
      action.action = wmap;
      action.gross_exposure = Math.round(gross * 10000) / 10000;
      action.net_exposure = Math.round(net * 10000) / 10000;
      action.rationale = ctx.chains;
      action.evidence = ctx.contextLines.slice(0, 4);
      action.signals = {
        hawk_dove: Math.round(hd * 1000) / 1000,
        port_density: Math.round(density * 1000) / 1000,
        retrieved_nodes: ctx.nodes,
        asset_hints: ctx.assetHints,
      };
      return { action, ctx, weights, chartEmb, density, hd };
    }

    return { bundle, tok, generate, embedText, embedChart, portDensity,
             retrieveCtx };
  }
  MMTG.Pipeline = Pipeline;

  /* --------------------------------------------------------- self check */
  function maxAbsDiff(a, b) {
    let m = 0;
    for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
    return m;
  }

  async function runSelfCheck(bundle, baseUrl) {
    const c = bundle.constants;
    const refs = bundle.refs;
    const checks = {};
    const report = reason => ({ ok: reason === null, detail: reason });

    const pipe = await Pipeline(bundle, baseUrl);

    const ids64 = pipe.tok.encode(refs.text.text, c.max_text_len);
    const ids96 = pipe.tok.encode(refs.vla.text, c.max_statement_len);
    checks.tokenizer = report(
      JSON.stringify(ids64) === JSON.stringify(refs.text.ids) ? null : "ids64 mismatch");

    const hd = hawkDove(refs.text.text, bundle.hawk_terms, bundle.dove_terms);
    checks.hawk_dove = report(
      Math.abs(hd - refs.vla.feats[1]) < 1e-9 ? null : `hd ${hd} vs ${refs.vla.feats[1]}`);

    const textEmb = await pipe.embedText(refs.text.text);
    checks.text_encoder = report(maxAbsDiff(textEmb, refs.text.out) < 1e-4
      ? null : "text emb mismatch");

    const ctx = retrieve(bundle, textEmb, "2025-12", hd, MMTG.RETRIEVE_CFG);
    const ctxErr = maxAbsDiff(ctx.contextEmbedding, refs.vla.ctx_emb);
    checks.graph_retrieval = report(ctxErr < 1e-4 ? null : "ctx emb mismatch");

    const out = await pipe.generate(refs.vla.text, "2025-12", {
      chartEmb: refs.vla.chart_emb, queryEmb: textEmb,
      density: refs.vla.feats[0], hd: refs.vla.feats[1],
    });
    const wErr = maxAbsDiff(out.weights, refs.vla.out);
    checks.vla_head = report(wErr < 1e-4 ? null : "weights mismatch");

    if (global.document) {
      const imgTensor = await decodeImageTensor(refs.image.png, c.img_size,
                                                c.img_mean, c.img_std);
      const imageEmb = await pipe.embedChart(imgTensor);
      checks.image_encoder = report(maxAbsDiff(imageEmb, refs.image.out) < 0.03
        ? null : "image emb mismatch");
      const portTensor = await decodeImageTensor(refs.port.png, c.port_size,
                                                 c.img_mean, c.img_std);
      const dens = await pipe.portDensity(portTensor);
      checks.port_cnn = report(Math.abs(dens - refs.port.out[0]) < 0.03
        ? null : "port density mismatch");
    }
    return checks;
  }
  MMTG.runSelfCheck = runSelfCheck;

  global.MMTG = MMTG;
})(typeof window !== "undefined" ? window : globalThis);