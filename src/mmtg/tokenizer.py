"""Word level financial tokenizer.

A compact deterministic tokenizer is sufficient here because both the chart
captions and the synthetic central bank corpus come from a controlled financial
vocabulary. The same class also handles arbitrary real text at inference time,
mapping unseen words to [UNK].
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from .utils import load_json, save_json

PAD, UNK, CLS, SEP = "[PAD]", "[UNK]", "[CLS]", "[SEP]"
SPECIALS = [PAD, UNK, CLS, SEP]

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?|[+-][0-9]+(?:\.[0-9]+)?%?|bp\b")
_NUM_RE = re.compile(r"^[+-]?[0-9]+(?:\.[0-9]+)?%?$")


def _normalize_number(tok: str) -> str:
    """Bucket raw numerals so the vocabulary stays small yet discriminative.

    Percent figures keep their sign and integer magnitude ("-1.4%" ->
    "<neg1%>"). Plain numbers below 1000 (basis point figures, mostly) are
    bucketed to the nearest 25 so captions that differ in magnitude stay
    textually distinct.
    """
    if not _NUM_RE.match(tok):
        return tok
    pct = tok.endswith("%")
    body = tok[:-1] if pct else tok
    try:
        val = float(body)
    except ValueError:
        return tok
    sign = "neg" if val < 0 else ""
    mag = int(abs(val))
    if pct:
        mag = min(mag, 15)
        return f"<{sign}{mag}%>"
    if mag >= 1000:
        return "<bignum>"
    bucket = int(round(mag / 25.0) * 25)
    return f"<n{sign}{bucket}>"


def tokenize_text(text: str) -> list[str]:
    text = text.lower().replace("&", " and ")
    toks = _TOKEN_RE.findall(text)
    return [_normalize_number(t) for t in toks]


class FinancialTokenizer:
    """Vocabulary built from the training corpus, with save and load."""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.inv = {i: t for t, i in vocab.items()}
        self.pad_id = vocab[PAD]
        self.unk_id = vocab[UNK]
        self.cls_id = vocab[CLS]
        self.sep_id = vocab[SEP]

    @classmethod
    def build(cls, corpus: Iterable[str], vocab_size: int = 4096,
              min_freq: int = 1) -> "FinancialTokenizer":
        counts: Counter = Counter()
        for text in corpus:
            counts.update(tokenize_text(text))
        vocab = {t: i for i, t in enumerate(SPECIALS)}
        for tok, freq in counts.most_common():
            if freq < min_freq or len(vocab) >= vocab_size:
                break
            vocab[tok] = len(vocab)
        return cls(vocab)

    def encode(self, text: str, max_len: int) -> list[int]:
        """[CLS] tokens [SEP], padded or truncated to max_len."""
        ids = [self.cls_id]
        for tok in tokenize_text(text)[: max_len - 2]:
            ids.append(self.vocab.get(tok, self.unk_id))
        ids.append(self.sep_id)
        if len(ids) < max_len:
            ids.extend([self.pad_id] * (max_len - len(ids)))
        return ids

    def attention_mask(self, ids: list[int]) -> list[int]:
        return [0 if i == self.pad_id else 1 for i in ids]

    def decode(self, ids: Iterable[int]) -> str:
        toks = [self.inv.get(int(i), UNK) for i in ids]
        return " ".join(t for t in toks if t not in (PAD, CLS, SEP))

    def __len__(self) -> int:
        return len(self.vocab)

    def save(self, path: str) -> None:
        save_json(self.vocab, path)

    @classmethod
    def load(cls, path: str) -> "FinancialTokenizer":
        return cls(load_json(path))
