"""Shared helpers: seeding, JSON IO, month arithmetic, simple logging."""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Iterable

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (when available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_jsonl(rows: Iterable[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def month_str(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def month_index_to_str(start_year: int, start_month: int, idx: int) -> str:
    total = (start_year * 12 + (start_month - 1)) + idx
    return month_str(total // 12, total % 12 + 1)


def month_diff(later: str, earlier: str) -> int:
    """Whole months between two YYYY-MM strings (later minus earlier)."""
    ly, lm = int(later[:4]), int(later[5:7])
    ey, em = int(earlier[:4]), int(earlier[5:7])
    return (ly * 12 + lm) - (ey * 12 + em)


class Timer:
    """Tiny context timer for training logs."""

    def __enter__(self) -> "Timer":
        self.t0 = time.time()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.time() - self.t0

    def __str__(self) -> str:
        return f"{getattr(self, 'elapsed', 0.0):.1f}s"


def log(msg: str) -> None:
    print(f"[mmtg] {msg}", flush=True)
