"""The judging subset every model sees (the same items for every model, so models are compared on equal terms).

Defaults give 600 items, the per-model cap for cloud models:
- Harvest: for each window w = 1, 10, 50, 30 violation records and 30 honest records (honest ones drawn in proportion
  to the dataset's honest strata, so about 80% have at least one apple taken), each judged in both renderings
  (summary and raw): 3 x 60 x 2 = 360 items;
- BazaarBench: for each failure type F1-F5, 12 unsafe and 12 safe GPT-5 decisions, each judged in both renderings
  (bundle and public): 5 x 24 x 2 = 240 items. F6 is left out (only 7 unsafe decisions in the whole set).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np


def load_jsonl(path) -> list[dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _pick(rng, items, n):
    if len(items) <= n:
        return list(items)
    return [items[i] for i in sorted(rng.choice(len(items), n, replace=False))]


def harvest_subset(examples, per_label=30, windows=(1, 10, 50), seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for w in windows:
        pos = [e for e in examples if e["window"] == w and e["label"] == 1]
        neg = [e for e in examples if e["window"] == w and e["label"] == 0]
        out += _pick(rng, pos, per_label) + _pick(rng, neg, per_label)
    return out


def bazaar_subset(examples, per_label=12, ftypes=("F1", "F2", "F3", "F4", "F5"), seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for f in ftypes:
        for lab in (1, 0):
            out += _pick(rng, [e for e in examples if e["ftype"] == f and e["label"] == lab], per_label)
    return out


HARVEST_RENDERINGS = ("summary", "raw")
BAZAAR_RENDERINGS = ("bundle", "public")


def items(harvest=(), bazaar=(), harvest_renderings=HARVEST_RENDERINGS, bazaar_renderings=BAZAAR_RENDERINGS):
    """(example, rendering) pairs, in a fixed order."""
    out = [(e, r) for r in harvest_renderings for e in harvest]
    out += [(e, r) for r in bazaar_renderings for e in bazaar]
    return out
