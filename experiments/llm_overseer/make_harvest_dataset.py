"""Build the Harvest judging dataset (overseer/llm/harvest_records.py) and save it as gzipped JSONL.

  PYTHONPATH=. .venv-socialjax/bin/python -m experiments.llm_overseer.make_harvest_dataset \
      --out experiments/llm_overseer/data/harvest_judge.jsonl.gz

Fixed seed; defaults: windows 1, 10, 50; 1,000 records per window (500 violations, 400 honest with at least one
apple taken, 100 honest with none); 6 bot mixes x 8 episodes of 1,000 steps. Writes a manifest next to the data.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from overseer.llm.harvest_records import build_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/llm_overseer/data/harvest_judge.jsonl.gz")
    ap.add_argument("--windows", default="1,10,50")
    ap.add_argument("--per-window", type=int, default=1000)
    ap.add_argument("--seeds-per-mix", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    windows = tuple(int(x) for x in a.windows.split(","))
    examples, manifest = build_dataset(windows=windows, per_window=a.per_window, seeds_per_mix=a.seeds_per_mix,
                                       steps=a.steps, seed=a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")
    manifest["n_examples"] = len(examples)
    manifest["labels"] = {str(w): dict(Counter(ex["label"] for ex in examples if ex["window"] == w)) for w in windows}
    manifest["bot_kinds"] = {str(w): dict(Counter((ex["bot_kind"], ex["label"]) for ex in examples
                                                  if ex["window"] == w).most_common()) for w in windows}
    manifest["bot_kinds"] = {w: {f"{k[0]}:{k[1]}": v for k, v in d.items()} for w, d in manifest["bot_kinds"].items()}
    for r in ("summary", "raw"):
        manifest[f"chars_{r}"] = {str(w): round(sum(len(ex["renderings"][r]) for ex in examples if ex["window"] == w)
                                               / max(1, sum(ex["window"] == w for ex in examples)))
                                  for w in windows}
    Path(str(out).replace(".jsonl.gz", "_manifest.json")).write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("n_examples", "labels", "chars_summary", "chars_raw")}, indent=1))


if __name__ == "__main__":
    main()
