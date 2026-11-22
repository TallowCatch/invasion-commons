"""Build the BazaarBench judging set (overseer/llm/bazaar_records.py).

  bash experiments/bazaarbench/download.sh            # once; about 258 MB into data/bazaarbench/raw (git-ignored)
  python -m experiments.llm_overseer.make_bazaar_dataset

The texts are BazaarBench data (CC-BY-4.0), which this repository does not commit (.gitignore), so the set is
written to data/bazaarbench/llm_judge/ (git-ignored) and only the index of chosen (bundle, failure type) pairs with
the GPT-5 labels is committed, as experiments/llm_overseer/data/bazaar_index.tsv.
The build is deterministic (fixed seed), so the index is a check that a rebuild matches.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from overseer.llm.bazaar_records import build_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/bazaarbench/raw")
    ap.add_argument("--out", default="data/bazaarbench/llm_judge/bazaar_judge.jsonl.gz")
    ap.add_argument("--index", default="experiments/llm_overseer/data/bazaar_index.tsv")
    ap.add_argument("--per-cell", type=int, default=50)
    ap.add_argument("--max-chars", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    examples, manifest = build_dataset(a.raw, per_cell=a.per_cell, max_chars=a.max_chars, seed=a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(a.index, "w") as fh:
        fh.write("id\tcell\tbundle_id\tftype\tgpt5_label\tgpt5_max_stage\treasoning_only\n")
        for ex in examples:
            fh.write(f"{ex['id']}\t{ex['cell']}\t{ex['bundle_id']}\t{ex['ftype']}\t{ex['label']}\t"
                     f"{ex['gpt5_max_stage']}\t{int(ex['reasoning_only'])}\n")
    manifest["n_examples"] = len(examples)
    manifest["labels"] = dict(Counter((ex["ftype"], ex["label"]) for ex in examples))
    manifest["labels"] = {f"{k[0]}:{k[1]}": v for k, v in sorted(manifest["labels"].items())}
    manifest["reasoning_only_unsafe"] = sum(ex["reasoning_only"] for ex in examples)
    manifest["mean_chars"] = {r: round(sum(len(ex["renderings"][r]) for ex in examples) / max(1, len(examples)))
                              for r in ("bundle", "public")}
    Path(a.index.replace("_index.tsv", "_manifest.json")).write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("n_examples", "labels", "reasoning_only_unsafe", "mean_chars")},
                     indent=1))


if __name__ == "__main__":
    main()
