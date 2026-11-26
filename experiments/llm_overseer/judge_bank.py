"""Judge the situation bank's stratified subset with one LLM inspector (error_patterns/protocol.md, section 2).

  # gpt-oss-20B on raw maps, Ollama cloud, under the study's 3M-token cap
  PYTHONPATH=. .venv-socialjax/bin/python -m experiments.llm_overseer.judge_bank --model gpt-oss-20b --think low \
      --max-tokens 2048 --workers 4 --token-cap 3000000 \
      --token-log notes/llm_overseer/error_patterns/cloud_tokens.tsv
  # Qwen 7B + reasoning, local, the first 600 items of the same order
  PYTHONPATH=. .venv-socialjax/bin/python -m experiments.llm_overseer.judge_bank --model qwen2.5-7b --mode reason \
      --num-ctx 4096 --limit 600

The subset: within each value of `near`, items drawn at random from the bank (seed 0) up to a quota, then shuffled
once, so any first part of the order is a random part of the subset. The order is written to <out-dir>/subset.tsv.
Output: <out-dir>/<tag>_<mode>.jsonl (resumes), one line per item as in overseer/llm/judge.run_items.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from overseer.llm import subset as S
from overseer.llm.judge import OllamaBackend, run_items
from overseer.llm.models import BY_TAG

QUOTA = {0: 400, 1: 400, 2: 400, 3: 300, 4: 250, 5: 200, 6: 200, **{n: 150 for n in range(7, 13)}}


def stratified_order(bank, quota=QUOTA, seed=0):
    rng = np.random.default_rng(seed)
    chosen = []
    for near in sorted(quota):
        pool = [x for x in bank if x["near"] == near]
        take = rng.permutation(len(pool))[:quota[near]]
        chosen += [pool[i] for i in sorted(take)]
    return [chosen[i] for i in rng.permutation(len(chosen))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="direct", choices=("direct", "reason"))
    ap.add_argument("--think", default=None)
    ap.add_argument("--max-tokens", type=int, default=None, help="default: 16 direct (2048 with --think), 768 reason")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--bank", default="experiments/llm_overseer/data/error_bank.jsonl.gz")
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N items of the order")
    ap.add_argument("--token-cap", type=int, default=None)
    ap.add_argument("--token-log", default=None)
    ap.add_argument("--out-dir", default="notes/llm_overseer/error_patterns/judge")
    a = ap.parse_args()

    info = BY_TAG.get(a.model)
    tag = info["tag"] if info else a.model.replace("/", "_").replace(":", "_")
    max_tokens = a.max_tokens or (768 if a.mode == "reason" else (2048 if a.think else 16))
    order = stratified_order(S.load_jsonl(a.bank))
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "subset.tsv").write_text("order\tid\tnear\tlabel\n" + "".join(
        f"{i}\t{x['id']}\t{x['near']}\t{x['label']}\n" for i, x in enumerate(order)))
    if a.limit:
        order = order[:a.limit]
    out = out_dir / f"{tag}_{a.mode}.jsonl"
    print(f"{tag}: {len(order)} items, mode {a.mode}, max tokens {max_tokens} -> {out}", flush=True)
    t0 = time.time()
    backend = OllamaBackend(info["ollama"] if info else a.model, host=a.host, num_ctx=a.num_ctx, think=a.think)
    status = run_items(backend, [(x, "raw") for x in order], a.mode, out, max_tokens, workers=a.workers,
                       token_cap=a.token_cap, token_log=a.token_log)
    status["minutes"] = round((time.time() - t0) / 60, 1)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
