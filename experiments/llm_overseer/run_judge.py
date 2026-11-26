"""Judge the shared subset (overseer/llm/subset.py) with one model and write one JSONL line per item.

Local Ollama model (ladder tags in overseer/llm/models.py):
  python -m experiments.llm_overseer.run_judge --model qwen2.5-3b
Cloud model through the signed-in local daemon, with the shared token cap and log:
  python -m experiments.llm_overseer.run_judge --model gpt-oss-120b --think low --max-tokens 2048 \
      --token-cap 1500000 --token-log results/llm_overseer/cloud_tokens.tsv
Smoke test on a few items:
  python -m experiments.llm_overseer.run_judge --model qwen2.5-0.5b --limit 50 --out-dir results/llm_overseer/smoke
Reason mode on the w = 1 Harvest items only (what the RL runs use):
  python -m experiments.llm_overseer.run_judge --model qwen2.5-7b --mode reason --datasets harvest --windows 1
Other backends: --backend transformers --model Qwen/Qwen2.5-0.5B-Instruct (CPU), or --backend vllm (CREATE).

Output: <out-dir>/<tag>_<mode>.jsonl (resumes if it exists).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from overseer.llm import subset as S
from overseer.llm.judge import OllamaBackend, TransformersBackend, make_record, run_items, vllm_generate
from overseer.llm.models import BY_TAG
from overseer.llm.prompts import build_messages


def load_items(a):
    harvest = bazaar = []
    if "harvest" in a.datasets:
        harvest = S.harvest_subset(S.load_jsonl(a.harvest), per_label=a.harvest_per_label, seed=a.seed)
    if "bazaar" in a.datasets:
        bazaar = S.bazaar_subset(S.load_jsonl(a.bazaar), per_label=a.bazaar_per_label, seed=a.seed)
    if a.windows:
        keep = {int(w) for w in a.windows.split(",")}
        harvest = [e for e in harvest if e["window"] in keep]
    items = S.items(harvest, bazaar)
    if a.limit:  # spread a small smoke test over every dataset and rendering
        step = max(1, len(items) // a.limit)
        items = items[::step][:a.limit]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ladder tag, or a raw model name for the backend")
    ap.add_argument("--backend", default="ollama", choices=("ollama", "transformers", "vllm"))
    ap.add_argument("--mode", default="direct", choices=("direct", "reason"))
    ap.add_argument("--max-tokens", type=int, default=None, help="default: 16 direct, 768 reason")
    ap.add_argument("--think", default=None, help="Ollama 'think' for reasoning models, e.g. low")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--datasets", default="harvest,bazaar")
    ap.add_argument("--harvest", default="experiments/llm_overseer/data/harvest_judge.jsonl.gz")
    ap.add_argument("--bazaar", default="data/bazaarbench/llm_judge/bazaar_judge.jsonl.gz")
    ap.add_argument("--harvest-per-label", type=int, default=30)
    ap.add_argument("--bazaar-per-label", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", default="", help="Harvest windows to keep, e.g. 1 (default: all)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-items", type=int, default=600, help="hard cap on items per model (cloud cap: 600)")
    ap.add_argument("--token-cap", type=int, default=None)
    ap.add_argument("--token-log", default=None)
    ap.add_argument("--out-dir", default="results/llm_overseer/judge")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    info = BY_TAG.get(a.model)
    tag = a.tag or (info["tag"] if info else a.model.replace("/", "_").replace(":", "_"))
    max_tokens = a.max_tokens or (16 if a.mode == "direct" else 768)
    items = load_items(a)
    if len(items) > a.max_items:
        raise SystemExit(f"{len(items)} items is over --max-items {a.max_items}")
    out = Path(a.out_dir) / f"{tag}_{a.mode}.jsonl"
    print(f"{tag}: {len(items)} items, mode {a.mode}, max tokens {max_tokens} -> {out}", flush=True)
    t0 = time.time()
    if a.backend == "vllm":
        name = info["hf_id"] if info else a.model
        res = vllm_generate(name, [build_messages(e, r, a.mode) for e, r in items], max_tokens)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            for (e, r), x in zip(items, res):
                fh.write(json.dumps(make_record(e, r, a.mode, x, name)) + "\n")
        status = dict(written=len(res))
    else:
        if a.backend == "ollama":
            backend = OllamaBackend(info["ollama"] if info else a.model, host=a.host, num_ctx=a.num_ctx, think=a.think)
        else:
            backend = TransformersBackend(info["hf_id"] if info else a.model)
        status = run_items(backend, items, a.mode, out, max_tokens, workers=a.workers, token_cap=a.token_cap,
                           token_log=a.token_log)
    status["minutes"] = round((time.time() - t0) / 60, 1)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
