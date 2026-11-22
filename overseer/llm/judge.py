"""Run an LLM inspector over judging items and record its verdict and token counts.

Backends:
- ollama (main, 22 Nov change of plan): the local Ollama daemon (default http://localhost:11434). Local models run on
  this machine; cloud models (names ending in "-cloud", e.g. gpt-oss:120b-cloud) go through the same daemon using
  the account it is signed in to, so no key is read or stored here. A 402/429 or a usage/credit message stops the run.
- transformers: CPU, a fallback smoke test without Ollama.
- vllm: GPU batch inference (optional alternative on CREATE; see notes/llm_overseer/README.md). Not run so far.

One output line per item: id, rendering, mode, model, label, verdict (1 violation / 0 honest / null none found),
tokens in and out, seconds, the reply text (cut to 4,000 characters) and any error. Runs resume: items already in the
output file are skipped. A token cap (summed over a token log shared by all runs) stops a run before it goes over.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from overseer.llm.prompts import build_messages, parse_verdict


class QuotaStop(RuntimeError):
    """Out of plan credit or rate-limited: stop cleanly."""


class OllamaBackend:
    def __init__(self, model, host="http://localhost:11434", num_ctx=8192, think=None, timeout=900, seed=0):
        self.model, self.host, self.num_ctx, self.think = model, host, num_ctx, think
        self.timeout, self.seed = timeout, seed

    @property
    def cloud(self):
        return self.model.endswith("-cloud") or ":cloud" in self.model

    def chat(self, messages, max_tokens):
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": 0, "seed": self.seed, "num_ctx": self.num_ctx, "num_predict": max_tokens}}
        if self.think is not None:
            body["think"] = self.think
        req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0, last = time.time(), None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as exc:
                text = exc.read().decode(errors="ignore").lower()
                if exc.code in (402, 429) or any(w in text for w in ("usage limit", "credit", "quota", "insufficient")):
                    raise QuotaStop(f"HTTP {exc.code}: {text[:200]}")
                if exc.code in (400, 401, 403, 404):
                    raise RuntimeError(f"HTTP {exc.code}: {text[:300]}")
                last = exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = exc
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"Ollama request failed after retries: {last}")
        msg = data.get("message", {})
        return dict(text=msg.get("content", ""), thinking=msg.get("thinking", "") or "",
                    tokens_in=int(data.get("prompt_eval_count") or 0), tokens_out=int(data.get("eval_count") or 0),
                    seconds=time.time() - t0, model=data.get("model", self.model),
                    done_reason=data.get("done_reason"))


class TransformersBackend:
    def __init__(self, model, device="cpu", seed=0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(seed)
        self.model, self.torch, self.device = model, torch, device
        self.tok = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32).to(device).eval()

    def chat(self, messages, max_tokens):
        t0 = time.time()
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if not hasattr(ids, "shape"):  # newer transformers may return a BatchEncoding
            ids = ids["input_ids"]
        ids = ids.to(self.device)
        with self.torch.no_grad():
            out = self.lm.generate(ids, max_new_tokens=max_tokens, do_sample=False, pad_token_id=self.tok.eos_token_id)
        new = out[0, ids.shape[1]:]
        return dict(text=self.tok.decode(new, skip_special_tokens=True), thinking="", tokens_in=int(ids.shape[1]),
                    tokens_out=int(new.shape[0]), seconds=time.time() - t0, model=self.model, done_reason=None)


def vllm_generate(model, item_messages, max_tokens, tensor_parallel=1, max_model_len=16384):
    """Batch inference with vLLM (optional CREATE path; not run so far). One result dict per item."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel, max_model_len=max_model_len, seed=0)
    params = SamplingParams(temperature=0, max_tokens=max_tokens)
    t0 = time.time()
    outs = llm.chat(item_messages, params)
    per = (time.time() - t0) / max(1, len(outs))
    return [dict(text=o.outputs[0].text, thinking="", tokens_in=len(o.prompt_token_ids),
                 tokens_out=len(o.outputs[0].token_ids), seconds=per, model=model, done_reason=None) for o in outs]


def done_keys(out_path: Path) -> set:
    keys = set()
    if Path(out_path).exists():
        with open(out_path) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if r.get("error") is None:
                        keys.add((r["id"], r["rendering"], r["mode"]))
    return keys


def tokens_logged(log_path) -> int:
    """Total tokens (in + out) in a token log written by run_items."""
    if log_path is None or not Path(log_path).exists():
        return 0
    total = 0
    with open(log_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 6 and parts[4].isdigit() and parts[5].isdigit():
                total += int(parts[4]) + int(parts[5])
    return total


def make_record(example, rendering, mode, res, model_name):
    return dict(id=example["id"], dataset=example["dataset"], rendering=rendering, mode=mode, label=example["label"],
                error=None, model=res.get("model", model_name), verdict=parse_verdict(res["text"], mode),
                tokens_in=res["tokens_in"], tokens_out=res["tokens_out"], seconds=round(res["seconds"], 3),
                done_reason=res.get("done_reason"), text=res["text"][:4000], thinking_chars=len(res.get("thinking", "")))


def run_items(backend, items, mode, out_path, max_tokens, workers=1, token_cap=None, token_log=None,
              max_errors=5, progress_every=50):
    """Judge every (example, rendering) item not already in out_path. Returns a short status dict."""
    out_path, lock = Path(out_path), threading.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out_path)
    todo = [(e, r) for e, r in items if (e["id"], r, mode) not in done]
    state = dict(tokens=tokens_logged(token_log), errors=0, consecutive=0, written=0, stop=None)
    if token_log is not None:
        Path(token_log).parent.mkdir(parents=True, exist_ok=True)
        if not Path(token_log).exists():
            Path(token_log).write_text("time\tmodel\tid\trendering\ttokens_in\ttokens_out\n")

    def one(item):
        e, r = item
        with lock:
            if state["stop"]:
                return
            if token_cap is not None and state["tokens"] >= token_cap:
                state["stop"] = f"token cap {token_cap} reached ({state['tokens']} tokens logged)"
                return
        try:
            rec = make_record(e, r, mode, backend.chat(build_messages(e, r, mode), max_tokens), backend.model)
        except QuotaStop as exc:
            with lock:
                state["stop"] = str(exc)
            return
        except Exception as exc:  # noqa: BLE001 - record it, and stop on repeated errors
            rec = dict(id=e["id"], dataset=e["dataset"], rendering=r, mode=mode, label=e["label"],
                       model=backend.model, verdict=None, tokens_in=0, tokens_out=0,
                       error=f"{type(exc).__name__}: {exc}"[:500])
        with lock:
            with open(out_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            state["written"] += 1
            state["tokens"] += rec["tokens_in"] + rec["tokens_out"]
            if token_log is not None and rec["error"] is None:
                with open(token_log, "a") as fh:
                    fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{rec['model']}\t{rec['id']}\t{r}\t"
                             f"{rec['tokens_in']}\t{rec['tokens_out']}\n")
            if rec["error"] is not None:
                state["errors"] += 1
                state["consecutive"] += 1
                if state["consecutive"] >= max_errors:
                    state["stop"] = f"{max_errors} errors in a row; last: {rec['error']}"
            else:
                state["consecutive"] = 0
            if state["written"] % progress_every == 0:
                print(f"  {state['written']}/{len(todo)} done; tokens {state['tokens']}", flush=True)

    if workers <= 1:
        for it in todo:
            one(it)
            if state["stop"]:
                break
    else:
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(one, todo))
    return dict(todo=len(todo), written=state["written"], errors=state["errors"], tokens_total=state["tokens"],
                stop=state["stop"])
