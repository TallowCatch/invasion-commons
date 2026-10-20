"""Study R1b: generate the two strategy pools (Pool A audit-aware, Pool B blind) with gpt-oss-120b.

Protocol (frozen, with amendments): notes/claude_audit_20261005/studies/R1b_audit_aware_strategies/protocol.md
Amendment 1 holds the exact Pool A prompt additions (overseer/adapters/emergent_llm.py: AUDIT_SECTION).

Pipeline: Willis et al.'s two-step generation (external/emergent_llm, generation/create_strategies.py), unchanged:
  1. one call for the natural-language strategy (their system prompt and create_strategy_user_prompt);
  2. one call for the code (their system prompt and create_code_user_prompt), then their clean_generated_code,
     validate_strategy_code (AST), write_strategy_class (rename to Strategy_<ATTITUDE>_<i>) and test harness
     (Pool B: their test_generated_strategy; Pool A: the same games in the audit-aware game,
     emergent_llm.test_aware_strategy_class). Up to 3 code attempts per strategy (their retry rule); a strategy whose
     3 attempts all fail is recorded as failed and excluded (protocol stop rule), and is never regenerated.
Calls go to Ollama (gpt-oss:120b) through the project's client (run_l2_llm_agents.Client, extended with Ollama's
`think` flag, as their Ollama call uses think=True); a 402/429 or usage-limit reply stops the run (QuotaStop).

Modes (run from the repo root):
  PYTHONPATH=. python3 -m experiments.oversight.run_r1b_generate pilot --out DIR [--client fake]
  PYTHONPATH=. python3 -m experiments.oversight.run_r1b_generate full  --out DIR [--per-synonym 16]
  PYTHONPATH=. python3 -m experiments.oversight.run_r1b_generate report --out DIR
- pilot: 10 strategies per pool (5 Collective, 5 Selfish, the 4 synonyms of each spread evenly), then a cost report.
- full: the whole pools (2 attitudes x 4 synonyms x --per-synonym per pool; 16 -> 128). The pilot's strategies are the
  first slots of the full plan and are kept (skipped). Refuses to start unless the pilot report exists and its
  projection gate passed (protocol: cost per strategy x 256 <= $2.5), unless --per-synonym is below 16 (pool sizes
  halved by amendment) or --skip-gate.
- report: print the pilot cost report from the log (no calls).
Order: index-major (strategy 1 of every synonym, then 2, ...), Collective and Selfish synonyms alternating, pools
alternating, so a run stopped by the budget leaves both pools and all synonyms balanced.

Budget (protocol "Budget"): every call is logged to DIR/calls.jsonl with its prompt hashes, the response (content and
thinking), Ollama's prompt_eval_count and eval_count, and an estimated cost = tokens x $0.32 per million. Before each
call the generator reads the running total (all earlier runs' log rows included, so the cap spans pilot + full) and
stops if the estimated spend plus one average call would reach --cap-usd (default 3.00), or if the calls made
(failed calls and retries included) would exceed --call-cap (default 1,000). Where it stopped is written to
DIR/status.json (and appended to DIR/status_history.jsonl). Rerunning resumes: finished and failed strategies are
skipped, and a strategy whose description exists continues with its remaining code attempts.

Layout (their StrategyRegistry layout, one model): DIR/pool_A/common_pool/gpt-oss-120b.py (all valid classes),
DIR/pool_A/common_pool/gpt-oss-120b_descriptions.py, DIR/pool_A/records/<ATTITUDE>_<i>.json (one record per
strategy: description, every attempt, status), the same for pool_B, DIR/prompts.jsonl (each distinct prompt text
once, by sha256), DIR/calls.jsonl, DIR/status.json, DIR/pilot_report.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from overseer.adapters import emergent_llm as E
from overseer.seeding import stable_seed

from emergent_llm.common import Attitude  # noqa: E402  (path set by the adapter)
from emergent_llm.games import CommonPoolDescription  # noqa: E402
from emergent_llm.generation import create_strategies as W  # noqa: E402
from emergent_llm.players import BaseStrategy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "runs" / "claude_r1b_v1" / "strategies"
MODEL = "gpt-oss:120b-cloud"           # Ollama Cloud name; the client sends "gpt-oss:120b" to https://ollama.com
PRICE_PER_M = 0.32                     # $ per million tokens (protocol "Budget")
CAP_USD = 3.00
CALL_CAP = 1000
GATE_USD_256 = 2.5
MAX_CODE_ATTEMPTS = 3
PILOT_PER_POOL = 10
COLLECTIVE_SYN = (Attitude.COLLECTIVE, Attitude.PROSOCIAL, Attitude.ALTRUISTIC, Attitude.BENEVOLENT)
SELFISH_SYN = (Attitude.SELFISH, Attitude.SELFINTERESTED, Attitude.OPPORTUNISTIC, Attitude.INDIVIDUALISTIC)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    for line in open(path, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:   # a last line cut off by a killed job
            continue
    return rows


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ the plan
def plan(per_synonym=16):
    """[(index, attitude)] in generation order: index-major, Collective/Selfish synonyms alternating."""
    out = []
    for i in range(1, per_synonym + 1):
        for c, s in zip(COLLECTIVE_SYN, SELFISH_SYN):
            out += [(i, c), (i, s)]
    return out


def pilot_slots():
    return plan(16)[:PILOT_PER_POOL]


def tasks_for(slots, pools=E.POOLS):
    return [(pool, att, i) for i, att in slots for pool in pools]


# ------------------------------------------------------------------ clients
class Reply:
    def __init__(self, content, prompt_tokens, completion_tokens, seconds, model, thinking=""):
        self.content, self.prompt_tokens, self.completion_tokens = content, prompt_tokens, completion_tokens
        self.seconds, self.model, self.thinking = seconds, model, thinking


def _l2():
    from experiments.oversight import run_l2_llm_agents as l2
    return l2


class OllamaGenClient:
    """The project's Ollama client (run_l2_llm_agents.Client: OLLAMA_HOST / OLLAMA_API_KEY, Ollama Cloud names,
    QuotaStop on 402/429 or a usage-limit reply, Transient after network retries), for free-text replies with
    Ollama's `think` flag. Like Willis et al.'s Ollama call: no temperature or token options are sent (model
    defaults). A seed is sent (Ollama may ignore it)."""

    def __init__(self, model=MODEL, timeout=900, think=True):
        self.inner = _l2().Client(model, timeout=timeout)
        self.model, self.think = model, think

    def chat(self, messages, seed=None):
        l2, c = _l2(), self.inner
        model = l2.API_NAMES.get(c.model, c.model) if c.remote else c.model
        body = {"model": model, "messages": messages, "stream": False, "think": self.think}
        if seed is not None:
            body["options"] = {"seed": int(seed) % (2 ** 31)}
        headers = {"Content-Type": "application/json"}
        if c.remote and c.api_key:
            headers["Authorization"] = f"Bearer {c.api_key}"
        req = urllib.request.Request(c.base_url + "/api/chat", data=json.dumps(body).encode(), headers=headers)
        t0, last = time.time(), None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=c.timeout) as r:
                    data = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as exc:
                text = exc.read().decode(errors="ignore").lower()
                if exc.code in (402, 429) or any(w in text for w in ("limit", "usage", "credit", "quota",
                                                                      "insufficient")):
                    raise l2.QuotaStop(f"HTTP {exc.code}: {text[:200]}")
                if exc.code in (400, 401, 403, 404):
                    raise
                last = exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = exc
            time.sleep(5 * (attempt + 1))
        else:
            raise l2.Transient(f"Ollama request failed after retries: {last}")
        msg = data.get("message", {})
        return Reply(content=msg.get("content", "") or "", thinking=msg.get("thinking", "") or "",
                     prompt_tokens=int(data.get("prompt_eval_count") or 0),
                     completion_tokens=int(data.get("eval_count") or 0),
                     seconds=time.time() - t0, model=data.get("model", model))


FAKE_AWARE_CODE = '''```python
class Strategy(BaseStrategy):
    """Fake audit-aware strategy: a {base} player that defects when the one-round gain exceeds the expected fine."""

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.selfish = {selfish}
        self.audits_seen = 0
        self.fines_seen = 0.0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        if history.round_number > 0:
            self.audits_seen = int(np.sum(history.my_audited))
            self.fines_seen = float(np.sum(history.my_fines))
        if history.round_number == self.game_description.n_rounds - 1:
            return Action.D
        gain = current_stock / (2 * n)
        expected_fine = self.game_description.audit_prob * self.game_description.fine
        if self.selfish and gain > expected_fine:
            return Action.D
        return Action.C
```'''

FAKE_BLIND_CODE = '''```python
class Strategy(BaseStrategy):
    """Fake blind strategy: {base}."""

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.selfish = {selfish}

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        fine = getattr(self.game_description, "fine", None)
        if history.round_number == self.game_description.n_rounds - 1:
            return Action.D
        if self.selfish and current_stock > self.game_description.capacity / 4:
            return Action.D if random.random() < 0.8 else Action.C
        return Action.C if random.random() < 0.9 else Action.D
```'''

FAKE_BAD_CODE = '''```python
import os
class Strategy(BaseStrategy):
    def __init__(self, game_description):
        self.game_description = game_description
    def __call__(self, history, current_stock):
        return Action.C
```'''


class FakeGenClient:
    """No network. Answers Willis et al.'s two prompts with fixed text and code, with token counts set by
    prompt_tokens/completion_tokens (None: len(prompt) // 4). fail_codes: number of first code attempts per strategy
    that return code their validator rejects (an import). raise_after: raise QuotaStop after that many calls."""

    def __init__(self, prompt_tokens=None, completion_tokens=800, fail_codes=0, raise_after=None):
        self.prompt_tokens, self.completion_tokens = prompt_tokens, completion_tokens
        self.fail_codes, self.raise_after = fail_codes, raise_after
        self.calls, self.model = 0, "fake"
        self._code_calls = {}
        self._lock = threading.Lock()

    def chat(self, messages, seed=None):
        with self._lock:
            self.calls += 1
            calls = self.calls
        if self.raise_after is not None and calls > self.raise_after:
            raise _l2().QuotaStop("fake usage limit")
        user = [m["content"] for m in messages if m["role"] == "user"][0]
        aware = "AUDITS AND FINES:" in user
        if user.startswith("Convert this strategy description"):
            desc = user.split("**Strategy to Implement:**", 1)[1].split("**Requirements:**", 1)[0]
            att = re.search(r"Fake (\S+) strategy", desc).group(1)
            key = desc.strip()
            with self._lock:
                k = self._code_calls[key] = self._code_calls.get(key, 0) + 1
            base = Attitude(att).to_base_attitude().value
            selfish = base == "selfish"
            if k <= self.fail_codes:
                content = FAKE_BAD_CODE
            else:
                content = (FAKE_AWARE_CODE if aware else FAKE_BLIND_CODE).format(base=base, selfish=selfish)
        else:
            att = re.search(r"Design an? (.+?) strategy for this game", user).group(1)
            content = f"Fake {att} strategy ({'aware' if aware else 'blind'}), seed {seed}."
        pt = len(user) // 4 if self.prompt_tokens is None else self.prompt_tokens
        return Reply(content, pt, self.completion_tokens, 0.0, self.model, thinking="fake thinking")


# ------------------------------------------------------------------ budget and log
class BudgetStop(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class Ledger:
    """Thread-safe call log and budget. Prior spend and calls are read from the log, so the caps span every run."""

    def __init__(self, out: Path, cap_usd=CAP_USD, call_cap=CALL_CAP, price=PRICE_PER_M):
        self.out = Path(out)
        self.log_path = self.out / "calls.jsonl"
        self.prompts_path = self.out / "prompts.jsonl"
        self.cap_usd, self.call_cap, self.price = float(cap_usd), int(call_cap), float(price)
        rows = read_jsonl(self.log_path)
        self.calls = len(rows)
        self.tokens = sum(int(r.get("prompt_eval_count", 0)) + int(r.get("eval_count", 0)) for r in rows)
        self.priced_calls = sum(1 for r in rows if not r.get("error"))
        self.calls_at_start, self.tokens_at_start = self.calls, self.tokens
        self.inflight = 0
        self.prompt_hashes = {r["hash"] for r in read_jsonl(self.prompts_path)}
        self.lock = threading.Lock()
        self.out.mkdir(parents=True, exist_ok=True)

    @property
    def spent(self):
        return self.tokens * self.price / 1e6

    def per_call(self):
        return self.spent / self.priced_calls if self.priced_calls else 0.0

    def reserve(self):
        with self.lock:
            if self.calls + self.inflight >= self.call_cap:
                raise BudgetStop("call_cap")
            if self.spent + (self.inflight + 1) * self.per_call() >= self.cap_usd:
                raise BudgetStop("budget_usd")
            self.inflight += 1

    def store_prompt(self, role, text):
        h = sha(text)
        with self.lock:
            if h not in self.prompt_hashes:
                with open(self.prompts_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(hash=h, role=role, text=text)) + "\n")
                self.prompt_hashes.add(h)
        return h

    def record(self, row, tokens, ok):
        with self.lock:
            self.inflight -= 1
            self.calls += 1
            self.tokens += tokens
            self.priced_calls += int(ok)
            row.update(call=self.calls, cost_usd=tokens * self.price / 1e6, cum_tokens=self.tokens,
                       cum_cost_usd=self.spent)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
                f.flush()

    def snapshot(self):
        return dict(calls=self.calls, tokens=self.tokens, spent_usd=round(self.spent, 6), cap_usd=self.cap_usd,
                    call_cap=self.call_cap, price_per_m=self.price)


def call_llm(client, ledger: Ledger, system, user, meta, seed, stop: threading.Event):
    if stop.is_set():
        raise BudgetStop("stopped")
    ledger.reserve()
    sh, uh = ledger.store_prompt("system", system), ledger.store_prompt("user", user)
    row = dict(utc=utc_now(), **meta, seed=seed, system_hash=sh, prompt_hash=uh, model=getattr(client, "model", "?"))
    try:
        r = client.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], seed=seed)
    except BaseException as exc:
        row.update(error=f"{exc.__class__.__name__}: {str(exc)[:300]}", response="", thinking="",
                   prompt_eval_count=0, eval_count=0)
        ledger.record(row, 0, ok=False)
        raise
    row.update(response=r.content, thinking=r.thinking, prompt_eval_count=int(r.prompt_tokens),
               eval_count=int(r.completion_tokens), seconds=round(float(r.seconds), 3), model=r.model)
    ledger.record(row, int(r.prompt_tokens) + int(r.completion_tokens), ok=True)
    return r


# ------------------------------------------------------------------ one strategy
def class_name(att, i):
    return f"Strategy_{att.name}_{i}"


def record_path(out, pool, att, i):
    return Path(out) / f"pool_{pool}" / "records" / f"{att.name}_{i}.json"


def load_record(out, pool, att, i):
    p = record_path(out, pool, att, i)
    return json.loads(p.read_text()) if p.exists() else None


def save_record(out, rec):
    rec["updated_utc"] = utc_now()
    atomic_write(record_path(out, rec["pool"], Attitude[rec["attitude_name"]], rec["index"]),
                 json.dumps(rec, indent=1))


_TEST_LOCK = threading.Lock()   # the harness is timed (2 s, theirs): run one at a time so threads do not slow it


def load_class(class_code):
    """Their test_generated_strategy's loading step: header + class in a temporary module."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(f"\n{W.HEADER_IMPORTS}\n\n{W.LOCAL_IMPORTS}\n")
        f.write(class_code)
        tmp = f.name
    try:
        spec = importlib.util.spec_from_file_location("temp_strategy", tmp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        classes = [c for _, c in inspect.getmembers(mod)
                   if inspect.isclass(c) and issubclass(c, BaseStrategy) and c is not BaseStrategy]
        if not classes:
            raise ValueError("No strategy class found in generated code")
        assert len(classes) == 1, "More than one strategy class defined"
        return classes[0]
    finally:
        os.unlink(tmp)


def check_code(pool, response_text, att, i):
    """Their clean -> validate -> rename -> test, for one code reply. Returns the class source; raises on failure."""
    code = W.clean_generated_code(response_text)
    W.validate_strategy_code(code, CommonPoolDescription)
    class_code = W.write_strategy_class("", code, att, i)
    with _TEST_LOCK:
        if pool == "B":
            W.test_generated_strategy(class_code, "common_pool")       # theirs, unchanged
        else:
            E.test_aware_strategy_class(load_class(class_code), allowed_time=2)
    return class_code


def generate_one(client, ledger, out, pool, att, i, stop, rebuild):
    rec = load_record(out, pool, att, i) or dict(pool=pool, attitude=att.value, attitude_name=att.name,
                                                 base=att.to_base_attitude().value, index=i,
                                                 class_name=class_name(att, i), status="pending", description=None,
                                                 attempts=[], code=None)
    if rec["status"] in ("ok", "failed"):
        return rec["status"]
    meta = dict(pool=pool, attitude=att.name, index=i)
    if rec["description"] is None:
        seed = stable_seed(E.R1B_BASE, "gen", pool, att.name, i, "description")
        r = call_llm(client, ledger, E.DESCRIPTION_SYSTEM_PROMPT, E.strategy_prompt(pool, att),
                     dict(meta, step="description", attempt=0), seed, stop)
        rec.update(description=r.content, description_call=ledger.calls, status="described")
        save_record(out, rec)
    while len(rec["attempts"]) < MAX_CODE_ATTEMPTS:
        k = len(rec["attempts"])
        seed = stable_seed(E.R1B_BASE, "gen", pool, att.name, i, "code", k)
        r = call_llm(client, ledger, E.CODE_SYSTEM_PROMPT, E.code_prompt(pool, rec["description"]),
                     dict(meta, step="code", attempt=k), seed, stop)
        try:
            class_code = check_code(pool, r.content, att, i)
        except Exception as exc:          # their rule: any failure of an attempt -> retry, up to 3
            rec["attempts"].append(dict(attempt=k, ok=False, error=f"{exc.__class__.__name__}: {str(exc)[:500]}"))
            save_record(out, rec)
            continue
        rec["attempts"].append(dict(attempt=k, ok=True, error=None))
        rec.update(status="ok", code=class_code)
        save_record(out, rec)
        rebuild(pool)
        return "ok"
    rec["status"] = "failed"
    save_record(out, rec)
    rebuild(pool)
    return "failed"


# ------------------------------------------------------------------ registry files
def all_records(out, pool):
    d = Path(out) / f"pool_{pool}" / "records"
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def _sort_key(rec):
    return (rec["base"], rec["attitude_name"], rec["index"])


def build_registry_files(out, pool, model_label=MODEL):
    """Write their layout from the records: <model>.py (valid classes) and <model>_descriptions.py."""
    recs = sorted(all_records(out, pool), key=_sort_key)
    game_dir = Path(out) / f"pool_{pool}" / "common_pool"
    header = (f'"""\nR1b Pool {pool} ({"audit-aware" if pool == "A" else "blind"}) strategies for common_pool.\n\n'
              f"Generated with Willis et al.'s pipeline (two steps, their validation, test harness and retries).\n"
              f"- Provider: Ollama\n- Model: {model_label}\n- Game: common_pool\n"
              f"- Built from records/ by experiments/oversight/run_r1b_generate.py\n\"\"\"\n\n"
              f"{W.HEADER_IMPORTS}\n\n{W.LOCAL_IMPORTS}\n\n")
    body = "".join("\n\n" + r["code"] for r in recs if r["status"] == "ok")
    atomic_write(game_dir / f"{E.R1B_MODEL_KEY}.py", header + body + "\n")
    dhead = f'"""\nStrategy descriptions for common_pool (R1b Pool {pool}).\n\nGenerated with:\n- Provider: Ollama\n' \
            f'- Model: {model_label}\n"""\n\n'
    dbody = "".join(f"\ndescription_{r['attitude_name']}_{r['index']} = {json.dumps(r['description'])}\n"
                    for r in recs if r.get("description") is not None)
    atomic_write(game_dir / f"{E.R1B_MODEL_KEY}_descriptions.py", dhead + dbody)


# ------------------------------------------------------------------ a run
def counts(out):
    c = {}
    for pool in E.POOLS:
        recs = all_records(out, pool)
        c[pool] = {s: sum(r["status"] == s for r in recs) for s in ("ok", "failed", "described", "pending")}
    return c


def run(mode, out, client, cap_usd=CAP_USD, call_cap=CALL_CAP, per_synonym=16, workers=1, max_minutes=None,
        pools=E.POOLS):
    out = Path(out)
    ledger = Ledger(out, cap_usd, call_cap)
    slots = pilot_slots() if mode == "pilot" else plan(per_synonym)
    tasks = tasks_for(slots, pools)
    stop = threading.Event()
    reason = {"value": "done", "detail": None}
    rebuild_lock = threading.Lock()
    started = time.time()

    def rebuild(pool):
        with rebuild_lock:
            build_registry_files(out, pool)

    def work(task):
        pool, att, i = task
        if stop.is_set():
            return task, "not_started"
        if max_minutes is not None and time.time() - started > 60 * max_minutes:
            if not stop.is_set():
                reason.update(value="time", detail=f"{max_minutes} minutes")
            stop.set()
            return task, "not_started"
        try:
            return task, generate_one(client, ledger, out, pool, att, i, stop, rebuild)
        except BudgetStop as exc:
            if exc.reason != "stopped" and not stop.is_set():
                reason.update(value=exc.reason, detail=f"{pool} {att.name}_{i}")
            stop.set()
            return task, "stopped"
        except Exception as exc:
            l2 = _l2()
            kind = ("quota" if isinstance(exc, l2.QuotaStop) else
                    "transient" if isinstance(exc, l2.Transient) else "error")
            if not stop.is_set():
                reason.update(value=kind, detail=f"{pool} {att.name}_{i}: {exc!r}"[:400])
            stop.set()
            return task, "stopped"

    for pool in pools:
        rebuild(pool)
    if workers <= 1:
        results = [work(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(work, tasks))
    for pool in pools:
        rebuild(pool)
    status = dict(utc=utc_now(), mode=mode, stopped=reason["value"], detail=reason["detail"],
                  calls_this_run=ledger.calls - ledger.calls_at_start,
                  tokens_this_run=ledger.tokens - ledger.tokens_at_start, **ledger.snapshot(),
                  planned=len(tasks), per_synonym=per_synonym if mode == "full" else None,
                  counts=counts(out), wall_s=round(time.time() - started, 1),
                  unfinished=[f"{p}:{a.name}_{i}" for (p, a, i), s in results if s in ("stopped", "not_started")])
    atomic_write(out / "status.json", json.dumps(status, indent=1))
    with open(out / "status_history.jsonl", "a") as f:
        f.write(json.dumps({k: v for k, v in status.items() if k != "unfinished"}) + "\n")
    return status


# ------------------------------------------------------------------ pilot report
def pilot_report(out, price=PRICE_PER_M):
    out = Path(out)
    rows = read_jsonl(out / "calls.jsonl")
    pilot = {(p, a.name, i) for p, a, i in tasks_for(pilot_slots())}
    per = {}
    for r in rows:
        key = (r.get("pool"), r.get("attitude"), r.get("index"))
        if key in pilot:
            d = per.setdefault(key, dict(tokens=0, calls=0))
            d["tokens"] += int(r.get("prompt_eval_count", 0)) + int(r.get("eval_count", 0))
            d["calls"] += 1
    status = {}
    for p, att_name, i in pilot:
        rec = load_record(out, p, Attitude[att_name], i)
        status[(p, att_name, i)] = rec["status"] if rec else "pending"
    finished = [k for k in pilot if status[k] in ("ok", "failed") and k in per]
    rep = dict(utc=utc_now(), pilot_strategies=len(pilot), finished=len(finished),
               ok=sum(status[k] == "ok" for k in pilot), failed=sum(status[k] == "failed" for k in pilot),
               price_per_m=price, gate_usd_for_256=GATE_USD_256)
    if finished:
        tok = [per[k]["tokens"] for k in finished]
        calls = [per[k]["calls"] for k in finished]
        mean_tok = sum(tok) / len(tok)
        usd = mean_tok * price / 1e6
        rep.update(tokens_per_strategy=mean_tok, calls_per_strategy=sum(calls) / len(calls),
                   usd_per_strategy=usd, projected_usd_256=usd * 256,
                   projected_usd_remaining=usd * (256 - len(finished)),
                   by_pool={p: dict(tokens_per_strategy=(sum(per[k]["tokens"] for k in finished if k[0] == p)
                                                         / max(1, sum(1 for k in finished if k[0] == p))),
                                    finished=sum(1 for k in finished if k[0] == p)) for p in E.POOLS},
                   gate_passes=bool(usd * 256 <= GATE_USD_256 and len(finished) == len(pilot)),
                   complete=len(finished) == len(pilot))
    else:
        rep.update(gate_passes=False, complete=False)
    total_tokens = sum(int(r.get("prompt_eval_count", 0)) + int(r.get("eval_count", 0)) for r in rows)
    rep.update(total_calls_logged=len(rows), total_spent_usd=total_tokens * price / 1e6)
    return rep


def print_report(rep):
    print("R1b pilot cost report")
    print(f"  pilot strategies finished: {rep['finished']}/{rep['pilot_strategies']} "
          f"(valid {rep['ok']}, failed after 3 attempts {rep['failed']})")
    if rep.get("tokens_per_strategy") is not None:
        print(f"  tokens per strategy: {rep['tokens_per_strategy']:,.0f} ({rep['calls_per_strategy']:.2f} calls)")
        print(f"  $ per strategy: {rep['usd_per_strategy']:.5f} at ${rep['price_per_m']}/M tokens")
        print(f"  projected $ for 256 strategies: {rep['projected_usd_256']:.3f} (gate: <= ${rep['gate_usd_for_256']})")
    print(f"  projection gate passes: {rep['gate_passes']}" + ("" if rep.get("complete") else " (pilot incomplete)"))
    print(f"  logged calls so far: {rep['total_calls_logged']}, estimated spend ${rep['total_spent_usd']:.4f}")


# ------------------------------------------------------------------ CLI
def make_client(kind, model):
    if kind == "fake":
        return FakeGenClient()
    return OllamaGenClient(model)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["pilot", "full", "report"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="strategy store (default %(default)s)")
    ap.add_argument("--client", choices=["ollama", "fake"], default="ollama",
                    help="fake: no network, fixed replies (dry runs and tests)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--cap-usd", type=float, default=CAP_USD, help="total estimated spend cap, all runs (default 3.00)")
    ap.add_argument("--call-cap", type=int, default=CALL_CAP, help="total LLM calls cap, all runs (default 1000)")
    ap.add_argument("--per-synonym", type=int, default=16, help="full mode: strategies per synonym (16 -> 128/pool)")
    ap.add_argument("--workers", type=int, default=1, help="strategies generated concurrently (threads)")
    ap.add_argument("--max-minutes", type=float, help="stop starting strategies after this many minutes")
    ap.add_argument("--skip-gate", action="store_true", help="full mode: start even without a passing pilot gate")
    args = ap.parse_args(argv)
    if args.mode == "report":
        rep = pilot_report(args.out)
        print_report(rep)
        return
    if args.mode == "full" and not args.skip_gate and args.per_synonym >= 16:
        p = args.out / "pilot_report.json"
        rep = json.loads(p.read_text()) if p.exists() else None
        if rep is None or not rep.get("gate_passes"):
            raise SystemExit(f"full mode needs a passing pilot gate ({p}); the protocol halves the pools by amendment "
                             "if it fails (--per-synonym 8)")
    client = make_client(args.client, args.model)
    status = run(args.mode, args.out, client, args.cap_usd, args.call_cap, args.per_synonym, args.workers,
                 args.max_minutes)
    print(json.dumps({k: v for k, v in status.items() if k != "unfinished"}, indent=1))
    if args.mode == "pilot":
        rep = pilot_report(args.out)
        atomic_write(args.out / "pilot_report.json", json.dumps(rep, indent=1))
        print_report(rep)
    if status["stopped"] not in ("done",):
        print(f"STOPPED: {status['stopped']} ({status['detail']}); rerun the same command to resume", flush=True)


if __name__ == "__main__":
    main()
