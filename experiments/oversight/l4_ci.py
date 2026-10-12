"""Remote runner for L4, L5 and L6 on GitHub Actions (one lane per job; three lanes = Ollama Pro's 3 concurrent requests).

Protocols: studies/L4_llm_identification, studies/L5_llm_families, studies/L6_llm_forest_spillover.
Called by scripts/l2_ci_step.sh with STAGE=l4 (the existing "L2 LLM agents (remote run)" workflow on main checks out this
branch and runs that script). Results live in the store (the l2-results branch): claude_l4_v1/, claude_l5_v1/,
claude_l6_v1/, claude_l6_pilot_v1/.

Each lane works through its units in order. A unit whose prerequisite is missing (a frozen prediction, the L6 fine, or L4
being complete) is skipped for now and retried later in the same job. The frozen files are written by whichever lane
first finds their inputs complete, and are committed and pushed at once, before any game that uses them:
  claude_l4_v1/prediction.json, claude_l5_v1/prediction.json   one entry per model, added when its stage 1 is complete
  claude_l6_v1/f_choice.json                                    one entry per model, from the L6 pilot gate
Entries are never overwritten. The job writes NEXT (now | wait | done) for the workflow's chaining step.

  python -m experiments.oversight.l4_ci run --store store --lane A --max-minutes 330
  python -m experiments.oversight.l4_ci check
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from experiments.oversight import analyze_l4 as A
from experiments.oversight import llm_clients as C
from experiments.oversight import run_l4_llm_identification as R
from experiments.oversight import run_l6_llm_forest as F

GPT, NEM, DS, GLM = "gpt-oss:120b-cloud", "nemotron-3-super:cloud", "deepseek-v4.1-flash:cloud", "glm-5.3:cloud"
LANES = {
    # GLM-5.3 dropped (L5 Amendment 2). Budget cut (12 Oct, L4/L5/L6 amendments): no Nemotron matched-e cells
    # ("thr" = stage 2's threshold and transfer cells only), no DeepSeek stage 2, no Nemotron L6.
    "A": [("s1", "L4", GPT, None), ("s1", "L5", DS, None), ("l6pilot", GPT), ("l6", GPT, None),
          ("s2", "L4", GPT, "0-6")],
    "B": [("s1", "L4", NEM, "0-4"), ("s2", "L4", NEM, "0-4", "thr"), ("s2", "L4", GPT, "7-13")],
    "C": [("s1", "L4", NEM, "5-9"), ("s2", "L4", NEM, "5-9", "thr"), ("s2", "L4", GPT, "14-19")],
}
DIRS = dict(L4="claude_l4_v1", L5="claude_l5_v1")
L6_DIR, L6_PILOT_DIR = "claude_l6_v1", "claude_l6_pilot_v1"
CODES = {"done": 0, "quota": 3, "budget": 4, "transient": 5, "time": 6}


def ctx_filter(rng, ctx):
    if not rng:
        return tuple(ctx)
    lo, hi = (int(x) for x in rng.split("-"))
    return tuple(c for c in ctx if lo <= c <= hi)


def expected(store, unit):
    """Episode files a unit must produce, or None if its prerequisite is missing."""
    kind = unit[0]
    if kind in ("s1", "s2"):
        study, model, rng = unit[1:4]
        thr = len(unit) > 4 and unit[4] == "thr"
        base = store / DIRS[study] / R.slug(model) / "episodes"
        if kind == "s1":
            groups = [(R.CTX10, R.stage1_specs(study))]
        else:
            pred = read(store / DIRS[study] / "prediction.json")
            if model not in pred.get("models", {}):
                return None
            if not pred["models"][model].get("run_stage2", True):
                return []
            groups = R.stage2_groups(study, pred["models"][model])
            if thr:
                groups = [g for g in groups if g[0] == R.CTX10]
        return [base / f"{c}__{k}.json" for ctx, specs in groups for k in ctx_filter(rng, ctx) for c in specs]
    if kind == "l6pilot":
        return [store / L6_PILOT_DIR / R.slug(unit[1]) / "episodes" / f"{F.PILOT_CELL}__{k}.json" for k in F.PILOT_CONTEXTS]
    _, model, rng = unit
    if unit[1] == NEM and not l4_complete(store):  # protocol: Nemotron's L6 runs after L4 is complete
        return None
    if model not in read(store / L6_DIR / "f_choice.json"):
        return None
    return [store / L6_DIR / R.slug(model) / "episodes" / f"{c}__{k}.json" for k in ctx_filter(rng, range(10))
            for c in F.CELLS]


def l4_complete(store):
    units = [u for lane in LANES.values() for u in lane if u[0] in ("s1", "s2") and u[1] == "L4"]
    for u in units:
        files = expected(store, u)
        if files is None or not all(f.exists() for f in files):
            return False
    return True


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save(store, msg):
    if os.environ.get("L4_CI_FAKE"):
        return
    subprocess.run(["bash", "scripts/l2_save.sh", str(store), msg], check=False)


def stage1_complete(store, study, model):
    files = [f for lane in LANES.values() for u in lane if u[:3] == ("s1", study, model)
             for f in expected(store, u)]
    return bool(files) and all(f.exists() for f in files)


def freeze_predictions(store):
    """Add a model's frozen prediction once its stage 1 is complete (never overwrite an entry)."""
    for study in ("L4", "L5"):
        path = store / DIRS[study] / "prediction.json"
        pred = read(path) or {"study": study, "rule": "F* = smallest fine on GRID with F/6 > g1; F- = the grid step below",
                              "grid": list(R.GRID), "models": {}}
        added = []
        for model in (R.L4_MODELS if study == "L4" else R.L5_MODELS):
            if model in pred["models"] or not stage1_complete(store, study, model):
                continue
            entry = A.predict(study, store / DIRS[study])["models"].get(model)
            if entry is not None:
                entry["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                pred["models"][model] = entry
                added.append(model)
        if added:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(pred, indent=1))
            print(f"froze {study} prediction for {added}", flush=True)
            save(store, f"{study} prediction frozen for {', '.join(added)}")


def l6_choice(store, model):
    out = store / L6_PILOT_DIR / R.slug(model)
    eps = [json.loads((out / "episodes" / f"{F.PILOT_CELL}__{k}.json").read_text()) for k in F.PILOT_CONTEXTS]
    rate = float(np.mean([F.overtake_rate(e) for e in eps]))
    path = store / L6_DIR / "f_choice.json"
    choice = read(path)
    if model not in choice:
        choice[model] = dict(pilot_overtake_rate=rate, fine=F.pilot_choice(rate),
                             rule="F=0 if <15%, 24 if >85%, else 12",
                             frozen_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(choice, indent=1))
        save(store, f"L6 fine fixed for {model}")


def client_for(model, forest=False):
    """L4_CI_FAKE=1 (tests only): deterministic fake clients instead of Ollama."""
    if os.environ.get("L4_CI_FAKE"):
        from fishery_sim import llm_actor as L
        return F.FakeForest(over=0.5) if forest else L.FakeClient(over=0.5)
    return C.make_client(model)


def run_unit(store, unit, minutes):
    kind = unit[0]
    if kind in ("s1", "s2"):
        study, model, rng = unit[1:4]
        return R.run_stage(client_for(model), study, model, 1 if kind == "s1" else 2,
                           store / DIRS[study] / R.slug(model), rng, minutes,
                           prediction_path=store / DIRS[study] / "prediction.json",
                           threshold_only=len(unit) > 4 and unit[4] == "thr")
    if kind == "l6pilot":
        model = unit[1]
        st = F.run(client_for(model, True), store / L6_PILOT_DIR / R.slug(model), (F.PILOT_CELL,), F.PILOT_CONTEXTS, model,
                   F.DEFAULT_F, max_minutes=minutes)
        if st == "done":
            l6_choice(store, model)
        return st
    _, model, rng = unit
    fine = read(store / L6_DIR / "f_choice.json")[model]["fine"]
    tag = f"ctx{rng}" if rng else None
    return F.run(client_for(model, True), store / L6_DIR / R.slug(model), tuple(F.CELLS), ctx_filter(rng, range(10)),
                 model, fine, tag=tag, max_minutes=minutes)


def lane(store, name, max_minutes):
    start = time.time()
    left = lambda: max_minutes - (time.time() - start) / 60
    pending = list(LANES[name])
    while pending:
        freeze_predictions(store)
        progressed = False
        for unit in list(pending):
            files = expected(store, unit)
            if files is None:
                continue  # prerequisite missing; try again later
            if all(f.exists() for f in files):
                pending.remove(unit)
                progressed = True
                continue
            if left() <= 40:
                return "now"
            st = run_unit(store, unit, left() - 35)
            freeze_predictions(store)
            if st == "done":
                pending.remove(unit)
                progressed = True
                print(f"lane {name}: {unit} complete", flush=True)
            elif st == "quota":
                return "wait"
            elif st in ("transient", "time"):
                return "now"
            else:
                print(f"lane {name}: {unit} stopped: {st}", flush=True)
                return "done"
        if not progressed:
            return "wait"  # everything left waits for another lane
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "check"))
    ap.add_argument("--store", default="store")
    ap.add_argument("--lane", default="A")
    ap.add_argument("--max-minutes", type=float, default=330)
    a = ap.parse_args()
    if a.mode == "check":
        bad = 0
        for m in (GPT, NEM, DS, GLM):
            try:
                r = C.make_client(m).chat([{"role": "user", "content": 'Reply with JSON only: {"ok": true}'}])
                print(f"OK   {m}: {r.content[:30]!r} ({r.prompt_tokens}+{r.completion_tokens} tokens)")
            except Exception as exc:  # never prints the key
                bad += 1
                print(f"FAIL {m}: {type(exc).__name__}: {str(exc)[:160]}")
        return 1 if bad else 0
    nxt = lane(Path(a.store), a.lane, a.max_minutes)
    Path("NEXT").write_text(nxt)
    print(f"lane {a.lane}: next = {nxt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
