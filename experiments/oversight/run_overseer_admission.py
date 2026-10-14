"""Admission test B for the overseer package (overseer/ADMISSION.md): S3 Parts A and B re-run through the overseer.

The programmed agents, populations, seeds, search and summaries are S3's (run_s3_threshold_timing_memory.py, Parts A
and B, which reuse the S2 non-compliance model). Only the reviewer is replaced: the allowances (SharedCut, MSY target),
Bernoulli audits (UniformRandom with S3's seed key, which includes q), detection (Perfect, or Noisy with miss = 1 - s
and S3's detection seeds) and fines (Fine) come from `overseer`. The output files have S3's format, so the stored
analysis (analyze_s3.parts_ab) can be run on them unchanged.

Run:  PYTHONPATH=. python3 -m experiments.oversight.run_overseer_admission --part A --profile full --out DIR
      PYTHONPATH=. python3 -m experiments.oversight.run_overseer_admission --compare DIR_A DIR_B
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.oversight import run_s2_compliance_deterrence as s2
from experiments.oversight import run_s3_threshold_timing_memory as s3
from experiments.oversight.claude_oversight_common import fishery_requests, fishery_setup, write_jsonl_gz
from fishery_sim.calibrated_oversight import fishery_reference
from fishery_sim.fishery_oversight import safe as fishery_safe
from overseer import Ledger, Overseer
from overseer.adapters.fishery_text import FisheryText
from overseer.allowance import SharedCut
from overseer.audit import UniformRandom
from overseer.detect import Noisy, Perfect
from overseer.sanction import Fine

STORED = dict(A="results/runs/claude_s3_partA_v1", B="results/runs/claude_s3_partB_v1")
SUMMARY = "notes/claude_audit_20261005/runs/claude_s3_partA_v1/s3_summary.json"


def make_overseer(game, n, protocol, fine, context, seeds, q, s):
    """S3's Auditor ('allow' or 'bern') as an overseer."""
    audit = detector = None
    if protocol == "bern":
        # S3 key: stable_seed(audit, context, t, round(q, 6)); detection: stable_seed(detect, context, t) < s
        audit = UniformRandom(q, seeds["audit"], prefix=(context,), suffix=(round(q, 6),))
        detector = Perfect(tol=1e-12) if s == 1.0 else Noisy(1 - s, 0.0, seeds["detect"], prefix=(context,), tol=1e-12)
    elif protocol != "allow":
        raise ValueError(f"Admission runner covers 'allow' and 'bern' only, not {protocol}")
    return Overseer(n, game.resource, allowance=SharedCut(), audit=audit, detector=detector, sanctions=(Fine(fine),))


def fishery_episode(context, protocol, d, fine, P, seeds, train=False, q=None, s=1.0):
    """s3.fishery_episode (timing 'always'), with the overseer as the reviewer."""
    cfg, pol, cheat = fishery_setup(context, seeds["train_fishery_pop"] if train else seeds["fishery_pop"], P["horizon"])
    game = FisheryText(cfg, s3.FISHERY_TARGET)
    ov = make_overseer(game, cfg.n_agents, protocol, fine, context, seeds, q, s)
    honest = np.array([i not in cheat for i in range(cfg.n_agents)])
    over_steps, caught_honest = np.zeros(cfg.n_agents, int), 0
    state, pay, stocks, unsafe, rows = game.initial_state(), np.zeros(cfg.n_agents), [], [], []
    for t in range(cfg.horizon):
        req = fishery_requests(pol, state.stock)
        row = dict(step=t, pre_safe=int(fishery_safe(cfg, state)))
        obs = game.observe(t, state, req)
        allowance = ov.begin_window(obs).caps
        taken = allowance.copy()
        for i in cheat:  # the S2 cheater: takes a share d of the cut anyway
            taken[i] = allowance[i] + d * (req[i] - allowance[i])
        over_steps += taken > allowance + 1e-12
        out = ov.end_window(obs, Ledger(taken, allowance, req))
        caught_honest += int((out.caught & honest).sum())
        if not train:
            row["label_exec"] = fishery_reference(cfg, state.stock, taken, s3.FISHERY_TARGET)["label"]
        rows.append(row)
        state, payoffs = game.step(state, taken)
        pay += payoffs
        stocks.append(state.stock)
        unsafe.append(int(not fishery_safe(cfg, state)))
        if state.collapsed:
            break
    au = SimpleNamespace(fines=ov.fines, audits=ov.budget.audits, messages=0, catches=int(ov.catches.sum()),
                         caught_honest=caught_honest)
    e = s2.summarize("fishery", context, protocol, d, fine, cfg.horizon, pay, cheat, float(np.mean(stocks)),
                     unsafe, rows, au)
    e.update(q=q, s=s, timing="always", cheater_over_steps=float(over_steps[cheat].mean()))
    return e


def run_nc(protocol, d, fine, P, contexts, seeds, train=False, **kw):
    return [fishery_episode(c, protocol, d, fine, P, seeds, train, **kw) for c in contexts]


def search_nc(protocol, fine, P, seeds, **kw):
    """s3.search_nc with timing 'always': ties go to the smaller d."""
    best, best_score, scores = None, -np.inf, {}
    for d in s3.SEARCH_D:
        sc = float(np.mean([e["cheater_payoff"] for e in run_nc(protocol, d, fine, P, range(P["train_contexts"]),
                                                                   seeds, True, **kw)]))
        scores[f"always:{d}"] = sc
        if best is None or sc > best_score + 1e-9:
            best, best_score = d, sc
    return best, scores


def part_ab(part, P, seeds=s3.SEEDS):
    """s3.part_ab, same order of cells and episodes."""
    episodes, searches = [], []
    test = range(P["contexts"])
    for e in run_nc("allow", 0.0, 0.0, P, test, seeds):
        episodes.append({**e, "actor": "comply"})
    d_allow, sc = search_nc("allow", 0.0, P, seeds)
    searches.append(dict(protocol="allow", q=None, s=1.0, e=None, fine=0.0, d_star=d_allow, train_scores=sc))
    for e in run_nc("allow", d_allow, 0.0, P, test, seeds):
        episodes.append({**e, "actor": "adaptive"})
    combos = [(q, 1.0) for q in s3.PART_A_Q] if part == "A" else [(1 / 6, s) for s in s3.PART_B_S]
    for q, s in combos:
        for ev in s3.E_GRID:
            fine = ev / (q * s)
            d, sc = search_nc("bern", fine, P, seeds, q=q, s=s)
            searches.append(dict(protocol="bern", q=q, s=s, e=ev, fine=fine, d_star=d, train_scores=sc))
            for e in run_nc("bern", d, fine, P, test, seeds, q=q, s=s):
                episodes.append({**e, "actor": "adaptive", "e": ev})
    return episodes, searches


def write(out, episodes, searches, part, profile, seconds):
    out = Path(out)
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    write_jsonl_gz(out / "episodes.jsonl.gz", episodes)
    (out / "adaptive_search.json").write_text(json.dumps(searches, indent=1))
    (out / "manifest.json").write_text(json.dumps(dict(experiment=f"overseer-admission-S3-{part}", profile=profile,
                                                       seconds=seconds, n_episodes=len(episodes)), indent=1))


# ------------------------------------------------------------------ comparison with the stored runs
def load_episodes(path):
    return [json.loads(x) for x in gzip.open(Path(path) / "episodes.jsonl.gz", "rt")]


def _diff(a, b, path=""):
    """Paths where two JSON-like values differ (floats compared exactly)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys {sorted(set(a) ^ set(b))}"]
        return [p for k in a for p in _diff(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in _diff(x, y, f"{path}[{i}]")]
    return [] if a == b or (a != a and b != b) else [f"{path}: {a!r} vs {b!r}"]


def compare(new_a, new_b, stored=STORED, summary=SUMMARY):
    """Returns a dict of mismatch lists (all empty = admitted) and counts."""
    from experiments.oversight import analyze_s3
    res = {}
    for part, new in (("A", new_a), ("B", new_b)):
        old_e, new_e = load_episodes(stored[part]), load_episodes(new)
        res[f"{part}_episodes"] = _diff(new_e, old_e)
        res[f"{part}_n_episodes"] = (len(new_e), len(old_e))
        res[f"{part}_search"] = _diff(json.loads((Path(new) / "adaptive_search.json").read_text()),
                                      json.loads((Path(stored[part]) / "adaptive_search.json").read_text()))
    ab = analyze_s3.parts_ab(Path(new_a), Path(new_b), np.random.default_rng(analyze_s3.SEED))
    res["summary_AB"] = _diff(json.loads(json.dumps(ab, default=float)), json.loads(Path(summary).read_text())["AB"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B"])
    ap.add_argument("--profile", default="full")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"))
    a = ap.parse_args()
    if a.compare:
        res = compare(*a.compare)
        print(json.dumps({k: (v if not isinstance(v, list) else dict(n=len(v), first=v[:5])) for k, v in res.items()},
                         indent=1))
        return
    P = s2.profile(a.profile)
    t0 = time.time()
    episodes, searches = part_ab(a.part, P)
    write(a.out, episodes, searches, a.part, a.profile, time.time() - t0)
    print(json.dumps(dict(n_episodes=len(episodes), seconds=time.time() - t0)))


if __name__ == "__main__":
    main()
