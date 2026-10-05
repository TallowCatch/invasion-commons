"""Experiment S1b: Part A (audit ablation) and Part B (S1 Fishery under the MSY target).

Protocol: notes/claude_audit_20261005/11_PROTOCOL_S1B_S2_ABLATION_AND_DETERRENCE.md
Run:  PYTHONPATH=. python -m experiments.run_s1b_ablation_msy --profile smoke|full --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.claude_oversight_common import sha256, write_jsonl_gz
from experiments.run_s1_reporting_audit import SEARCH_D, profile, run_condition

ABLATION = {"harvest": ("rand2", "targ1", "peer"), "fishery": ("rand2", "peer")}
BELIEF = (True, False)
SANCTIONS = ("excl+fine", "fine", "none")
PART_B_PROTOCOLS = ("report", "rand1", "rand2", "peer", "peer_collude")


def search(game, proto, P, **opts):
    scores = {}
    for d in SEARCH_D:
        tr = run_condition(game, proto, d, P, range(P["train_contexts"]), train=True, **opts)
        scores[d] = float(np.mean([e["misreporter_payoff"] for e in tr]))
    best = SEARCH_D[0]
    for d in SEARCH_D[1:]:
        if scores[d] > scores[best] + 1e-9:
            best = d
    return best, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = profile(a.profile)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0, episodes, searches = time.time(), [], []
    test = range(P["contexts"])
    # ---------------- Part A: ablation
    for game, protos in ABLATION.items():
        for proto in protos:
            for belief in BELIEF:
                for sanction in SANCTIONS:
                    opts = dict(belief=belief, sanction=sanction)
                    tag = dict(part="A", belief=belief, sanction=sanction, target="one_step")
                    for d in (0.0, 0.5):
                        for e in run_condition(game, proto, d, P, test, **opts):
                            episodes.append({**e, **tag, "actor": "fixed"})
                    best, scores = search(game, proto, P, **opts)
                    searches.append(dict(game=game, protocol=proto, **tag, d_star=best, train_scores=scores))
                    for e in run_condition(game, proto, best, P, test, **opts):
                        episodes.append({**e, **tag, "actor": "adaptive"})
                    print(f"A {game} {proto} belief={belief} {sanction} d*={best} {time.time()-t0:.0f}s", flush=True)
    # ---------------- Part B: Fishery, MSY target
    opts = dict(target="msy")
    tag = dict(part="B", belief=True, sanction="excl+fine", target="msy")
    for proto in ("none", "full"):
        for e in run_condition("fishery", proto, 0.0, P, test, **opts):
            episodes.append({**e, **tag, "actor": "fixed"})
    for proto in PART_B_PROTOCOLS:
        for d in (0.0, 0.25, 0.5):
            for e in run_condition("fishery", proto, d, P, test, **opts):
                episodes.append({**e, **tag, "actor": "fixed"})
        best, scores = search("fishery", proto, P, **opts)
        searches.append(dict(game="fishery", protocol=proto, **tag, d_star=best, train_scores=scores))
        for e in run_condition("fishery", proto, best, P, test, **opts):
            episodes.append({**e, **tag, "actor": "adaptive"})
        print(f"B fishery {proto} d*={best} {time.time()-t0:.0f}s", flush=True)
    write_jsonl_gz(out / "episodes.jsonl.gz", episodes)
    (out / "adaptive_search.json").write_text(json.dumps(searches, indent=1))
    manifest = dict(experiment="S1b", profile=a.profile, params=P, seconds=time.time() - t0, n_episodes=len(episodes),
                    files={"episodes.jsonl.gz": sha256(out / "episodes.jsonl.gz"),
                           "adaptive_search.json": sha256(out / "adaptive_search.json")},
                    source={p: sha256(Path(p)) for p in ["fishery_sim/calibrated_oversight.py",
                            "experiments/claude_oversight_common.py", "experiments/run_s1_reporting_audit.py",
                            "experiments/run_s1b_ablation_msy.py"]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(dict(n_episodes=len(episodes), seconds=manifest["seconds"])))


if __name__ == "__main__":
    main()
