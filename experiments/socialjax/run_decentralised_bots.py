"""Scripted-bot sweeps for central vs decentralised overseers in Harvest, Harvest (large map), Coins and Clean Up.

Protocols (written before the runs): notes/socialjax/<game>/protocol.md

  .venv-socialjax/bin/python -m experiments.socialjax.run_decentralised_bots --game harvest \
      --out results/runs/socialjax_decentralised
  (--game harvest | harvest_large | coins | cleanup; --seeds N; --list prints the conditions and stops)

One JSON per game (<out>/<game>.json), saved after every condition, so an interrupted run resumes where it stopped.
Every condition at a given k has the same total budget: k checks per step, summed over regions.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import jax

from overseer.socialjax.decentralised import DecConfig, DecentralisedOverseer
from overseer.socialjax.game_bots import kind_codes, make_policy
from overseer.socialjax.game_rollout import run_seeds
from overseer.socialjax.games import make_rules

STEPS = 999
SHARE = dict(share_every=1, cost_per_message=0.25)  # one message carries a score for every agent

GAMES = {
    "harvest": dict(
        ks=(1, 2, 4), regions=(2, 4), layout="strips", freeze=50, window=1, seeds=10,
        populations={"7 honest": ("honest",) * 7, "1 greedy + 6 honest": ("greedy",) + ("honest",) * 6,
                     "7 sneaky": ("sneaky",) * 7, "7 migrator": ("migrator",) * 7}),
    "harvest_large": dict(
        ks=(2, 4), regions=(2, 4), layout="strips", freeze=50, window=1, seeds=5,
        populations={"14 honest": ("honest",) * 14, "2 greedy + 12 honest": ("greedy",) * 2 + ("honest",) * 12,
                     "14 sneaky": ("sneaky",) * 14, "14 migrator": ("migrator",) * 14}),
    "coins": dict(
        ks=(1, 2), regions=(2,), layout="rows", freeze=20, window=1, seeds=10,
        populations={"2 honest": ("honest",) * 2, "1 greedy + 1 honest": ("greedy", "honest"),
                     "2 greedy": ("greedy",) * 2, "2 sneaky": ("sneaky",) * 2, "2 migrator": ("migrator",) * 2}),
    "cleanup": dict(
        ks=(1, 2), regions=(2,), layout="strips", freeze=50, window=150, seeds=10, cost_per_step_seen=0.01,
        windows=(1, 50, 100, 200), presume_windows=(10, 50, 100),
        populations={"7 honest": ("honest",) * 7, "1 greedy + 6 honest": ("greedy",) + ("honest",) * 6,
                     "7 sneaky": ("sneaky",) * 7, "7 migrator": ("migrator",) * 7}),
}


def conditions(game):
    """(name, DecConfig kwargs) for every overseer setting of a game's sweep."""
    g = GAMES[game]
    base = dict(window=g["window"], freeze_steps=g["freeze"], layout=g["layout"],
                cost_per_step_seen=g.get("cost_per_step_seen", 0.0))
    out = [("none", dict(base, chooser="none", k=0))]
    for k in g["ks"]:
        out.append((f"central random k={k}", dict(base, k=k, chooser="random")))
        out.append((f"central history k={k}", dict(base, k=k, chooser="history")))
        for R in g["regions"]:
            out.append((f"R={R} random k={k}", dict(base, k=k, chooser="random", regions=R)))
            out.append((f"R={R} history k={k}", dict(base, k=k, chooser="history", regions=R)))
            out.append((f"R={R} history+share k={k}", dict(base, k=k, chooser="history", regions=R, share=True,
                                                            **SHARE)))
            out.append((f"R={R} history+share+coverage k={k}",
                        dict(base, k=k, chooser="history", regions=R, share=True, share_coverage=True, **SHARE)))
    for w in g.get("windows", ()):  # how long a look has to be (Clean Up): central random checks, k = 2
        out.append((f"central random k=2 w={w}", dict(base, k=2, chooser="random", window=w)))
    for w in g.get("presume_windows", ()):
        out.append((f"central random k=2 w={w} presume", dict(base, k=2, chooser="random", window=w, presume=True)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, choices=sorted(GAMES))
    ap.add_argument("--out", default="results/runs/socialjax_decentralised")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    g = GAMES[a.game]
    conds = conditions(a.game)
    if a.list:
        for name, cfg in conds:
            print(name, cfg)
        print(f"{len(conds)} conditions x {len(g['populations'])} populations")
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.game}.json"
    d = json.loads(path.read_text()) if path.exists() else dict(game=a.game, steps=STEPS, rows=[], done=[],
                                                                 seconds=0.0)
    seeds = range(a.seeds or g["seeds"])
    rules = make_rules(a.game, num_inner_steps=STEPS + 1)
    policy = make_policy(rules)
    d["n_agents"], d["reward_unit"], d["lookback"] = rules.env.num_agents, rules.reward_unit, rules.lookback
    for name, kw in conds:
        if name in d["done"]:
            continue
        t0 = time.time()
        cfg = DecConfig(**kw)
        ov = DecentralisedOverseer(rules, cfg)
        for pop, kinds in g["populations"].items():
            for r in run_seeds(ov, seeds, kind_codes(kinds), policy, STEPS):
                d["rows"].append(dict(condition=name, population=pop, config=asdict(cfg), **r))
        jax.clear_caches()  # one compiled program per setting; without this, memory grows until the run is killed
        d["done"].append(name)
        d["seconds"] += time.time() - t0
        path.write_text(json.dumps(d, default=float))
        print(f"{a.game}: {name} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
