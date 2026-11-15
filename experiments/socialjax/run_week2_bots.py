"""Week 2 with scripted bots: choose the rule's threshold m, Schelling diagrams, checks x consequence.

Protocol (written before the runs): notes/socialjax/week2_bots/protocol.md

  .venv-socialjax/bin/python -m experiments.socialjax.run_week2_bots --out results/runs/socialjax_week2_bots
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from overseer.socialjax import HarvestOverseer, OverseerConfig, make_harvest_env
from overseer.socialjax.harvest_bots import kind_codes
from overseer.socialjax.rollout import run_seeds

N, STEPS = 7, 999
APPLE = 7.0  # reward per apple with individual rewards
M_VALUES = (1, 2, 3, 4, 5, 6)
K_VALUES = tuple(range(N + 1))
FREEZES = (0, 10, 25, 50, 100, 200)
PENALTIES = (0, 1, 2, 5, 10, 20)  # apples per catch
SCHELLING = {
    "none": dict(chooser="none"),
    "random k=2, freeze 50": dict(k=2, chooser="random", freeze_steps=50),
    "random k=2, penalty 5": dict(k=2, chooser="random", penalty=5 * APPLE),
}


def runs(env, kinds, seeds, m, **cfg):
    ov = HarvestOverseer(env, OverseerConfig(min_neighbours=m, **cfg))
    out = run_seeds(ov, seeds, kind_codes(kinds), STEPS, min_neighbours=m)
    jax.clear_caches()  # one compiled program per setting; without this, memory grows until the run is killed
    for r, seed in zip(out, seeds):
        r["seed"] = seed
        r["net_apples_per_agent"] = (np.array(r["eaten_per_agent"]) - np.array(r["penalty_per_agent"]) / APPLE).tolist()
    return out


def part_a(env, seeds):
    rows = []
    for m in M_VALUES:
        for r in runs(env, ("restrained",) * N, seeds, m, chooser="none"):
            rows.append(dict(m=m, **r))
    greedy = runs(env, ("greedy",) * N, seeds, 3, chooser="none")
    by_m = {m: [r for r in rows if r["m"] == m] for m in M_VALUES}
    ok = {m: (np.mean([r["apples_eaten"] for r in rs]), all(r["orchard_alive"] for r in rs)) for m, rs in by_m.items()}
    alive = [m for m, (_, a) in ok.items() if a]
    best = max(ok[m][0] for m in alive)
    chosen = min(m for m in alive if ok[m][0] >= 0.98 * best)
    return rows, greedy, chosen


def part_b(env, seeds, m):
    rows = []
    for name, cfg in SCHELLING.items():
        for l in range(N + 1):
            kinds = ("restrained",) * l + ("greedy",) * (N - l)
            for r in runs(env, kinds, seeds, m, **cfg):
                net = np.array(r["net_apples_per_agent"])
                rows.append(dict(condition=name, cooperators=l, seed=r["seed"],
                                 R_c=float(net[:l].mean()) if l > 0 else None,
                                 R_d=float(net[l:].mean()) if l < N else None,
                                 apples_eaten=r["apples_eaten"], orchard_alive=r["orchard_alive"]))
    return rows


def part_c1(env, seeds, m):
    pops = {"1 greedy + 6 restrained": ("greedy",) + ("restrained",) * 6, "7 sneaky": ("sneaky",) * N}
    rows = []
    for pop, kinds in pops.items():
        for chooser in ("random", "history"):
            for k in K_VALUES:
                print(f"  C1 {pop} / {chooser} / k={k}", flush=True)
                for freeze in FREEZES:
                    for r in runs(env, kinds, seeds, m, k=k, chooser=chooser if k else "none", freeze_steps=freeze):
                        rows.append(dict(population=pop, chooser=chooser, k=k, freeze=freeze, seed=r["seed"],
                                         apples_eaten=r["apples_eaten"], orchard_alive=r["orchard_alive"],
                                         violations=r["violations"], detected=r["detected"]))
    return rows


def part_c2(env, seeds, m):
    rows = []
    for k in K_VALUES:
        cfg = dict(k=k, chooser="random" if k else "none")
        cheat = runs(env, ("greedy",) + ("restrained",) * 6, seeds, m, **cfg)
        honest = runs(env, ("restrained",) * N, seeds, m, **cfg)
        rc7 = np.mean([np.mean(r["eaten_per_agent"]) for r in honest])
        for r in cheat:
            rows.append(dict(k=k, seed=r["seed"], greedy_eaten=r["eaten_per_agent"][0],
                             greedy_catches=r["caught_per_agent"][0], restrained_all_eaten=float(rc7),
                             restrained_same_group_eaten=float(np.mean(r["eaten_per_agent"][1:])),
                             orchard_alive=r["orchard_alive"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = make_harvest_env(num_inner_steps=STEPS + 1)
    t0 = time.time()
    path = out / "week2.json"
    d = json.loads(path.read_text()) if path.exists() else dict(steps=STEPS, penalties=PENALTIES)

    def save():  # after every part, so an interrupted run resumes where it stopped
        d["seconds"] = d.get("seconds", 0.0) + time.time() - t0
        path.write_text(json.dumps(d, default=float))

    if "part_a" not in d:
        d["part_a"], d["part_a_greedy"], d["chosen_m"] = part_a(env, range(10))
        save()
    m = d["chosen_m"]
    print(f"part A done: chosen m = {m}", flush=True)
    for name, fn, seeds in (("part_b", part_b, range(10)), ("part_c1", part_c1, range(5)),
                            ("part_c2", part_c2, range(10))):
        if name not in d:
            t0 = time.time()
            d[name] = fn(env, seeds, m)
            save()
        print(f"{name} done", flush=True)


if __name__ == "__main__":
    main()
