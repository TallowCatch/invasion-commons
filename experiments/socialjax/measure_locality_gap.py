"""Task 3 of notes/socialjax/locality/protocol.md: does locality predict the decentralisation gap (D1)?

  .venv-socialjax/bin/python -m experiments.socialjax.measure_locality_gap --out results/locality/gap.json

Random checks, K checks per step in total, no penalty, a freeze on a catch (50 steps; 20 in Coins, as in the
existing sweeps), window w in {1, 10, 50} (Clean Up: 150, since its rule needs 100 steps of record). Per testbed
and w: no overseer, every agent checked every step (central), central random, and for each region layout
independent overseers with local records (kappa = 0) and with the existing full hand-over. One JSON, saved after
every condition, so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

from experiments.socialjax.measure_locality import STEPS, testbeds
from overseer.socialjax.decentralised import DecConfig
from overseer.socialjax.game_bots import kind_codes
from overseer.socialjax.game_rollout import run_seeds
from overseer.socialjax.local_records import LocalRecordOverseer

GAP = {  # testbed -> (layouts used, windows, freeze)
    "harvest": (("strips R=2", "strips R=4", "grid R=4"), (1, 10, 50), 50),
    "harvest_large": (("strips R=2", "strips R=4"), (1, 10, 50), 50),
    "rooms": (("rooms (R=2)", "strips R=4"), (1, 10, 50), 50),
    "rooms_door": (("rooms (R=2)",), (1, 10, 50), 50),
    "rooms_large": (("rooms (R=4)", "strips R=2"), (1, 10, 50), 50),
    "coins": (("bands R=2",), (1, 10, 50), 20),
    "cleanup": (("strips R=2", "bands R=2"), (150,), 50),
}


def populations(pops):
    """1 greedy + rest honest, all sneaky, all migrator (the last exploits regions on purpose)."""
    return {k: v for k, v in pops.items() if "honest" not in k or "greedy" in k}


def conditions(name):
    g = testbeds()[name]
    layouts = dict(g["layouts"])
    used, windows, freeze = GAP[name]
    for w in windows:
        base = dict(window=w, freeze_steps=freeze)
        yield f"w={w} none", None, "handover", dict(base, k=0, chooser="none")
        yield f"w={w} all", None, "handover", dict(base, k=g["K"], chooser="all")
        yield f"w={w} central", None, "handover", dict(base, k=g["K"], chooser="random")
        for label in used:
            for rec in ("local", "handover"):
                yield f"w={w} {label} {rec}", layouts[label], rec, dict(base, k=g["K"], chooser="random")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/locality/gap.json")
    ap.add_argument("--testbeds", default=",".join(GAP))
    ap.add_argument("--seeds", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(out.read_text()) if out.exists() else dict(rows=[], done=[])
    tb = testbeds()
    for name in a.testbeds.split(","):
        g = tb[name]
        rules = g["rules"]()
        policy = g["policy"](rules)
        seeds = range(a.seeds or g["seeds"])
        for cname, make_grid, rec, kw in conditions(name):
            tag = f"{name} | {cname}"
            if tag in d["done"]:
                continue
            t0 = time.time()
            cfg = DecConfig(**kw)
            rgrid = None if make_grid is None else np.asarray(make_grid(rules))
            ov = LocalRecordOverseer(rules, cfg, records=rec, region_grid=rgrid)
            for pop, kinds in populations(g["pops"]).items():
                for r in run_seeds(ov, seeds, kind_codes(kinds), policy, STEPS):
                    r.pop("regions", None)
                    d["rows"].append(dict(testbed=name, condition=cname, window=kw["window"],
                                          layout=None if make_grid is None else cname.split(" ", 1)[1].rsplit(" ", 1)[0],
                                          records=rec, population=pop, config=asdict(ov.cfg), **r))
            jax.clear_caches()
            d["done"].append(tag)
            out.write_text(json.dumps(d, default=float))
            print(f"{tag} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
