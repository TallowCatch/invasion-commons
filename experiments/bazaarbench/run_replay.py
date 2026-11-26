"""Run every replay in notes/bazaarbench/offline_replay/protocol.md and save the numbers.

Usage: python -m experiments.bazaarbench.run_replay
Writes notes/bazaarbench/offline_replay/results.json (summary numbers only, no raw data).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bazaarbench import bb_data as bd  # noqa: E402
from experiments.bazaarbench import replay as rp  # noqa: E402

RAW = ROOT / "data" / "bazaarbench" / "raw"
OUT = ROOT / "notes" / "bazaarbench" / "offline_replay" / "results.json"
CELLS = {"L1": "gpt55_L1_gptoss120b", "L3": "gpt55_L3_gptoss120b"}
KS = [1, 2, 5, 10, 20]
CHOOSERS = ["random", "repeat", "active"]
N_RUNS = {"random": 1000, "repeat": 200, "active": 200}
N_RUNS_DEC = 200


def main() -> None:
    t0 = time.time()
    cells = {lab: bd.load_cell(RAW / name) for lab, name in CELLS.items()}
    arrays = {lab: rp.CellArrays.from_cell(c) for lab, c in cells.items()}
    unit = float(np.mean(np.concatenate([a.cost["full"].ravel() for a in arrays.values()])))
    res: dict = {"unit_tokens": unit, "cells": {}, "count": [], "tokens": [], "decentral": []}

    for lab, ca in arrays.items():
        cell = cells[lab]
        res["cells"][lab] = {
            "cell_id": cell.cell_id, "n_objects": int(len(ca.stage)),
            "n_past_s1": int((ca.stage >= 2).sum()), "n_s4": int((ca.day_s4 >= 0).sum()),
            "n_s5": int((ca.day_s5 >= 0).sum()),
            "tokens_all_per_day_full": float(ca.cost["full"].sum() / ca.D),
            "tokens_all_per_day_public": float(ca.cost["public"].sum() / ca.D),
            "tick_level_top_before_s5": cell_top(cells[lab]),
            "judge_input_tokens_treated": int(sum(r["judge_input_tokens"] for r in cell.agent_day_cost().values())),
            "zips": ca.zips,
            "region_groups": [list(map(int, ca.agents[g])) for g in rp.region_groups(ca.zips)],
        }

    # A. count budgets
    for lab, ca in arrays.items():
        for view in rp.VIEWS:
            for chooser in CHOOSERS:
                for k in KS:
                    s = rp.run_many(ca, N_RUNS[chooser], seed=k, view=view, chooser=chooser,
                                    budget_kind="count", budget=k)
                    res["count"].append({"cell": lab, "view": view, "chooser": chooser, "k": k, **s})
        print(f"[{time.time() - t0:.0f}s] count sweep {lab} done", flush=True)

    # A2 (post hoc, added after the first results): repeat offenders scored on acts only
    for lab, ca in arrays.items():
        for k in KS:
            s = rp.run_many(ca, N_RUNS["repeat"], seed=k, view="full", chooser="repeat",
                            budget_kind="count", budget=k, score_on="acts")
            res["count"].append({"cell": lab, "view": "full", "chooser": "repeat (acts only)", "k": k, **s})
        for units in (1, 10):
            s = rp.run_many(ca, N_RUNS["repeat"], seed=units, view="full", chooser="repeat",
                            budget_kind="tokens", budget=units * unit, score_on="acts")
            res["tokens"].append({"cell": lab, "units": units, "chooser": "repeat (acts only)", "view": "full", **s})
            s = rp.run_many(ca, N_RUNS["repeat"], seed=units, view="full", chooser="cheapest",
                            budget_kind="tokens", budget=units * unit)
            res["tokens"].append({"cell": lab, "units": units, "chooser": "cheapest (post hoc)", "view": "full", **s})

    # B. token budgets (1 and 10 units), plus record-on-everyone combined with the full judge
    for lab, ca in arrays.items():
        for units in (1, 10):
            for chooser in CHOOSERS:
                for view, rec in (("full", False), ("public", False), ("full", True)):
                    s = rp.run_many(ca, N_RUNS[chooser], seed=units, view=view, record_all=rec,
                                    chooser=chooser, budget_kind="tokens", budget=units * unit)
                    res["tokens"].append({"cell": lab, "units": units, "chooser": chooser,
                                          "view": "full+record" if rec else view, **s})
        print(f"[{time.time() - t0:.0f}s] token budgets {lab} done", flush=True)

    # C. decentralised overseers (token budgets)
    for lab, ca in arrays.items():
        groupings = {f"random G={G}": dict(group_fn=rp.random_groups(ca.n, G)) for G in (2, 4, 5)}
        groupings["region G=4"] = dict(groups=rp.region_groups(ca.zips))
        for units in (1, 10):
            B = units * unit
            for chooser in ("repeat", "random"):
                for mname, mkw in (("central", {}), ("central + savings", dict(savings=True))):
                    base = rp.run_many(ca, N_RUNS_DEC, seed=7, chooser=chooser, budget_kind="tokens",
                                       budget=B, **mkw)
                    res["decentral"].append({"cell": lab, "units": units, "chooser": chooser,
                                             "grouping": "none (G=1)", "mode": mname, **base})
                for gname, gkw in groupings.items():
                    variants = [("independent", dict(mode="independent")),
                                ("independent + savings", dict(mode="independent", savings=True))]
                    if chooser == "repeat":
                        variants += [("shared, 0 tokens", dict(mode="shared", comm_cost=0.0)),
                                     ("shared, 500 tokens", dict(mode="shared", comm_cost=500.0)),
                                     ("shared, 5,000 tokens", dict(mode="shared", comm_cost=5000.0))]
                    for mname, mkw in variants:
                        s = rp.run_many(ca, N_RUNS_DEC, seed=7, chooser=chooser, budget_kind="tokens",
                                        budget=B, **gkw, **mkw)
                        res["decentral"].append({"cell": lab, "units": units, "chooser": chooser,
                                                 "grouping": gname, "mode": mname, **s})
        # count budgets, independent vs central, repeat offenders
        for k in (2, 5, 10):
            base = rp.run_many(ca, N_RUNS_DEC, seed=11, chooser="repeat", budget_kind="count", budget=k)
            res["decentral"].append({"cell": lab, "k": k, "chooser": "repeat", "grouping": "none (G=1)",
                                     "mode": "central", **base})
            for gname, gkw in groupings.items():
                s = rp.run_many(ca, N_RUNS_DEC, seed=11, chooser="repeat", budget_kind="count", budget=k,
                                mode="independent", **gkw)
                res["decentral"].append({"cell": lab, "k": k, "chooser": "repeat", "grouping": gname,
                                         "mode": "independent", **s})
        print(f"[{time.time() - t0:.0f}s] decentralised {lab} done", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")


def cell_top(cell: bd.Cell) -> dict:
    t = rp.top_before_harm_ticks(cell)
    return {"full": list(t["judge"]), "public": list(t["no_reasoning"])}


if __name__ == "__main__":
    main()
