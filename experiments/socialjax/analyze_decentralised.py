"""Tables (markdown) and a summary JSON for the decentralised-overseer bot sweeps.

  .venv-socialjax/bin/python -m experiments.socialjax.analyze_decentralised --run results/runs/socialjax_decentralised \
      --game harvest [--print]

Writes <run>/<game>_summary.json: one row per (population, condition) with means over seeds.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def cell_summary(rows, n, unit_k):
    """Means over seeds for one (population, condition)."""
    cfg = rows[0]["config"]
    k = cfg["k"]
    viol = sum(r["violations"] for r in rows)
    det = sum(r["detected"] for r in rows)
    checks = np.mean([r["checks"] for r in rows])
    unused = np.mean([sum(r["regions"]["unused_checks"]) for r in rows])
    budget = checks + unused
    reward = np.array([r["reward_per_agent"] for r in rows])  # (seeds, N), in goods
    caught = np.array([r["caught_per_agent"] for r in rows])
    viol_pa = np.array([r["violations_per_agent"] for r in rows])
    p_catch = n / k if k else 0.0  # penalty per catch (goods) for an average penalty of 1 good per violation
    net = reward - caught * p_catch
    out = dict(
        seeds=len(rows), k=k, chooser=cfg["chooser"], regions=cfg["regions"], share=cfg["share"],
        coverage=cfg.get("share_coverage", False), window=cfg["window"], presume=cfg["presume"],
        taken=float(np.mean([r["taken"] for r in rows])),
        welfare=float(np.mean([r["welfare"] for r in rows])),
        alive=int(sum(r["alive"] for r in rows)),
        violations=float(viol / len(rows)),
        detected=float(det / len(rows)),
        caught_share=float(det / viol) if viol else float("nan"),
        checks=float(checks),
        unused_share=float(unused / budget) if budget else 0.0,
        cost=float(np.mean([r["cost"] for r in rows])),
        comm_cost=float(np.mean([r["comm_cost"] for r in rows])),
        wrongly_stopped=float(np.mean([r["wrongly_stopped"] for r in rows])),
        catches=float(np.mean([r["catches"] for r in rows])),
        reward_per_agent=reward.mean(0).tolist(),
        net_c1_per_agent=net.mean(0).tolist(),
        violations_per_agent=viol_pa.mean(0).tolist(),
        resource_mean=float(np.mean([r["resource_mean"] for r in rows])),
        region_violations=np.mean([r["regions"]["violations"] for r in rows], 0).tolist(),
        region_detected=np.mean([r["regions"]["detected"] for r in rows], 0).tolist(),
        region_checks=np.mean([r["regions"]["checks"] for r in rows], 0).tolist(),
        region_unused=np.mean([r["regions"]["unused_checks"] for r in rows], 0).tolist(),
        region_agents=np.mean([r["regions"]["agents_mean"] for r in rows], 0).tolist(),
    )
    if "pollution_mean" in rows[0]:
        out["pollution"] = float(np.mean([r["pollution_mean"] for r in rows]))
        out["cleaned_per_agent"] = np.mean([r["cleaned_per_agent"] for r in rows], 0).tolist()
    if "stolen_from_per_agent" in rows[0]:
        out["stolen_from_per_agent"] = np.mean([r["stolen_from_per_agent"] for r in rows], 0).tolist()
    return out


def summarise(d):
    cells = defaultdict(list)
    for r in d["rows"]:
        cells[(r["population"], r["condition"])].append(r)
    order = list(dict.fromkeys(r["condition"] for r in d["rows"]))
    pops = list(dict.fromkeys(r["population"] for r in d["rows"]))
    out = []
    for pop in pops:
        for cond in order:
            if (pop, cond) in cells:
                out.append(dict(population=pop, condition=cond, **cell_summary(cells[(pop, cond)], d["n_agents"], 1)))
    return out


def fmt(x, nd=0):
    if isinstance(x, float) and np.isnan(x):
        return "–"
    return f"{x:.{nd}f}"


def table(rows, pop, conds=None, cols=("taken", "alive", "violations", "caught_share", "unused_share", "comm_cost")):
    head = {"taken": "goods taken", "alive": "alive", "violations": "violations", "caught_share": "caught",
            "unused_share": "unused checks", "comm_cost": "comm. cost", "welfare": "welfare",
            "wrongly_stopped": "wrong flags", "pollution": "pollution", "cost": "check cost"}
    lines = ["| Overseer | " + " | ".join(head[c] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for r in rows:
        if r["population"] != pop or (conds and r["condition"] not in conds):
            continue
        cells = []
        for c in cols:
            v = r[c]
            if c == "alive":
                cells.append(f"{v}/{r['seeds']}")
            elif c in ("caught_share", "unused_share"):
                cells.append("–" if np.isnan(v) else f"{100 * v:.0f}%")
            elif c == "pollution":
                cells.append(f"{v:.2f}")
            else:
                cells.append(fmt(v, 1 if abs(v) < 10 else 0))
        lines.append(f"| {r['condition']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/runs/socialjax_decentralised")
    ap.add_argument("--game", required=True)
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    d = json.loads((Path(a.run) / f"{a.game}.json").read_text())
    rows = summarise(d)
    (Path(a.run) / f"{a.game}_summary.json").write_text(json.dumps(dict(game=a.game, n_agents=d["n_agents"],
                                                                        seconds=d["seconds"], rows=rows)))
    if a.print:
        for pop in dict.fromkeys(r["population"] for r in rows):
            print(f"\n### {pop}\n")
            print(table(rows, pop))


if __name__ == "__main__":
    main()
