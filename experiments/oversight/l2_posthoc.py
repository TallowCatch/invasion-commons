"""Post hoc diagnostics for L2 (decided after the results were seen; not in the protocol). Descriptive only.

1. Over-take rate in steps 0-4, before any lake collapses, so collapse cannot shorten a game.
2. What an over-take step looks like: the agent caught more than it had requested, or it kept its request
   after the reviewer cut it (catch = request > allowance).
3. After being caught (added 8 Oct): the chance of over-taking again in the next round, after an over-take that was
   caught against one that was not checked. Fines below the gain (E1-E8) and checks with no fine (E0, S0, P0) are kept
   separate. A rational agent facing independent random audits should not change; humans often comply less right after
   an audit (the "bomb-crater" effect; Mittone 2006, Kastlunger et al. 2009).
4. Size of an over-take (added 8 Oct): the share of over-take steps at the 6 t maximum. Under a flat fine, nothing
   deters taking more once over (no marginal deterrence; Stigler 1970).
5. Over-take rate by stock level (E0-E8). This is confounded with how far the reviewer cuts at low stock.
6. Whole-game gain (added 8 Oct, consistency check). The simulations' gain g* (R3) is the cheaters' net gain over
   complying across the whole game, per over-take step. For the LLMs: the LLM agents' mean net catch in E0 minus E36,
   where nobody over-takes (the compliant baseline, paired by context), divided by their over-take steps in E0. The
   pre-registered LLM gain g (L2, L3) is instead the one-round excess: tonnes above the allowance per over-take step.
Run:  PYTHONPATH=. python -m experiments.oversight.l2_posthoc --run STORE/claude_l2_v1 --out DIR
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from experiments.oversight import run_l2_llm_agents as l2
from fishery_sim import llm_actor as L

THR = l2.OVER_T / L.MAX_CATCH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for d in sorted(p for p in Path(a.run).iterdir() if (p / "episodes").exists()):
        eps = [json.loads(p.read_text()) for p in sorted((d / "episodes").glob("*.json")) if not p.name.startswith("EM__")]
        early, kind = collections.defaultdict(list), collections.Counter()
        for e in eps:
            for s in e["steps"]:
                for i in e["llm_agents"]:
                    over = s["taken"][i] > s["allowance"][i] + THR
                    if s["t"] < 5:
                        early[e["cell"]].append(over)
                    if over:
                        cut = s["allowance"][i] < s["requests"][i] - 1e-9
                        if s["taken"][i] > s["requests"][i] + 1e-6:
                            kind["caught more than it requested"] += 1
                        elif cut and abs(s["taken"][i] - s["requests"][i]) < 1e-6:
                            kind["kept its request after a cut"] += 1
                        else:
                            kind["other"] += 1
        after = {}
        for label, cells in (("fine_below_gain", ("E1", "E2", "E4", "E8")), ("checked_no_fine", ("E0", "S0", "P0"))):
            nxt = collections.defaultdict(list)
            for e in eps:
                if e["cell"] not in cells:
                    continue
                st = e["steps"]
                for k in range(len(st) - 1):
                    for i in e["llm_agents"]:
                        if st[k]["taken"][i] > st[k]["allowance"][i] + THR:
                            nxt["caught" if st[k]["caught"][i] else "not_checked"].append(
                                st[k + 1]["taken"][i] > st[k + 1]["allowance"][i] + THR)
            after[label] = {k: dict(rate=float(np.mean(v)), n=len(v)) for k, v in nxt.items()}
        sizes, bystock = [], collections.defaultdict(list)
        for e in eps:
            if e["cell"] not in ("E0", "E1", "E2", "E4", "E8"):
                continue
            for s in e["steps"]:
                b = "<30" if s["stock"] < 30 else "30-50" if s["stock"] < 50 else "50-70" if s["stock"] < 70 else ">=70"
                for i in e["llm_agents"]:
                    over = s["taken"][i] > s["allowance"][i] + THR
                    bystock[b].append(over)
                    if over:
                        sizes.append(s["taken"][i] * L.MAX_CATCH)
        byc = {(e["cell"], e["context"]): e for e in eps}
        wg = None
        if all(("E0", c) in byc and ("E36", c) in byc for c in range(10)):
            G = [byc[("E0", c)]["llm_net"] - byc[("E36", c)]["llm_net"] for c in range(10)]
            nov = [np.mean([sum(s["taken"][i] > s["allowance"][i] + THR for s in byc[("E0", c)]["steps"])
                            for i in byc[("E0", c)]["llm_agents"]]) for c in range(10)]
            # one-round gain g1 by context: summed tonnes above the allowance, and the number of over-take steps, in E0
            g1s = []
            for c in range(10):
                e0 = byc[("E0", c)]
                ex = [(s_["taken"][i] - s_["allowance"][i]) * L.MAX_CATCH for s_ in e0["steps"] for i in e0["llm_agents"]
                      if s_["taken"][i] > s_["allowance"][i] + THR]
                g1s.append((float(np.sum(ex)), len(ex)))
            wg = dict(net_gain_per_game=float(np.mean(G)), net_gain_by_context=[float(x) for x in G],
                      overtake_steps_by_context=[float(x) for x in nov], g1_excess_by_context=[x[0] for x in g1s],
                      g1_steps_by_context=[x[1] for x in g1s],
                      overtake_steps_per_agent=float(np.mean(nov)),
                      g_star_whole_game=float(np.sum(G) / np.sum(nov)) if np.sum(nov) else None)
        n = sum(kind.values())
        res[d.name] = dict(early_overtake_rate={c: float(np.mean(early[c])) for c in l2.CELLS if c in early},
                           overtake_kind_share={k: v / n for k, v in kind.items()} if n else {}, overtake_steps=n,
                           after_overtake_next_round=after,
                           overtake_size=dict(share_at_max=float(np.mean(np.isclose(sizes, L.MAX_CATCH))) if sizes else None,
                                              median_t=float(np.median(sizes)) if sizes else None, n=len(sizes),
                                              catches_t=[round(x, 3) for x in sizes]),
                           overtake_by_stock={b: dict(rate=float(np.mean(v)), n=len(v)) for b, v in bystock.items()},
                           whole_game_gain=wg)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "l2_posthoc.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({m: dict(steps=r["overtake_steps"], kind={k: round(v, 3) for k, v in r["overtake_kind_share"].items()})
                      for m, r in res.items()}, indent=1))


if __name__ == "__main__":
    main()
