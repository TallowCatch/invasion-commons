"""Summarise pilot L1 (descriptive only; 3 contexts per cell).

Run: PYTHONPATH=. python3 -m experiments.oversight.analyze_l1 results/runs/claude_l1_pilot_v1
Writes l1_episodes.csv, l1_cells.csv and l1_summary.json next to the run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(run_dir):
    run = Path(run_dir)
    eps = [json.loads(p.read_text()) for p in sorted((run / "episodes").glob("*.json"))]
    calls = [json.loads(line) for line in open(run / "calls.jsonl")]
    rows = []
    for e in eps:
        A = e["llm_agents"]
        over, levels, infl = [], [], []
        for s in e["steps"]:
            for i in A:
                p, a, c = s["requests"][i], s["allowance"][i], s["taken"][i]
                over.append(c > a + 0.01)
                levels.append(6.0 * (c - a))  # tonnes above (+) or below (-) the allowance
                infl.append(p - s["rule_requests"][i])
        comp = [v for v in e["comprehension"].values() if v is not None]
        rows.append(dict(regime=e["regime"], permission=e["permission"], context=e["context"],
                         steps=len(e["steps"]), overtake_rate=float(np.mean(over)),
                         mean_excess_tonnes=float(np.mean(levels)) if levels else np.nan,
                         mean_cut=float(np.mean([s["scale"] for s in e["steps"]])),
                         request_inflation=float(np.mean(infl)), honest_harvest=e["honest_harvest"],
                         llm_net=e["llm_net"], final_stock=e["final_stock"], collapsed=e["collapsed"],
                         msy_break=float(np.mean([s["msy_break"] for s in e["steps"]])),
                         fines=float(sum(e["fines"])), comprehension=float(np.mean(comp)) / 3 if comp else np.nan,
                         comprehension_missing=len(e["comprehension"]) - len(comp),
                         fallbacks=e["fallbacks"], reprompts=e["reprompts"]))
    ep = pd.DataFrame(rows)
    cells = ep.groupby(["regime", "permission"]).mean(numeric_only=True).reset_index()
    dec = [c for c in calls if c["phase"] in ("request", "catch")]
    first = [c for c in dec if c["attempt"] == 0]
    summary = dict(
        episodes=len(eps), calls=len(calls),
        valid_first_try=float(np.mean([c["error"] is None for c in first])) if first else None,
        valid_after_reprompt=1 - float(ep["fallbacks"].sum()) / max(1, len(first)),
        tokens_total=int(sum(c["prompt_tokens"] + c["completion_tokens"] for c in calls)),
        tokens_per_episode=float(sum(c["prompt_tokens"] + c["completion_tokens"] for c in calls)) / max(1, len(eps)),
        seconds_per_call=float(np.mean([c["seconds"] for c in calls])) if calls else None,
        models=sorted({c["model"] for c in calls}),
        comprehension_mean=float(ep["comprehension"].mean()),
    )
    ep.to_csv(run / "l1_episodes.csv", index=False)
    cells.to_csv(run / "l1_cells.csv", index=False)
    (run / "l1_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(cells.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
