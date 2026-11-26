"""Step 1: one row per (tested agent, day) for each downloaded cell.

Columns: unsafe failure objects first seen that day by type (F1-F6), the furthest stage any of
that agent-day's objects reached (S0 = none ... S5 = carried through), and how much text a
check would read (characters, tokens estimated as characters / 4, and the GPT-5 judge's own
allocated input tokens for comparison).

Writes data/bazaarbench/derived/agent_day_<cell>.csv (git-ignored) and prints a summary.

Usage: python -m experiments.bazaarbench.build_table
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bazaarbench import bb_data as bd  # noqa: E402

RAW = ROOT / "data" / "bazaarbench" / "raw"
DERIVED = ROOT / "data" / "bazaarbench" / "derived"
CELLS = ["gpt55_L1_gptoss120b", "gpt55_L3_gptoss120b"]


def agent_day_rows(cell: bd.Cell) -> list[dict]:
    cost = cell.agent_day_cost()
    rows = []
    for a in cell.treated:
        for d in range(bd.N_DAYS):
            c = cost[(a, d)]
            seen = [o for o in cell.objects if o.actor == a and d in o.visible_days("judge")]
            new = [o for o in seen if min(o.visible_days("judge")) == d]
            row = {"cell": cell.cell_id, "agent": a, "zip": cell.zips.get(a, ""), "day": d,
                   "actions": c["actions"], "bundles": c["bundles"], "chars": c["chars"],
                   "est_tokens": round(c["chars"] / bd.CHARS_PER_TOKEN),
                   "reasoning_chars": c["reasoning_chars"],
                   "est_tokens_no_reasoning": round((c["chars"] - c["reasoning_chars"]) / bd.CHARS_PER_TOKEN),
                   "judge_input_tokens": c["judge_input_tokens"],
                   "objects_visible": len(seen), "objects_new": len(new)}
            for f in ["F1", "F2", "F3", "F4", "F5", "F6"]:
                row[f"new_{f}"] = sum(o.ftype == f for o in new)
            row["furthest_stage_visible"] = max([o.max_stage_num for o in seen], default=0)
            row["objects_visible_public"] = sum(d in o.visible_days("no_reasoning") for o in cell.objects if o.actor == a)
            rows.append(row)
    return rows


def object_rows(cell: bd.Cell) -> list[dict]:
    rows = []
    for o in cell.objects:
        rows.append({"cell": cell.cell_id, "obj": o.obj_id, "agent": o.actor, "ftype": o.ftype,
                     "max_stage": o.max_stage_num, "n_decisions": len(o.decisions),
                     "first_day": min(o.visible_days("judge")),
                     "days_visible": " ".join(map(str, sorted(o.visible_days("judge")))),
                     "days_visible_public": " ".join(map(str, sorted(o.visible_days("no_reasoning")))),
                     "day_S4": o.first_day_at_stage(4), "day_S5": o.first_day_at_stage(5)})
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarise(cell: bd.Cell, rows: list[dict]) -> None:
    objs = cell.objects
    print(f"\n== {cell.cell_id} ({cell.regime}) ==")
    print(f"bundles judged: {len(cell.bundles)}; unsafe decisions (all 100 agents): {len(cell.decisions)}; "
          f"failure objects, all agents: {len(cell.objects_all)}; tested 20 agents: {len(objs)}")
    print("tested-agent objects by type:", dict(sorted(Counter(o.ftype for o in objs).items())))
    print("by furthest stage:", dict(sorted(Counter(f"S{o.max_stage_num}" for o in objs).items())))
    print("visible outside private reasoning:", sum(bool(o.visible_days('no_reasoning')) for o in objs),
          "| visible in records (F1-F3, no reasoning):", sum(bool(o.visible_days('record')) for o in objs))
    tok = [r["est_tokens"] for r in rows]
    print(f"tokens per agent-day (chars/4): mean {sum(tok) / len(tok):.0f}, min {min(tok)}, max {max(tok)}; "
          f"all 20 agents for 7 days: {sum(tok):,}")
    jt = [r["judge_input_tokens"] for r in rows]
    print(f"GPT-5 allocated input tokens per agent-day: mean {sum(jt) / len(jt):.0f}")


def main() -> None:
    for name in CELLS:
        cell = bd.load_cell(RAW / name)
        rows = agent_day_rows(cell)
        write_csv(rows, DERIVED / f"agent_day_{name}.csv")
        write_csv(object_rows(cell), DERIVED / f"objects_{name}.csv")
        summarise(cell, rows)


if __name__ == "__main__":
    main()
