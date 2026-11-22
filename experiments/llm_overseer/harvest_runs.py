"""Harvest IPPO job list from each LLM inspector's measured alpha and beta (notes/llm_overseer/protocol.md, section 6).

Same format as experiments/socialjax/phase2_runs.py: one line per run, "name<TAB>arguments", for
scripts/create/ippo_overseer_array.sbatch. Reads summary.json from experiments/llm_overseer/analyze.py and uses the
w = 1 numbers (the RL overseer checks one step): --miss-rate alpha, --false-alarm beta_eat (false alarms on honest
harvests, as the wrapper models them since 21 Nov).

  python -m experiments.llm_overseer.harvest_runs --summary notes/llm_overseer/results/summary.json > runs_llm.tsv
  python -m experiments.llm_overseer.harvest_runs --dedupe 0.05 --one-seed-if-fails > runs_llm_small.tsv

Each line: L_<model>[-<mode> if not direct]_<rendering>_k<k>_c<c>_seed<s>  --k k --chooser random --penalty-apples 7c/k --miss-rate a
--false-alarm b --seed s (1e8 steps from scratch, as in the mistakes test).
"""
from __future__ import annotations

import argparse
import json

from experiments.socialjax.phase2_runs import penalty
from overseer.llm.metrics import c1_class, effective_penalty


def runs(c1_entries, modes=("direct",), renderings=("summary", "raw"), ks=(1, 7), levels=(1.0, 2.0),
         seeds=(1, 2, 3), dedupe=0.0, one_seed_if_fails=False, models=None):
    seen = set()
    for e in c1_entries:
        if e["mode"] not in modes or e["rendering"] not in renderings or (models and e["model"] not in models):
            continue
        a, b = round(e["alpha"], 3), round(e["beta_eat"], 3)
        if a != a or b != b:  # nan: no records in a cell
            continue
        if dedupe:
            key = (round(a / dedupe), round(b / dedupe))
            if key in seen:
                continue
            seen.add(key)
        for k in ks:
            for c in levels:
                fails = c1_class(effective_penalty(a, b, c)) == "fails"
                for s in (seeds[:1] if (fails and one_seed_if_fails) else seeds):
                    tag = e["model"] if e["mode"] == "direct" else f"{e['model']}-{e['mode']}"
                    name = f"L_{tag}_{e['rendering']}_k{k}_c{c}_seed{s}"
                    yield name, (f"--k {k} --chooser random --penalty-apples {penalty(k, c)} "
                                 f"--miss-rate {a} --false-alarm {b} --seed {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="notes/llm_overseer/results/summary.json")
    ap.add_argument("--modes", default="direct")
    ap.add_argument("--renderings", default="summary,raw")
    ap.add_argument("--models", default="", help="comma-separated ladder tags (default: all measured)")
    ap.add_argument("--ks", default="1,7")
    ap.add_argument("--levels", default="1,2")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--dedupe", type=float, default=0.0, help="drop inspectors whose (alpha, beta) round alike")
    ap.add_argument("--one-seed-if-fails", action="store_true", help="1 seed where C1 predicts the orchard dies")
    a = ap.parse_args()
    c1 = json.load(open(a.summary))["c1"]
    for name, args in runs(c1, modes=tuple(a.modes.split(",")), renderings=tuple(a.renderings.split(",")),
                           ks=tuple(int(x) for x in a.ks.split(",")),
                           levels=tuple(float(x) for x in a.levels.split(",")),
                           seeds=tuple(int(x) for x in a.seeds.split(",")), dedupe=a.dedupe,
                           one_seed_if_fails=a.one_seed_if_fails,
                           models=set(a.models.split(",")) if a.models else None):
        print(f"{name}\t{args}")


if __name__ == "__main__":
    main()
