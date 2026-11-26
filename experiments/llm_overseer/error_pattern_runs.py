"""Harvest IPPO job list: patterned LLM inspector vs a coin flip at the matched average (error_patterns/protocol.md §5).

Same format as experiments/llm_overseer/harvest_runs.py ("name<TAB>arguments", for
scripts/create/ippo_overseer_array.sbatch). Reads the fitted tables (experiments/llm_overseer/fit_error_model.py):

  PYTHONPATH=. python3 -m experiments.llm_overseer.error_pattern_runs > experiments/llm_overseer/runs_error_patterns.tsv
  PYTHONPATH=. python3 -m experiments.llm_overseer.error_pattern_runs --x5 > experiments/llm_overseer/runs_x5.tsv

Per inspector, k in {1, 7}, c in {1, 2}, seeds 1-3, random checks, 1e8 steps from scratch:
  E_<tag>_table_k<k>_c<c>_seed<s>   --inspector-table <tables/<tag>.json>
  E_<tag>_coin_k<k>_c<c>_seed<s>    --miss-rate <matched alpha> --false-alarm <matched beta> (the bank average)
  E_<tag>_coinov_k<k>_c<c>_seed<s>  the average over the overseen policies only, if it differs by more than 0.05
The table path is relative to the repository root, which is the array script's working directory. Inspectors with no
table (the fit did not beat a constant rate on held-out items: Qwen 7B + reasoning on 179 items) are skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.socialjax.phase2_runs import penalty

TABLES = "notes/llm_overseer/error_patterns/tables"
SHORT = {"gpt-oss-20b_direct": "gptoss20b", "qwen2.5-7b_reason": "qwen7b-reason"}


def runs(tags, tables=TABLES, ks=(1, 7), levels=(1.0, 2.0), seeds=(1, 2, 3), sensitivity=0.05, arms=None,
         prefix="E", short=None):
    """arms (None = all), prefix and short (tag -> name) are for X5; the defaults give the E_ job list unchanged."""
    for tag in tags:
        path = Path(tables) / f"{tag}.json"
        if not path.exists():  # no table: the fit did not beat the coin flip on held-out items (results.md)
            continue
        t = json.loads(path.read_text())
        pooled, over = t["matched_coin_flip"], t["matched_coin_flip_overseen"]
        all_arms = [("table", f"--inspector-table {path}"),
                ("coin", f"--miss-rate {pooled['alpha']:.3f} --false-alarm {pooled['beta']:.3f}")]
        if max(abs(pooled["alpha"] - over["alpha"]), abs(pooled["beta"] - over["beta"])) > sensitivity:
            all_arms.append(("coinov", f"--miss-rate {over['alpha']:.3f} --false-alarm {over['beta']:.3f}"))
        label = (short or {}).get(tag) or SHORT.get(tag, tag)
        for arm, flags in all_arms:
            if arms is not None and arm not in arms:
                continue
            for k in ks:
                for c in levels:
                    for s in seeds:
                        yield (f"{prefix}_{label}_{arm}_k{k}_c{c}_seed{s}",
                               f"--k {k} --chooser random --penalty-apples {penalty(k, c)} {flags} --seed {s}")


X5_DIR = "notes/formulation/x5_capability"
X5_TAGS = tuple(f"qwen2.5-{s}_reason" for s in ("3b", "7b", "14b", "32b", "72b"))


def x5(tags=X5_TAGS, out_dir=X5_DIR, levels=(1.0, 2.0), seeds=(1, 2, 3)):
    """X5, the capability curve (notes/formulation/x5_capability/predictions.md): each Qwen2.5 judge at k = 1,
    c in {1, 2}, seeds 1-3, random checks. A judge whose fitted table beats a constant rate on held-out items plays
    its table (runs() with arm "table"); otherwise it plays a coin flip at its bank-average alpha and beta from
    <out_dir>/fit.json. Names X5_qwen<size>_<table|coin>_k1_c<c>_seed<s>."""
    fit = json.loads((Path(out_dir) / "fit.json").read_text())
    short = {t: "qwen" + t.split("_")[0].split("-")[1] for t in tags}
    for tag in tags:
        if (Path(out_dir) / "tables" / f"{tag}.json").exists():
            yield from runs([tag], tables=f"{out_dir}/tables", ks=(1,), levels=levels, seeds=seeds, arms=("table",),
                            prefix="X5", short=short)
            continue
        a, b = fit[tag]["alpha"]["matched_pooled"], fit[tag]["beta"]["matched_pooled"]
        for c in levels:
            for s in seeds:
                yield (f"X5_{short[tag]}_coin_k1_c{c}_seed{s}",
                       f"--k 1 --chooser random --penalty-apples {penalty(1, c)} --miss-rate {a:.3f} "
                       f"--false-alarm {b:.3f} --seed {s}")


def inverted(tables=TABLES, ks=(1, 7), levels=(1.0, 2.0), seeds=(1, 2, 3)):
    """H3b (notes/formulation/03_hypotheses.md): gpt-oss-20B's table with its blind spot moved onto the most harmful
    violations (tables/gpt-oss-20b_inverted.json), matched to the same bank-average alpha as the E_*_coin arm."""
    path = Path(tables) / "gpt-oss-20b_inverted.json"
    for k in ks:
        for c in levels:
            for s in seeds:
                yield (f"E_gptoss20b_inverted_k{k}_c{c}_seed{s}",
                       f"--k {k} --chooser random --penalty-apples {penalty(k, c)} --inspector-table {path} --seed {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="gpt-oss-20b_direct,qwen2.5-7b_reason")
    ap.add_argument("--tables", default=TABLES)
    ap.add_argument("--inverted", action="store_true", help="only the H3b inverted-table arm")
    ap.add_argument("--x5", action="store_true", help="only the X5 capability-curve list")
    a = ap.parse_args()
    jobs = inverted(a.tables) if a.inverted else x5() if a.x5 else runs(a.tags.split(","), a.tables)
    for name, args in jobs:
        print(f"{name}\t{args}")


if __name__ == "__main__":
    main()
