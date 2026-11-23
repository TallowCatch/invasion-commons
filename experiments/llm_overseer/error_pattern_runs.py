"""Harvest IPPO job list: patterned LLM inspector vs a coin flip at the matched average (error_patterns/protocol.md §5).

Same format as experiments/llm_overseer/harvest_runs.py ("name<TAB>arguments", for
scripts/create/ippo_overseer_array.sbatch). Reads the fitted tables (experiments/llm_overseer/fit_error_model.py):

  PYTHONPATH=. python3 -m experiments.llm_overseer.error_pattern_runs > experiments/llm_overseer/runs_error_patterns.tsv

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


def runs(tags, tables=TABLES, ks=(1, 7), levels=(1.0, 2.0), seeds=(1, 2, 3), sensitivity=0.05):
    for tag in tags:
        path = Path(tables) / f"{tag}.json"
        if not path.exists():  # no table: the fit did not beat the coin flip on held-out items (results.md)
            continue
        t = json.loads(path.read_text())
        pooled, over = t["matched_coin_flip"], t["matched_coin_flip_overseen"]
        arms = [("table", f"--inspector-table {path}"),
                ("coin", f"--miss-rate {pooled['alpha']:.3f} --false-alarm {pooled['beta']:.3f}")]
        if max(abs(pooled["alpha"] - over["alpha"]), abs(pooled["beta"] - over["beta"])) > sensitivity:
            arms.append(("coinov", f"--miss-rate {over['alpha']:.3f} --false-alarm {over['beta']:.3f}"))
        for arm, flags in arms:
            for k in ks:
                for c in levels:
                    for s in seeds:
                        yield (f"E_{SHORT.get(tag, tag)}_{arm}_k{k}_c{c}_seed{s}",
                               f"--k {k} --chooser random --penalty-apples {penalty(k, c)} {flags} --seed {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="gpt-oss-20b_direct,qwen2.5-7b_reason")
    ap.add_argument("--tables", default=TABLES)
    a = ap.parse_args()
    for name, args in runs(a.tags.split(","), a.tables):
        print(f"{name}\t{args}")


if __name__ == "__main__":
    main()
