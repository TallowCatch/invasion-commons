"""X5, the capability curve (notes/formulation/x5_capability/): predictions from the judges' verdicts, then the check
against the RL runs.

  # after judging and fitting (fit_error_model.py --out-dir notes/formulation/x5_capability):
  PYTHONPATH=. .venv-socialjax/bin/python -m experiments.llm_overseer.x5_capability predict
  # after the X5_ runs are pulled to results/create_pull:
  PYTHONPATH=. .venv-socialjax/bin/python -m experiments.llm_overseer.x5_capability compare --pull results/create_pull

Proposition 5.7 (notes/formulation/05_formulation.md §5.4): with random looks the orchard is kept iff the sanction a
harmful violation meets, rho_H = (k/N)(1 - alpha_H)F = c(1 - alpha_H) at k = 1 (penalty 7c), is above the learners'
threshold rho* in (0.79, 1.0]. Harmful = a violation with near = 0 (the patch's last apples).
Class: "survive" if rho_H > 1.0, "die" if rho_H < 0.79, "undetermined" in between.

alpha_H is computed twice:
- measured: the judge's miss rate on the near = 0 violations it judged (outer pooled; no verdict counts as a miss,
  as in the fit), with a 95% Wilson interval;
- played: the miss rate the RL run actually applies to a near = 0 violation. A judge with a table plays its table's
  near = 0 cells (weighted by the bank's near = 0 violations over the outer bins); a judge without one plays a coin
  flip at its bank-average alpha everywhere, so its played alpha_H is that average. The RL prediction uses the played
  value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.llm_overseer.error_pattern_runs import X5_DIR, X5_TAGS
from overseer.llm.metrics import wilson
from overseer.llm.models import BY_TAG
from overseer.llm.subset import load_jsonl

BANK = "experiments/llm_overseer/data/error_bank.jsonl.gz"
LO, HI = 0.79, 1.0  # the threshold band for rho* (05_formulation.md §5.1)
LEVELS = (1.0, 2.0)


def klass(rho):
    return "survive" if rho > HI else "die" if rho < LO else "undetermined"


def table_alpha_h(table, bank):
    """Bank-weighted near = 0 miss rate of a trainer table (outer split at the table's edges, if any)."""
    feats = {f["name"]: f["edges"] for f in table["features"]}
    edges = feats.get("outer", [])
    row = table["alpha"][0]
    v0 = [x for x in bank if x["label"] == 1 and x["near"] == 0]
    bins = [int(np.searchsorted(edges, x["outer"], side="right")) for x in v0]
    return float(np.mean([row[min(b, len(row) - 1)] for b in bins]))


def judge_stats(tag, out_dir, bank_by_id, bank):
    recs = [json.loads(l) for l in open(Path(out_dir) / "judge" / f"{tag}.jsonl")]
    recs = [r for r in recs if r.get("error") is None]
    fit = json.loads((Path(out_dir) / "fit.json").read_text())[tag]
    near = {}
    for n in (0, 1, 2):
        v = [r for r in recs if r["label"] == 1 and bank_by_id[r["id"]]["near"] == n]
        miss = sum(r["verdict"] != 1 for r in v)
        near[n] = dict(n=len(v), miss=miss, alpha=miss / len(v) if v else float("nan"), ci=wilson(miss, len(v)))
    tpath = Path(out_dir) / "tables" / f"{tag}.json"
    table = json.loads(tpath.read_text()) if tpath.exists() else None
    a_avg, b_avg = fit["alpha"]["matched_pooled"], fit["beta"]["matched_pooled"]
    played = table_alpha_h(table, bank) if table else a_avg
    tok = np.array([r["tokens_in"] + r["tokens_out"] for r in recs])
    info = BY_TAG.get(tag.split("_")[0], {})
    out = dict(tag=tag, params_b=info.get("params_b"), ollama=info.get("ollama"), n_items=len(recs),
               no_verdict=sum(r["verdict"] is None for r in recs),
               cut_off=sum(r.get("done_reason") == "length" for r in recs),
               alpha_h=near[0]["alpha"], alpha_h_ci=near[0]["ci"], n_h=near[0]["n"],
               alpha_near1=near[1]["alpha"], alpha_near2=near[2]["alpha"],
               alpha_avg=a_avg, beta_avg=b_avg, alpha_avg_overseen=fit["alpha"]["matched_overseen"],
               table=table is not None, arm="table" if table else "coin", alpha_h_played=played,
               tokens_per_check=float(tok.mean()), tokens_in=float(np.mean([r["tokens_in"] for r in recs])),
               tokens_out=float(np.mean([r["tokens_out"] for r in recs])),
               seconds_per_check=float(np.mean([r["seconds"] for r in recs])))
    for c in LEVELS:
        for kind, a in (("measured", out["alpha_h"]), ("played", played)):
            out[f"rho_{kind}_c{c:g}"] = c * (1 - a)
            out[f"class_{kind}_c{c:g}"] = klass(c * (1 - a))
        lo_a, hi_a = out["alpha_h_ci"]
        out[f"rho_measured_ci_c{c:g}"] = (c * (1 - hi_a), c * (1 - lo_a))
        out[f"class_measured_ci_c{c:g}"] = sorted({klass(c * (1 - hi_a)), klass(c * (1 - lo_a))})
    return out


def predict(a):
    bank = load_jsonl(a.bank)
    by_id = {x["id"]: x for x in bank}
    rows = [judge_stats(t, a.out_dir, by_id, bank) for t in a.tags
            if (Path(a.out_dir) / "judge" / f"{t}.jsonl").exists()]
    Path(a.out_dir, "predictions.json").write_text(json.dumps(rows, indent=1))
    print("| judge | size (B) | items | α_𝓗 measured [95% CI] (n) | α at near 1 / 2 | average α / β | arm | α_𝓗 played "
          "| ρ_𝓗 c = 1 → class | ρ_𝓗 c = 2 → class | tokens per check (in + out) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lo, hi = r["alpha_h_ci"]
        print(f"| {r['tag'].split('_')[0]} | {r['params_b']} | {r['n_items']} | {r['alpha_h']:.3f} [{lo:.3f}, {hi:.3f}] "
              f"({r['n_h']}) | {r['alpha_near1']:.2f} / {r['alpha_near2']:.2f} | {r['alpha_avg']:.3f} / "
              f"{r['beta_avg']:.3f} | {r['arm']} | {r['alpha_h_played']:.3f} | {r['rho_played_c1']:.2f} → "
              f"{r['class_played_c1']} | {r['rho_played_c2']:.2f} → {r['class_played_c2']} | "
              f"{r['tokens_per_check']:.0f} ({r['tokens_in']:.0f} + {r['tokens_out']:.0f}) |")


def compare(a):
    from experiments.socialjax.analyze_formulation_runs import run_stats
    preds = {r["tag"]: r for r in json.loads(Path(a.out_dir, "predictions.json").read_text())}
    rows, right, wrong, undet = [], 0, 0, 0
    for tag, p in preds.items():
        short = "qwen" + tag.split("_")[0].split("-")[1]
        for c in LEVELS:
            runs = sorted(Path(a.pull).glob(f"X5_{short}_{p['arm']}_k1_c{c}_seed*/metrics.npz"))
            stats = [run_stats(str(x)) for x in runs]
            if not stats:
                continue
            alive = [s["alive"] for s in stats]
            observed = "survive" if np.mean(alive) >= 0.5 else "die"
            pred = p[f"class_played_c{c:g}"]
            hit = None if pred == "undetermined" else pred == observed
            right += hit is True
            wrong += hit is False
            undet += hit is None
            near = np.nanmean([s["near"] for s in stats], 0) if all("near" in s for s in stats) else [np.nan] * 3
            rows.append(dict(tag=tag, c=c, seeds=len(stats), rho=p[f"rho_played_c{c:g}"], predicted=pred,
                             observed=observed, hit=hit, eaten=[s["eaten"] for s in stats], alive=alive,
                             viol=float(np.mean([s["viol"] for s in stats])),
                             alpha_real=float(np.nanmean([s.get("alpha_real", np.nan) for s in stats])),
                             near_share=[float(x) for x in near],
                             caught=float(np.nanmean([s["caught"] for s in stats]))))
    Path(a.out_dir, "results.json").write_text(json.dumps(dict(rows=rows, right=right, wrong=wrong, undetermined=undet),
                                                          indent=1))
    print("| judge | c | seeds | ρ_𝓗 played | predicted | apples eaten mean [range] | orchard alive (per seed) | observed "
          "| violations | realised α | share of violations at near 0 / 1 / 2 | on the predicted side? |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        e = r["eaten"]
        print(f"| {r['tag'].split('_')[0]} | {r['c']:g} | {r['seeds']} | {r['rho']:.2f} | {r['predicted']} | "
              f"{np.mean(e):.0f} [{min(e):.0f}, {max(e):.0f}] | {', '.join(f'{x:.2f}' for x in r['alive'])} | "
              f"{r['observed']} | {r['viol']:.1f} | {r['alpha_real']:.2f} | "
              f"{' / '.join(f'{x:.2f}' for x in r['near_share'])} | "
              f"{'—' if r['hit'] is None else 'yes' if r['hit'] else '**no**'} |")
    print(f"\nDetermined cells on the predicted side: {right} of {right + wrong}; undetermined: {undet}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("predict", "compare"))
    ap.add_argument("--tags", nargs="+", default=list(X5_TAGS))
    ap.add_argument("--out-dir", default=X5_DIR)
    ap.add_argument("--bank", default=BANK)
    ap.add_argument("--pull", default="results/create_pull")
    a = ap.parse_args()
    predict(a) if a.cmd == "predict" else compare(a)


if __name__ == "__main__":
    main()
