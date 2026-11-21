"""Analysis for S1b (ablation + Fishery MSY) and S2 (compliance and deterrence). Claude audit, October 2026."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

B, SEED = 4000, 20261007


def load(p):
    return pd.DataFrame([json.loads(l) for l in gzip.open(p, "rt")])


def boot_rate_diff(x, y, rng):
    """x, y: per-context DataFrames (sorted by context) with exec_risky and scored_steps."""
    xr, xs, yr, ys = (np.asarray(v, float) for v in (x.exec_risky, x.scored_steps, y.exec_risky, y.scored_steps))
    f = lambda i: xr[i].sum() / max(xs[i].sum(), 1) - yr[i].sum() / max(ys[i].sum(), 1)
    n = len(xr)
    draws = [f(rng.integers(0, n, n)) for _ in range(B)]
    return dict(estimate=float(f(np.arange(n))), ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))])


def boot_mean_diff(x, y, rng):
    d = np.asarray(x, float) - np.asarray(y, float)
    draws = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return dict(estimate=float(d.mean()), ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))])


def table(E, keys, payoff):
    T = E.groupby(keys).agg(contexts=("context", "nunique"), scored=("scored_steps", "sum"), exec_risky=("exec_risky", "sum"),
                            total_harvest=("total_harvest", "mean"), honest_harvest=("honest_harvest", "mean"),
                            payoff=(payoff, "mean"), mean_health=("mean_health", "mean"), unsafe_fixed=("unsafe_fixed", "mean"),
                            audits=("audits", "sum"), steps=("t_end", "sum"), catches=("catches", "mean")).reset_index()
    T["unsafe_action_rate"] = T.exec_risky / T.scored
    T["audits_per_step"] = T.audits / T.steps
    return T


def s1b(run, out, rng):
    E = load(run / "episodes.jsonl.gz")
    T = table(E, ["part", "game", "target", "protocol", "belief", "sanction", "actor", "d"], "misreporter_payoff")
    T.to_csv(out / "s1b_condition_table.csv", index=False)
    S = dict(adaptive=json.loads((run / "adaptive_search.json").read_text()))
    s1 = load(Path("results/runs/claude_s1_reporting_audit_v1/episodes.jsonl.gz"))
    A = E[(E.part == "A") & (E.actor == "fixed")]

    def cell(game, proto, belief, sanction, d):
        return A[(A.game == game) & (A.protocol == proto) & (A.belief == belief) & (A.sanction == sanction)
                 & np.isclose(A.d, d)].sort_values("context")
    rep = s1[(s1.protocol == "report") & (s1.actor == "fixed") & np.isclose(s1.d, .5)]
    for game, protos in (("harvest", ("rand2", "targ1", "peer")), ("fishery", ("rand2", "peer"))):
        r = rep[rep.game == game].sort_values("context")
        for proto in protos:
            ref = cell(game, proto, True, "excl+fine", .5)
            S[f"A_{game}_{proto}"] = {
                "belief_only_minus_report": boot_rate_diff(cell(game, proto, True, "none", .5), r, rng),
                "fine_only_with_belief_minus_report": boot_rate_diff(cell(game, proto, True, "fine", .5), r, rng),
                "exclusion_without_belief_minus_full_S1": boot_rate_diff(cell(game, proto, False, "excl+fine", .5), ref, rng),
                "S1_cell_minus_report": boot_rate_diff(ref, r, rng)}
    Bf = E[(E.part == "B")]
    full = Bf[(Bf.protocol == "full")].sort_values("context")
    for d in (.25, .5):
        rp = Bf[(Bf.protocol == "report") & np.isclose(Bf.d, d) & (Bf.actor == "fixed")].sort_values("context")
        S[f"B_report_minus_full_d{d}"] = boot_rate_diff(rp, full, rng)
        S[f"B_report_minus_full_harvest_d{d}"] = boot_mean_diff(rp.total_harvest, full.total_harvest, rng)
    (out / "s1b_summary.json").write_text(json.dumps(S, indent=1))


def s2(run, out, rng):
    E = load(run / "episodes.jsonl.gz")
    T = table(E, ["game", "protocol", "fine", "actor", "d"], "cheater_payoff")
    T.to_csv(out / "s2_condition_table.csv", index=False)
    srch = json.loads((run / "adaptive_search.json").read_text())
    S = dict(adaptive=srch)
    for game in ("fishery", "harvest"):
        comply = E[(E.game == game) & (E.actor == "comply")].sort_values("context")
        allow1 = E[(E.game == game) & (E.protocol == "allow") & (E.actor == "d1")].sort_values("context")
        S[f"{game}_allow_d1_minus_comply_unsafe"] = boot_rate_diff(allow1, comply, rng)
        S[f"{game}_allow_d1_minus_comply_cheater_payoff"] = boot_mean_diff(allow1.cheater_payoff, comply.cheater_payoff, rng)
        S[f"{game}_allow_d1_minus_comply_honest_harvest"] = boot_mean_diff(allow1.honest_harvest, comply.honest_harvest, rng)
        S[f"{game}_allow_d1_minus_comply_total_harvest"] = boot_mean_diff(allow1.total_harvest, comply.total_harvest, rng)
        gains = {}
        for s in srch:
            if s["game"] != game:
                continue
            a = E[(E.game == game) & (E.protocol == s["protocol"]) & (E.fine == s["fine"]) & (E.actor == "adaptive")].sort_values("context")
            gains[f'{s["protocol"]}_F{s["fine"]:g}'] = dict(d_star=s["d_star"],
                heldout_gain_vs_comply=boot_mean_diff(a.cheater_payoff, comply.cheater_payoff, rng),
                unsafe_minus_comply=boot_rate_diff(a, comply, rng))
        S[f"{game}_adaptive_cells"] = gains
    (out / "s2_summary.json").write_text(json.dumps(S, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1b", required=True)
    ap.add_argument("--s2", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    for run, fn in ((Path(a.s1b), s1b), (Path(a.s2), s2)):
        out = run / "analysis"
        out.mkdir(exist_ok=True)
        fn(run, out, rng)
    print("done")


if __name__ == "__main__":
    main()
