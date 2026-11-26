"""Analysis for S5 (S5 protocol). Paired context bootstrap, 4,000 resamples, seed 20261013, 95% percentile
intervals; Holm correction across the S5 confirmatory family (H1-H4) with bootstrap p-values (share of resamples on
the wrong side of the hypothesis threshold, doubled, capped at 1).

Run:  PYTHONPATH=. python -m experiments.oversight.analyze_s5 RUN_DIR      (RUN_DIR contains T0T1/ and optionally T2/)
Writes RUN_DIR/analysis/s5_summary.json and s5_*.csv.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED, B, ALPHA = 20261013, 4000, 0.05
TIERS = ("T0", "T1", "T2")


def load(p):
    return pd.DataFrame([json.loads(l) for l in gzip.open(p, "rt")]) if Path(p).exists() else pd.DataFrame()


def cell_key(regime, q):
    return "none" if regime == "none" or q is None or (isinstance(q, float) and np.isnan(q)) else f"{regime}|{float(q):.4f}"


def prep(df):
    if df.empty:
        return df
    df = df.copy()
    df["cell"] = [cell_key(r, q) for r, q in zip(df.regime, df.q)]
    return df


class Boot:
    """Paired context bootstrap. All series passed to one call must be aligned by context."""

    def __init__(self, seed=SEED):
        self.rng = np.random.default_rng(seed)

    def stat(self, f, *xs):
        xs = [np.asarray(x, float) for x in xs]
        n = len(xs[0])
        assert all(len(x) == n for x in xs)
        est = float(f(*xs))
        draws = np.empty(B)
        for b in range(B):
            i = self.rng.integers(0, n, n)
            draws[b] = f(*(x[i] for x in xs))
        return est, draws

    def indep(self, f, x, y):
        """Unpaired (train vs held-out are different contexts)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        est = float(f(x, y))
        draws = np.empty(B)
        for b in range(B):
            draws[b] = f(x[self.rng.integers(0, len(x), len(x))], y[self.rng.integers(0, len(y), len(y))])
        return est, draws


def summarize_draws(est, draws, threshold=None, claim=None):
    fin = draws[np.isfinite(draws)]
    ci = [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))] if len(fin) else [float("nan")] * 2
    out = dict(estimate=est, ci=ci)
    if threshold is not None:
        # wrong side of the threshold for the claim ('above' -> draws <= thr are wrong; 'below' -> draws >= thr)
        wrong = np.mean(draws <= threshold) if claim == "above" else np.mean(~(draws < threshold))
        out.update(threshold=threshold, claim=claim, p=float(min(1.0, 2 * wrong)))
    return out


def holm(ps):
    ps = np.asarray(ps, float)
    m = len(ps)
    order = np.argsort(ps, kind="stable")
    adj, running = np.empty(m), 0.0
    for rank, j in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[j]))
        adj[j] = running
    return adj.tolist()


def aligned(df, cell, tier, split="test"):
    s = df[(df.cell == cell) & (df.tier == tier) & (df.split == split)].sort_values("context")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    a = ap.parse_args()
    run = Path(a.run_dir)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    E = prep(pd.concat([load(run / "T0T1" / "episodes.jsonl.gz"), load(run / "T2" / "episodes.jsonl.gz")], ignore_index=True))
    TR = prep(pd.concat([load(run / "T0T1" / "train_episodes.jsonl.gz"), load(run / "T2" / "train_episodes.jsonl.gz")],
                        ignore_index=True))
    E["split"], TR["split"] = "test", "train"
    searches = json.loads((run / "T0T1" / "searches.json").read_text())
    t2log = []
    if (run / "T2" / "training_log.json.gz").exists():
        with gzip.open(run / "T2" / "training_log.json.gz", "rt") as f:
            t2log = json.load(f)
    bs = Boot()
    bs_x = Boot(SEED + 1)  # separate stream for exploratory intervals, so the main draws do not depend on them
    cells = list(dict.fromkeys(E.cell))
    rows, by = [], {}
    mean_diff = lambda x, y: (x - y).mean()

    # ------------------------------------------------------------ per cell x tier outcomes (held out)
    for cell in cells:
        comply = aligned(E, cell, "comply")
        comply_tr = aligned(TR, cell, "comply", "train")
        for tier in TIERS:
            x = aligned(E, cell, tier)
            if x.empty:
                continue
            assert list(x.context) == list(comply.context)
            gain = summarize_draws(*bs.stat(mean_diff, x.cheater_payoff, comply.cheater_payoff))
            nh = x.n_honest.values
            honest = summarize_draws(*bs.stat(mean_diff, x.honest_harvest / nh, comply.honest_harvest / nh))
            xt = aligned(TR, cell, tier, "train")
            tr_gain = float(xt.cheater_payoff.mean() - comply_tr.cheater_payoff.mean())
            gap = summarize_draws(*bs.indep(lambda u, v: u.mean() - v.mean(),
                                            xt.cheater_payoff.values - comply_tr.cheater_payoff.values,
                                            x.cheater_payoff.values - comply.cheater_payoff.values))
            r = dict(cell=cell, regime=x.regime.iloc[0], q=None if cell == "none" else float(x.q.iloc[0]), tier=tier,
                     n_contexts=len(x), cheater_gain=gain, honest_per_agent=float((x.honest_harvest / nh).mean()),
                     honest_per_agent_comply=float((comply.honest_harvest / nh).mean()), honest_minus_comply=honest,
                     target_breaking=float(x.exec_risky.sum() / max(x.scored_steps.sum(), 1)),
                     target_breaking_comply=float(comply.exec_risky.sum() / max(comply.scored_steps.sum(), 1)),
                     audits_per_step=float(x.audits.sum() / x.t_end.sum()),
                     audits_per_step_comply=float(comply.audits.sum() / comply.t_end.sum()),
                     enforced_checks_per_step=float(x.enforced_checks.sum() / x.t_end.sum()),
                     catches_per_context=float(x.catches.mean()), fines_per_context=float(x.fines.mean()),
                     mean_level=float(x.mean_level.mean()), collapsed=int((x.t_end < 80).sum()),
                     train_gain=tr_gain, heldout_gain=gain["estimate"], train_minus_heldout=gap)
            # exploratory: each cheater rank's own gain (T1/T2 optimise own payoff, T0 the group's)
            own = np.asarray(list(x.cheater_payoffs), float) - np.asarray(list(comply.cheater_payoffs), float)
            r["own_gain_by_rank"] = own.mean(axis=0).tolist()
            r["max_rank_own_gain"] = summarize_draws(*bs_x.stat(lambda *cols: max(c.mean() for c in cols), *own.T))
            rows.append(r)
            by[(cell, tier)] = r

    # ------------------------------------------------------------ validity (manipulation) checks
    validity = []
    pairs = [("T1", "T0"), ("T2", "T1"), ("T2", "T0")]
    for cell in cells:
        for hi, lo in pairs:
            x, y = aligned(E, cell, hi), aligned(E, cell, lo)
            if x.empty or y.empty:
                continue
            d = summarize_draws(*bs.stat(mean_diff, x.cheater_payoff, y.cheater_payoff))
            validity.append(dict(cell=cell, stronger=hi, weaker=lo, gain_difference=d,
                                 is_stronger=bool(d["estimate"] > 0 and d["ci"][0] > 0)))
    t1_any = any(v["is_stronger"] for v in validity if v["stronger"] == "T1" and v["weaker"] == "T0")
    t2_vs_t0 = [v for v in validity if v["stronger"] == "T2" and v["weaker"] == "T0"]
    t2_vs_t1 = [v for v in validity if v["stronger"] == "T2" and v["weaker"] == "T1"]
    manip = dict(
        T1_stronger_than_T0_cells=[v["cell"] for v in validity if v["stronger"] == "T1" and v["is_stronger"]],
        T1_failed_manipulation=not t1_any,
        T2_stronger_than_T1_cells=[v["cell"] for v in t2_vs_t1 if v["is_stronger"]],
        T2_stronger_than_T0_cells=[v["cell"] for v in t2_vs_t0 if v["is_stronger"]],
        T2_failed_manipulation=(not any(v["is_stronger"] for v in t2_vs_t0)) if t2_vs_t0 else None,
        T2_present=bool(t2_vs_t0))

    # ------------------------------------------------------------ confirmatory family
    fam = []
    none_c, none_t0 = aligned(E, "none", "comply"), aligned(E, "none", "T0")
    for q in (0.10, 1 / 6):  # H1
        cell = cell_key("fine", q)
        for tier in TIERS:
            x, c = aligned(E, cell, tier), aligned(E, cell, "comply")
            if x.empty:
                continue
            s = summarize_draws(*bs.stat(mean_diff, x.cheater_payoff, c.cheater_payoff), threshold=0.0, claim="above")
            fam.append(dict(hyp="S5-H1", component=f"{cell} {tier} gain", **s))
    for regime in ("memory_cut", "memory_cap"):  # H2
        cell = cell_key(regime, 1 / 6)
        x, c = aligned(E, cell, "T0"), aligned(E, cell, "comply")
        assert list(x.context) == list(none_t0.context)

        def ratio(a, b, n1, n0):
            den = (n1 - n0).mean()
            return (a - b).mean() / den if den > 0 else np.inf
        s = summarize_draws(*bs.stat(ratio, x.cheater_payoff, c.cheater_payoff, none_t0.cheater_payoff,
                                     none_c.cheater_payoff), threshold=0.25, claim="below")
        fam.append(dict(hyp="S5-H2", component=f"{cell} T0 gain / none T0 gain", **s))
    cell = cell_key("memory_cap", 1 / 6)  # H3
    for tier in ("T1", "T2"):
        x, y = aligned(E, cell, tier), aligned(E, cell, "T0")
        if x.empty:
            continue
        s = summarize_draws(*bs.stat(mean_diff, x.cheater_payoff, y.cheater_payoff), threshold=10.0, claim="below")
        fam.append(dict(hyp="S5-H3", component=f"{cell} {tier} - T0 gain", **s))
    cell = cell_key("fine+memory_cut", 0.05)  # H4
    x, c = aligned(E, cell, "T1"), aligned(E, cell, "comply")
    s = summarize_draws(*bs.stat(mean_diff, x.cheater_payoff, c.cheater_payoff), threshold=0.0, claim="above")
    fam.append(dict(hyp="S5-H4", component=f"{cell} T1 gain", **s))
    for f, p in zip(fam, holm([f["p"] for f in fam])):
        f["p_holm"] = p
        f["significant_holm"] = bool(p < ALPHA)

    verdicts = {}
    h1 = [f for f in fam if f["hyp"] == "S5-H1"]
    for f in h1:
        f["violation"] = bool(f["estimate"] > 5 and f["significant_holm"])
        f["violation_uncorrected"] = bool(f["estimate"] > 5 and f["ci"][0] > 0)
    verdicts["S5-H1"] = dict(
        verdict="falsified" if any(f["violation"] for f in h1) else "supported",
        rule="falsified if any tier has held-out gain > 5 with Holm-adjusted p < 0.05 for gain > 0 "
             "(uncorrected interval excluding 0 also reported)",
        tiers_tested=sorted({f["component"].split()[1] for f in h1}),
        uncorrected_verdict="falsified" if any(f["violation_uncorrected"] for f in h1) else "supported",
        note="The protocol trains T2 only at fine q = 0.05, so H1 (fine, q >= 0.10) is tested on T0 and T1.")
    for hyp, thr in (("S5-H2", 0.25), ("S5-H3", 10.0)):
        comps = [f for f in fam if f["hyp"] == hyp]
        for f in comps:
            if f["estimate"] < thr and f["significant_holm"]:
                f["verdict"] = "supported"
            elif f["estimate"] >= thr and f["ci"][0] > thr:
                f["verdict"] = "falsified"
            else:
                f["verdict"] = "inconclusive"
        v = [f["verdict"] for f in comps]
        verdicts[hyp] = dict(
            verdict="falsified" if "falsified" in v else ("supported" if v and all(x == "supported" for x in v)
                                                         else "inconclusive"),
            rule=f"component supported if estimate < {thr} and Holm-adjusted p (share of resamples >= {thr}, doubled) "
                 f"< 0.05; falsified if estimate >= {thr} and the uncorrected interval lies above {thr}; "
                 "hypothesis supported only if every component is",
            components=[f["component"] for f in comps])
    if verdicts["S5-H3"]["components"] and not any("T2" in c for c in verdicts["S5-H3"]["components"]):
        verdicts["S5-H3"]["note"] = "T2 not present in this run; only T1 tested"
    h4 = [f for f in fam if f["hyp"] == "S5-H4"][0]
    verdicts["S5-H4"] = dict(
        verdict="supported" if (h4["estimate"] > 5 and h4["significant_holm"]) else "not supported",
        rule="supported if T1 held-out gain > 5 with Holm-adjusted p < 0.05 for gain > 0 (same convention as the H1 "
             "falsifier)", strict_interval_above_5=bool(h4["ci"][0] > 5))
    no_check = by.get(("none", "T0"), {}).get("cheater_gain")

    # ------------------------------------------------------------ write
    S = dict(protocol="notes/claude_audit_20261005/studies/S5_stronger_attackers/protocol.md", bootstrap=dict(seed=SEED, B=B),
             n_test_contexts=int(E[E.tier == "comply"].groupby("cell").size().min()),
             no_check_T0_gain=no_check, cells=rows, validity=validity, manipulation_check=manip,
             family=fam, verdicts=verdicts,
             searches=[dict(cell=cell_key(s["regime"], s["q"]), T0=dict(d_star=s["T0"]["d_star"], reaction=s["T0"]["reaction"]),
                            T1=dict(levels=s["T1"]["levels"], lie_low=s["T1"]["lie_low"], rounds=s["T1"]["rounds"],
                                    evaluations=s["T1"]["evaluations"])) for s in searches],
             T2_training=[dict(cell=cell_key(l["regime"], l["q"]), best_iteration=l["best_iteration"],
                               best_train_group=l["best_train_group"], episodes_used=l["episodes_used"]) for l in t2log])
    (out / "s5_summary.json").write_text(json.dumps(S, indent=1, default=float))
    flat = lambda d: {k: (v["estimate"] if isinstance(v, dict) and "estimate" in v else v) for k, v in d.items()}
    ci = lambda d, k: {f"{k}_lo": d[k]["ci"][0], f"{k}_hi": d[k]["ci"][1]}
    pd.DataFrame([{**flat(r), **ci(r, "cheater_gain"), **ci(r, "honest_minus_comply"), **ci(r, "train_minus_heldout"),
                   **ci(r, "max_rank_own_gain")}
                  for r in rows]).to_csv(out / "s5_cells.csv", index=False)
    pd.DataFrame([{**flat(v), **ci(v, "gain_difference")} for v in validity]).to_csv(out / "s5_validity.csv", index=False)
    pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in f.items()} for f in fam]).to_csv(
        out / "s5_family.csv", index=False)
    pd.DataFrame([dict(cell=r["cell"], tier=r["tier"], train_gain=r["train_gain"], heldout_gain=r["heldout_gain"],
                       gap=r["train_minus_heldout"]["estimate"], gap_lo=r["train_minus_heldout"]["ci"][0],
                       gap_hi=r["train_minus_heldout"]["ci"][1]) for r in rows]).to_csv(out / "s5_gaps.csv", index=False)
    print(json.dumps(dict(verdicts={k: v["verdict"] for k, v in verdicts.items()}, manipulation=manip), indent=1))


if __name__ == "__main__":
    main()
