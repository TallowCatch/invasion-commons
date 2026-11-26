"""Analysis for experiment R1 (Claude audit, October 2026). Writes CSV tables and summary.json."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

B, SEED = 4000, 20261005


def load(path):
    return pd.DataFrame([json.loads(l) for l in gzip.open(path, "rt")])


def decision_table(df):
    df = df.copy()
    df["risky"] = df.label == "risky"; df["safe"] = df.label == "safe"; df["unres"] = df.label == "unresolved"
    df["approved"] = df.scale >= 1.0
    df["risky_approved"] = df.risky & df.approved
    df["safe_restricted"] = df.safe & ~df.approved
    df["safe_scale"] = np.where(df.safe, df.scale, np.nan)
    keys = ["game", "target", "reviewer", "budget", "fill"]
    g = df.groupby(keys)
    t = g.agg(risky=("risky", "sum"), risky_approved=("risky_approved", "sum"), safe=("safe", "sum"),
              safe_restricted=("safe_restricted", "sum"), unresolved=("unres", "sum"),
              safe_retained=("safe_scale", "mean"), contexts_risky=("context", lambda s: 0)).reset_index()
    t["contexts_risky"] = df[df.risky].groupby(keys).context.nunique().reindex(
        pd.MultiIndex.from_frame(t[keys])).fillna(0).values
    t["unsafe_approval_rate"] = t.risky_approved / t.risky.replace(0, np.nan)
    t["usefulness_loss_rate"] = t.safe_restricted / t.safe.replace(0, np.nan)
    return t, df


def boot_ratio_diff(df, a, b, num, den, rng):
    """Paired context bootstrap of case-weighted rate difference a-b."""
    ctx = sorted(df.context.unique())
    per = {}
    for name in (a, b):
        sub = df[df.cond == name].groupby("context")[[num, den]].sum().reindex(ctx).fillna(0)
        per[name] = sub.values
    est = per[a][:, 0].sum() / max(per[a][:, 1].sum(), 1) - per[b][:, 0].sum() / max(per[b][:, 1].sum(), 1)
    draws = []
    for _ in range(B):
        idx = rng.integers(0, len(ctx), len(ctx))
        x, y = per[a][idx], per[b][idx]
        if x[:, 1].sum() == 0 or y[:, 1].sum() == 0:
            continue
        draws.append(x[:, 0].sum() / x[:, 1].sum() - y[:, 0].sum() / y[:, 1].sum())
    if not draws or per[a][:, 1].sum() == 0:
        return dict(estimate=None, ci=None, note="no cases of this class")
    return dict(estimate=float(est), ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))])


def boot_mean_diff(x, y, rng):
    d = np.asarray(x) - np.asarray(y)
    draws = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return dict(estimate=float(d.mean()), ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ep = load(run / "episodes.jsonl.gz").drop_duplicates(subset=["game", "context", "reviewer", "budget", "fill", "target"])
    opened = load(run / "open_loop_decisions.jsonl.gz")
    closed = load(run / "closed_loop_decisions.jsonl.gz")
    closed = closed[closed.pre_safe == 1].copy()
    closed["target"] = closed["target"].fillna("one_step") if "target" in closed else "one_step"
    ot, odf = decision_table(opened)
    ct, cdf = decision_table(closed)
    ot.to_csv(out / "open_loop_decisions_table.csv", index=False)
    ct.to_csv(out / "closed_loop_decisions_table.csv", index=False)
    outcomes = ep.groupby(["game", "target", "reviewer", "budget", "fill"]).agg(
        contexts=("context", "nunique"), total_harvest=("total_harvest", "mean"),
        mean_health=("mean_health", "mean"), unsafe_fixed=("unsafe_fixed", "mean"),
        failures=("failure", "sum")).reset_index()
    outcomes.to_csv(out / "closed_loop_outcomes.csv", index=False)

    S = {}
    # ---- H1: Harvest open loop, k=6 (both fills identical at k=6; use max)
    def row(t, **kw):
        m = np.ones(len(t), bool)
        for k, v in kw.items():
            m &= (t[k] == v).values
        return t[m].iloc[0].to_dict()
    H = {r: row(ot, game="harvest", reviewer=r, budget=6, fill="max") for r in ("joint", "local_bounded", "local_optimistic")}
    S["H1"] = dict(joint_usefulness_loss=H["joint"]["usefulness_loss_rate"], joint_unsafe_approval=H["joint"]["unsafe_approval_rate"],
                   optimistic_unsafe_approval=H["local_optimistic"]["unsafe_approval_rate"],
                   bounded_usefulness_loss=H["local_bounded"]["usefulness_loss_rate"],
                   counts={r: {k: H[r][k] for k in ("risky", "risky_approved", "safe", "safe_restricted", "unresolved")} for r in H})
    S["H1"]["supported"] = bool(H["joint"]["usefulness_loss_rate"] < .05 and H["joint"]["unsafe_approval_rate"] < .01
                                and H["local_optimistic"]["unsafe_approval_rate"] > .03)
    # ---- H2: Harvest open loop, k=3, previous fill
    H2 = {r: row(ot, game="harvest", reviewer=r, budget=3, fill="previous") for r in ("joint", "local_bounded", "local_optimistic")}
    H2max = {r: row(ot, game="harvest", reviewer=r, budget=3, fill="max") for r in ("joint", "local_bounded", "local_optimistic")}
    S["H2"] = dict(previous={r: H2[r]["usefulness_loss_rate"] for r in H2}, max={r: H2max[r]["usefulness_loss_rate"] for r in H2max},
                   unsafe_previous={r: H2[r]["unsafe_approval_rate"] for r in H2})
    S["H2"]["supported"] = bool(H2["joint"]["usefulness_loss_rate"] < .9 and
                                abs(H2["joint"]["usefulness_loss_rate"] - H2["local_bounded"]["usefulness_loss_rate"]) > .10)
    # ---- H3: Fishery target
    def harv(game, target, rev, k, fill="max"):
        return ep[(ep.game == game) & (ep.target == target) & (ep.reviewer == rev) & (ep.budget == k) & (ep.fill == fill)].sort_values("context")
    f = {(t, k): harv("fishery", t, "joint", k) for t in ("one_step", "msy") for k in (0, 3, 6)}
    S["H3"] = dict(mean_total_harvest={f"{t}_k{k}": float(v.total_harvest.mean()) for (t, k), v in f.items()},
                   mean_stock={f"{t}_k{k}": float(v.mean_health.mean()) for (t, k), v in f.items()},
                   msy_minus_onestep_k6=boot_mean_diff(f[("msy", 6)].total_harvest.values, f[("one_step", 6)].total_harvest.values, rng),
                   msy_k6_minus_k0=boot_mean_diff(f[("msy", 6)].total_harvest.values, f[("msy", 0)].total_harvest.values, rng),
                   onestep_k6_minus_k0=boot_mean_diff(f[("one_step", 6)].total_harvest.values, f[("one_step", 0)].total_harvest.values, rng))
    S["H3"]["supported"] = bool(S["H3"]["msy_minus_onestep_k6"]["estimate"] > 0 and
                                f[("msy", 6)].total_harvest.mean() >= 0.98 * f[("msy", 0)].total_harvest.mean())
    # ---- H4: closed-loop safety
    def uf(game, rev, k=6, target="one_step", fill="max"):
        return float(harv(game, target, rev, k, fill).unsafe_fixed.mean())
    S["H4"] = {f"{g}_{r}": uf(g, r) for g in ("fishery", "harvest") for r in ("joint", "local_bounded", "local_optimistic", "none") if not (r == "none")}
    S["H4"]["fishery_none"] = float(ep[(ep.game == "fishery") & (ep.reviewer == "none")].unsafe_fixed.mean())
    S["H4"]["harvest_none"] = float(ep[(ep.game == "harvest") & (ep.reviewer == "none")].unsafe_fixed.mean())
    S["H4"]["supported"] = bool(all(S["H4"][f"{g}_{r}"] <= .01 for g in ("fishery", "harvest") for r in ("joint", "local_bounded"))
                                and S["H4"]["fishery_local_optimistic"] > .10)
    # ---- paired bootstrap contrasts (open loop, k=6, fill max)
    for game, target in (("harvest", "one_step"), ("fishery", "one_step"), ("fishery", "msy")):
        sub = odf[(odf.game == game) & (odf.target == target) & (odf.budget == 6) & (odf.fill == "max")].copy()
        sub["cond"] = sub.reviewer
        for other in ("local_bounded", "local_optimistic"):
            S[f"{game}_{target}_k6_joint_minus_{other}"] = dict(
                usefulness_loss=boot_ratio_diff(sub, "joint", other, "safe_restricted", "safe", rng),
                unsafe_approval=boot_ratio_diff(sub, "joint", other, "risky_approved", "risky", rng))
    # long-run paired harvest differences, k=6, fill max
    for game, target in (("harvest", "one_step"), ("fishery", "one_step"), ("fishery", "msy")):
        j = harv(game, target, "joint", 6)
        for other in ("local_bounded", "local_optimistic"):
            o = harv(game, target, other, 6)
            S[f"{game}_{target}_k6_longrun_joint_minus_{other}"] = dict(
                total_harvest=boot_mean_diff(j.total_harvest.values, o.total_harvest.values, rng),
                mean_health=boot_mean_diff(j.mean_health.values, o.mean_health.values, rng))
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps({k: v.get("supported") for k, v in S.items() if isinstance(v, dict) and "supported" in v}))


if __name__ == "__main__":
    main()
