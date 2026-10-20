"""Analysis for experiment R2 (R2 protocol).

Paired context bootstrap (4,000 resamples, seed 20261012, 95% percentile intervals). Every
statistic is a function of per-context sums, and every statistic over the same number of
contexts uses the SAME resample matrix. Bootstrap p-value: the share of valid resamples on
the wrong side of 0, doubled (capped at 1). Holm's method is applied within family A and
within family B over every component test that has a p-value and is testable.

Verdicts per component (all reported):
- ``verdict``            primary: estimate on the predicted side AND Holm-adjusted p < 0.05;
- ``verdict_interval``   uncorrected: 95% interval entirely on the predicted side;
- ``verdict_point``      the point estimate alone on the predicted side;
- ``verdict_as_worded``  the protocol's falsifier wording read literally, without multiplicity
                         correction: the uncorrected interval for A-H2 and A-H3 ("the paired
                         interval includes 0"), the point estimate for A-H1, B-H1, B-H2, B-H4, B-H5.
Deterministic checks (A-H4, B-H3) have no p-value and are decided exactly. B-H3's verdict is its declared
falsifier (no MSY decision differs between exact and a regrowth condition); the hypothesis's other statements
(regrowth changes one-step decisions; K changes both) are reported on a separate "B-H3 (secondary ...)" line.
"not_testable": the measure cannot be computed (no risky / safe cases, no gain from cheating,
or nothing for memory to remove: the no-memory arm was never unsafe).

Run:  PYTHONPATH=. python -m experiments.oversight.analyze_r2 RUN_DIR   (RUN_DIR holds partA/ and/or partB/)
Writes RUN_DIR/analysis/r2_summary.json and tidy CSV tables.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

B, SEED, ALPHA = 4000, 20261012, 0.05
INTERVAL_WORDED = ("A-H2", "A-H3")
VERDICTS = ("verdict", "verdict_interval", "verdict_point", "verdict_as_worded")
REVIEWERS = ("joint", "local_bounded", "local_optimistic")


def load(path):
    with gzip.open(path, "rt") as f:
        return pd.DataFrame([json.loads(line) for line in f])


# ------------------------------------------------------------------ bootstrap
class Boot:
    """Resample counts for n contexts: C[b, i] = times context i is drawn in resample b."""
    _cache: dict[int, np.ndarray] = {}

    @classmethod
    def counts(cls, n):
        if n not in cls._cache:
            idx = np.random.default_rng(SEED).integers(0, n, (B, n))
            C = np.zeros((B, n))
            for b in range(B):
                C[b] = np.bincount(idx[b], minlength=n)
            cls._cache[n] = C
        return cls._cache[n]


def boot(X, f):
    """X: (n contexts, m) per-context values; f maps column sums (..., m) and n to a statistic."""
    X = np.asarray(X, float)
    n = X.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        est = float(f(X.sum(0), n))
        draws = np.asarray(f(Boot.counts(n) @ X, n), float)
    valid = draws[np.isfinite(draws)]
    ci = [float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))] if len(valid) else [None, None]
    return dict(estimate=est if np.isfinite(est) else None, ci=ci, draws=valid, n_contexts=n, n_valid=int(len(valid)))


def ratio(s, n, i=0, j=1):
    return s[..., i] / s[..., j]


def test(family, hyp, X, f, strict=True, testable=True, note="", **ident):
    """Component test of the prediction theta > 0 (strict) or theta >= 0, theta = f(sums)."""
    row = dict(family=family, hypothesis=hyp, **ident, strict=strict, testable=bool(testable), note=note,
               estimate=None, ci_low=None, ci_high=None, p_boot=None, n_contexts=None, n_valid_resamples=None)
    if testable:
        r = boot(X, f)
        if r["estimate"] is None or r["n_valid"] == 0:
            row.update(testable=False, note=(note + "; " if note else "") + "statistic undefined")
        else:
            d = r["draws"]
            wrong = np.mean(d <= 0) if strict else np.mean(d < 0)
            row.update(estimate=r["estimate"], ci_low=r["ci"][0], ci_high=r["ci"][1], p_boot=float(min(1.0, 2 * wrong)),
                       n_contexts=r["n_contexts"], n_valid_resamples=r["n_valid"])
    return row


def contrast(X, f, **ident):
    r = boot(X, f)
    return dict(**ident, estimate=r["estimate"], ci_low=r["ci"][0], ci_high=r["ci"][1], n_contexts=r["n_contexts"])


def holm(rows):
    idx = [i for i, r in enumerate(rows) if r.get("testable") and r.get("p_boot") is not None]
    order = sorted(idx, key=lambda i: rows[i]["p_boot"])
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * rows[i]["p_boot"]))
        rows[i]["p_holm"] = running
    for r in rows:
        r.setdefault("p_holm", None)
        if not r.get("testable"):
            r.update({v: "not_testable" for v in VERDICTS})
            continue
        if r.get("deterministic"):
            r["verdict_as_worded"] = r["verdict"]
            continue
        ok = (r["estimate"] > 0) if r["strict"] else (r["estimate"] >= 0)
        lo_ok = (r["ci_low"] > 0) if r["strict"] else (r["ci_low"] >= 0)
        r["verdict_point"] = "pass" if ok else "fail"
        r["verdict_interval"] = "pass" if lo_ok else "fail"
        r["verdict"] = "pass" if ok and r["p_holm"] < ALPHA else "fail"
        r["verdict_as_worded"] = r["verdict_interval"] if r["hypothesis"] in INTERVAL_WORDED else r["verdict_point"]
    return rows


def roll_up(rows):
    """Per hypothesis x cell, then per hypothesis: fail if any testable component fails."""
    out = []
    keyed = {}
    for r in rows:
        keyed.setdefault((r["family"], r["hypothesis"], r.get("cell")), []).append(r)
    for (fam, hyp, cell), comps in keyed.items():
        rec = dict(family=fam, hypothesis=hyp, cell=cell, components=len(comps))
        for v in VERDICTS:
            vs = [c[v] for c in comps]
            rec[v] = "fail" if "fail" in vs else ("not_testable" if "not_testable" in vs else "pass")
        out.append(rec)
    overall = []
    for (fam, hyp) in sorted({(r["family"], r["hypothesis"]) for r in out}):
        cells = [r for r in out if r["family"] == fam and r["hypothesis"] == hyp]
        rec = dict(family=fam, hypothesis=hyp, cells=len(cells))
        for v in VERDICTS:
            vs = [c[v] for c in cells]
            rec[v] = "fail" if "fail" in vs else ("pass" if "pass" in vs else "not_testable")
            rec[f"{v}_cells"] = {k: vs.count(k) for k in ("pass", "fail", "not_testable")}
        overall.append(rec)
    return out, overall


# ------------------------------------------------------------------ helpers
def decision_flags(df):
    df = df.copy()
    df["risky"] = (df.label == "risky").astype(int)
    df["safe"] = (df.label == "safe").astype(int)
    df["unresolved"] = (df.label == "unresolved").astype(int)
    df["approved"] = (df.scale >= 1.0).astype(int)
    df["risky_approved"] = df.risky * df.approved
    df["safe_restricted"] = df.safe * (1 - df.approved)
    return df


def per_context(df, cols, n):
    g = df.groupby("context")[list(cols)].sum().reindex(range(n)).fillna(0)
    return g.values


def decision_table(df, keys, n):
    rows = []
    for k, g in df.groupby(keys, dropna=False):
        k = k if isinstance(k, tuple) else (k,)
        rec = dict(zip(keys, k))
        rec.update(cases=len(g), risky=int(g.risky.sum()), risky_approved=int(g.risky_approved.sum()),
                   safe=int(g.safe.sum()), safe_restricted=int(g.safe_restricted.sum()), unresolved=int(g.unresolved.sum()),
                   contexts_with_risky=int(g[g.risky == 1].context.nunique()),
                   contexts_with_safe=int(g[g.safe == 1].context.nunique()))
        rec["unsafe_approval_rate"] = rec["risky_approved"] / rec["risky"] if rec["risky"] else None
        rec["usefulness_loss_rate"] = rec["safe_restricted"] / rec["safe"] if rec["safe"] else None
        rec["safe_scale_mean"] = float(g[g.safe == 1].scale.mean()) if rec["safe"] else None
        rec["descriptive_only"] = rec["contexts_with_risky"] < 20
        rows.append(rec)
    return pd.DataFrame(rows)


def sel(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).values if v is not None else df[k].isna().values
    return df[m]


def by_context(df, col, n):
    s = df.set_index("context")[col].reindex(range(n))
    if s.isna().any():
        raise ValueError(f"missing contexts for {col}")
    return s.values.astype(float)


def mean_diff(n):
    return lambda s, _n: s[..., 0] / _n - s[..., 1] / _n


# ================================================================== Part A
def part_a(run, out):
    man = json.loads((run / "manifest.json").read_text())
    n = man["params"]["contexts"]
    E = load(run / "episodes.jsonl.gz")
    O = decision_flags(load(run / "open_loop.jsonl.gz"))
    Cr = decision_flags(load(run / "closed_rows.jsonl.gz"))
    D = json.loads((run / "deterrence.json").read_text())
    cells = man["cells"]
    tests, contrasts, S = [], [], dict(cells=cells, n_contexts=n)

    # ---- tables
    ot = decision_table(O, ["cell", "target", "reviewer"], n)
    ot.to_csv(out / "A_open_loop.csv", index=False)
    ct = decision_table(Cr[Cr.pre_safe == 1], ["cell", "target", "reviewer", "budget"], n)
    ct.to_csv(out / "A_closed_loop_decisions.csv", index=False)
    C = E[E.kind == "closed"].copy()
    C["target"] = C["target"].fillna("none")
    co = C.groupby(["cell", "game", "target", "reviewer", "budget"]).agg(
        contexts=("context", "nunique"), total_harvest=("total_harvest", "mean"), mean_health=("mean_health", "mean"),
        unsafe_fixed=("unsafe_fixed", "mean"), collapse_rate=("failure", "mean"), collapses=("failure", "sum")).reset_index()
    co.to_csv(out / "A_closed_loop_outcomes.csv", index=False)
    M = E[E.kind == "memory"].copy()
    mt = M.groupby(["cell", "game", "target", "mode", "liar"]).agg(
        contexts=("context", "nunique"), unsafe_fixed=("unsafe_fixed", "mean"), exec_risky=("exec_risky", "sum"),
        scored_steps=("scored_steps", "sum"), useful_loss=("useful_loss", "sum"), req_safe=("req_safe", "sum"),
        honest_harvest=("honest_harvest", "mean"), total_harvest=("total_harvest", "mean"),
        catches=("catches", "mean"), collapse_rate=("failure", "mean")).reset_index()
    mt["target_breaking_rate"] = mt.exec_risky / mt.scored_steps.clip(lower=1)
    mt["usefulness_loss"] = mt.useful_loss / mt.req_safe.clip(lower=1)
    mt.to_csv(out / "A_memory.csv", index=False)

    # ---- A-H1 (Harvest, open loop, k = 6)
    for c in [c for c in cells if c["game"] == "harvest"]:
        sub = sel(O, cell=c["cell"])
        j, o = sel(sub, reviewer="joint"), sel(sub, reviewer="local_optimistic")
        Xj, Xo = per_context(j, ["risky_approved", "risky"], n), per_context(o, ["risky_approved", "risky"], n)
        has = Xj[:, 1].sum() > 0
        tests.append(test("A", "A-H1", Xj, lambda s, _n: 0.02 - ratio(s, _n), True, has,
                          "theta = 0.02 - joint unsafe-approval rate" + ("" if has else "; no risky requests"),
                          cell=c["cell"], component="joint_unsafe_approval_below_2pct", piloted=c["piloted"]))
        tests.append(test("A", "A-H1", np.hstack([Xo, Xj]), lambda s, _n: s[..., 0] / s[..., 1] - s[..., 2] / s[..., 3],
                          True, has, "theta = optimistic - joint unsafe-approval rate" + ("" if has else "; no risky requests"),
                          cell=c["cell"], component="optimistic_above_joint", piloted=c["piloted"]))
    # exploratory open-loop reviewer contrasts in every cell
    for (cell, target), sub in O.groupby(["cell", "target"]):
        j = sub[sub.reviewer == "joint"]
        for other in ("local_bounded", "local_optimistic"):
            o = sub[sub.reviewer == other]
            for num, den, name in (("risky_approved", "risky", "unsafe_approval"), ("safe_restricted", "safe", "usefulness_loss")):
                X = np.hstack([per_context(j, [num, den], n), per_context(o, [num, den], n)])
                if X[:, 1].sum() and X[:, 3].sum():
                    contrasts.append(contrast(X, lambda s, _n: s[..., 0] / s[..., 1] - s[..., 2] / s[..., 3], part="A",
                                              measure=f"open_loop_{name}", cell=cell, target=target,
                                              contrast=f"joint - {other}", exploratory=True))

    # ---- A-H2 (Fishery, joint k = 6: MSY - one-step total harvest)
    for c in [c for c in cells if c["game"] == "fishery"]:
        jm = sel(C, cell=c["cell"], reviewer="joint", budget=6, target="msy")
        jo = sel(C, cell=c["cell"], reviewer="joint", budget=6, target="one_step")
        X = np.column_stack([by_context(jm, "total_harvest", n), by_context(jo, "total_harvest", n)])
        tests.append(test("A", "A-H2", X, mean_diff(n), True, True, "theta = total harvest, MSY - one-step (joint, k = 6)",
                          cell=c["cell"], component="msy_minus_one_step_harvest", piloted=c["piloted"]))
    # exploratory closed-loop contrasts: k = 6 - k = 0 per cell/target
    for (cell, target), sub in C[C.reviewer == "joint"].groupby(["cell", "target"]):
        for col in ("total_harvest", "unsafe_fixed"):
            X = np.column_stack([by_context(sel(sub, budget=6), col, n), by_context(sel(sub, budget=0), col, n)])
            contrasts.append(contrast(X, mean_diff(n), part="A", measure=f"closed_loop_{col}", cell=cell, target=target,
                                      contrast="joint k=6 - k=0", exploratory=True))

    # ---- A-H3 (memory - memoryless, fixed liars) and exploratory noisy / trust contrasts
    for c in cells:
        sub = sel(M, cell=c["cell"])
        for liar in ("fixed", "noisy"):
            arms = {m: sel(sub, mode=m, liar=liar) for m in ("trust", "memoryless", "memory")}
            if c["game"] == "harvest":
                metric = "unsafe_fixed (share of horizon unsafe)"
                cols = {m: by_context(a, "unsafe_fixed", n)[:, None] for m, a in arms.items()}
                f = lambda s, _n: s[..., 0] / _n - s[..., 1] / _n
            else:
                metric = "MSY-target-breaking executed actions / pre-safe steps (as S3 Part D)"
                cols = {m: per_context(a, ["exec_risky", "scored_steps"], n) for m, a in arms.items()}
                f = lambda s, _n: s[..., 0] / s[..., 1] - s[..., 2] / s[..., 3]
            nothing = cols["memoryless"][:, 0].sum() == 0
            if liar == "fixed":
                tests.append(test("A", "A-H3", np.hstack([cols["memoryless"], cols["memory"]]), f, True, not nothing,
                                  f"theta = memoryless - memory; {metric}" + ("; no-memory arm never unsafe" if nothing else ""),
                                  cell=c["cell"], component="memory_below_memoryless_fixed_liars", piloted=c["piloted"]))
            for a, b_ in (("memory", "memoryless"), ("memory", "trust"), ("memoryless", "trust")):
                contrasts.append(contrast(np.hstack([cols[a], cols[b_]]), f, part="A", measure=f"memory_{metric.split(' ')[0]}",
                                          cell=c["cell"], target="one_step" if c["game"] == "harvest" else "msy",
                                          contrast=f"{a} - {b_} ({liar} liars)", exploratory=liar != "fixed" or a != "memory" or b_ != "memoryless"))

    # ---- A-H4 (deterrence threshold vs G) -- deterministic
    drows = []
    De = E[E.kind == "deter"]
    for d in D:
        step = 2 * d["G"] / 10 if d["testable"] else None
        rec = dict(family="A", hypothesis="A-H4", cell=d["cell"], component="e_star_within_one_grid_step_of_G",
                   piloted=d["piloted"], deterministic=True, testable=bool(d["testable"]), strict=False,
                   G=d["G"], G_test=d["G_test"], grid_step=step, e_star=d["e_star"],
                   over_steps_train=d["over_steps_train"], over_steps_test=d["over_steps_test"],
                   G_per_overtake_step=(d["G"] * man["params"]["horizon"] / d["over_steps_train"]) if d["over_steps_train"] else None,
                   p_boot=None, note="no gain from cheating (G <= 0)" if not d["testable"] else "")
        ex = d.get("exploratory_allow", {})  # exploratory, not used for the verdict
        rec.update(expl_d_allow=ex.get("d_allow"), expl_G_at_d_allow=ex.get("G_at_d_allow"),
                   expl_G_per_over_step_at_0_75=ex.get("G_per_over_step_at_0_75"),
                   expl_G_per_over_step_at_d_allow=ex.get("G_per_over_step_at_d_allow"),
                   expl_e_star_extended=d.get("e_star_extended"))
        if d["testable"]:
            if d["e_star"] is None:
                rec.update(abs_e_star_minus_G=None, estimate=None,
                           note="no e in [0, 2G] gave d* = 0 (e* > 2G, more than one grid step above G)")
                ok = False
            else:
                gap = abs(d["e_star"] - d["G"])
                rec.update(abs_e_star_minus_G=gap, estimate=step - gap)
                ok = gap <= step * (1 + 1e-9)
            rec.update(verdict="pass" if ok else "fail", verdict_interval="pass" if ok else "fail",
                       verdict_point="pass" if ok else "fail")
        tests.append(rec)
        comply = sel(De, cell=d["cell"], arm="comply").sort_values("context")
        for srch in d["searches"]:
            a = De[(De.cell == d["cell"]) & (De.arm == "adaptive") & np.isclose(De.e.astype(float), srch["e"])].sort_values("context")
            X = np.column_stack([a.cheater_payoff.values / a.n_cheaters.values, comply.cheater_payoff.values / comply.n_cheaters.values])
            g = contrast(X, mean_diff(n))
            drows.append(dict(cell=d["cell"], G=d["G"], e=srch["e"], fine=srch["fine"], d_star=srch["d_star"],
                              e_over_G=srch["e"] / d["G"], heldout_gain_per_cheater=g["estimate"],
                              gain_ci_low=g["ci_low"], gain_ci_high=g["ci_high"],
                              catches_per_context=float(a.catches.mean()),
                              target_breaking_rate=float(a.exec_risky.sum() / max(a.scored_steps.sum(), 1)),
                              collapse_rate=float(a.failure.mean()),
                              train_scores=json.dumps(srch["train_scores"])))
        if not d["searches"]:
            drows.append(dict(cell=d["cell"], G=d["G"], e=None, d_star=None))
    pd.DataFrame(drows).to_csv(out / "A_deterrence.csv", index=False)
    S["deterrence"] = [{k: v for k, v in d.items() if k not in ("searches", "extended_searches")}
                       | dict(d_star_by_e=[(s["e"], s["d_star"]) for s in d["searches"]],
                              exploratory_d_star_by_e_beyond_2G=[(s["e"], s["d_star"]) for s in d.get("extended_searches", [])])
                       for d in D]
    S["tables"] = dict(open_loop=ot.to_dict("records"), closed_loop_outcomes=co.to_dict("records"),
                       memory=mt.to_dict("records"))
    return tests, contrasts, S


# ================================================================== Part B
def part_b(run, out):
    man = json.loads((run / "manifest.json").read_text())
    n, horizon = man["params"]["contexts"], man["params"]["horizon"]
    E = load(run / "episodes.jsonl.gz")
    O = decision_flags(load(run / "open_loop.jsonl.gz"))
    Cr = decision_flags(load(run / "closed_rows.jsonl.gz"))
    cells = man["cells"]
    tests, contrasts, S = [], [], dict(cells=cells, n_contexts=n)

    ot = decision_table(O, ["cell", "condition", "target", "reviewer"], n)
    ot.to_csv(out / "B_open_loop.csv", index=False)
    ct = decision_table(Cr[Cr.pre_safe == 1], ["cell", "condition", "target"], n)
    C = E[~E.condition.isin(["none", "none_allee"])].copy()
    co = C.groupby(["cell", "game", "condition", "target"]).agg(
        contexts=("context", "nunique"), total_harvest=("total_harvest", "mean"), mean_health=("mean_health", "mean"),
        unsafe_fixed=("unsafe_fixed", "mean"), collapse_rate=("failure", "mean"), collapses=("failure", "sum")).reset_index()
    co = co.merge(ct[["cell", "condition", "target", "risky", "risky_approved", "safe", "safe_restricted",
                      "unsafe_approval_rate", "usefulness_loss_rate"]], on=["cell", "condition", "target"], how="left")
    co.to_csv(out / "B_closed_loop.csv", index=False)
    nr = E[E.condition.isin(["none", "none_allee"])].groupby(["cell", "condition"]).agg(
        total_harvest=("total_harvest", "mean"), unsafe_fixed=("unsafe_fixed", "mean"),
        collapse_rate=("failure", "mean")).reset_index()
    nr.to_csv(out / "B_no_reviewer.csv", index=False)

    # ---- B-H1 (Harvest, open loop, joint, k = 6)
    for c in [c for c in cells if c["game"] == "harvest"]:
        j = {cond: sel(O, cell=c["cell"], reviewer="joint", condition=cond) for cond in ("exact", "noise_low", "noise_high")}
        X = per_context(j["noise_low"], ["risky_approved", "risky"], n)
        has = X[:, 1].sum() > 0
        tests.append(test("B", "B-H1", X, lambda s, _n: ratio(s, _n) - 0.02, True, has,
                          "theta = noise_low joint unsafe-approval rate - 0.02" + ("" if has else "; no risky requests"),
                          cell=c["cell"], condition="noise_low", component="noise_low_unsafe_approval_above_2pct"))
        X = np.hstack([per_context(j["noise_high"], ["safe_restricted", "safe"], n),
                       per_context(j["exact"], ["safe_restricted", "safe"], n)])
        has = X[:, 1].sum() > 0
        tests.append(test("B", "B-H1", X, lambda s, _n: s[..., 0] / s[..., 1] - s[..., 2] / s[..., 3] - 0.10, False, has,
                          "theta = usefulness loss (noise_high - exact) - 0.10" + ("" if has else "; no safe requests"),
                          cell=c["cell"], condition="noise_high", component="noise_high_usefulness_loss_up_10_points"))

    # ---- B-H2 (open loop: optimistic - joint unsafe approvals, every cell x condition x target)
    for (cell, cond, target), sub in O.groupby(["cell", "condition", "target"]):
        Xj = per_context(sub[sub.reviewer == "joint"], ["risky_approved", "risky"], n)
        Xo = per_context(sub[sub.reviewer == "local_optimistic"], ["risky_approved", "risky"], n)
        has = Xj[:, 1].sum() > 0
        tests.append(test("B", "B-H2", np.hstack([Xo, Xj]), lambda s, _n: s[..., 0] / s[..., 1] - s[..., 2] / s[..., 3],
                          True, has, "theta = optimistic - joint unsafe-approval rate" + ("" if has else "; no risky requests"),
                          cell=cell, condition=cond, target=target, component=f"optimistic_above_joint[{cond},{target}]"))
    # exploratory: every condition - exact, per reviewer (open loop) and joint (closed loop)
    for (cell, target, rev), sub in O.groupby(["cell", "target", "reviewer"]):
        ex = sub[sub.condition == "exact"]
        for cond in sorted(set(sub.condition) - {"exact", "allee"}):
            cs = sub[sub.condition == cond]
            for num, den, name in (("risky_approved", "risky", "unsafe_approval"), ("safe_restricted", "safe", "usefulness_loss")):
                X = np.hstack([per_context(cs, [num, den], n), per_context(ex, [num, den], n)])
                if X[:, 1].sum() and X[:, 3].sum():
                    contrasts.append(contrast(X, lambda s, _n: s[..., 0] / s[..., 1] - s[..., 2] / s[..., 3], part="B",
                                              measure=f"open_loop_{name}", cell=cell, target=target, reviewer=rev,
                                              contrast=f"{cond} - exact", exploratory=True))
    for (cell, target), sub in C.groupby(["cell", "target"]):
        ex = sel(sub, condition="exact")
        for cond in sorted(set(sub.condition) - {"exact"}):
            for col in ("total_harvest", "unsafe_fixed", "failure"):
                X = np.column_stack([by_context(sel(sub, condition=cond), col, n), by_context(ex, col, n)])
                contrasts.append(contrast(X, mean_diff(n), part="B", measure=f"closed_loop_{col}", cell=cell, target=target,
                                          reviewer="joint", contrast=f"{cond} - exact", exploratory=True))

    # ---- B-H3 (Fishery decision changes) -- deterministic code check
    diffs = []
    for c in [c for c in cells if c["game"] == "fishery"]:
        for cond in ("regen_low", "regen_high", "K_low", "K_high"):
            for target in ("one_step", "msy"):
                for rev in REVIEWERS:
                    a = sel(O, cell=c["cell"], condition=cond, target=target, reviewer=rev).set_index(["context", "step"]).scale
                    b = sel(O, cell=c["cell"], condition="exact", target=target, reviewer=rev).set_index(["context", "step"]).scale
                    a, b = a.align(b, join="inner")
                    diffs.append(dict(cell=c["cell"], condition=cond, target=target, reviewer=rev, loop="open",
                                      compared=len(a), differing=int((a != b).sum()), contexts_differing=None))
                a = sel(Cr, cell=c["cell"], condition=cond, target=target)
                b = sel(Cr, cell=c["cell"], condition="exact", target=target)
                ctx_diff, compared, differing = 0, 0, 0
                for ctx in range(n):
                    sa = a[a.context == ctx].sort_values("step").scale.values
                    sb = b[b.context == ctx].sort_values("step").scale.values
                    m = min(len(sa), len(sb))
                    dd = int((sa[:m] != sb[:m]).sum()) + abs(len(sa) - len(sb))
                    compared += max(len(sa), len(sb)); differing += dd; ctx_diff += int(dd > 0)
                diffs.append(dict(cell=c["cell"], condition=cond, target=target, reviewer="joint", loop="closed",
                                  compared=compared, differing=differing, contexts_differing=ctx_diff))
    Dd = pd.DataFrame(diffs)
    Dd.to_csv(out / "B_decision_diffs.csv", index=False)
    for c in [c for c in cells if c["game"] == "fishery"]:
        d = Dd[Dd.cell == c["cell"]]
        checks = [
            ("msy_unchanged_by_regen", "falsifier: no MSY decision (any reviewer, open or closed loop) differs between exact and regen_low/high",
             int(d[d.condition.isin(["regen_low", "regen_high"]) & (d.target == "msy")].differing.sum()) == 0),
        ]
        for cond in ("regen_low", "regen_high"):
            checks.append((f"{cond}_changes_one_step", f"at least one joint one-step decision differs under {cond}",
                           int(d[(d.condition == cond) & (d.target == "one_step") & (d.reviewer == "joint")].differing.sum()) > 0))
        for cond in ("K_low", "K_high"):
            for target in ("one_step", "msy"):
                checks.append((f"{cond}_changes_{target}", f"at least one joint {target} decision differs under {cond}",
                               int(d[(d.condition == cond) & (d.target == target) & (d.reviewer == "joint")].differing.sum()) > 0))
        for comp, note, ok in checks:
            v = "pass" if ok else "fail"
            # the protocol's falsifier is the MSY check only; the stated "changes" are reported as a secondary line
            hyp = "B-H3" if comp == "msy_unchanged_by_regen" else "B-H3 (secondary: stated changes, not the falsifier)"
            tests.append(dict(family="B", hypothesis=hyp, cell=c["cell"], component=comp, note=note, deterministic=True,
                              testable=True, strict=False, p_boot=None, verdict=v, verdict_interval=v, verdict_point=v))

    # ---- B-H4 (learned): early vs late closed-loop unsafe approvals; long-run harvest within 5% of exact
    wrows = []
    late0 = horizon - 20
    for (cell, target), sub in Cr[(Cr.condition == "learned") & (Cr.pre_safe == 1)].groupby(["cell", "target"]):
        early, late = sub[sub.step < 20], sub[sub.step >= late0]
        X = np.hstack([per_context(early, ["risky_approved", "risky"], n), per_context(late, ["risky_approved", "risky"], n)])
        has = X[:, 1].sum() > 0 and X[:, 3].sum() > 0
        tests.append(test("B", "B-H4", X, lambda s, _n: s[..., 0] / s[..., 1] - s[..., 2] / s[..., 3], True, has,
                          f"theta = unsafe-approval rate steps 0-19 - steps {late0}-{horizon - 1} (closed loop, joint k = 6)"
                          + ("" if has else "; no risky requests in one window"),
                          cell=cell, condition="learned", target=target, component=f"late_below_early[{target}]"))
        wrows.append(dict(cell=cell, target=target, early_risky=int(X[:, 1].sum()), early_risky_approved=int(X[:, 0].sum()),
                          late_risky=int(X[:, 3].sum()), late_risky_approved=int(X[:, 2].sum()),
                          early_rate=X[:, 0].sum() / X[:, 1].sum() if X[:, 1].sum() else None,
                          late_rate=X[:, 2].sum() / X[:, 3].sum() if X[:, 3].sum() else None))
    for (cell, target), sub in C.groupby(["cell", "target"]):
        X = np.column_stack([by_context(sel(sub, condition="learned"), "total_harvest", n),
                             by_context(sel(sub, condition="exact"), "total_harvest", n)])
        tests.append(test("B", "B-H4", X, lambda s, _n: 0.05 - np.abs(s[..., 0] / s[..., 1] - 1), True, True,
                          "theta = 0.05 - |mean harvest learned / exact - 1|", cell=cell, condition="learned", target=target,
                          component=f"harvest_within_5pct[{target}]"))
        r = sel(sub, condition="learned").total_harvest.mean() / sel(sub, condition="exact").total_harvest.mean()
        for w in wrows:
            if w["cell"] == cell and w["target"] == target:
                w["harvest_ratio_learned_over_exact"] = float(r)
    # learned model trajectory (exploratory)
    L = Cr[Cr.condition == "learned"]
    if "model" in L:
        params = pd.json_normalize(L["model"].tolist()).set_index(L.index)
        L = pd.concat([L[["cell", "target", "step"]], params], axis=1)
        L.groupby(["cell", "target", "step"]).mean(numeric_only=True).reset_index().to_csv(out / "B_learned_model_by_step.csv", index=False)
    pd.DataFrame(wrows).to_csv(out / "B_learned_windows.csv", index=False)

    # ---- B-H5 (Allee): collapse >= 25% under one-step, < 5% under MSY (joint, k = 6)
    for c in [c for c in cells if c["game"] == "fishery"]:
        for target, thr, strict, comp in (("one_step", 0.25, False, "collapse_at_least_25pct_one_step"),
                                          ("msy", 0.05, True, "collapse_below_5pct_msy")):
            x = by_context(sel(C, cell=c["cell"], condition="allee", target=target), "failure", n)[:, None]
            f = (lambda s, _n: s[..., 0] / _n - 0.25) if target == "one_step" else (lambda s, _n: 0.05 - s[..., 0] / _n)
            tests.append(test("B", "B-H5", x, f, strict, True,
                              f"theta = {'collapse share - 0.25' if target == 'one_step' else '0.05 - collapse share'}",
                              cell=c["cell"], condition="allee", target=target, component=comp))
    S["tables"] = dict(open_loop=ot.to_dict("records"), closed_loop=co.to_dict("records"), decision_diffs=diffs,
                       learned_windows=wrows, no_reviewer=nr.to_dict("records"))
    return tests, contrasts, S


# ================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run directory containing partA/ and/or partB/")
    a = ap.parse_args(argv)
    run = Path(a.run)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    summary = dict(protocol="notes/claude_audit_20261005/studies/R2_robustness_and_reviewer_model/protocol.md",
                   bootstrap=dict(resamples=B, seed=SEED, interval="95% percentile", unit="context (paired)"),
                   multiplicity="Holm within family A and within family B, bootstrap p = 2 x share of resamples on the wrong side of 0",
                   verdict_rules=__doc__, parts={})
    all_tests, all_contrasts = [], []
    for part, fn in (("A", part_a), ("B", part_b)):
        d = run / f"part{part}"
        if not (d / "manifest.json").exists():
            summary["parts"][part] = "missing"
            continue
        tests, contrasts, S = fn(d, out)
        holm(tests)
        all_tests += tests
        all_contrasts += contrasts
        S["manifest"] = json.loads((d / "manifest.json").read_text())
        summary["parts"][part] = S
    by_cell, overall = roll_up(all_tests)
    T = pd.DataFrame(all_tests)
    T.to_csv(out / "hypothesis_tests.csv", index=False)
    pd.DataFrame(by_cell).to_csv(out / "hypothesis_verdicts_by_cell.csv", index=False)
    pd.DataFrame(overall).to_csv(out / "hypothesis_verdicts.csv", index=False)
    pd.DataFrame(all_contrasts).to_csv(out / "contrasts.csv", index=False)
    summary.update(hypothesis_verdicts=overall, hypothesis_verdicts_by_cell=by_cell, component_tests=all_tests,
                   contrasts=all_contrasts)
    (out / "r2_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    for r in overall:
        print(f"{r['hypothesis']:6s} verdict={r['verdict']:13s} cells={r['verdict_cells']}  "
              f"(as worded: {r['verdict_as_worded']}, uncorrected interval: {r['verdict_interval']}, point: {r['verdict_point']})")


if __name__ == "__main__":
    main()
