"""Fit situation-dependent miss and false-alarm rates to an LLM inspector's verdicts (error_patterns/protocol.md §3).

  PYTHONPATH=. .venv-socialjax/bin/python -m experiments.llm_overseer.fit_error_model \
      --judged gpt-oss-20b_direct qwen2.5-7b_reason

For each inspector, separately for violations (alpha = P(no flag)) and honest eats (beta = P(flag)):
  M0 constant (the coin flip); M1 one rate per `near` bin; M2 logistic regression on near bins + outer + players +
  walls; M3 the trainer's table, near bins x outer bins (outer split only if M2's outer coefficient is clearly
  non-zero, |z| > 2; otherwise M3 = M1).
Held-out log-likelihood per item by 5-fold cross-validation (seeded folds), against M0, with a paired bootstrap 95%
interval of the mean difference; Brier scores; per-cell rates with Wilson intervals; the matched coin flip (rates
averaged over the bank at its natural frequencies, and over the overseen policies only).
Writes <out-dir>/tables/<tag>.json (the trainer's --inspector-table, with the coin flip and fit summary as metadata),
<out-dir>/fit.json and <out-dir>/fit_tables.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from overseer.llm.metrics import wilson
from overseer.llm.subset import load_jsonl

NEAR_BINS = {1: ((0,), (1,), (2,)), 0: ((3,), (4,), (5,), (6,), (7, 8), (9, 10, 11, 12))}
NEAR_EDGES = [1, 2, 3, 4, 5, 6, 7, 9]  # trainer table bins: 0, 1, 2, 3, 4, 5, 6, 7-8, 9+
OVERSEEN = ("B_random_k1_c1.0_seed1", "B_random_k2_c1.0_seed1", "B_random_k7_c2.0_seed1", "D_history_k2_c1.0_seed1")
NUMERIC = ("outer", "players", "walls")
PRIOR = 2.0  # pseudo-items pulling a cell's rate towards its parent's (M1 cell -> overall, M3 cell -> M1 cell)


def near_bin(near, label):
    for i, b in enumerate(NEAR_BINS[label]):
        if near in b:
            return i
    raise ValueError(near)


def smooth(k, n, p0, a=PRIOR):
    return (k + a * p0) / (n + a)


def loglik(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return y * np.log(p) + (1 - y) * np.log(1 - p)


# ------------------------------------------------------------------------------------------------ models
def fit_const(y):
    return (y.sum() + 0.5) / (len(y) + 1.0)


def fit_bins(y, cells, n_cells, parent):
    """Smoothed rate per cell; `parent` (n_cells,) is each cell's prior rate."""
    k = np.bincount(cells, weights=y, minlength=n_cells)
    n = np.bincount(cells, minlength=n_cells)
    return smooth(k, n, parent)


def design(d, label, mu=None, sd=None):
    nb = len(NEAR_BINS[label])
    X = [np.ones(len(d["y"]))] + [(d["nbin"] == i).astype(float) for i in range(1, nb)]
    num = np.stack([d[f] for f in NUMERIC], 1).astype(float)
    if mu is None:
        mu, sd = num.mean(0), num.std(0) + 1e-9
    return np.column_stack(X + [(num - mu) / sd]), mu, sd


def fit_logistic(X, y, ridge=1e-2, iters=50):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - ridge * w
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-X @ w))
    cov = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1]))
    return w, np.sqrt(np.diag(cov))


def outer_edges(outer):
    """Two outer bins split at the median (or three at terciles if there are many distinct values)."""
    q = np.unique(np.quantile(outer, [1 / 3, 2 / 3]).round().astype(int))
    q = [int(e) for e in q if outer.min() < e <= outer.max()]
    return q or [int(np.median(outer)) + 1]


def m3_cells(d, edges):
    ob = sum((d["outer"] >= e).astype(int) for e in edges)
    return d["nbin"] * (len(edges) + 1) + ob


# ------------------------------------------------------------------------------------------------ one group
def predict_all(train, test, label, edges):
    nb = len(NEAR_BINS[label])
    p0 = fit_const(train["y"])
    m1 = fit_bins(train["y"], train["nbin"], nb, np.full(nb, p0))
    X, mu, sd = design(train, label)
    w, _ = fit_logistic(X, train["y"])
    Xt, _, _ = design(test, label, mu, sd)
    out = dict(M0=np.full(len(test["y"]), p0), M1=m1[test["nbin"]], M2=1 / (1 + np.exp(-Xt @ w)))
    if edges:
        nc = nb * (len(edges) + 1)
        m3 = fit_bins(train["y"], m3_cells(train, edges), nc, np.repeat(m1, len(edges) + 1))
        out["M3"] = m3[m3_cells(test, edges)]
    else:
        out["M3"] = out["M1"]
    return out


def crossval(d, label, edges, folds=5, seed=0, boot=2000):
    n = len(d["y"])
    fold = np.random.default_rng(seed).permutation(n) % folds
    preds = {m: np.zeros(n) for m in ("M0", "M1", "M2", "M3")}
    for f in range(folds):
        tr, te = fold != f, fold == f
        p = predict_all({k: v[tr] for k, v in d.items()}, {k: v[te] for k, v in d.items()}, label, edges)
        for m in preds:
            preds[m][te] = p[m]
    ll = {m: loglik(d["y"], p) for m, p in preds.items()}
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, n, (boot, n))
    res = {}
    for m in preds:
        diff = ll[m] - ll["M0"]
        bs = diff[idx].mean(1)
        res[m] = dict(heldout_ll=float(ll[m].mean()), gain_vs_M0=float(diff.mean()),
                      gain_ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                      brier=float(((preds[m] - d["y"]) ** 2).mean()))
    return res, preds


def group(items, label):
    sel = [x for x in items if x["label"] == label]
    y = np.asarray([(1 - x["flag"]) if label == 1 else x["flag"] for x in sel], float)
    d = dict(y=y, nbin=np.asarray([near_bin(x["near"], label) for x in sel]),
             near=np.asarray([x["near"] for x in sel]), source=np.asarray([x["source"] for x in sel]),
             **{f: np.asarray([x[f] for x in sel]) for f in NUMERIC})
    return d


def analyse(d, label, bank_group):
    nb = len(NEAR_BINS[label])
    X, mu, sd = design(d, label)
    w, se = fit_logistic(X, d["y"])
    names = ["intercept"] + [f"near bin {NEAR_BINS[label][i]}" for i in range(1, nb)] + [f"{f} (per sd)"
                                                                                          for f in NUMERIC]
    coefs = {nm: dict(coef=float(c), se=float(s), z=float(c / s)) for nm, c, s in zip(names, w, se)}
    z_outer = coefs["outer (per sd)"]["z"]
    edges = outer_edges(d["outer"]) if abs(z_outer) > 2 else []
    cv, _ = crossval(d, label, edges)
    # final fit on everything
    p0 = fit_const(d["y"])
    m1 = fit_bins(d["y"], d["nbin"], nb, np.full(nb, p0))
    cells = []
    for i in range(nb):
        m = d["nbin"] == i
        k, n = int(d["y"][m].sum()), int(m.sum())
        lo, hi = wilson(k, n)
        cells.append(dict(near=list(NEAR_BINS[label][i]), n=n, raw=k / n if n else float("nan"), lo=lo, hi=hi,
                          fitted=float(m1[i])))
    m3 = None
    if edges:
        nc = nb * (len(edges) + 1)
        c3 = m3_cells(d, edges)
        m3 = fit_bins(d["y"], c3, nc, np.repeat(m1, len(edges) + 1))
        for c in range(nc):
            m = c3 == c
            k, n = int(d["y"][m].sum()), int(m.sum())
            lo, hi = wilson(k, n)
            cells.append(dict(near=list(NEAR_BINS[label][c // (len(edges) + 1)]), outer_bin=c % (len(edges) + 1),
                              n=n, raw=k / n if n else float("nan"), lo=lo, hi=hi, fitted=float(m3[c])))
    # matched coin flip: the near-bin rates weighted by the bank's natural frequencies (M1; M3 by its own cells)
    def weighted(bg):
        if edges:
            c = m3_cells(bg, edges)
            return float(m3[c].mean())
        return float(m1[bg["nbin"]].mean())
    over = {k: v[np.isin(bank_group["source"], OVERSEEN)] for k, v in bank_group.items()}
    table_rates = (m3 if edges else m1)
    big = [c for c in cells if c["n"] >= 30 and (("outer_bin" in c) == bool(edges))]
    spread = dict(min=min(c["fitted"] for c in big), max=max(c["fitted"] for c in big)) if big else {}
    bank_cells = m3_cells(bank_group, edges) if edges else bank_group["nbin"]
    freq = np.bincount(bank_cells, minlength=len(table_rates)) / len(bank_cells)
    mean = float((freq * table_rates).sum())
    spread["sd_weighted"] = float(np.sqrt((freq * (table_rates - mean) ** 2).sum()))
    by_source = {}
    for s in np.unique(d["source"]):
        m = d["source"] == s
        by_source[str(s)] = dict(n=int(m.sum()), raw=float(d["y"][m].mean()),
                                 by_near={str(v): [int(d["y"][m & (d["near"] == v)].sum()),
                                                   int((m & (d["near"] == v)).sum())]
                                          for v in np.unique(d["near"][m])})
    return dict(n=len(d["y"]), raw_subset=float(d["y"].mean()), constant=float(p0), logistic=coefs,
                outer_edges=edges, crossval=cv, cells=cells, near_rates=[float(x) for x in m1],
                table_rates=[float(x) for x in table_rates], matched_pooled=weighted(bank_group),
                matched_overseen=weighted(over), spread=spread, by_source=by_source)


def trainer_table(res_v, res_h):
    """near (and outer, if either label uses it) table in the trainer's format."""
    ev, eh = res_v["outer_edges"], res_h["outer_edges"]
    if ev and eh and ev != eh:
        eh = ev  # one shared outer split; refit is not needed for the parts never read
    edges = ev or eh
    near_bins = len(NEAR_EDGES) + 1  # 0..2 violations, 3.. honest
    honest_of = {0: None, 1: None, 2: None, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5}  # trainer near bin -> honest bin
    nob = len(edges) + 1
    alpha = [[0.0] * nob for _ in range(near_bins)]
    beta = [[0.0] * nob for _ in range(near_bins)]

    def rate(res, b, o):
        if res["outer_edges"] and res["outer_edges"] == edges:
            return res["table_rates"][b * nob + o]
        return res["near_rates"][b]
    for tb in range(near_bins):
        for o in range(nob):
            if tb <= 2:
                alpha[tb][o] = rate(res_v, tb, o)
            else:
                beta[tb][o] = rate(res_h, honest_of[tb], o)
    feats = [dict(name="near", edges=NEAR_EDGES)] + ([dict(name="outer", edges=edges)] if edges else [])
    if not edges:
        alpha, beta = [a[0] for a in alpha], [b[0] for b in beta]
    return dict(features=feats, alpha=alpha, beta=beta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", nargs="+", default=["gpt-oss-20b_direct", "qwen2.5-7b_reason"])
    ap.add_argument("--bank", default="experiments/llm_overseer/data/error_bank.jsonl.gz")
    ap.add_argument("--out-dir", default="notes/llm_overseer/error_patterns")
    a = ap.parse_args()
    out = Path(a.out_dir)
    bank = {x["id"]: x for x in load_jsonl(a.bank)}
    for x in bank.values():
        x.pop("renderings", None)
    bank_items = list(bank.values())
    for x in bank_items:
        x["flag"] = 0
    bank_groups = {lab: group(bank_items, lab) for lab in (0, 1)}
    fits = {}
    for tag in a.judged:
        recs = [json.loads(l) for l in open(out / "judge" / f"{tag}.jsonl")]
        recs = {r["id"]: r for r in recs if r.get("error") is None}
        items = [dict(bank[i], flag=int(r["verdict"] == 1), no_verdict=r["verdict"] is None)
                 for i, r in recs.items()]
        res = {lab: analyse(group(items, lab), lab, bank_groups[lab]) for lab in (1, 0)}
        tokens = [r["tokens_in"] + r["tokens_out"] for r in recs.values()]
        fits[tag] = dict(n_items=len(items), no_verdict=sum(x["no_verdict"] for x in items),
                         tokens_per_check=float(np.mean(tokens)), alpha=res[1], beta=res[0])
        table = trainer_table(res[1], res[0])
        table.update(inspector=tag, matched_coin_flip=dict(alpha=res[1]["matched_pooled"],
                                                           beta=res[0]["matched_pooled"]),
                     matched_coin_flip_overseen=dict(alpha=res[1]["matched_overseen"],
                                                     beta=res[0]["matched_overseen"]),
                     note="fitted by experiments/llm_overseer/fit_error_model.py; see error_patterns/results.md")
        (out / "tables").mkdir(parents=True, exist_ok=True)
        (out / "tables" / f"{tag}.json").write_text(json.dumps(table, indent=1))
        print(f"{tag}: {len(items)} items; alpha matched {res[1]['matched_pooled']:.3f} "
              f"(overseen {res[1]['matched_overseen']:.3f}); beta matched {res[0]['matched_pooled']:.3f} "
              f"(overseen {res[0]['matched_overseen']:.3f})")
    (out / "fit.json").write_text(json.dumps(fits, indent=1, default=float))
    (out / "fit_tables.md").write_text(render(fits))


def pct(x):
    return "–" if x != x else f"{x:.2f}"


def render(fits):
    lines = ["# Error-model fit (generated by experiments/llm_overseer/fit_error_model.py)", ""]
    for tag, f in fits.items():
        lines += [f"## {tag}: {f['n_items']} items, {f['no_verdict']} without a verdict, "
                  f"{f['tokens_per_check']:.0f} tokens a check", ""]
        for name, r in (("α (miss rate, violations)", f["alpha"]), ("β (false-alarm rate, honest eats)", f["beta"])):
            lines += [f"### {name}: n = {r['n']}; constant {r['constant']:.3f}; matched coin flip (bank average) "
                      f"{r['matched_pooled']:.3f}, overseen policies only {r['matched_overseen']:.3f}", "",
                      "| near | outer bin | n | rate [95% CI] | fitted |", "|---|---|---|---|---|"]
            for c in r["cells"]:
                ob = c.get("outer_bin", "all")
                lines.append(f"| {'–'.join(map(str, (c['near'][0], c['near'][-1]))) if len(c['near']) > 1 else c['near'][0]}"
                             f" | {ob} | {c['n']} | {pct(c['raw'])} [{pct(c['lo'])}, {pct(c['hi'])}] | "
                             f"{c['fitted']:.3f} |")
            lines += ["", f"Outer split edges: {r['outer_edges'] or 'none (M3 = M1)'}", "",
                      "| model | held-out log-lik per item | gain vs M0 [95% CI] | Brier |", "|---|---|---|---|"]
            for m, cv in r["crossval"].items():
                lines.append(f"| {m} | {cv['heldout_ll']:.4f} | {cv['gain_vs_M0']:+.4f} "
                             f"[{cv['gain_ci'][0]:+.4f}, {cv['gain_ci'][1]:+.4f}] | {cv['brier']:.4f} |")
            lines += ["", "| logistic term | coef | se | z |", "|---|---|---|---|"]
            for nm, c in r["logistic"].items():
                lines.append(f"| {nm} | {c['coef']:+.3f} | {c['se']:.3f} | {c['z']:+.2f} |")
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
