"""Analysis for R1b (audit-aware and blind gpt-oss-120b strategies in Willis et al.'s CPR).

Protocol (frozen, with amendments): notes/claude_audit_20261005/studies/R1b_audit_aware_strategies/protocol.md

Reads (written by run_r1b_generate.py and run_r1b_parts.py, e.g. a checkout of the r1b-results branch):
  <root>/strategies/pool_{A,B}/records/*.json, <root>/strategies/status.json, calls.jsonl
  <root>/part1/pool_<P>_n<n>_e<e>.json.gz           (Part 1)
  <root>/prediction.json, <root>/runs/<cond>/run_*.json.gz   (Part 2)
Writes, in --summary-dir (default notes/claude_audit_20261005/runs/claude_r1b_v1):
  r1b_summary.json (generation, H1-H5, Part 3), r1b_part1.csv, r1b_part2.csv, r1b_part3_strategies.csv,
  r1b_part3_handcheck.csv (20 random Pool A strategies, seed R1B_BASE + 2, with blank columns for the hand check),
and the figure --figure (default notes/claude_audit_20261005/figures/fig_r1b_audit_aware.{pdf,png}).

Statistics (protocol): the unit is the strategy in Parts 1 and 3 (a strategy's 20 games are averaged first) and the
evolution run in Part 2; 95% percentile bootstrap, 4,000 resamples, seeded from R1B_BASE + 1 (one generator per
statistic, stable_seed(R1B_BASE + 1, *key)). Within a pool, the e levels are resampled jointly (the same strategies
at every e, paired); different pools or attitude groups are resampled independently.
Part 2 conditions are used for H4 only with all --min-runs runs; partial conditions are listed as pending.

Run: PYTHONPATH=. python3 -m experiments.oversight.analyze_r1b [--root DIR] [--summary-dir DIR] [--figure STEM]
     [--min-runs 20] [--no-figure]
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from overseer.seeding import stable_seed

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "runs" / "claude_r1b_v1"
SUMMARY = ROOT / "notes" / "claude_audit_20261005" / "runs" / "claude_r1b_v1"
FIG = ROOT / "notes" / "claude_audit_20261005" / "figures" / "fig_r1b_audit_aware"
R1B_BASE = 26_101_450
B, BOOT_SEED, HAND_SEED = 4000, R1B_BASE + 1, R1B_BASE + 2
P1_E = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)
MULTS = (0.0, 0.5, 1.0, 1.25, 2.0)
POOLS = ("A", "B")


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def boot_idx(m, key):
    rng = np.random.default_rng(stable_seed(BOOT_SEED, *key))
    return rng.integers(0, m, (B, m))


def ci(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def mean_ci(x, key):
    x = np.asarray(x, float)
    if len(x) == 0:
        return dict(mean=None, ci=None, k=0)
    if len(x) == 1:
        return dict(mean=float(x[0]), ci=[float(x[0])] * 2, k=1)
    return dict(mean=float(x.mean()), ci=ci(x[boot_idx(len(x), key)].mean(axis=1)), k=int(len(x)))


def diff_ci(a, b, key):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return dict(diff=None, ci=None)
    da = a[boot_idx(len(a), key + ("a",))].mean(axis=1)
    db = b[boot_idx(len(b), key + ("b",))].mean(axis=1)
    return dict(diff=float(a.mean() - b.mean()), ci=ci(da - db))


# ------------------------------------------------------------------ generation
def generation_summary(strategies):
    out = dict(pools={})
    for pool in POOLS:
        d = Path(strategies) / f"pool_{pool}" / "records"
        recs = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []
        by = {}
        for r in recs:
            k = f"{r['base']}/{r['attitude']}"
            by.setdefault(k, dict(ok=0, failed=0, unfinished=0))
            by[k]["ok" if r["status"] == "ok" else "failed" if r["status"] == "failed" else "unfinished"] += 1
        out["pools"][pool] = dict(records=len(recs), ok=sum(r["status"] == "ok" for r in recs),
                                  failed=sum(r["status"] == "failed" for r in recs),
                                  failed_classes=sorted(r["class_name"] for r in recs if r["status"] == "failed"),
                                  attempts_used=sum(len(r["attempts"]) for r in recs), by_synonym=by)
    st = Path(strategies) / "status.json"
    out["last_status"] = json.loads(st.read_text()) if st.exists() else None
    calls = Path(strategies) / "calls.jsonl"
    if calls.exists():
        rows = [json.loads(x) for x in open(calls) if x.strip()]
        tok = sum(int(r.get("prompt_eval_count", 0)) + int(r.get("eval_count", 0)) for r in rows)
        out.update(calls=len(rows), tokens=tok, estimated_usd=tok * 0.32 / 1e6,
                   failed_calls=sum(1 for r in rows if r.get("error")))
    return out


# ------------------------------------------------------------------ Part 1
def load_part1(root):
    """{(pool, n): dict(strategies=[...], rates={e: per-strategy mean non-final defection rate})}."""
    out = {}
    for p in sorted((Path(root) / "part1").glob("pool_*_n*_e*.json.gz")):
        rec = read_gz(p)
        key = (rec["pool"], int(rec["n"]))
        d = out.setdefault(key, dict(strategies=rec["strategies"], rates={}, fines={}, net={}))
        if [s["cls"] for s in d["strategies"]] != [s["cls"] for s in rec["strategies"]]:
            raise ValueError(f"{p}: strategy list differs between e levels")
        d["rates"][float(rec["e"])] = np.asarray(rec["focal_defect_rate_nonfinal"], float).mean(axis=1)
        d["fines"][float(rec["e"])] = np.asarray(rec["focal_fines"], float).mean(axis=1)
        d["net"][float(rec["e"])] = np.asarray(rec["focal_net_payoff"], float).mean(axis=1)
    return out


def group_mask(d, base):
    return np.array([s["base"] == base for s in d["strategies"]])


def part1_table(p1):
    rows = []
    for (pool, n), d in sorted(p1.items()):
        for base in ("collective", "selfish"):
            msk = group_mask(d, base)
            for e in sorted(d["rates"]):
                s = mean_ci(d["rates"][e][msk], ("p1", pool, n, base, e))
                rows.append(dict(pool=pool, n=n, base=base, e=e, strategies=int(msk.sum()), defect_rate=s["mean"],
                                 ci_lo=s["ci"][0] if s["ci"] else None, ci_hi=s["ci"][1] if s["ci"] else None,
                                 fines_per_game=float(d["fines"][e][msk].mean()) if msk.any() else None,
                                 net_payoff=float(d["net"][e][msk].mean()) if msk.any() else None))
    return rows


def h1(p1):
    """Pool A Selfish: ratio (e = 2) / (e = 0) of the mean defection rate, and every step's change, paired bootstrap."""
    out = {}
    for (pool, n), d in sorted(p1.items()):
        if pool != "A":
            continue
        es = sorted(d["rates"])
        if 0.0 not in es or 2.0 not in es:
            out[str(n)] = dict(status="pending", e_levels=es)
            continue
        msk = group_mask(d, "selfish")
        X = np.stack([d["rates"][e][msk] for e in es])          # e x strategies
        m = X.shape[1]
        idx = boot_idx(m, ("H1", n))
        bm = X[:, idx].mean(axis=2)                              # e x B
        i0, i2 = es.index(0.0), es.index(2.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_b = bm[i2] / bm[i0]
        m0, m2 = float(X[i0].mean()), float(X[i2].mean())
        ratio = m2 / m0 if m0 > 0 else None
        rci = ci(ratio_b)
        steps = []
        for a, b in zip(range(len(es) - 1), range(1, len(es))):
            steps.append(dict(e_from=es[a], e_to=es[b], change=float(X[b].mean() - X[a].mean()),
                              ci=ci(bm[b] - bm[a])))
        fails_ratio = rci is not None and rci[0] > 0.5
        fails_steps = any(s["ci"] is not None and s["ci"][0] > 0 for s in steps)
        if ratio is None:
            verdict = "undefined (no defection at e = 0)"
        elif fails_ratio or fails_steps:
            verdict = "fails"
        elif ratio <= 0.5:
            verdict = "holds"
        else:
            verdict = "not decided (ratio above 0.5, interval includes 0.5)"
        out[str(n)] = dict(strategies=int(m), rate_e0=m0, rate_e2=m2, ratio=ratio, ratio_ci=rci,
                           fails_ratio=bool(fails_ratio), steps=steps, fails_steps=bool(fails_steps), verdict=verdict)
    return out


def h2(p1):
    """Pool B: the defection rates must not change with e (by construction; a code check)."""
    out = {}
    for (pool, n), d in sorted(p1.items()):
        if pool != "B":
            continue
        es = sorted(d["rates"])
        X = np.stack([d["rates"][e] for e in es])
        dev = float(np.abs(X - X[0]).max()) if len(es) > 1 else 0.0
        out[str(n)] = dict(e_levels=es, max_abs_change_any_strategy=dev, holds=bool(dev == 0.0),
                           mean_by_e={str(e): float(X[k].mean()) for k, e in enumerate(es)})
    return out


def h3(p1):
    """Pool A Collective vs Pool B Collective at e = 0 (descriptive, two-sided interval)."""
    out = {}
    for n in sorted({k[1] for k in p1}):
        a, b = p1.get(("A", n)), p1.get(("B", n))
        if not a or not b or 0.0 not in a["rates"] or 0.0 not in b["rates"]:
            out[str(n)] = dict(status="pending")
            continue
        xa = a["rates"][0.0][group_mask(a, "collective")]
        xb = b["rates"][0.0][group_mask(b, "collective")]
        out[str(n)] = dict(pool_A=float(xa.mean()), pool_B=float(xb.mean()), **diff_ci(xa, xb, ("H3", n)))
    return out


# ------------------------------------------------------------------ Part 2
def load_runs(root):
    out = {}
    for d in sorted((Path(root) / "runs").glob("*")):
        files = sorted(d.glob("run_*.json.gz"))
        if not files:
            continue
        recs = [read_gz(f) for f in files]
        out[d.name] = dict(condition=recs[0]["condition"], runs=[r["run"] for r in recs],
                           summaries=[r["window_summary"] for r in recs])
    return out


def part2(runs, min_runs):
    table = []
    for name, d in sorted(runs.items(), key=lambda kv: (kv[1]["condition"]["pool"], kv[1]["condition"]["mult"])):
        c = d["condition"]
        row = dict(name=name, pool=c["pool"], mult=c["mult"], e=c["e"], F=c["F"], runs=len(d["runs"]),
                   complete=len(d["runs"]) >= min_runs)
        for k in ("collective_frequency", "efficiency_transfer", "efficiency_lost", "fines_per_game",
                  "audits_per_game", "defections_nonfinal_per_agent_game"):
            s = mean_ci([x[k] for x in d["summaries"]], ("P2", name, k))
            row[k], row[k + "_ci"] = s["mean"], s["ci"]
        table.append(row)
    by = {(r["pool"], r["mult"]): r for r in table}
    a, b = runs.get("A_e1.25"), runs.get("B_e1.25")
    if a and b and len(a["runs"]) >= min_runs and len(b["runs"]) >= min_runs:
        d = diff_ci([x["efficiency_transfer"] for x in a["summaries"]],
                    [x["efficiency_transfer"] for x in b["summaries"]], ("H4",))
        H4 = dict(**d, runs_A=len(a["runs"]), runs_B=len(b["runs"]),
                  verdict=("not computable (fewer than 2 runs)" if d["ci"] is None else
                           "holds" if d["ci"][0] > 0 else "fails"))
    else:
        H4 = dict(status="pending", runs_A=len(a["runs"]) if a else 0, runs_B=len(b["runs"]) if b else 0)
    H5 = {}
    for pool in POOLS:
        first = next((m for m in MULTS if (pool, m) in by and by[(pool, m)]["complete"]
                      and by[(pool, m)]["collective_frequency"] > 0.5), None)
        H5[pool] = dict(first_mult_collective_above_half=first,
                        complete_levels=[m for m in MULTS if (pool, m) in by and by[(pool, m)]["complete"]])
    pending = [r["name"] for r in table if not r["complete"]]
    return table, H4, H5, pending


# ------------------------------------------------------------------ Part 3: what the code does
_Q, _F = {"audit_prob"}, {"fine"}
_HIST = {"my_audited", "my_fines"}
_IGNORE = {"self", "game_description", "np", "math", "random", "Action", "C", "D", "int", "float", "abs", "min",
           "max", "sum", "len", "round"}


def _tokens(node):
    """Names, attribute names and getattr(..., "name") strings in a subtree."""
    out = set()
    for x in ast.walk(node):
        if isinstance(x, ast.Name):
            out.add(x.id)
        elif isinstance(x, ast.Attribute):
            out.add(x.attr)
        elif (isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "getattr"
              and len(x.args) >= 2 and isinstance(x.args[1], ast.Constant) and isinstance(x.args[1].value, str)):
            out.add(x.args[1].value)
    return out


def _target_names(t):
    if isinstance(t, ast.Name):
        return {t.id}
    if isinstance(t, ast.Attribute):
        return {t.attr}
    if isinstance(t, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(e) for e in t.elts)) if t.elts else set()
    return set()


def _has_ev_product(node, Q, F, EV):
    for x in ast.walk(node):
        if isinstance(x, ast.Name) and x.id in EV or isinstance(x, ast.Attribute) and x.attr in EV:
            return True
        if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Mult):
            t = _tokens(x)
            if t & Q and t & F:
                return True
    return False


def analyse_code(code: str) -> dict:
    """Heuristic (protocol Part 3): does the code read q or F, read its audit history, and compare an expected fine
    (a product of q and F, directly or through variables) with a gain term (any other non-constant quantity)?"""
    tree = ast.parse(code)
    toks = _tokens(tree)
    Q, F, EV = set(_Q), set(_F), set()
    assigns = [n for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and n.value]
    for _ in range(10):                       # propagate through assignments to a fixed point
        before = (len(Q), len(F), len(EV))
        for a in assigns:
            targets = set().union(*(_target_names(t) for t in (a.targets if isinstance(a, ast.Assign)
                                                               else [a.target])))
            if _has_ev_product(a.value, Q, F, EV):
                EV |= targets
            else:
                t = _tokens(a.value)
                if t & Q and not t & F:
                    Q |= targets
                elif t & F and not t & Q:
                    F |= targets
        if (len(Q), len(F), len(EV)) == before:
            break
    ev_compare = False
    for x in ast.walk(tree):
        if isinstance(x, ast.Compare) and _has_ev_product(x, Q, F, EV):
            other = _tokens(x) - Q - F - EV - _IGNORE
            if other:
                ev_compare = True
                break
    return dict(reads_q_or_F=bool(toks & (_Q | _F)), reads_q=bool(toks & _Q), reads_F=bool(toks & _F),
                reads_audit_history=bool(toks & _HIST), ev_comparison=ev_compare)


def part3(strategies):
    rows = []
    for pool in POOLS:
        d = Path(strategies) / f"pool_{pool}" / "records"
        for p in sorted(d.glob("*.json")) if d.exists() else []:
            r = json.loads(p.read_text())
            if r["status"] != "ok":
                continue
            rows.append(dict(pool=pool, cls=r["class_name"], attitude=r["attitude"], base=r["base"],
                             **analyse_code(r["code"])))
    summ = {}
    for pool in POOLS:
        for base in ("all", "collective", "selfish"):
            rs = [r for r in rows if r["pool"] == pool and (base == "all" or r["base"] == base)]
            if rs:
                summ[f"{pool}/{base}"] = dict(strategies=len(rs), **{k: sum(r[k] for r in rs) / len(rs) for k in
                                              ("reads_q_or_F", "reads_audit_history", "ev_comparison")})
    a = sorted((r for r in rows if r["pool"] == "A"), key=lambda r: r["cls"])
    pick = sorted(np.random.default_rng(HAND_SEED).choice(len(a), min(20, len(a)), replace=False).tolist()) if a else []
    hand = [dict(a[i], hand_reads_q_or_F="", hand_ev_comparison="", hand_note="") for i in pick]
    return rows, summ, hand


# ------------------------------------------------------------------ outputs
def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def figure(p1rows, p2rows, stem, e_pred=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    style = {("A", "selfish"): ("#b2182b", "-", "told the rule, Selfish"),
             ("A", "collective"): ("#2166ac", "-", "told the rule, Collective"),
             ("B", "selfish"): ("#b2182b", "--", "not told, Selfish"),
             ("B", "collective"): ("#2166ac", "--", "not told, Collective")}
    for ax, n in zip(axes[:2], (64, 4)):
        for (pool, base), (col, ls, lab) in style.items():
            rs = sorted((r for r in p1rows if r["pool"] == pool and r["n"] == n and r["base"] == base),
                        key=lambda r: r["e"])
            if not rs:
                continue
            x = [r["e"] for r in rs]
            ax.plot(x, [r["defect_rate"] for r in rs], ls, color=col, marker="o", ms=3, label=lab)
            ax.fill_between(x, [r["ci_lo"] for r in rs], [r["ci_hi"] for r in rs], color=col, alpha=0.12, lw=0)
        ax.axvline(2.0, color="0.6", lw=0.8, ls=":")
        ax.set(title=f"({'a' if n == 64 else 'b'}) One strategy in a group of {n}",
               xlabel="expected fine per over-take, e = qF", ylabel="share of rounds over-taking", ylim=(-0.02, 1.02))
    axes[0].legend(fontsize=7, frameon=False)
    ax = axes[2]
    for pool, col, lab in (("A", "#1b7837", "told the rule"), ("B", "0.4", "not told")):
        rs = sorted((r for r in p2rows if r["pool"] == pool), key=lambda r: r["mult"])
        if rs:
            x = [r["mult"] for r in rs]
            ax.plot(x, [r["collective_frequency"] for r in rs], "-o", ms=3, color=col, label=lab)
            ax.fill_between(x, [r["collective_frequency_ci"][0] for r in rs],
                            [r["collective_frequency_ci"][1] for r in rs], color=col, alpha=0.15, lw=0)
    ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax.set(title="(c) Cultural evolution, groups of 64", xlabel="expected fine / balance fine of the blind pool",
           ylabel="share Collective (last 100 generations)", ylim=(-0.02, 1.02))
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=RAW)
    ap.add_argument("--strategies", type=Path, help="default <root>/strategies")
    ap.add_argument("--summary-dir", type=Path, default=SUMMARY)
    ap.add_argument("--figure", type=Path, default=FIG, help="figure path stem (.pdf and .png)")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--min-runs", type=int, default=20)
    args = ap.parse_args(argv)
    strategies = args.strategies or args.root / "strategies"
    gen = generation_summary(strategies)
    p1 = load_part1(args.root)
    p1rows = part1_table(p1)
    runs = load_runs(args.root)
    p2rows, H4, H5, pending = part2(runs, args.min_runs)
    p3rows, p3summ, hand = part3(strategies)
    pred = args.root / "prediction.json"
    summary = dict(study="R1b", generation=gen,
                   prediction=json.loads(pred.read_text()) if pred.exists() else None,
                   part1=dict(H1=h1(p1), H2=h2(p1), H3=h3(p1), table=p1rows),
                   part2=dict(H4=H4, H5=H5, pending_conditions=pending, table=p2rows),
                   part3=dict(summary=p3summ, handcheck_seed=HAND_SEED, handcheck=[h["cls"] for h in hand]),
                   bootstrap=dict(resamples=B, seed=BOOT_SEED))
    sd = args.summary_dir
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "r1b_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    write_csv(sd / "r1b_part1.csv", p1rows)
    write_csv(sd / "r1b_part2.csv", p2rows)
    write_csv(sd / "r1b_part3_strategies.csv", p3rows)
    write_csv(sd / "r1b_part3_handcheck.csv", hand)
    if not args.no_figure and (p1rows or p2rows):
        figure(p1rows, p2rows, args.figure)
    print(json.dumps(dict(H1=summary["part1"]["H1"], H2={k: v["holds"] for k, v in summary["part1"]["H2"].items()},
                          H3=summary["part1"]["H3"], H4=H4, H5=H5, part3=p3summ), indent=1, default=float))
    print("wrote", sd)


if __name__ == "__main__":
    main()
