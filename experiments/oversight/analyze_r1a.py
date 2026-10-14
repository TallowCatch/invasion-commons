"""Analysis for R1a (audits and fines in Willis et al.'s CPR).

Protocol (frozen, with amendments): notes/claude_audit_20261005/studies/R1a_willis_cpr_audits/protocol.md

Reads results/runs/claude_r1a_v1/runs/<condition>/run_*.json.gz (written by run_r1a_willis_audits.py) and
runs/claude_r1a_v1/prediction.json. Writes, in notes/claude_audit_20261005/runs/claude_r1a_v1/:
  r1a_summary.json   (conditions, H0-H3, descriptive outcomes, what ran),
  r1a_conditions.csv (one row per condition), r1a_runs.csv (one row per run),
and the figure notes/claude_audit_20261005/figures/fig_r1a_willis_audits.{pdf,png}.

Run: PYTHONPATH=.:external/emergent_llm/src python3 -m experiments.oversight.analyze_r1a [--min-runs 20]
A condition is used for a hypothesis only if it has all R runs (--min-runs); partial conditions are shown as pending.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from overseer.seeding import stable_seed

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "runs" / "claude_r1a_v1"
SUMMARY = ROOT / "notes" / "claude_audit_20261005" / "runs" / "claude_r1a_v1"
FIG = ROOT / "notes" / "claude_audit_20261005" / "figures"
B, BOOT_SEED = 4000, 26101441
ALPHA = 0.05
MODELS = ("claude-haiku-4-5", "gemini-3.1-flash-lite-preview", "gpt-5.4-mini")
MODEL_LABEL = {"claude-haiku-4-5": "Claude Haiku 4.5", "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
               "gpt-5.4-mini": "GPT-5.4 Mini"}
PAPER = {64: dict(collective=0.19, efficiency=0.15), 4: dict(collective=0.56, efficiency=0.91)}
METRICS = ("collective_frequency", "efficiency_transfer", "efficiency_lost", "fines_per_game", "audits_per_game",
           "caught_per_game", "cooperation_rate", "defections_nonfinal_per_agent_game", "welfare_transfer",
           "welfare_lost", "final_stock_mean")


# ------------------------------------------------------------------ loading
def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def bounds(n):
    """Their per-player, per-game welfare bounds (CommonPoolDescription, capacity 4n, 20 rounds)."""
    r, k = 20, 4 * n
    return k / n, 0.5 * (k / n) * (r + 1)


def run_summary(rec, window=100):
    """Final-window statistics of one run (recomputed from its generation_stats)."""
    stats = rec["generation_stats"]
    n = rec["condition"]["n"]
    lo, hi = bounds(n)
    w = stats[-window:]
    out = {k: float(np.mean([s[k] for s in w])) for k in
           ("collective_frequency", "welfare_transfer", "welfare_lost", "fines_per_game", "audits_per_game",
            "caught_per_game", "cooperation_rate", "defections_nonfinal_per_agent_game", "final_stock_mean")}
    out["efficiency_transfer"] = (out["welfare_transfer"] - lo) / (hi - lo)
    out["efficiency_lost"] = (out["welfare_lost"] - lo) / (hi - lo)
    for m in MODELS:
        out[f"share_{m}"] = float(np.mean([s["model_shares"].get(m, 0.0) for s in w]))
    # genotype (model x base attitude) shares over the window of recorded states that played
    genes = rec["genes"]
    counts = np.asarray(rec["gene_counts"][-window - 1:-1], float)   # states that played the last `window` gens
    tot = counts.sum(axis=1, keepdims=True)
    for m in MODELS:
        for base in ("collective", "selfish"):
            cols = [i for i, g in enumerate(genes) if g.startswith(m + "[") and _base(g) == base]
            out[f"geno_{m}_{base}"] = float((counts[:, cols].sum(axis=1) / tot[:, 0]).mean())
    sw = next((s["generation"] for s in stats if s["collective_frequency"] > 0.5), None)
    out["switch_generation"] = sw
    out["collective_path"] = [s["collective_frequency"] for s in stats]
    return out


_COLL = {"collective", "prosocial", "altruistic", "benevolent"}


def _base(gene):
    att = gene[gene.index("[") + 1:-1]
    return "collective" if att in _COLL else "selfish"


def load_runs(raw=RAW):
    out = {}
    for d in sorted((raw / "runs").glob("*")):
        files = sorted(d.glob("run_*.json.gz"))
        if not files:
            continue
        recs = [read_gz(f) for f in files]
        out[d.name] = dict(condition=recs[0]["condition"], runs=[r["run"] for r in recs],
                           summaries=[run_summary(r) for r in recs],
                           runtime_s=[r.get("runtime_s") for r in recs])
    return out


# ------------------------------------------------------------------ statistics
def boot_means(x, key):
    x = np.asarray(x, float)
    rng = np.random.default_rng(stable_seed(BOOT_SEED, *key))
    return x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1)


def interval(x, key):
    if len(x) == 0:
        return None
    if len(x) == 1:
        return [float(x[0]), float(x[0])]
    bm = boot_means(x, key)
    return [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]


def p_one_sided(x, key, side, ref=0.5):
    """Bootstrap one-sided p-value for H: mean < ref (side='below') or mean > ref (side='above')."""
    bm = boot_means(x, key)
    wrong = (bm >= ref).sum() if side == "below" else (bm <= ref).sum()
    return float((wrong + 1) / (B + 1))


def holm(pvals: dict, alpha=ALPHA, m=None):
    """Holm step-down over m hypotheses (pending ones count, with p = 1). Returns {name: rejected}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = m or len(items)
    out, stop = {}, False
    for i, (name, p) in enumerate(items):
        ok = (not stop) and p <= alpha / (m - i)
        stop = stop or not ok
        out[name] = bool(ok)
    return out


def diff_interval(a, b, key):
    """Interval of mean(a) - mean(b), resampling each independently."""
    da = boot_means(a, key + ("a",))
    db = boot_means(b, key + ("b",))
    d = da - db
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], d


# ------------------------------------------------------------------ analysis
def condition_table(data, r_full):
    rows = []
    for name, d in data.items():
        c = d["condition"]
        row = dict(name=name, n=c["n"], q=c["q"], mult=c["mult"], e=c["e"], F=c["F"], exempt=c["exempt"],
                   arm=c["arm"], runs=len(d["summaries"]), complete=len(d["summaries"]) >= r_full,
                   runtime_mean_s=float(np.mean([t for t in d["runtime_s"] if t])) if d["runtime_s"] else None)
        for k in METRICS + tuple(f"share_{m}" for m in MODELS) + tuple(
                f"geno_{m}_{b}" for m in MODELS for b in ("collective", "selfish")):
            x = [s[k] for s in d["summaries"]]
            row[k] = float(np.mean(x))
            row[k + "_ci"] = interval(x, (name, k))
        sw = [s["switch_generation"] for s in d["summaries"]]
        row["switched_runs"] = int(sum(v is not None for v in sw))
        row["switch_generation_median"] = (float(np.median([v for v in sw if v is not None]))
                                           if any(v is not None for v in sw) else None)
        rows.append(row)
    order = {"baseline": 0, "primary": 1, "secondary": 2}
    rows.sort(key=lambda r: (order.get(r["arm"].split("_")[0], 3), -r["n"], r["q"], r["mult"]))
    return rows


def hypotheses(data, rows, r_full):
    byname = {r["name"]: r for r in rows}
    ok = lambda name: name in byname and byname[name]["complete"]
    x = lambda name, k: [s[k] for s in data[name]["summaries"]]
    out = {}

    # H0
    h0 = {}
    for n, test in ((64, "cf<0.5 and eff<0.3"), (4, "eff>0.5")):
        name = f"n{n}_base"
        if not ok(name):
            h0[str(n)] = dict(status="pending")
            continue
        r = byname[name]
        holds = (r["collective_frequency"] < 0.5 and r["efficiency_transfer"] < 0.3) if n == 64 \
            else r["efficiency_transfer"] > 0.5
        h0[str(n)] = dict(test=test, collective_frequency=r["collective_frequency"],
                          collective_frequency_ci=r["collective_frequency_ci"],
                          efficiency_transfer=r["efficiency_transfer"], efficiency_transfer_ci=r["efficiency_transfer_ci"],
                          paper=PAPER[n], holds=bool(holds), status="verified")
    h0["holds"] = all(v.get("holds") for v in h0.values() if isinstance(v, dict)) if all(
        v.get("status") == "verified" for v in h0.values()) else None
    out["H0"] = h0

    # H1 (Amendment 2 for n = 4; Amendment 3: q = 0.1 only, Holm over 2 cells)
    cells = {}
    for n in (64, 4):
        q = 0.1
        names = [r["name"] for r in rows if r["n"] == n and r["q"] == q and r["exempt"] and r["arm"] == "primary"]
        levels = {}
        lows = [f"n{n}_base"] + [nm for nm in names if byname[nm]["mult"] <= 0.75]
        highs = [nm for nm in names if byname[nm]["mult"] >= 1.25]
        if n == 4:   # Amendment 2: > 0.5 at every tested e, including 0
            lows, highs = [], [f"n{n}_base"] + names
        planned_low = ["base", 0.5] if n == 64 else []
        planned_high = [1.25, 2.0] if n == 64 else ["base", 0.5, 1.0, 1.25, 2.0]
        pending = []
        ps = []
        for side, group in (("below", lows), ("above", highs)):
            for nm in group:
                if not ok(nm):
                    pending.append(nm)
                    continue
                p = p_one_sided(x(nm, "collective_frequency"), (nm, "H1"), side)
                ps.append(p)
                levels[nm] = dict(side=side, mean=byname[nm]["collective_frequency"],
                                  ci=byname[nm]["collective_frequency_ci"], p=p,
                                  interval_excludes_half=bool((byname[nm]["collective_frequency_ci"][1] < 0.5)
                                                              if side == "below"
                                                              else (byname[nm]["collective_frequency_ci"][0] > 0.5)))
        n_planned = len(planned_low) + len(planned_high)
        complete = len(levels) == n_planned
        cells[f"n{n}_q{q}"] = dict(levels=levels, pending=pending, complete=complete,
                                   p_cell=max(ps) if (ps and complete) else None,
                                   amended="Amendment 2: > 0.5 at every e" if n == 4 else None)
    pv = {k: (v["p_cell"] if v["p_cell"] is not None else 1.0) for k, v in cells.items()}
    rej = holm(pv, m=2)
    for k, v in cells.items():
        v["holm_rejects_null"] = rej[k] if v["complete"] else None
        v["holds"] = rej[k] if v["complete"] else None
    out["H1"] = dict(cells=cells, holm_m=2, alpha=ALPHA)

    # H2 (q = 0.1 only, Amendment 3)
    h2 = {}
    for q in (0.1,):
        cell, base = f"n64_q{q}_e1.25", "n64_base"
        if not (ok(cell) and ok(base)):
            h2[str(q)] = dict(status="pending")
            continue
        res = {}
        for k in ("efficiency_transfer", "efficiency_lost"):
            ci, _ = diff_interval(x(cell, k), x(base, k), ("H2", q, k))
            res[k] = dict(diff=byname[cell][k] - byname[base][k], ci=ci)
        h2[str(q)] = dict(**res, holds=bool(res["efficiency_transfer"]["ci"][0] > 0.2), status="verified")
    out["H2"] = h2

    # H3 (descriptive): c* per audit, W per game = n x per-player welfare
    h3 = {}
    for r in rows:
        if r["mult"] == 0:
            continue
        base = f"n{r['n']}_base"
        if base not in byname or r["audits_per_game"] <= 0:
            continue
        n = r["n"]
        res = dict(audits_per_game=r["audits_per_game"], complete=r["complete"] and byname[base]["complete"])
        for k, lab in (("welfare_transfer", "transfer"), ("welfare_lost", "lost")):
            a, b = np.asarray(x(r["name"], k)), np.asarray(x(base, k))
            aud = np.asarray(x(r["name"], "audits_per_game"))
            point = n * (a.mean() - b.mean()) / aud.mean()
            ra = np.random.default_rng(stable_seed(BOOT_SEED, "H3", r["name"], k))
            ia = ra.integers(0, len(a), (B, len(a)))
            ib = ra.integers(0, len(b), (B, len(b)))
            bs = n * (a[ia].mean(1) - b[ib].mean(1)) / aud[ia].mean(1)
            res[f"c_star_{lab}"] = float(point)
            res[f"c_star_{lab}_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            res[f"W_per_game_{lab}_cell"] = float(n * a.mean())
            res[f"W_per_game_{lab}_none"] = float(n * b.mean())
        h3[r["name"]] = res
    out["H3"] = h3

    # secondary arm against the matching exempt cell
    sec = {}
    for r in rows:
        if r["arm"] != "secondary":
            continue
        twin = r["name"].replace("_noexempt", "")
        if not (ok(r["name"]) and ok(twin)):
            sec[r["name"]] = dict(status="pending")
            continue
        res = {}
        for k in ("collective_frequency", "efficiency_transfer", "efficiency_lost"):
            ci, _ = diff_interval(x(r["name"], k), x(twin, k), ("secondary", r["name"], k))
            res[k] = dict(noexempt=r[k], exempt=byname[twin][k], diff=r[k] - byname[twin][k], ci=ci)
        sec[r["name"]] = res
    out["secondary"] = sec
    return out


# ------------------------------------------------------------------ figure
def figure(rows, data, e64, path_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
    INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False, "lines.linewidth": 1.6,
    })
    NCOL = {64: BLUE, 4: ORANGE}
    MCOL = {"claude-haiku-4-5": VIOLET, "gemini-3.1-flash-lite-preview": YELLOW, "gpt-5.4-mini": AQUA}
    prim = [r for r in rows if r["arm"] in ("baseline", "primary") and r["q"] in (0.0, 0.1)]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    for ax, key, title, ylab in ((axes[0, 0], "collective_frequency", "(a) Share of Collective strategies",
                                  "Share Collective, last 100 generations"),
                                 (axes[0, 1], "efficiency_transfer", "(b) Welfare efficiency",
                                  "Welfare efficiency, last 100 generations")):
        for n, dx in ((64, -0.025), (4, 0.025)):
            pts = [r for r in prim if r["n"] == n]
            for r in pts:
                lo, hi = r[key + "_ci"]
                face = NCOL[n] if r["complete"] else "white"
                ax.errorbar(r["mult"] + dx, r[key], yerr=[[r[key] - lo], [hi - r[key]]], fmt="o", color=NCOL[n],
                            mfc=face, ms=5.5, capsize=2.5, lw=1.1)
                if key == "efficiency_transfer" and r["mult"] > 0:
                    ax.plot(r["mult"] + dx, r["efficiency_lost"], marker="_", color=NCOL[n], ms=9, mew=1.6)
            ax.axhline(PAPER[n]["collective" if key == "collective_frequency" else "efficiency"], color=NCOL[n],
                       ls=":", lw=1.0)
        if key == "collective_frequency":
            ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
            ax.text(2.06, 0.505, "half", color=INK2, fontsize=7, va="bottom", ha="right")
        ax.set_xticks([0, 0.5, 1.0, 1.25, 2.0])
        ax.set_xlim(-0.15, 2.15)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"Expected fine per defection, in units of the predicted balance point ({e64:.2f})")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
    handles = [plt.Line2D([], [], marker="o", color=BLUE, ls="", label="Groups of 64"),
               plt.Line2D([], [], marker="o", color=ORANGE, ls="", label="Groups of 4"),
               plt.Line2D([], [], color=INK2, ls=":", label="Paper, no oversight"),
               plt.Line2D([], [], marker="_", color=INK2, ls="", ms=9, mew=1.6, label="Fines counted as lost (b)"),
               plt.Line2D([], [], marker="o", color=INK2, mfc="white", ls="", label="Fewer than 20 runs so far")]
    axes[0, 0].legend(handles=handles, loc="upper left", fontsize=7)

    # (c) genotype shares
    ax = axes[1, 0]
    shown = [r for r in rows if r["q"] in (0.0, 0.1)]
    labels = []
    for i, r in enumerate(shown):
        bottom = 0.0
        for m in MODELS:
            for b, alpha, hatch in (("collective", 1.0, None), ("selfish", 0.45, "///")):
                v = r[f"geno_{m}_{b}"]
                ax.bar(i, v, bottom=bottom, color=MCOL[m], alpha=alpha, hatch=hatch, edgecolor="white", lw=0.4,
                       width=0.75)
                bottom += v
        lab = f"{r['n']}: " + ("none" if r["mult"] == 0 else f"{r['mult']:g}") + ("*" if not r["exempt"] else "")
        labels.append(lab + ("" if r["complete"] else f"\n({r['runs']} runs)"))
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels(labels, fontsize=6.6, rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of the population, last 100 generations")
    ax.set_title("(c) Which model's strategies survive (group size: fine level)", loc="left")
    leg = [Patch(color=MCOL[m], label=MODEL_LABEL[m]) for m in MODELS] + [
        Patch(facecolor="white", edgecolor=INK2, label="solid = Collective"),
        Patch(facecolor="white", edgecolor=INK2, hatch="///", label="hatched = Selfish")]
    ax.legend(handles=leg, fontsize=6.6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="x", visible=False)

    # (d) Collective share over generations, groups of 64
    ax = axes[1, 1]
    shades = {0.0: INK2, 0.5: "#9cc3ef", 1.0: "#5a9be3", 1.25: BLUE, 2.0: "#173f75"}
    for r in [r for r in prim if r["n"] == 64]:
        paths = np.asarray([s["collective_path"] for s in data[r["name"]]["summaries"]])
        m = paths.mean(axis=0)
        ax.plot(np.arange(len(m)), m, color=shades.get(r["mult"], BLUE), lw=1.3,
                label=("no oversight" if r["mult"] == 0 else f"{r['mult']:g} × balance point")
                + ("" if r["complete"] else f" ({r['runs']} runs)"))
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Share Collective (mean over runs)")
    ax.set_title("(d) Groups of 64, audits at 10%: share Collective over time", loc="left")
    ax.legend(fontsize=6.8, loc="upper left")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{path_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main
def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-runs", type=int, default=20)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    pred = json.loads((SUMMARY / "prediction.json").read_text())
    e64 = pred["per_n"]["64"]["e_pred"]
    data = load_runs()
    rows = condition_table(data, args.min_runs)
    hyp = hypotheses(data, rows, args.min_runs)
    grid = json.loads((RAW / "grid_conditions.json").read_text()) if (RAW / "grid_conditions.json").exists() else {}
    planned = [c["name"] for c in grid.get("conditions", []) if "dropped" not in c["arm"]]
    ran = {r["name"]: r["runs"] for r in rows}
    status = dict(R=args.min_runs, planned=planned,
                  complete=[c for c in planned if ran.get(c, 0) >= args.min_runs],
                  partial={c: ran[c] for c in planned if 0 < ran.get(c, 0) < args.min_runs},
                  not_started=[c for c in planned if ran.get(c, 0) == 0],
                  dropped_by_amendment_3=[c["name"] for c in grid.get("conditions", []) if "dropped" in c["arm"]])
    SUMMARY.mkdir(parents=True, exist_ok=True)
    slim = [{k: v for k, v in r.items()} for r in rows]
    out = dict(study="R1a", prediction=dict(e_pred={k: v["e_pred"] for k, v in pred["per_n"].items()},
                                            computed_utc=pred["computed_utc"], scale_used_for_both_n=e64),
               status=status, conditions=slim, hypotheses=hyp, bootstrap=dict(B=B, seed=BOOT_SEED))
    (SUMMARY / "r1a_summary.json").write_text(json.dumps(out, indent=2))
    write_csv(SUMMARY / "r1a_conditions.csv", slim)
    runrows = []
    for name, d in data.items():
        for run, s in zip(d["runs"], d["summaries"]):
            runrows.append(dict(condition=name, run=run, **{k: v for k, v in s.items() if k != "collective_path"}))
    if runrows:
        write_csv(SUMMARY / "r1a_runs.csv", runrows)
    if not args.no_figure and rows:
        figure(rows, data, e64, FIG / "fig_r1a_willis_audits")
    print(json.dumps(dict(status=status, H0=hyp["H0"], H2=hyp["H2"],
                          H1={k: {kk: v[kk] for kk in ("complete", "p_cell", "holds")} for k, v in
                              hyp["H1"]["cells"].items()}), indent=1))


if __name__ == "__main__":
    main()
