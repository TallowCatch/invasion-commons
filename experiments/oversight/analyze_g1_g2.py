"""Analysis of G1 (Schelling diagrams) and G2 (parameter sweeps). Claude audit, October 2026.

Protocol: notes/claude_audit_20261005/studies/G1_schelling_and_sweeps/protocol.md (with Amendments 1-2)
Reads results/runs/claude_g1_v1 and claude_g2_v1; writes summaries to notes/claude_audit_20261005/runs/
claude_g1_v1 and claude_g2_v1 and the figures fig_g1_schelling and fig_g2_sweeps to notes/claude_audit_20261005/figures.

Run:  PYTHONPATH=. python3 -m experiments.oversight.analyze_g1_g2 [--g1 DIR] [--g2 DIR]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.oversight import run_g1_schelling as g1  # noqa: E402

NOTES = Path("notes/claude_audit_20261005")
FIG = NOTES / "figures"
NAMES = dict(fishery="Fishery", forest="Forest", river="River")
UNITS = dict(fishery="catch (t)", forest="harvest", river="discharge")
# make_progress_figures.py palette (dataviz reference palette, light mode)
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "legend.frameon": False, "lines.linewidth": 2,
})
B, SEED = 4000, 26101498


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def boot_mean(x, rng):
    x = np.asarray(x, float)
    d = x[rng.integers(0, len(x), size=(B, len(x)))].mean(1)
    return dict(estimate=float(x.mean()), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])


def curves_from_rows(rows, n):
    by = defaultdict(dict)
    for r in rows:
        by[r["k"]][r["context"]] = r
    ctx = sorted(by[0])
    returns = {k: np.array([by[k][c]["returns"] for c in ctx]) for k in range(n + 1)}
    defect = {k: np.array([by[k][c]["defect"] for c in ctx], bool) for k in range(n + 1)}
    harm = {k: float(np.mean([by[k][c]["harm"] for c in ctx])) for k in range(n + 1)}
    return g1.schelling_curves(returns, defect, n), harm, len(ctx)


# ================================================================== G1
def analyse_g1(g1dir):
    rows = read_gz(g1dir / "episodes.jsonl.gz")
    fstar = json.loads((g1dir / "fstar.json").read_text())
    out, csvrows = {}, []
    for game in g1.GAMES:
        n = 6
        res = {"F_star": fstar[game]}
        sub = [r for r in rows if r["game"] == game and r["diagram"] == 1]
        (rc, rd), harm, P = curves_from_rows(sub, n)
        t = g1.dilemma_test(rc, rd)
        res["diagram1"] = dict(test=t, harm_by_k=harm, populations=P)
        csvrows += _curve_rows(game, 1, None, t)
        F = fstar[game]["F_star"]
        res["diagram2"] = {}
        for fine in sorted({r["fine"] for r in rows if r["game"] == game and r["diagram"] == 2}):
            sub = [r for r in rows if r["game"] == game and r["diagram"] == 2 and r["fine"] == fine]
            (rc, rd), harm, P = curves_from_rows(sub, n)
            t = g1.dilemma_test(rc, rd)
            ratio = fine / F if F > 0 else None
            p2 = None
            if fine == 0:
                p2 = dict(prediction="D3 holds at F = 0", holds=t["D3"]["holds"])
            elif ratio is not None and ratio >= 1 - 1e-9:
                p2 = dict(prediction="D3 fails at F >= F* (no m where over-taking pays)", holds=not t["D3"]["holds"],
                          point_estimates_all_nonpositive=t["D3"]["never_pays_point"])
            res["diagram2"][f"{fine:.6g}"] = dict(fine=fine, fine_over_F_star=ratio, test=t, harm_by_k=harm,
                                                  populations=P, P2=p2)
            csvrows += _curve_rows(game, 2, fine, t)
        out[game] = res
    P1 = all(out[g]["diagram1"]["test"]["dilemma"] for g in g1.GAMES)
    P2 = {g: [v["P2"]["holds"] for v in out[g]["diagram2"].values() if v["P2"] is not None] for g in g1.GAMES}
    return dict(games=out, P1_all_pass=P1, P2_by_game={g: all(v) if v else None for g, v in P2.items()}), csvrows


def _curve_rows(game, diagram, fine, t):
    out = []
    for m in range(len(t["rc"])):
        tm = t["D3"]["by_m"][m]
        out.append(dict(game=game, diagram=diagram, fine=fine, m=m, Rc=t["rc"][m], Rc_lo=t["rc_ci"][m][0],
                        Rc_hi=t["rc_ci"][m][1], Rd=t["rd"][m], Rd_lo=t["rd_ci"][m][0], Rd_hi=t["rd_ci"][m][1],
                        Rd_minus_Rc=tm["estimate"], diff_lo=tm["ci"][0], diff_hi=tm["ci"][1]))
    return out


# ================================================================== G2
def analyse_g2(g2dir):
    points = json.loads((g2dir / "points.json").read_text())
    d1 = defaultdict(list)
    for r in read_gz(g2dir / "diagram1.jsonl.gz"):
        d1[r["point"]].append(r)
    eh = defaultdict(list)
    for r in read_gz(g2dir / "deterrence_harm.jsonl.gz"):
        eh[r["point"]].append(r)
    rng = np.random.default_rng(SEED)
    out = []
    for p in points:
        pid, n = p["point"], p["setting"]["n"]
        (rc, rd), harm_k, P = curves_from_rows(d1[pid], n)
        t = g1.dilemma_test(rc, rd)
        rows = eh[pid]
        none = {r["context"]: r for r in rows if r["arm"] == "none"}
        comply = {r["context"]: r for r in rows if r["arm"] == "comply"}
        aud = {r["context"]: r for r in rows if r["arm"] == "audit" and r["j"] == 20}
        ctx = sorted(none)
        h_none = np.array([none[c]["harm"] for c in ctx])
        h_aud = np.array([aud[c]["harm"] for c in ctx])
        h_comply = np.array([comply[c]["harm"] for c in ctx])
        diff = boot_mean(h_aud - h_none, rng)
        e_obs, gs = p["e_obs"], p["g_star"]
        ratio = e_obs / gs if (e_obs is not None and gs > 0) else None
        s2_applicable = bool(p["testable"] and t["dilemma"])
        s2 = None if not s2_applicable else bool(ratio is not None and abs(ratio - 1) <= 0.25 + 1e-9)
        # held-out gain per cheater over complying, at the last cheating grid value and at e_obs (exploratory)
        gains = {}
        for s in p["searches"]:
            a = {r["context"]: r for r in rows if r["arm"] == "audit" and r["j"] == s["j"]}
            g = [(a[c]["cheater_net"] - comply[c]["cheater_net"]) / comply[c]["n_cheaters"] for c in ctx]
            gains[s["j"]] = dict(e=s["e"], d=s["d_star"], **boot_mean(g, rng))
        out.append(dict(point=pid, factor=p["factor"], value=p["value"], game=p["setting"]["game"],
                        setting=p["setting"], dilemma=t["dilemma"], D1=t["D1"], D2=t["D2"],
                        D3=dict(holds=t["D3"]["holds"], fear=t["D3"]["fear"], greed=t["D3"]["greed"]),
                        Rc=t["rc"], Rd=t["rd"], g_star=gs, testable=p["testable"], F_star_pt=p["F_star_pt"],
                        fine_source=p["fine_source"], e_obs=e_obs, ratio=ratio, stays_zero_above=p["stays_zero_above"],
                        d_star_by_j=[s["d_star"] for s in p["searches"]], S2_applicable=s2_applicable, S2=s2,
                        harm_none=boot_mean(h_none, rng), harm_audit_2F=boot_mean(h_aud, rng),
                        harm_comply_no_audit=float(h_comply.mean()), harm_diff=diff,
                        S3=bool(diff["estimate"] < 0), S3_ci_excludes_0=bool(diff["ci"][1] < 0),
                        d_at_2F=next(s["d_star"] for s in p["searches"] if s["j"] == 20),
                        heldout_gain_by_j=gains))
    S1 = all(o["dilemma"] for o in out)
    S2 = [o["S2"] for o in out if o["S2"] is not None]
    S3 = all(o["S3"] for o in out)
    return dict(points=out, S1_all_hold=S1, S1_failing=[o["point"] for o in out if not o["dilemma"]],
                S2_all_hold=all(S2) if S2 else None, S2_tested=len(S2),
                S2_failing=[o["point"] for o in out if o["S2"] is False],
                S2_not_applicable=[o["point"] for o in out if o["S2"] is None],
                S3_all_hold=S3, S3_failing=[o["point"] for o in out if not o["S3"]],
                S3_ci_includes_0=[o["point"] for o in out if o["S3"] and not o["S3_ci_excludes_0"]])


# ================================================================== figures
def fig_g1(res, csvrows):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    letters = iter("abcdef")
    cols = [BLUE, AQUA, YELLOW, ORANGE]
    for j, game in enumerate(g1.GAMES):
        ax = axes[0, j]
        r = [x for x in csvrows if x["game"] == game and x["diagram"] == 1]
        m = [x["m"] for x in r]
        for key, lab, col in (("Rc", "restrained agent", BLUE), ("Rd", "greedy agent", ORANGE)):
            ax.plot(m, [x[key] for x in r], "-o", color=col, ms=3.5, label=lab)
            ax.fill_between(m, [x[key + "_lo"] for x in r], [x[key + "_hi"] for x in r], color=col, alpha=0.18, lw=0)
        t = res["games"][game]["diagram1"]["test"]
        verdict = "social dilemma" if t["dilemma"] else "not a dilemma (" + ", ".join(
            k for k in ("D1", "D2", "D3") if not t[k]["holds"]) + " fails)"
        ax.set_title(f"({next(letters)}) {NAMES[game]}, no reviewer: {verdict}", loc="left", fontsize=8.5)
        ax.set_xlabel("other agents who are restrained")
        ax.set_ylabel(f"return per agent, {UNITS[game]}")
        if j == 0:
            ax.legend(loc="upper left", fontsize=7.5)
    for j, game in enumerate(g1.GAMES):
        ax = axes[1, j]
        F = res["games"][game]["F_star"]["F_star"]
        fines = sorted({x["fine"] for x in csvrows if x["game"] == game and x["diagram"] == 2})
        for i, fine in enumerate(fines):
            r = [x for x in csvrows if x["game"] == game and x["diagram"] == 2 and x["fine"] == fine]
            m = [x["m"] for x in r]
            lab = "no fine" if fine == 0 else f"fine {fine / F:g} x break-even ({fine:.2f})"
            ax.plot(m, [x["Rd_minus_Rc"] for x in r], "-o", ms=3.5, color=cols[i % 4], label=lab)
            ax.fill_between(m, [x["diff_lo"] for x in r], [x["diff_hi"] for x in r], color=cols[i % 4], alpha=0.15, lw=0)
        ax.axhline(0, color=INK2, lw=0.8)
        extra = " (break-even fine is 0)" if F == 0 else ""
        ax.set_title(f"({next(letters)}) {NAMES[game]}, reviewer and audits{extra}", loc="left", fontsize=8.5)
        ax.set_xlabel("other agents who comply")
        ax.set_ylabel(f"over-taker minus complier, {UNITS[game]}")
        ax.legend(loc="best", fontsize=6.8)
    fig.tight_layout()
    _save(fig, "fig_g1_schelling")


def fig_g2(res):
    factors = ["r", "N", "T", "externality", "kappa"]
    xl = dict(r="Fishery regrowth rate", N="Fishery group size (K/N fixed)", T="Fishery rounds",
              externality="Forest neighbour damage", kappa="River synergy (x paper value)")
    fig, axes = plt.subplots(3, 5, figsize=(13, 7.2), sharey="row")
    letters = iter("abcdefghijklmno")
    for j, f in enumerate(factors):
        pts = sorted([p for p in res["points"] if p["factor"] == f], key=lambda p: p["value"])
        x = [p["value"] for p in pts]
        ax = axes[0, j]
        for i, key in enumerate(("D1", "D2", "D3")):
            held = [p[key]["holds"] for p in pts]
            ax.scatter([v for v, h in zip(x, held) if h], [i] * sum(held), color=BLUE, s=36, zorder=3)
            ax.scatter([v for v, h in zip(x, held) if not h], [i] * (len(held) - sum(held)), facecolor="none",
                       edgecolor=ORANGE, s=36, lw=1.5, zorder=3)
        ax.set_yticks([0, 1, 2], ["D1 cooperate > defect", "D2 cooperate > exploited", "D3 temptation"])
        ax.set_ylim(-0.6, 2.6)
        ax.set_title(f"({next(letters)}) dilemma test (filled: holds)", loc="left", fontsize=8.5)
        ax = axes[1, j]
        xs = [p["value"] for p in pts if p["ratio"] is not None]
        ax.axhspan(0.75, 1.25, color=AQUA, alpha=0.15, lw=0)
        ax.axhline(1, color=INK2, lw=0.8)
        ax.plot(xs, [p["ratio"] for p in pts if p["ratio"] is not None], "-o", color=VIOLET, ms=4)
        missing = [p for p in pts if p["ratio"] is None]
        for p in missing:
            ax.plot([p["value"]], [1.0], "x", color=INK2)
        if missing:
            why = "over-taking never pays (g* = 0)" if all(not p["testable"] for p in missing) else "no stop on the grid"
            ax.text(0.5, 0.62, why, transform=ax.transAxes, ha="center", fontsize=7, color=INK2)
        ax.set_title(f"({next(letters)}) observed / predicted stopping fine", loc="left", fontsize=8.5)
        ax = axes[2, j]
        for key, lab, col, dx in (("harm_none", "no reviewer, no audits", ORANGE, 0), ("harm_audit_2F", "reviewer + audits, fine 2x break-even", BLUE, 0)):
            y = [p[key]["estimate"] for p in pts]
            lo = [p[key]["estimate"] - p[key]["ci"][0] for p in pts]
            hi = [p[key]["ci"][1] - p[key]["estimate"] for p in pts]
            ax.errorbar(x, y, yerr=[lo, hi], fmt="-o", color=col, ms=4, capsize=2, label=lab)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"({next(letters)}) share of rounds below half capacity", loc="left", fontsize=8.5)
        ax.set_xlabel(xl[f])
        ax.set_xticks(x)
        for r in range(2):
            axes[r, j].set_xticks(x)
            axes[r, j].set_xlim(axes[2, j].get_xlim())
    axes[2, 0].legend(loc="center left", fontsize=7)
    axes[1, 0].set_ylabel("ratio (band: within 25%)")
    axes[2, 0].set_ylabel("harm share")
    fig.tight_layout()
    _save(fig, "fig_g2_sweeps")


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1", default="results/runs/claude_g1_v1")
    ap.add_argument("--g2", default="results/runs/claude_g2_v1")
    a = ap.parse_args(argv)
    r1, curves = analyse_g1(Path(a.g1))
    o1 = NOTES / "runs/claude_g1_v1"
    o1.mkdir(parents=True, exist_ok=True)
    (o1 / "g1_summary.json").write_text(json.dumps(r1, indent=1, default=float))
    write_csv(o1 / "g1_curves.csv", curves)
    for f in ("manifest.json", "fstar.json"):
        (o1 / f).write_text((Path(a.g1) / f).read_text())
    fig_g1(r1, curves)
    for g, v in r1["games"].items():
        t = v["diagram1"]["test"]
        print(g, "D1", t["D1"]["holds"], "D2", t["D2"]["holds"], "D3", t["D3"]["holds"], "fear", t["D3"]["fear"],
              "greed", t["D3"]["greed"], "| d2:", {k: (x["test"]["D3"]["holds"], x["P2"]) for k, x in v["diagram2"].items()})
    if Path(a.g2, "points.json").exists():
        r2 = analyse_g2(Path(a.g2))
        o2 = NOTES / "runs/claude_g2_v1"
        o2.mkdir(parents=True, exist_ok=True)
        (o2 / "g2_summary.json").write_text(json.dumps(r2, indent=1, default=float))
        write_csv(o2 / "g2_points.csv", [dict(point=p["point"], game=p["game"], dilemma=p["dilemma"], D1=p["D1"]["holds"],
                                              D2=p["D2"]["holds"], D3=p["D3"]["holds"], fear=p["D3"]["fear"],
                                              greed=p["D3"]["greed"], g_star=p["g_star"], e_obs=p["e_obs"],
                                              ratio=p["ratio"], stays_zero_above=p["stays_zero_above"], S2=p["S2"],
                                              harm_none=p["harm_none"]["estimate"],
                                              harm_audit_2F=p["harm_audit_2F"]["estimate"],
                                              harm_diff=p["harm_diff"]["estimate"], harm_diff_lo=p["harm_diff"]["ci"][0],
                                              harm_diff_hi=p["harm_diff"]["ci"][1], S3=p["S3"], d_at_2F=p["d_at_2F"])
                                         for p in r2["points"]])
        (o2 / "manifest.json").write_text((Path(a.g2) / "manifest.json").read_text())
        fig_g2(r2)
        print(json.dumps({k: v for k, v in r2.items() if k != "points"}, indent=1))


if __name__ == "__main__":
    main()
