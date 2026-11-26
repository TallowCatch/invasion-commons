"""Figures and a summary for week 2 with scripted bots (notes/socialjax/week2_bots/).

  .venv-socialjax/bin/python -m experiments.socialjax.make_week2_figures --run results/runs/socialjax_week2_bots
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

OUT = Path("notes/socialjax/week2_bots")
# palette of experiments/oversight/analyze_g1_g2.py
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "legend.frameon": False, "lines.linewidth": 2,
})
RNG = np.random.default_rng(20261115)


def ci(x, b=4000):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    boots = RNG.choice(x, (b, len(x))).mean(1)
    return x.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def fig_rule_and_schelling(d):
    m_chosen = d["chosen_m"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    ax = axes[0]
    for m in sorted({r["m"] for r in d["part_a"]}):
        rs = [r for r in d["part_a"] if r["m"] == m]
        mean, lo, hi = ci([r["apples_eaten"] for r in rs])
        alive = sum(r["orchard_alive"] for r in rs)
        col = BLUE if m == m_chosen else INK2
        ax.errorbar([m], [mean], yerr=[[mean - lo], [hi - mean]], fmt="o", color=col, ms=5, capsize=2)
        ax.annotate(f"{alive}/{len(rs)}", (m, hi), textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=7, color=col)
    g = np.mean([r["apples_eaten"] for r in d["part_a_greedy"]])
    ax.axhline(g, color=ORANGE, lw=1, ls="--")
    ax.text(0.98, g, " all greedy", color=ORANGE, fontsize=7, va="bottom", ha="right",
            transform=ax.get_yaxis_transform())
    ax.set(title=f"(a) The rule: 7 restrained bots (chosen m = {m_chosen})",
           xlabel="m: apples that must be left within two squares", ylabel="apples eaten in 999 steps")
    ax.set_ylim(bottom=0)
    ax.text(0.02, 0.03, "labels: orchard alive at the end (of 10)", transform=ax.transAxes, fontsize=7, color=INK2)

    for ax, (name, title) in zip(axes[1:], [("none", "(b) No overseer"),
                                            ("random k=2, freeze 50", "(c) 2 random checks, 50-step freeze"),
                                            ("random k=2, penalty 5", "(d) 2 random checks, 5-apple penalty")]):
        rs = [r for r in d["part_b"] if r["condition"] == name]
        for key, col, lab in (("R_c", BLUE, "restrained (cooperator)"), ("R_d", ORANGE, "greedy (defector)")):
            ls = sorted({r["cooperators"] for r in rs if r[key] is not None})
            stats = [ci([r[key] for r in rs if r["cooperators"] == l]) for l in ls]
            ax.plot(ls, [s[0] for s in stats], "-o", color=col, ms=3.5, label=lab)
            ax.fill_between(ls, [s[1] for s in stats], [s[2] for s in stats], color=col, alpha=0.18, lw=0)
        ax.set(title=title, xlabel="number of restrained agents (of 7)", ylabel="net apples per agent")
        ax.set_xticks(range(8))
    axes[1].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rule_and_schelling.png", dpi=200)
    plt.close(fig)


def fig_checks_by_freeze(d):
    rows = d["part_c1"]
    panels = [(p, c) for p in ("1 greedy + 6 restrained", "7 sneaky") for c in ("random", "history")]
    ks = sorted({r["k"] for r in rows})
    fz = sorted({r["freeze"] for r in rows})
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    for j, (pop, chooser) in enumerate(panels):
        cell = defaultdict(list)
        for r in rows:
            if r["population"] == pop and r["chooser"] == chooser:
                cell[(r["k"], r["freeze"])].append(r)
        for i, (metric, cmap, vmax, lab) in enumerate([
                ("orchard_alive", "Greens", 1.0, "share of seeds with the orchard alive"),
                ("apples_eaten", "Blues", None, "apples eaten")]):
            grid = np.array([[np.mean([float(r[metric]) for r in cell[(k, f)]]) for f in fz] for k in ks])
            ax = axes[i, j]
            im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=0,
                           vmax=vmax if vmax else grid.max())
            for (a, b), v in np.ndenumerate(grid):
                ax.text(b, a, f"{v:.1f}" if metric == "orchard_alive" else f"{v:.0f}", ha="center", va="center",
                        fontsize=6.5, color="white" if v > 0.6 * (vmax or grid.max()) else INK)
            ax.set_xticks(range(len(fz)), fz)
            ax.set_yticks(range(len(ks)), ks)
            ax.grid(False)
            ax.set(xlabel="freeze after a catch (steps)", ylabel="agents checked per step (k of 7)")
            ax.set_title(f"({'abcdefgh'[i * 4 + j]}) {pop}, {chooser} checks", fontsize=8.5)
            if j == 3:
                fig.colorbar(im, ax=ax, label=lab, fraction=0.05)
    fig.tight_layout()
    fig.savefig(OUT / "fig_checks_by_freeze.png", dpi=200)
    plt.close(fig)


def penalty_tables(d, against):
    """Greedy agent's net apples minus the comparison's, per k (rows) and penalty (columns), and the break-even."""
    rows = d["part_c2"]
    ks = sorted({r["k"] for r in rows})
    adv = np.array([[np.mean([r["greedy_eaten"] - p * r["greedy_catches"] - r[against]
                              for r in rows if r["k"] == k]) for p in d["penalties"]] for k in ks])
    breakeven = []
    for k in ks:
        rs = [r for r in rows if r["k"] == k]
        gain = float(np.mean([r["greedy_eaten"] - r[against] for r in rs]))
        catches = float(np.mean([r["greedy_catches"] for r in rs]))
        breakeven.append(dict(k=k, gain=gain, catches=catches,
                              breakeven=(gain / catches) if catches > 0 and gain > 0 else None))
    return ks, adv, breakeven


def fig_penalty(d):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7))
    out = {}
    for ax, (against, title) in zip(axes[:2], [
            ("restrained_same_group_eaten", "(a) Greedy vs a restrained agent in the same group\n(1 greedy + 6 restrained)"),
            ("restrained_all_eaten", "(b) Greedy vs restraining when all 7 restrain")]):
        ks, adv, be = penalty_tables(d, against)
        out[against] = be
        lim = np.abs(adv).max()
        im = ax.imshow(adv, origin="lower", aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim))
        for (a, b), v in np.ndenumerate(adv):
            ax.text(b, a, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 * lim else INK)
        ax.set_xticks(range(len(d["penalties"])), d["penalties"])
        ax.set_yticks(range(len(ks)), ks)
        ax.grid(False)
        ax.set(title=title, xlabel="penalty per catch (apples)", ylabel="agents checked per step (k of 7)")
        fig.colorbar(im, ax=ax, fraction=0.05, label="greedy minus restrained (apples)\nred: greed pays")
    ax = axes[2]
    pts = [(b["k"], b["breakeven"]) for b in out["restrained_same_group_eaten"] if b["breakeven"] is not None]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color=VIOLET, ms=5)
    ax.set(title="(c) Penalty per catch at which greed stops paying\n(comparison (a))",
           xlabel="agents checked per step (k of 7)", ylabel="apples per catch")
    ax.set_xticks(range(8))
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_penalty.png", dpi=200)
    plt.close(fig)
    return out


def summary(d, breakeven):
    s = dict(chosen_m=d["chosen_m"], seconds=d["seconds"])
    s["part_a"] = {m: dict(apples=ci([r["apples_eaten"] for r in d["part_a"] if r["m"] == m])[0],
                           alive=sum(r["orchard_alive"] for r in d["part_a"] if r["m"] == m))
                   for m in sorted({r["m"] for r in d["part_a"]})}
    s["part_a_greedy_apples"] = float(np.mean([r["apples_eaten"] for r in d["part_a_greedy"]]))
    sch = {}
    for name in sorted({r["condition"] for r in d["part_b"]}):
        rs = [r for r in d["part_b"] if r["condition"] == name]
        sch[name] = {l: dict(R_c=ci([r["R_c"] for r in rs if r["cooperators"] == l])[0],
                             R_d=ci([r["R_d"] for r in rs if r["cooperators"] == l])[0],
                             alive=sum(r["orchard_alive"] for r in rs if r["cooperators"] == l)) for l in range(8)}
    s["schelling"] = sch
    s["breakeven"] = breakeven
    (OUT / "summary.json").write_text(json.dumps(s, indent=1, default=float))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    d = json.loads((Path(a.run) / "week2.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    fig_rule_and_schelling(d)
    fig_checks_by_freeze(d)
    s = summary(d, fig_penalty(d))
    print(json.dumps({k: v for k, v in s.items() if k != "schelling"}, indent=1, default=float))
    for name, t in s["schelling"].items():
        print(name)
        for l, v in t.items():
            print(f"  l={l} R_c={v['R_c']:.1f} R_d={v['R_d']:.1f} alive={v['alive']}")


if __name__ == "__main__":
    main()
