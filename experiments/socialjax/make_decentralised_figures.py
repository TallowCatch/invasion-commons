"""Figures for the decentralised-overseer bot sweeps (notes/socialjax/<game>/).

  .venv-socialjax/bin/python -m experiments.socialjax.make_decentralised_figures --run results/runs/socialjax_decentralised

Bars are means over seeds with 95% bootstrap intervals. Conditions are categories, so they are never joined by lines;
the Clean Up window w is a real scale but only a few values were run, so it is shown as points.
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

from experiments.socialjax.make_week2_figures import AQUA, BLUE, GRID, INK, INK2, ORANGE, VIOLET, YELLOW  # noqa: E402

RNG = np.random.default_rng(20261122)
NOTES = Path("notes/socialjax")
LIGHT_BLUE, PALE, GREY = "#8fb8ea", "#c9dcf4", "#a7a6a1"


def ci(x, b=4000):
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    boots = RNG.choice(x, (b, len(x))).mean(1)
    return x.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def load(run, game):
    d = json.loads((Path(run) / f"{game}.json").read_text())
    cells = defaultdict(list)
    for r in d["rows"]:
        cells[(r["population"], r["condition"])].append(r)
    return d, cells


def caught(r):
    return r["detected"] / r["violations"] if r["violations"] else np.nan


def grouped(ax, cells, pop, fn, groups, variants, ylabel, title, pct=False, legend=True, ylim=None, headroom=1.0):
    """groups: [(label, k)], variants: [(label, condition template with {k}, colour)]."""
    width = 0.8 / len(variants)
    for j, (vlabel, tmpl, col) in enumerate(variants):
        xs, means, los, his = [], [], [], []
        for i, (_, k) in enumerate(groups):
            rows = cells.get((pop, tmpl.format(k=k)))
            if not rows:
                continue
            m, lo, hi = ci([fn(r) for r in rows])
            scale = 100 if pct else 1
            xs.append(i + (j - (len(variants) - 1) / 2) * width)
            means.append(m * scale), los.append((m - lo) * scale), his.append((hi - m) * scale)
        ax.bar(xs, means, width * 0.92, color=col, label=vlabel, yerr=[los, his], capsize=1.5,
               error_kw=dict(lw=0.8, ecolor=INK2))
    ax.set_xticks(range(len(groups)), [g for g, _ in groups])
    ax.set(ylabel=ylabel, title=title)
    if ylim:
        ax.set_ylim(*ylim)
    if headroom != 1.0:
        ax.set_ylim(top=ax.get_ylim()[1] * headroom)
    if pct:
        ax.set_yticks(range(0, 101, 20))
    if legend:
        ax.legend(fontsize=6.5, ncol=1, loc="upper left")


RANDOM = [("central", "central random k={k}", BLUE), ("2 regions", "R=2 random k={k}", LIGHT_BLUE),
          ("4 regions", "R=4 random k={k}", PALE)]
HISTORY = [("central", "central history k={k}", VIOLET), ("2 regions, independent", "R=2 history k={k}", ORANGE),
           ("2 regions, share suspicion", "R=2 history+share k={k}", YELLOW),
           ("2 regions, share suspicion + coverage", "R=2 history+share+coverage k={k}", AQUA),
           ("4 regions, independent", "R=4 history k={k}", "#f2b8a0"),
           ("4 regions, share suspicion + coverage", "R=4 history+share+coverage k={k}", "#9fdcc4")]
MIXED = [("central, random", "central random k={k}", BLUE), ("2 regions, random", "R=2 random k={k}", LIGHT_BLUE),
         ("central, history", "central history k={k}", VIOLET), ("2 regions, history", "R=2 history k={k}", ORANGE),
         ("2 regions, + share suspicion", "R=2 history+share k={k}", YELLOW),
         ("2 regions, + share coverage", "R=2 history+share+coverage k={k}", AQUA)]


def unused(r):
    u = sum(r["regions"]["unused_checks"])
    return u / (u + r["checks"]) if (u + r["checks"]) else 0.0


def fig_harvest(run, game, n, cheat_pop, sneak_pop, migr_pop, ks, out):
    d, cells = load(run, game)
    groups = [(f"k = {k}", k) for k in ks]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
    grouped(axes[0, 0], cells, cheat_pop, caught, groups, RANDOM, "over-harvests caught (%)",
            f"(a) Random checks: {cheat_pop}", pct=True, ylim=(0, 105))
    grouped(axes[0, 1], cells, cheat_pop, caught, groups, HISTORY, "over-harvests caught (%)",
            f"(b) Repeat-offenders-first checks: {cheat_pop}", pct=True, ylim=(0, 105), headroom=1.45)
    grouped(axes[0, 2], cells, cheat_pop, lambda r: float(r["alive"]), groups, HISTORY,
            "share of runs with the orchard alive at the end (%)", f"(c) Orchard survival: {cheat_pop}", pct=True,
            legend=False, ylim=(0, 105))
    grouped(axes[1, 0], cells, migr_pop, lambda r: r["violations"], groups, MIXED, "over-harvests per game",
            f"(d) {migr_pop} (cheat where unchecked by that region)", legend=True, headroom=1.5)
    grouped(axes[1, 1], cells, sneak_pop, lambda r: r["violations"], groups, MIXED, "over-harvests per game",
            f"(e) {sneak_pop} (cheat when unchecked by anyone)", legend=False)
    regs = [("2 regions", "R=2 random k={k}", LIGHT_BLUE), ("4 regions", "R=4 random k={k}", PALE)]
    grouped(axes[1, 2], cells, cheat_pop, unused, groups, regs, "checks unused (% of budget)",
            "(f) Budget wasted on regions with too few agents", pct=True)
    for ax in axes.ravel():
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Central vs decentralised overseers at the same total budget (k checks per step), {d['n_agents']} "
                 f"agents, {('Harvest' if game == 'harvest' else 'Harvest, doubled map')}; scripted bots, "
                 f"{len(next(iter(cells.values())))} seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_coins(run, out):
    d, cells = load(run, "coins")
    groups = [("k = 1", 1), ("k = 2", 2)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    grouped(axes[0], cells, "1 greedy + 1 honest", caught, groups, MIXED, "thefts caught (%)",
            "(a) Thefts caught: 1 greedy + 1 honest", pct=True, ylim=(0, 105), headroom=1.45)

    def net(r, who):  # coins after an average penalty of 1 coin per theft (penalty 2/k per catch)
        k = r["config"]["k"] or 1
        return r["reward_per_agent"][who] - r["caught_per_agent"][who] * 2 / k
    variants = [("central, random", "central random k={k}", BLUE), ("2 regions, random", "R=2 random k={k}", LIGHT_BLUE),
                ("central, history", "central history k={k}", VIOLET), ("2 regions, history", "R=2 history k={k}", ORANGE)]
    ax = axes[1]
    grouped(ax, cells, "1 greedy + 1 honest", lambda r: net(r, 0), groups, variants,
            "greedy player's coins after the penalty", "(b) Does theft pay? (average penalty 1 coin per theft)",
            headroom=1.6)
    none = cells[("1 greedy + 1 honest", "none")]
    honest = np.mean([np.mean(r["reward_per_agent"]) for r in cells[("2 honest", "none")]])
    for y, lab, col in ((np.mean([r["reward_per_agent"][0] for r in none]), "greedy, no overseer", ORANGE),
                        (honest, "each player when both are honest", AQUA)):
        ax.axhline(y, color=col, lw=1, ls="--")
        ax.text(0.98, y, lab + " ", color=col, fontsize=7, ha="right", va="bottom", transform=ax.get_yaxis_transform())
    grouped(axes[2], cells, "2 migrator", lambda r: r["violations"], groups, MIXED, "thefts per game",
            "(c) 2 migrators", legend=False)
    grouped(axes[3], cells, "1 greedy + 1 honest", unused, groups,
            [("2 regions", "R=2 random k={k}", LIGHT_BLUE)], "checks unused (% of budget)",
            "(d) Budget wasted on empty halves", pct=True)
    for a in axes:
        a.grid(axis="x", visible=False)
    fig.suptitle("Coins (2 players): central vs decentralised overseers (top and bottom halves) at the same total "
                 "budget; scripted bots, 10 seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_cleanup(run, out):
    d, cells = load(run, "cleanup")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
    W = d["lookback"]
    # (a) catch share vs look length
    ax = axes[0, 0]
    for label, tmpl, col, ws in (("proof only", "central random k=2 w={w}", BLUE, (1, 50, 100, 150, 200)),
                                 ("presume guilt", "central random k=2 w={w} presume", ORANGE, (10, 50, 100))):
        for w in ws:
            name = "central random k=2" if (w == 150 and "presume" not in tmpl) else tmpl.format(w=w)
            m, lo, hi = ci([caught(r) for r in cells[("1 greedy + 6 honest", name)]])
            ax.errorbar([w], [100 * m], yerr=[[100 * (m - lo)], [100 * (hi - m)]], fmt="o", color=col, ms=5,
                        capsize=2, label=label if w == ws[0] else None)
    ax.axvline(W, color=INK2, lw=0.8, ls=":")
    ax.text(W, 50, f" rule's window\n W = {W}", fontsize=7, color=INK2)
    ax.set(xlabel="steps of record one check reads (w)", ylabel="free-rides caught (%)",
           title="(a) A check must read the rule's whole window", ylim=(-5, 105))
    ax.legend(fontsize=7)
    # (b) wrong flags and apples under presumption
    ax = axes[0, 1]
    for pop, col in (("7 honest", AQUA), ("1 greedy + 6 honest", ORANGE)):
        for w in (10, 50, 100):
            m, lo, hi = ci([r["wrongly_stopped"] for r in cells[(pop, f"central random k=2 w={w} presume")]])
            ax.errorbar([w], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=col, ms=5, capsize=2,
                        label=pop if w == 10 else None)
    ax.set(xlabel="steps of record one check reads (w)", ylabel="honest agents wrongly flagged per game",
           title="(b) Presuming guilt flags honest cleaners")
    ax.legend(fontsize=7)
    # (c) apples by population and overseer
    ax = axes[0, 2]
    conds = [("none", "none", GREY), ("proof, w = 150", "central random k=2", BLUE),
             ("presume, w = 50", "central random k=2 w=50 presume", ORANGE),
             ("presume, w = 10", "central random k=2 w=10 presume", "#f2b8a0")]
    pops = ["7 honest", "1 greedy + 6 honest", "7 sneaky"]
    width = 0.8 / len(conds)
    for j, (lab, name, col) in enumerate(conds):
        ms, es = [], [[], []]
        for pop in pops:
            m, lo, hi = ci([r["taken"] for r in cells[(pop, name)]])
            ms.append(m), es[0].append(m - lo), es[1].append(hi - m)
        ax.bar(np.arange(len(pops)) + (j - 1.5) * width, ms, width * 0.92, color=col, label=lab, yerr=es,
               capsize=1.5, error_kw=dict(lw=0.8, ecolor=INK2))
    ax.set_xticks(range(len(pops)), pops)
    ax.set(ylabel="apples eaten per game", title="(c) Apples eaten (2 random checks per step)")
    ax.legend(fontsize=6.5)
    # (d) the free-rider's apples vs the honest agents'
    ax = axes[1, 0]
    names = [("none", "none"), ("central random k=1", "central\nrandom k=1"), ("central random k=2", "central\nrandom k=2"),
             ("R=2 random k=2", "2 regions\nrandom k=2"), ("R=2 history+share k=2", "2 regions\nhistory+share")]
    for j, (who, col, sel) in enumerate((("greedy free-rider", ORANGE, lambda r: r["reward_per_agent"][0]),
                                         ("honest (mean of 6)", AQUA, lambda r: np.mean(r["reward_per_agent"][1:])))):
        ms, es = [], [[], []]
        for name, _ in names:
            m, lo, hi = ci([sel(r) for r in cells[("1 greedy + 6 honest", name)]])
            ms.append(m), es[0].append(m - lo), es[1].append(hi - m)
        ax.bar(np.arange(len(names)) + (j - 0.5) * 0.4, ms, 0.37, color=col, label=who, yerr=es, capsize=1.5,
               error_kw=dict(lw=0.8, ecolor=INK2))
    ax.set_xticks(range(len(names)), [lab for _, lab in names], fontsize=7)
    ax.set(ylabel="apples per agent per game", title="(d) 1 greedy + 6 honest: who eats (window 150)")
    ax.legend(fontsize=7)
    # (e) shirking within the rule: sneaky and migrator bots
    groups = [("k = 1", 1), ("k = 2", 2)]
    variants = [("central, random", "central random k={k}", BLUE), ("2 regions, random", "R=2 random k={k}", LIGHT_BLUE),
                ("central, history", "central history k={k}", VIOLET), ("2 regions, history", "R=2 history k={k}", ORANGE)]
    grouped(axes[1, 1], cells, "7 migrator", lambda r: r["taken"], groups, variants, "apples eaten per game",
            "(e) 7 migrators: no rule broken, cleaning skipped", headroom=1.35)
    honest = np.mean([r["taken"] for r in cells[("7 honest", "none")]])
    axes[1, 1].axhline(honest, color=AQUA, lw=1, ls="--")
    axes[1, 1].text(0.98, honest, "7 honest ", color=AQUA, fontsize=7, ha="right", va="bottom",
                    transform=axes[1, 1].get_yaxis_transform())
    grouped(axes[1, 2], cells, "7 sneaky", lambda r: r["taken"], groups, variants, "apples eaten per game",
            "(f) 7 sneaky bots", legend=False)
    axes[1, 2].axhline(honest, color=AQUA, lw=1, ls="--")
    for a in axes.ravel():
        a.grid(axis="x", visible=False)
    fig.suptitle(f"Clean Up (7 agents): rule = no eating while pollution > 0.2 unless you cleaned in the last {W} "
                 "steps; scripted bots, 10 seeds, caught agents frozen 50 steps", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/runs/socialjax_decentralised")
    a = ap.parse_args()
    fig_harvest(a.run, "harvest", 7, "1 greedy + 6 honest", "7 sneaky", "7 migrator", (1, 2, 4),
                NOTES / "harvest" / "fig_decentralised.png")
    fig_harvest(a.run, "harvest_large", 14, "2 greedy + 12 honest", "14 sneaky", "14 migrator", (2, 4),
                NOTES / "harvest" / "fig_decentralised_large_map.png")
    fig_coins(a.run, NOTES / "coins" / "fig_decentralised.png")
    fig_cleanup(a.run, NOTES / "cleanup" / "fig_decentralised.png")


if __name__ == "__main__":
    main()
