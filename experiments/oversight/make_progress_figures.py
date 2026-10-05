"""Progress figures for the Claude audit experiments (R1, S1, S1b, S2). October 2026.

Writes PDF + PNG to notes/claude_audit_20261005/figures/. Every number plotted is read from saved run tables.
Run:  PYTHONPATH=. python -m experiments.oversight.make_progress_figures
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("notes/claude_audit_20261005/figures")
RUNS = Path("results/runs")
# Reference categorical palette (dataviz skill, light mode), first three slots validated all-pairs.
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
REV = {"joint": ("Joint", BLUE), "local_bounded": ("Bounded local", ORANGE), "local_optimistic": ("Optimistic local", AQUA)}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "legend.frameon": False, "lines.linewidth": 2,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def caption(fig, text, y=-0.02):
    fig.text(0.01, y, text, ha="left", va="top", fontsize=7.5, color=INK2, wrap=True)


# ----------------------------------------------------------------- Figure 1: calibration
def fig1():
    sept = pd.read_csv("paper/paper_v5_scalable_oversight_commons/data/analysis/primary_decision_quality.csv")
    r1 = pd.read_csv(RUNS / "claude_r1_repaired_reviewer_v1/analysis/open_loop_decisions_table.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True, sharey=True)
    panels = [("23 Sept: fixed weather buffer", sept[(sept.game == "harvest") & (sept.inspection_budget == 6)],
               "method", "safe_reject_rate", "harmful_accept_rate"),
              ("R1: calibrated 5% chance constraint", r1[(r1.game == "harvest") & (r1.budget == 6) & (r1.fill == "max")],
               "reviewer", "usefulness_loss_rate", "unsafe_approval_rate")]
    # label anchor positions (data units) chosen per panel so labels never collide
    spots = [{"local_optimistic": (22, 4.0), "joint": (48, 9.0), "local_bounded": (80, 4.0)},
             {"local_optimistic": (8, 11.5), "joint": (8, 4.5), "local_bounded": (70, 4.0)}]
    for ax, (title, df, key, xcol, ycol), spot in zip(axes, panels, spots):
        for _, row in df.iterrows():
            lab, col = REV[row[key]]
            x, y = 100 * row[xcol], 100 * row[ycol]
            ax.scatter(x, y, s=55, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            tx, ty = spot[row[key]]
            ax.annotate(f"{lab}\n{x:.0f}% cut\n{y:.1f}% let through", (x, y), xytext=(tx, ty),
                        textcoords="data", fontsize=7.2, color=INK, ha="left", va="bottom",
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=4))
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlim(-5, 108); ax.set_ylim(-1.5, 16)
        ax.set_xlabel("Safe requests cut (usefulness loss, %)")
    axes[0].set_ylabel("Risky requests let through\n(unsafe approvals, %)")
    fig.suptitle("Harvest, all 6 requests inspected: the old buffer hid real differences between reviewers",
                 x=0.01, ha="left", fontsize=10, color=INK)
    caption(fig, "Open-loop scoring on requests from unregulated runs (initially safe states). Left: 23 Sept confirmation "
                 "(3,687 safe / 1,368 risky). Right: R1, 64 fresh contexts (2,016 safe / 720 risky). Bottom-left corner is best.")
    fig.tight_layout()
    save(fig, "fig1_calibration_harvest")


# ----------------------------------------------------------------- Figure 2: Fishery target
def fig2():
    o = pd.read_csv(RUNS / "claude_r1_repaired_reviewer_v1/analysis/closed_loop_outcomes.csv")
    o = o[(o.game == "fishery") & (o.reviewer == "joint")]
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    spec = [("msy", "max", BLUE, "-", "MSY target, unseen = maximum"),
            ("msy", "previous", BLUE, "--", "MSY target, unseen = last request"),
            ("one_step", "max", ORANGE, "-", "Old line (stock ≥ 10), unseen = maximum"),
            ("one_step", "previous", ORANGE, "--", "Old line, unseen = last request")]
    for tgt, fill, col, ls, lab in spec:
        d = o[(o.target == tgt) & (o.fill == fill)].sort_values("budget")
        ax.plot(d.budget, d.total_harvest, color=col, ls=ls, marker="o", ms=5, label=lab)
    ax.axhline(17.5 * 80, color=INK2, lw=1, ls=":")
    ax.text(0, 1450, "Sustainable maximum ≈ 1,400 (dotted)", va="bottom", fontsize=7.2, color=INK2)
    ax.set_xticks([0, 3, 6]); ax.set_xlim(-0.3, 6.3); ax.set_ylim(0, 1600)
    ax.set_xlabel("Requests inspected (of 6)"); ax.set_ylabel("Total harvest over 80 steps")
    ax.set_title("Fishery, joint reviewer: the safety target sets long-run harvest", loc="left", color=INK)
    ax.legend(loc="lower left", fontsize=7.2)
    caption(fig, "R1 closed loop, mean over 64 contexts. Under the old line, more inspection lowered harvest only when unseen "
                 "requests were assumed to be the maximum (that pessimism acted as accidental conservation).")
    fig.tight_layout()
    save(fig, "fig2_fishery_target")


# ----------------------------------------------------------------- Figure 3: limited checking
def fig3():
    r1 = pd.read_csv(RUNS / "claude_r1_repaired_reviewer_v1/analysis/open_loop_decisions_table.csv")
    h = r1[(r1.game == "harvest") & (r1.reviewer == "joint")]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for fill, col, lab in (("max", ORANGE, "Assume unseen = maximum"), ("previous", BLUE, "Assume unseen = last request")):
        d = h[h.fill == fill].sort_values("budget")
        x, y = 100 * d.usefulness_loss_rate.values, 100 * d.unsafe_approval_rate.values
        ax.plot(x, y, color=col, marker="o", ms=6, label=lab)
        for xi, yi, k in zip(x, y, d.budget):
            if k == 6 and fill == "max":
                continue  # same point as the other fill; labelled once
            txt = "k=6 (both fills)" if k == 6 else ("k=0 and k=3" if (fill == "max" and k == 3) else f"k={k}")
            if fill == "max" and k == 0:
                continue
            ax.annotate(txt, (xi, yi), xytext=(-4 if fill == "max" else 6, 6), textcoords="offset points",
                        fontsize=7.2, color=INK, ha="right" if fill == "max" else "left")
    ax.set_xlabel("Safe requests cut (usefulness loss, %)")
    ax.set_ylabel("Risky requests let through (%)")
    ax.set_xlim(-4, 108); ax.set_ylim(-2, 42)
    ax.set_title("Harvest, joint reviewer: what it assumes about unchecked\nrequests decides which mistake it makes",
                 loc="left", color=INK)
    ax.legend(loc="upper right", fontsize=7.2)
    caption(fig, "R1 open loop, 64 contexts; k = number of the 6 requests inspected. At k = 6 both fills coincide.")
    fig.tight_layout()
    save(fig, "fig3_limited_checking")


# ----------------------------------------------------------------- Figure 4: S1 protocols
def fig4():
    T = pd.read_csv(RUNS / "claude_s1_reporting_audit_v1/analysis/condition_table.csv")
    order = [("full", "Full verification (6 audits)"), ("report", "Trust reports (0 audits)"), ("rand1", "1 random audit"),
             ("rand2", "2 random audits"), ("targ1", "1 targeted audit"), ("targ2", "2 targeted audits"),
             ("peer", "Neighbour reports"), ("peer_collude", "Neighbour reports, collusion")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, game in zip(axes, ("harvest", "fishery")):
        rows = []
        for key, lab in order:
            m = T[(T.game == game) & (T.protocol == key) & (T.actor == "fixed")]
            if key != "full":
                m = m[np.isclose(m.d, 0.5)]
            if len(m):
                rows.append((lab, 100 * m.unsafe_approval_rate.iloc[0], key))
        labs, vals, keys = zip(*rows)
        cols = [ORANGE if k == "report" else (INK2 if k == "full" else BLUE) for k in keys]
        y = np.arange(len(labs))[::-1]
        ax.barh(y, vals, color=cols, height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v + 0.3, yi, f"{v:.1f}%", va="center", fontsize=7.2, color=INK)
        ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7.4)
        ax.set_xlim(0, max(vals) * 1.25 + 0.5)
        ax.grid(axis="y", visible=False)
        ax.set_title({"harvest": "Harvest (2 of 6 under-report by half)", "fishery": "Fishery (4 of 6 under-report by half)"}[game],
                     loc="left", color=INK)
        ax.set_xlabel("Steps with an unsafe executed action (%)")
    fig.suptitle("S1: acting on unverified reports is unsafe; checks restore safety (mainly by excluding cheaters)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    caption(fig, "64 test contexts per game, d = 0.5. Caught agents were excluded for 6 steps and fined 12; in the arms near 0%, "
                 "the reviewer never needed to cut a request, so safety came from exclusion, not better decisions (file 10, §5b).")
    fig.tight_layout()
    save(fig, "fig4_s1_protocols")


# ----------------------------------------------------------------- Figure 5: S1b ablation
def fig5():
    p = RUNS / "claude_s1b_ablation_msy_v1/analysis/s1b_condition_table.csv"
    if not p.exists():
        return
    T = pd.read_csv(p)
    A = T[(T.part == "A") & (T.actor == "fixed") & np.isclose(T.d, 0.5)]
    s1 = pd.read_csv(RUNS / "claude_s1_reporting_audit_v1/analysis/condition_table.csv")
    protos = [("harvest", "rand2", "Harvest\n2 random audits"), ("harvest", "targ1", "Harvest\n1 targeted audit"),
              ("harvest", "peer", "Harvest\nneighbour reports"), ("fishery", "rand2", "Fishery\n2 random audits"),
              ("fishery", "peer", "Fishery\nneighbour reports")]
    bars = [("Trust reports\n(no checks)", ORANGE), ("Audit result used,\nno sanction", BLUE), ("Caught agents\nexcluded 6 steps", AQUA)]
    fig, axes = plt.subplots(1, len(protos), figsize=(9.6, 3.0))
    for ax, (game, proto, title) in zip(axes, protos):
        base = s1[(s1.game == game) & (s1.protocol == "report") & np.isclose(s1.d, .5) & (s1.actor == "fixed")].unsafe_fixed.iloc[0]
        honest = s1[(s1.game == game) & (s1.protocol == "full")].unsafe_fixed.iloc[0]
        vals = [base,
                A[(A.game == game) & (A.protocol == proto) & A.belief & (A.sanction == "none")].unsafe_fixed.iloc[0],
                A[(A.game == game) & (A.protocol == proto) & A.belief & (A.sanction == "excl+fine")].unsafe_fixed.iloc[0]]
        x = np.arange(3)
        ax.bar(x, [100 * v for v in vals], width=0.6, color=[c for _, c in bars])
        top = 100 * max(vals) * 1.25 + 0.05
        for xi, v in zip(x, vals):
            ax.text(xi, 100 * v + top * 0.02, f"{100*v:.1f}%", ha="center", va="bottom", fontsize=7, color=INK)
        ax.axhline(100 * honest, color=INK2, lw=1, ls=":")
        ax.set_ylim(0, top)
        ax.set_xticks(x); ax.set_xticklabels(["Trust", "Audit\nused", "Exclude"], fontsize=7)
        ax.set_title(title, loc="left", fontsize=8, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of 80 steps unsafe (%)")
    fig.suptitle("S1b: with this memoryless reviewer, excluding caught cheaters (not using audit results) produced the safety",
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    caption(fig, "Misreporters under-report by half (d = 0.5); 64 contexts per game. 'Audit used' = the audited agent's true request "
                 "enters that step's decision only (no memory, no sanction). 'Exclude' = S1 rule (6 steps out + fine), which also removes "
                 "extraction, so it can go below the honest level (dotted line = everyone honest / full verification). "
                 "Note different y-scales by panel.", y=0.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig5_s1b_ablation")


# ----------------------------------------------------------------- Figure 6: S2 deterrence
def fig6():
    p = RUNS / "claude_s2_compliance_deterrence_v1/adaptive_search.json"
    if not p.exists():
        return
    srch = pd.DataFrame(json.loads(p.read_text()))
    T = pd.read_csv(RUNS / "claude_s2_compliance_deterrence_v1/analysis/s2_condition_table.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.3), gridspec_kw=dict(width_ratios=[1, 1.2]))
    f = srch[(srch.game == "fishery") & (srch.protocol != "allow")]
    piv = f.groupby("fine").d_star.agg(["min", "max"]).reset_index()
    assert (piv["min"] == piv["max"]).all()  # identical for every checking rule
    ax.scatter(piv.fine, piv["max"], color=BLUE, s=45, zorder=3)
    allow = srch[(srch.game == "fishery") & (srch.protocol == "allow")].d_star.iloc[0]
    ax.axhline(allow, color=INK2, lw=1, ls=":")
    ax.text(24, allow + 0.04, "no checks at all", ha="right", fontsize=7.2, color=INK2)
    ax.set_xticks([0, 6, 12, 24]); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Fine per catch (only these 4 values tested)"); ax.set_ylabel("Cheating level chosen (d*)")
    ax.set_title("Fishery: cheating stopped at every\ntested fine (6, 12, 24)", loc="left", color=INK)
    ax.text(7, 0.3, "Identical for 1 or 2 random audits per step\nand neighbour reports (with or without\ncollusion)",
            fontsize=7, color=INK2)
    rows = [("Everyone\ncomplies", T[(T.game == "fishery") & (T.actor == "comply")]),
            ("Cheating,\nno checks", T[(T.game == "fishery") & (T.protocol == "allow") & (T.actor == "adaptive")]),
            ("1 random audit\n+ fine 6", T[(T.game == "fishery") & (T.protocol == "rand1") & (T.fine == 6) & (T.actor == "adaptive")])]
    x = np.arange(3)
    honest = [r.honest_harvest.iloc[0] / 2 for _, r in rows]
    cheat = [r.payoff.iloc[0] / 4 for _, r in rows]
    ax2.bar(x - 0.19, honest, width=0.36, color=AQUA, label="Per honest agent (harvest)")
    ax2.bar(x + 0.19, cheat, width=0.36, color=ORANGE, label="Per would-be cheater (net payoff)")
    for xi, hv, cv in zip(x, honest, cheat):
        ax2.text(xi - 0.19, hv + 6, f"{hv:.0f}", ha="center", fontsize=7.2, color=INK)
        ax2.text(xi + 0.19, cv + 6, f"{cv:.0f}", ha="center", fontsize=7.2, color=INK)
    tot = [r.total_harvest.iloc[0] for _, r in rows]
    ax2.set_xticks(x); ax2.set_xticklabels([f"{n}\n(total {t:,.0f})" for (n, _), t in zip(rows, tot)], fontsize=7.2)
    ax2.set_ylabel("Per agent, over 80 steps"); ax2.set_ylim(0, 470)
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="upper left", fontsize=7.2)
    ax2.set_title("Unchecked cheating moves harvest from\nhonest agents to cheaters", loc="left", color=INK)
    fig.suptitle("S2: hidden over-extraction that pays privately, and fixed-level cheaters' response to audits plus fines",
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    caption(fig, "Fishery, MSY target, 64 held-out contexts; one cheating level per group chosen on 8 training contexts. Audits are "
                 "perfect, the fine is flat per catch, and 1 audit/step covers 17% of agents (far above AI-control budgets). "
                 "In Harvest, cheating never paid at any fine, so deterrence could not be tested there.", y=0.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig6_s2_deterrence")


# ----------------------------------------------------------------- per-context helpers (raw episode/step files)
def _rows(path, keep):
    with gzip.open(path, "rt") as f:
        return [r for r in map(json.loads, f) if keep(r)]


# ----------------------------------------------------------------- Figure 7: Fishery stock over time
def fig7():
    p = RUNS / "claude_r1_repaired_reviewer_v1/closed_loop_decisions.jsonl.gz"
    if not p.exists():
        return
    spec = [((6, "max", "msy"), BLUE, "-", "MSY target, all 6 inspected"),
            ((6, "max", "one_step"), ORANGE, "-", "Old line, all 6 inspected"),
            ((0, "max", "one_step"), ORANGE, "--", "Old line, none inspected\n(unseen = maximum)")]
    want = {k for k, *_ in spec}
    rows = _rows(p, lambda r: r["game"] == "fishery" and r["reviewer"] == "joint"
                 and (r["budget"], r["fill"], r["target"]) in want)
    o = pd.read_csv(RUNS / "claude_r1_repaired_reviewer_v1/analysis/closed_loop_outcomes.csv")
    o = o[(o.game == "fishery") & (o.reviewer == "joint")]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for (k, fill, tgt), col, ls, lab in spec:
        S = pd.DataFrame([(r["context"], r["step"], r["state"]) for r in rows
                          if (r["budget"], r["fill"], r["target"]) == (k, fill, tgt)], columns=["c", "t", "s"])
        W = S.pivot(index="t", columns="c", values="s")
        assert W.shape == (80, 64) and not W.isna().any().any()
        ax.plot(W.index, W.values, color=col, lw=0.4, alpha=0.12, zorder=1)
        med = W.median(axis=1)
        ax.plot(W.index, med, color=col, ls=ls, lw=2, zorder=3)
        h = o[(o.budget == k) & (o.fill == fill) & (o.target == tgt)].total_harvest.iloc[0]
        ax.text(81, med.iloc[-1], f"{lab}\nharvest {h:,.0f}", va="center", fontsize=7.2, color=INK)
    ax.axhline(10, color=INK2, lw=1, ls=":")
    ax.text(1, 9, "old safety line: stock \u2265 10 next step", fontsize=7, color=INK2, va="top")
    ax.set_xlim(0, 79); ax.set_ylim(0, 100)
    ax.set_xlabel("Step"); ax.set_ylabel("Fish stock at start of step (max 100)")
    ax.set_title("Fishery, joint reviewer: the target decides where the stock settles", loc="left", color=INK)
    caption(fig, "R1 closed loop, 64 contexts per line (thin lines = single contexts, bold = median). The MSY target keeps 50 "
                 "after each harvest, which regrows to about 68. The old line lets the stock be run down to just above 10. "
                 "With no inspection, assuming every unseen request is the maximum held the stock near 37 by accident. "
                 "Harvest = mean total over 80 steps.")
    fig.tight_layout()
    save(fig, "fig7_fishery_stock_over_time")


# ----------------------------------------------------------------- Figure 8: S1 per-context breadth
def fig8():
    p = RUNS / "claude_s1_reporting_audit_v1/episodes.jsonl.gz"
    if not p.exists():
        return
    E = pd.DataFrame(_rows(p, lambda r: r["actor"] == "fixed"))
    order = [("full", "Full verification (6 audits)"), ("report", "Trust reports (0 audits)"), ("rand1", "1 random audit"),
             ("rand2", "2 random audits"), ("targ1", "1 targeted audit"), ("targ2", "2 targeted audits"),
             ("peer", "Neighbour reports"), ("peer_collude", "Neighbour reports, collusion")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharex=True)
    for ax, game in zip(axes, ("harvest", "fishery")):
        rows = []
        for key, lab in order:
            m = E[(E.game == game) & (E.protocol == key)]
            if key != "full":
                m = m[np.isclose(m.d, 0.5)]
            if len(m):
                assert len(m) == 64
                rows.append((lab, int((m.exec_risky > 0).sum()), int((m.t_end < 80).sum()), int(m.exec_risky.max()), key))
        labs, hit, coll, mx, keys = zip(*rows)
        cols = [ORANGE if k == "report" else (INK2 if k == "full" else BLUE) for k in keys]
        y = np.arange(len(labs))[::-1]
        ax.barh(y, hit, color=cols, height=0.6)
        for yi, n, c, m in zip(y, hit, coll, mx):
            note = f"{n}/64" + (f" (max {m} step{'s' * (m > 1)})" if n else "") + (f", collapsed {c}" if c else "")
            ax.text(n + 1, yi, note, va="center", fontsize=7, color=INK)
        ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7.4)
        ax.set_xlim(0, 100); ax.set_xticks([0, 16, 32, 48, 64]); ax.spines["bottom"].set_bounds(0, 64)
        ax.grid(axis="y", visible=False)
        ax.set_title({"harvest": "Harvest (2 of 6 under-report by half)", "fishery": "Fishery (4 of 6 under-report by half)"}[game],
                     loc="left", color=INK)
        ax.set_xlabel("Contexts with \u2265 1 unsafe executed action (of 64)")
    fig.suptitle("S1 by context: the harm from trusting reports is broad, not a few outlier populations",
                 x=0.01, ha="left", fontsize=10, color=INK)
    caption(fig, "Same arms as Figure 4, counted per context (the independent unit) instead of pooled over steps. "
                 "'max' = most unsafe executed steps in any one context; 'collapsed' = contexts where the stock collapsed before step 80. "
                 "Full verification is the honest baseline: in Harvest the reviewer's 5% risk allowance still let an unsafe action "
                 "through in 16 contexts (0.4% of steps, Figure 4).")
    fig.tight_layout()
    save(fig, "fig8_s1_per_context")


# ----------------------------------------------------------------- Figure 9: S1b paired per context
def fig9():
    p = RUNS / "claude_s1b_ablation_msy_v1/episodes.jsonl.gz"
    if not p.exists():
        return
    E = pd.DataFrame(_rows(p, lambda r: r["part"] == "A" and r["actor"] == "fixed" and r["d"] == 0.5))
    protos = [("harvest", "rand2", "Harvest, 2 random audits"), ("harvest", "targ1", "Harvest, 1 targeted audit"),
              ("harvest", "peer", "Harvest, neighbour reports"), ("fishery", "rand2", "Fishery, 2 random audits"),
              ("fishery", "peer", "Fishery, neighbour reports")]

    def arm(game, proto, belief, sanction):
        m = E[(E.game == game) & (E.protocol == proto) & (E.belief == belief) & (E.sanction == sanction)]
        assert len(m) == 64
        return m.set_index("context").unsafe_fixed.sort_index()

    labels, counts = [], []
    for game, proto, title in protos:
        trust = arm(game, proto, False, "none")
        for name, (b, s) in (("audit result used", (True, "none")), ("caught agents excluded", (True, "excl+fine"))):
            diff = arm(game, proto, b, s) - trust
            counts.append(((diff < -1e-9).sum(), (diff.abs() <= 1e-9).sum(), (diff > 1e-9).sum()))
            labels.append(f"{title}: {name}")
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    y = np.arange(len(labels))[::-1] + np.repeat(np.arange(len(protos))[::-1] * 0.5, 2)
    segs = [("safer than trusting", BLUE), ("unchanged", GRID), ("less safe", ORANGE)]
    left = np.zeros(len(labels))
    for i, (name, col) in enumerate(segs):
        w = np.array([c[i] for c in counts], float)
        ax.barh(y, w, left=left, color=col, height=0.7, edgecolor=SURFACE, linewidth=1, label=name)
        for yi, l, wi in zip(y, left, w):
            if wi >= 2:
                ax.text(l + wi / 2, yi, f"{wi:.0f}", ha="center", va="center", fontsize=7,
                        color=SURFACE if col != GRID else INK)
        left += w
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_xlim(0, 64); ax.set_xticks([0, 16, 32, 48, 64])
    ax.set_xlabel("Contexts (of 64), compared with the same context when reports are trusted")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=7.2)
    fig.suptitle("S1b, context by context: excluding caught agents made most contexts safer; a memoryless\n"
                 "reviewer using audit results changed nothing in many and made some less safe",
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    caption(fig, "Outcome = share of the 80 steps in an unsafe state. Each context is the same population and seeds in all three "
                 "arms, so the comparison is paired. 'Audit result used' = memoryless, no sanction. Misreporters under-report by half.",
            y=-0.06)
    fig.tight_layout()
    save(fig, "fig9_s1b_paired")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9):
        f()
    print(sorted(p.name for p in OUT.glob("*.png")))
