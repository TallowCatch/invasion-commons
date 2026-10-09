"""Paper exhibits (figures 2-6, tables 1-2) for paper/paper_v6, drawn only from saved run tables.

Style follows the related literature (AI Control, Games for AI Control, GovSim, Makins et al.):
vector PDF at the paper's text width, (a)(b) panels, one fixed colour per entity across figures,
95% paired context-bootstrap intervals as bars or bands, booktabs tables with the best value in bold.
Figure 1 (the protocol schematic) is TikZ: paper/paper_v6/figures/fig1_game_step.tex.

Run:  PYTHONPATH=. python -m experiments.oversight.make_paper_exhibits
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

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / "notes/claude_audit_20261005/runs"
RAW = ROOT / "results/runs"
FIG = ROOT / "paper/paper_v6/figures"
TAB = ROOT / "paper/paper_v6/tables"
TEXTW = 5.5  # inches, NeurIPS-style single-column text width

# One colour per entity, the same in every figure (validated categorical slots, fixed order).
BLUE, ORANGE, AQUA, VIOLET, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100"
INK, INK2, GRID = "#1a1a1a", "#5f5e5a", "#e6e5e1"
REVIEWER = {"joint": ("Joint", BLUE, "o"), "local_bounded": ("Cautious local", ORANGE, "s"),
            "local_optimistic": ("Optimistic local", AQUA, "D")}
REGIME = {"none": ("No audits", INK2), "fine": ("Fine", ORANGE), "memory": ("Lower allowance for caught agents", AQUA),
          "memory_cap": ("Check caught agents every round", VIOLET), "memory_cut": ("Lower allowance for all agents", YELLOW),
          "fine+memory_cut": ("Fine and lower allowance for all agents", BLUE)}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 1.3, "lines.markersize": 4,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "white",
})


def panel(ax, letter, x=-0.16):
    ax.text(x, 1.04, f"({letter})", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ shared helpers for the composite figures
LIGHT = "#b9b7b1"  # light grey for "no reviewer / nobody cheats / audit without memory" baselines


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def grouped(ax, labels, series, width=0.8, fmt=None, ylim=None, hatch=None):
    """series: list of (name, colour, values, (lo, hi) or None). Draws grouped vertical bars."""
    n = len(series)
    w = width / n
    x = np.arange(len(labels))
    for k, (name, col, vals, ci) in enumerate(series):
        xs = x - width / 2 + w * (k + 0.5)
        err = None
        if ci is not None:
            lo, hi = np.asarray(ci[0], float), np.asarray(ci[1], float)
            err = [np.clip(np.asarray(vals) - lo, 0, None), np.clip(hi - np.asarray(vals), 0, None)]
        ax.bar(xs, vals, width=w * 0.92, color=col, label=name, yerr=err, capsize=1.2,
               error_kw=dict(lw=0.6, ecolor=INK), hatch=(hatch or {}).get(name), edgecolor="white" if not hatch else None,
               linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.6)
    ax.grid(axis="x", visible=False)
    if ylim:
        ax.set_ylim(*ylim)


def title(ax, letter, text=None):
    """Panel letter only (paper style, 9 Oct 2026): the description lives in the caption. `text` is kept for reference."""
    ax.set_title(f"({letter})", loc="left", fontsize=8, fontweight="bold", pad=4)


# ------------------------------------------------------------------ Figure 2: what the reviewer aims for and knows (claims 1-2)
def fig2_reviewer():
    SAFE, MSY_C = INK2, BLUE  # targets
    REV = {"joint": ("Joint", VIOLET), "local_bounded": ("Cautious local", ORANGE), "local_optimistic": ("Optimistic local", AQUA)}
    traj = {}
    with gzip.open(RAW / "claude_r1_repaired_reviewer_v1/closed_loop_decisions.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["game"] == "fishery" and r["reviewer"] == "joint" and r["budget"] == 6 and r["fill"] == "max":
                traj.setdefault(r["target"], {}).setdefault(r["context"], {})[r["step"]] = r["state"]
    A = pd.read_csv(NOTES / "claude_r2_v1/A_closed_loop_outcomes.csv")
    B = pd.read_csv(NOTES / "claude_r2_v1/B_open_loop.csv")
    C = pd.read_csv(NOTES / "claude_r2_v1/B_closed_loop.csv")
    L = pd.read_csv(NOTES / "claude_r2_v1/B_learned_windows.csv").set_index("cell")

    fig, axs = plt.subplots(2, 3, figsize=(TEXTW, 3.9))
    (a, b, c), (d, e, f) = axs
    # (a) stock over time
    for target, lab, col in (("msy", "MSY target", MSY_C), ("one_step", "Safety line", SAFE)):
        W = pd.DataFrame(traj[target]).sort_index()
        a.fill_between(W.index, W.quantile(0.1, axis=1), W.quantile(0.9, axis=1), color=col, alpha=0.18, lw=0)
        a.plot(W.index, W.quantile(0.5, axis=1), color=col, label=lab)
    a.axhline(10, color=INK2, lw=0.6, ls=":")
    a.set_xlim(0, 79); a.set_ylim(0, 100); a.set_xlabel("Step"); a.set_ylabel("Fish stock")
    title(a, "a", "Fish stock over a run")
    # (b) harvest in six settings
    cells = ["fishery_s4_r0.5", "fishery_s4_r0.7", "fishery_s4_r0.9", "fishery_s2_r0.5", "fishery_s2_r0.7", "fishery_s2_r0.9"]
    labs = ["4g\n0.5", "4g\n0.7$^\\dagger$", "4g\n0.9", "2g\n0.5", "2g\n0.7", "2g\n0.9"]
    get = lambda cell, t, bud: float(A[(A.cell == cell) & (A.target == t) & (A.budget == bud)].total_harvest.iloc[0]) / 1000
    grouped(b, labs, [("No reviewer", LIGHT, [get(x, "none", 0.0) for x in cells], None),
                      ("Safety line", SAFE, [get(x, "one_step", 6.0) for x in cells], None),
                      ("MSY target", MSY_C, [get(x, "msy", 6.0) for x in cells], None)])
    b.set_ylabel("Total harvest (thousands)"); b.set_xlabel("Greedy agents, regrowth rate", labelpad=1)
    title(b, "b", "Harvest, six Fishery settings")
    # (c) collapse when the model is wrong
    conds = ["exact", "regen_low", "regen_high", "K_low", "K_high", "allee"]
    clabs = ["correct", "regrowth −25%", "regrowth +25%", "capacity −25%", "capacity +25%", "tipping point"]
    F = C[C.cell == "fishery_s4_r0.7"].set_index(["condition", "target"])
    grouped(c, clabs, [("Safety line", SAFE, [100 * F.loc[(k, "one_step"), "collapse_rate"] for k in conds], None),
                       ("MSY target", MSY_C, [100 * F.loc[(k, "msy"), "collapse_rate"] for k in conds], None)], ylim=(0, 100))
    c.set_ylabel("Runs that collapsed (%)")
    plt.setp(c.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor", fontsize=6.2)
    c.text(0.98, 0.95, "MSY target: 0%\nin every case", transform=c.transAxes, ha="right", va="top", fontsize=6.2, color=MSY_C)
    title(c, "c", "Fishery: wrong model")
    # (d)(e) Forest decisions when the model is wrong
    H = B[B.cell == "harvest_s2_m0.85"]
    mconds = ["exact", "noise_low", "regen_high", "noise_high", "regen_low"]
    mlabs = ["correct", "noise −50%", "regrowth +25%", "noise ×2", "regrowth −25%"]
    for ax, col, letter, ttl, ylab in ((d, "unsafe_approval_rate", "d", "Forest: risky let through", "Risky requests approved (%)"),
                                       (e, "usefulness_loss_rate", "e", "Forest: safe blocked", "Safe requests cut (%)")):
        series = []
        for rev, (lab, rc) in REV.items():
            g = H[H.reviewer == rev].set_index("condition")
            series.append((lab, rc, [100 * g.loc[k, col] for k in mconds], None))
        grouped(ax, mlabs, series, ylim=(0, 105))
        ax.set_ylabel(ylab)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor", fontsize=6.2)
        title(ax, letter, ttl)
    # (f) learning the model while running
    rows = [("harvest_s2_m0.85", "Forest,\npilot setting"), ("harvest_s4_m1.0", "Forest,\nnew setting")]
    vals, cis = {"Steps 0–19": [], "Steps 60–79": []}, {"Steps 0–19": ([], []), "Steps 60–79": ([], [])}
    for cell, _ in rows:
        r = L.loc[cell]
        for key, k, n in (("Steps 0–19", r.early_risky_approved, r.early_risky), ("Steps 60–79", r.late_risky_approved, r.late_risky)):
            vals[key].append(100 * k / n); lo, hi = wilson(k, n); cis[key][0].append(100 * lo); cis[key][1].append(100 * hi)
    grouped(f, [x[1] for x in rows], [("Steps 0–19", "#c9c1ea", vals["Steps 0–19"], cis["Steps 0–19"]),
                                      ("Steps 60–79", VIOLET, vals["Steps 60–79"], cis["Steps 60–79"])], width=0.6, ylim=(0, 22))
    f.set_ylabel("Risky requests approved (%)"); f.legend(loc="upper right", fontsize=6.2)
    title(f, "f", "A joint reviewer that learns")
    # shared legends
    h1 = [plt.Rectangle((0, 0), 1, 1, color=x) for x in (LIGHT, SAFE, MSY_C)]
    fig.legend(h1, ["No reviewer", "Safety line", "MSY target"], loc="upper left", bbox_to_anchor=(0.01, 1.04), ncol=3,
               fontsize=6.8, title="Reviewer's target:", title_fontsize=6.8, alignment="left")
    h2 = [plt.Rectangle((0, 0), 1, 1, color=v[1]) for v in REV.values()]
    fig.legend(h2, [v[0] for v in REV.values()], loc="upper right", bbox_to_anchor=(0.99, 1.04), ncol=3, fontsize=6.8,
               title="Reviewer (panels d–f):", title_fontsize=6.8, alignment="left")
    fig.tight_layout(h_pad=1.2, w_pad=0.8, rect=(0, 0, 1, 0.94))
    save(fig, "fig2_reviewer")


# ------------------------------------------------------------------ Figure 3: what makes audits work (claims 3-5)
def fig3_audits():
    """Audits with programmed agents (5 panels). Legends: (a) above its axes; (c) in a side column on the left;
    (e) in a side column on the right (9 Oct 2026 layout). River joins panel (a) from R4 (a 2x2 grid like Forest's)."""
    M = pd.read_csv(NOTES / "claude_r2_v1/A_memory.csv")
    S3 = json.loads((NOTES / "claude_s3_partA_v1/s3_summary.json").read_text())
    T = pd.read_csv(NOTES / "claude_s5_v1/s5_cells.csv")
    FINE_C, MEM_C, BOTH_C = ORANGE, AQUA, BLUE
    fig = plt.figure(figsize=(7.0, 4.3))
    gs = fig.add_gridspec(2, 5, width_ratios=[0.78, 1, 1, 1, 0.78], height_ratios=[1, 1], hspace=0.62, wspace=0.62,
                          left=0.04, right=0.99, top=0.9, bottom=0.12)
    a = fig.add_subplot(gs[0, 0:3]); b = fig.add_subplot(gs[0, 3:5])
    lc_ax = fig.add_subplot(gs[1, 0]); c = fig.add_subplot(gs[1, 1]); d = fig.add_subplot(gs[1, 2])
    e = fig.add_subplot(gs[1, 3]); le_ax = fig.add_subplot(gs[1, 4])
    for ax in (lc_ax, le_ax):
        ax.axis("off")

    # (a) memory across settings: nine programmed settings (R2) and River (C1 Part B, half capacity)
    order = ["fishery_s4_r0.9", "fishery_s4_r0.7", "fishery_s4_r0.5", "fishery_s2_r0.5", "fishery_s2_r0.7",
             "harvest_s4_m1.0", "harvest_s4_m0.85", "harvest_s2_m0.85", "harvest_s2_m1.0"]
    def lab(cell):
        g, s_, p_ = cell.split("_")
        return f"{s_[1]}g\n{p_[1:]}"
    def harm(cell, mode):
        g = M[(M.cell == cell) & (M["mode"] == mode) & (M.liar == "fixed")].iloc[0]
        return 100 * (g.target_breaking_rate if g.target == "msy" else g.unsafe_fixed)
    R4 = {r["setting"]: r for r in json.loads((NOTES / "claude_r4_v1/r4_summary.json").read_text())["settings"]}
    rv_order = ["4g 1.0", "4g 0.85", "2g 0.85", "2g 1.0"]  # River grid (R4), ordered as Forest
    labels = [lab(x) for x in order] + [x.replace(" ", "\n") for x in rv_order]
    series = [("No audits", INK2, [harm(x, "trust") for x in order] + [100 * R4[x]["below50"]["trust"] for x in rv_order], None),
              ("Audits without memory", LIGHT, [harm(x, "memoryless") for x in order] + [100 * R4[x]["below50"]["memoryless"] for x in rv_order], None),
              ("Audits with memory", MEM_C, [harm(x, "memory") for x in order] + [100 * R4[x]["below50"]["memory"] for x in rv_order], None)]
    grouped(a, labels, series, ylim=(0, 100))
    a.set_ylim(0, 100); a.set_yticks([0, 25, 50, 75, 100])
    a.set_ylabel("Rounds below harm line (%)")
    a.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=3, fontsize=6.2, frameon=False, borderaxespad=0,
             handlelength=1.2, columnspacing=1.0)
    for xv in (4.5, 8.5):
        a.plot([xv, xv], [0, 100], color=INK2, lw=0.5)
    for xc, gname in ((2.0, "Fishery"), (6.5, "Forest"), (10.5, "River")):
        a.text(xc, -0.30, gname, transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=6.8, fontweight="bold")
    title(a, "a")

    # (b) predicted break-even against observed threshold (R3 settings and S3 audit-rate / detection variants)
    R3 = json.loads((NOTES / "claude_r3_v1/cells.json").read_text())
    pts = [(c_["g_star"], c_["e_star"]) for c_ in R3 if c_["testable"]]
    g3 = S3["AB"]["A_allow"]["predicted_e_threshold"]
    s3pts = [(g3, S3["AB"][k]["e_star"]) for k in S3["AB"] if k.startswith(("A_q", "B_q"))]
    lo_, hi_ = 0.12, 2.0
    xx = np.geomspace(lo_, hi_, 50)
    b.fill_between(xx, 0.8 * xx, 1.2 * xx, color=GRID, lw=0, label="Within one grid step")
    b.plot(xx, xx, color=INK2, lw=0.7, ls="--")
    b.scatter([p_[0] for p_ in s3pts], [p_[1] for p_ in s3pts], s=16, facecolor="white", edgecolor=FINE_C, lw=0.9, zorder=3,
              label="Audit rate or detection varied")
    b.scatter([p_[0] for p_ in pts], [p_[1] for p_ in pts], s=18, color=FINE_C, zorder=4, label="New settings, predicted first")
    b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(lo_, hi_); b.set_ylim(lo_, hi_)
    b.set_xticks([0.2, 0.5, 1]); b.set_xticklabels(["0.2", "0.5", "1"]); b.set_yticks([0.2, 0.5, 1]); b.set_yticklabels(["0.2", "0.5", "1"])
    b.minorticks_off()
    b.set_xlabel("Predicted break-even $g^*$"); b.set_ylabel("Observed threshold $e^*$")
    b.legend(fontsize=5.6, loc="lower right", borderaxespad=0.2, handletextpad=0.3)
    title(b, "b")

    # (c) cheaters' gain against audit rate, by audit rule (S5)
    T0 = T[T.tier == "T0"]
    rules = [("fine", "Fine", FINE_C), ("memory", "Lower allowance\nfor caught agents", MEM_C),
             ("memory_cap", "Check caught agents\nevery round", VIOLET), ("memory_cut", "Lower allowance\nfor all agents", YELLOW),
             ("fine+memory_cut", "Fine and lower\nallowance for all agents", BOTH_C)]
    for key, lab_, col in rules:
        g = T0[T0.regime == key].sort_values("q")
        c.fill_between(g.q, g.cheater_gain_lo, g.cheater_gain_hi, color=col, alpha=0.14, lw=0)
        c.plot(g.q, g.cheater_gain, color=col, marker="o", ms=2.4, lw=1.0, label=lab_)
    none_gain = float(T0[T0.regime == "none"].cheater_gain.iloc[0])
    c.axhline(none_gain, color=INK2, lw=0.6, ls=":"); c.axhline(0, color=INK2, lw=0.5)
    c.text(0.0205, none_gain + 4, "no audits", fontsize=6.0, color=INK2)
    c.set_xscale("log"); c.set_xticks([0.02, 0.05, 0.1, 1 / 6]); c.set_xticklabels(["0.02", "0.05", "0.10", "1/6"]); c.minorticks_off()
    c.set_xlabel("Audit rate"); c.set_ylabel("Cheaters' gain")
    hc, lc_ = c.get_legend_handles_labels()
    lc_ax.legend(hc, lc_, loc="center right", fontsize=6.0, frameon=False,
                 handlelength=1.4, labelspacing=0.9, alignment="left")
    title(c, "c")

    # (d) predictable against random audits (S3 Part C), fine 24
    SC = S3["C"]
    comply = SC["allow"]["comply_honest_per_agent"]
    keys = [("Nobody\ncheats", None, LIGHT, None), ("No\naudits", "allow", INK2, None), ("Random\naudits", "bern_F24", FINE_C, None),
            ("Fixed\nschedule", "periodic6_F24", FINE_C, "////")]
    for i_, (lab_, key, col, hat) in enumerate(keys):
        v = comply if key is None else comply + SC[key]["honest_drop"]["estimate"]
        err = None if key is None else [[v - (comply + SC[key]["honest_drop"]["ci"][0])], [(comply + SC[key]["honest_drop"]["ci"][1]) - v]]
        d.bar(i_, v, width=0.68, color="white" if hat else col, edgecolor=col if hat else "white", hatch=hat, lw=0.8,
              yerr=err, capsize=1.2, error_kw=dict(lw=0.6, ecolor=INK))
    d.set_xticks(range(4)); d.set_xticklabels([k[0] for k in keys], fontsize=5.4); d.grid(axis="x", visible=False)
    d.set_ylim(0, 120); d.set_ylabel("Harvest per honest agent")
    title(d, "d")

    # (e) aiming audits (T1): harm left as a share of no-audit harm, three games
    T1 = json.loads((NOTES / "claude_t1_v1/t1_summary.json").read_text())
    cov = json.loads((NOTES / "claude_t1_v1/t1_posthoc_distinct_liars.json").read_text())
    arms = [("random", "Random agent", MEM_C, None), ("report", "Largest report", MEM_C, "////"),
            ("signal", "Plot furthest\nbelow forecast", "#0b6b4a", None)]
    games = [("fishery", "Fishery"), ("harvest", "Forest"), ("river", "River")]
    for gi, (g, gl) in enumerate(games):
        present = [x for x in arms if x[0] in T1[g]["harm_relative_to_trust"]]
        w = 0.8 / 3
        for k, (arm, al, col, hat) in enumerate(present):
            r = T1[g]["harm_relative_to_trust"][arm]
            v, lo, hi = 100 * r["estimate"], 100 * r["ci"][0], 100 * r["ci"][1]
            xk = gi - 0.4 + w * (k + 0.5)
            e.bar(xk, v, width=w * 0.9, color="white" if hat else col, edgecolor=col if hat else "white", hatch=hat, lw=0.8,
                  yerr=[[v - lo], [hi - v]], capsize=1.2, error_kw=dict(lw=0.6, ecolor=INK), label=al if gi == 1 else None)
            caught = cov[f"{g}_{arm}"]["share_of_liars_ever_caught"]
            e.text(xk, hi + 3, f"{100 * caught:.0f}", ha="center", fontsize=5.2, color=INK2)
    e.set_xticks(range(3)); e.set_xticklabels([x[1] for x in games], fontsize=6.4); e.grid(axis="x", visible=False)
    e.set_ylim(0, 112); e.set_ylabel("Harm left (% of no audits)")
    he, le = e.get_legend_handles_labels()
    le_ax.legend(he, le, loc="center left", fontsize=6.0, frameon=False,
                 handlelength=1.4, labelspacing=0.9, alignment="left")
    title(e, "e")
    save(fig, "fig3_audits")


# ------------------------------------------------------------------ Table 2: audit regimes (claim 4)
def table2_regimes():
    T = pd.read_csv(NOTES / "claude_s5_v1/s5_cells.csv")
    T0 = T[T.tier == "T0"]
    comply = float(T0.honest_per_agent_comply.dropna().iloc[0])
    rows = [("none", None)] + [(r, q) for r in ("fine", "memory", "memory_cap", "memory_cut", "fine+memory_cut") for q in (0.1, 1 / 6)]
    recs = []
    for regime, q in rows:
        d = T0[T0.regime == regime] if q is None else T0[(T0.regime == regime) & np.isclose(T0.q, q)]
        r = d.iloc[0]
        recs.append(dict(regime=REGIME[regime][0], q="---" if q is None else ("0.10" if q == 0.1 else "1/6"),
                         gain=r.cheater_gain, lo=r.cheater_gain_lo, hi=r.cheater_gain_hi, honest=r.honest_per_agent,
                         breaking=100 * r.target_breaking, audits=r.audits_per_step))
    R = pd.DataFrame(recs)
    best = dict(gain=R.gain.abs().min(), honest=R.honest.max(), breaking=R.breaking.min())

    def b(val, key, fmt):
        s = fmt.format(val).replace("-", "$-$")
        return f"\\textbf{{{s}}}" if np.isclose(abs(val) if key == "gain" else val, best[key]) else s
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Audit rule & Audit rate $q$ & Cheaters' extra harvest [95\% interval] & Harvest per honest agent & Steps over MSY limit (\%) & Checks per step \\",
             r"\midrule"]
    prev = None
    for _, r in R.iterrows():
        if prev is not None and r.regime != prev:
            lines.append(r"\addlinespace[2pt]")
        ci = "" if np.isnan(r.lo) or (r.lo == 0 and r.hi == 0) else f" [{r.lo:.0f}, {r.hi:.0f}]".replace("-", "$-$")
        lines.append(f"{r.regime if r.regime != prev else ''} & {r.q} & {b(r.gain, 'gain', '{:.1f}')}{ci} & "
                     f"{b(r.honest, 'honest', '{:.1f}')} & {b(r.breaking, 'breaking', '{:.0f}')} & {r.audits:.2f} \\\\")
        prev = r.regime
    lines += [r"\midrule", f"Nobody cheats & --- & 0 & {comply:.1f} & 0 & 0 \\\\", r"\bottomrule", r"\end{tabular}"]
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "table2_regimes.tex").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ Figure 4: language-model fishers under audits (claim 6, L2)
LLM = {"gpt-oss_120b-cloud": ("gpt-oss-120b", BLUE, "o"),
       "nemotron-3-super_cloud": ("Nemotron 3 Super", ORANGE, "s"),
       "gemma4_31b-cloud": ("Gemma 4 31B", AQUA, "D"),
       "mistral-large-3_675b-cloud": ("Mistral Large 3", VIOLET, "^")}
E_CELLS = ["E0", "E1", "E2", "E4", "E8", "E36"]
E_FINE = {"E0": 0, "E1": 1, "E2": 2, "E4": 4, "E8": 8, "E36": 36}


def _boot_pooled(df, col, w=None, b=4000, seed=20261019):
    """Mean over contexts (weighted by agent-steps when w is given) and a 95% context-bootstrap interval."""
    rng = np.random.default_rng(seed)
    v = df[col].to_numpy(float)
    ww = df[w].to_numpy(float) if w else np.ones(len(v))
    est = np.sum(v * ww) / np.sum(ww)
    idx = rng.integers(0, len(v), size=(b, len(v)))
    draws = np.sum(v[idx] * ww[idx], axis=1) / np.sum(ww[idx], axis=1)
    return est, np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def fig4_llm(layout="full"):
    d = NOTES / "claude_l2_v1"
    pc = pd.read_csv(d / "l2_context_cells.csv")
    l3 = NOTES / "claude_l3_v1" / "l3_context_cells.csv"
    if l3.exists():
        pc = pd.concat([pc, pd.read_csv(l3)])
    summ = {r["model"]: r for r in json.loads((d / "l2_summary.json").read_text())["models"]}
    em_ok = "EM" in set(pc.cell)
    """layout: "full" (2x2, exhibits preview), "main" (panels a-b, the paper) or "controls" (panels c-d, supplement)."""
    if layout == "full":
        fig, axs = plt.subplots(2, 2, figsize=(TEXTW, 4.9))
        (a, b), (c, dd) = axs
    elif layout == "main":
        fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTW, 2.6))
        c = dd = None
    else:
        fig, (c, dd) = plt.subplots(1, 2, figsize=(TEXTW, 2.5))
        a = b = None
    lc, ld = ("c", "d") if layout == "full" else ("a", "b")
    if a is not None:
        cells_all = ["E0", "E1", "E2", "E4", "E8", "E12", "E18", "E24", "E30", "E36"]  # L3 adds E12-E30 for gpt-oss, Nemotron
        fine_of = {**E_FINE, "E12": 12, "E18": 18, "E24": 24, "E30": 30}
        x = np.arange(len(cells_all))
        off = np.linspace(-0.27, 0.27, len(LLM))
        for k, (m, (lab, col, mk)) in enumerate(LLM.items()):
            sub = pc[pc.model == m]
            have = [c for c in cells_all if (sub.cell == c).any()]
            xs = np.array([cells_all.index(c) for c in have])
            for ax, colname, w, scale in ((a, "overtake_rate", "agent_steps", 100), (b, "honest", None, 1)):
                pts = [_boot_pooled(sub[sub.cell == c], colname, w) for c in have]
                est, lo, hi = (np.array(t) * scale for t in zip(*pts))
                ax.errorbar(xs + off[k], est, yerr=[est - lo, hi - est], fmt=mk, color=col, ms=3.2, lw=0.7, capsize=1.0,
                            label=lab, mfc=col if m != "mistral-large-3_675b-cloud" else "white")
        gs = {m: summ[m]["g_tonnes"] for m in ("gpt-oss_120b-cloud", "nemotron-3-super_cloud") if summ.get(m)}
        for ax in (a, b):
            ax.set_xticks(x)
            ax.set_xticklabels([str(fine_of[c]) for c in cells_all], fontsize=6.4)
            ax.set_xlabel("Fine F (t); expected fine e = F/6", fontsize=7)
            ax.grid(axis="x", visible=False)
            ax.axvspan(7.5, 9.5, color=GRID, alpha=0.7, zorder=0, lw=0)  # e = 5 and 6: above both models' gain
            for m, g in gs.items():  # where e = g falls between grid points (6g t lies between F = 24 and 30)
                ax.axvline(7 + (6 * g - 24) / 6, color=LLM[m][1], lw=0.6, ls=(0, (2, 2)), zorder=1)
        a.text(8.5, 72, "e > g", ha="center", va="bottom", fontsize=6, color=INK2)
        a.set_ylim(-3, 103)
        a.set_ylabel("Over-taking (% of agent-steps)")
        title(a, "a", "Over-taking ends as e nears the gain g")
        b.set_ylabel("Catch per rule-following fisher (t)")
        b.set_ylim(0, 40)
        title(b, "b", "Harm to the rule-followers")
        h, l = a.get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6.6, handletextpad=0.2,
                   columnspacing=1.0)
    if c is not None:
        # (c) wording: fine 0 against fine 36 under explicit, silent and paraphrased rules
        groups = [("E0", "E36", "Explicit"), ("S0", "S36", "Silent"), ("P0", "P36", "Reworded")]
        labels = ["F = 0", "F = 36"] * len(groups)
        series = []
        for m, (lab, col, mk) in LLM.items():
            sub = pc[pc.model == m]
            pts = [_boot_pooled(sub[sub.cell == cell], "overtake_rate", "agent_steps") for g0, g36, _ in groups for cell in (g0, g36)]
            est, lo, hi = (100 * np.array(t) for t in zip(*pts))
            series.append((lab, col, est, (lo, hi)))
        grouped(c, labels, series, ylim=(0, 100))
        c.set_ylabel("Over-taking (% of agent-steps)")
        for k, (_, _, g) in enumerate(groups):  # wording names under each pair of fine levels
            c.text(2 * k + 0.5, -0.17, g, transform=c.get_xaxis_transform(), ha="center", va="top", fontsize=7)
        title(c, lc, "Same pattern under other wordings")
        # (d) what an over-take looks like (post hoc), or the memory cell once it has run
        ph = json.loads((d / "l2_posthoc.json").read_text())
        kinds = [("caught more than it requested", "Caught more than\nit requested", INK2),
                 ("kept its request after a cut", "Kept its request\nafter a cut", LIGHT)]
        ms = [m for m in LLM if ph.get(m, {}).get("overtake_steps")]
        left = np.zeros(len(ms))
        for key, lab, col in kinds:
            v = np.array([100 * ph[m]["overtake_kind_share"].get(key, 0) for m in ms])
            dd.barh(np.arange(len(ms)), v, left=left, color=col, label=lab.replace("\n", " "), height=0.6)
            left += v
        dd.barh(np.arange(len(ms)), 100 - left, left=left, color=GRID, label="Other", height=0.6)
        dd.set_yticks(np.arange(len(ms)))
        dd.set_yticklabels([LLM[m][0].split(" (")[0] for m in ms], fontsize=6.6)
        dd.invert_yaxis()
        dd.set_xlim(0, 100)
        dd.set_xlabel("Share of over-take steps (%)")
        dd.grid(axis="y", visible=False)
        for i, m in enumerate(ms):
            dd.text(101, i, f"n = {ph[m]['overtake_steps']:,}", va="center", fontsize=6, color=INK2)
        dd.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.22), ncol=3, fontsize=6.2, handletextpad=0.3, columnspacing=0.8)
        title(dd, ld, "What an over-take is (post hoc)")
    if layout == "controls":
        hs, ls = c.get_legend_handles_labels()
        fig.legend(hs, ls, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4, fontsize=6.6, handletextpad=0.3)
        fig.subplots_adjust(left=0.09, right=0.93, top=0.82, bottom=0.3, wspace=0.42)
        save(fig, "figS_llm_controls")
    elif layout == "main":
        fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.17, wspace=0.28)
        save(fig, "fig4_llm_main")
    else:
        fig.subplots_adjust(left=0.09, right=0.93, top=0.89, bottom=0.13, hspace=0.62, wspace=0.42)
        save(fig, "fig4_llm_agents")


# ------------------------------------------------------------------ Figure 5: the spine. Cheating against e/g, simulated and LLM agents
def _llm_relative(model, g):
    """Over-taking relative to the no-fine cell E0, against e/g, for every explicit fine (L2, and L3 when it exists)."""
    pc = pd.read_csv(NOTES / "claude_l2_v1" / "l2_context_cells.csv")
    sub = pc[pc.model == model]
    l3 = NOTES / "claude_l3_v1" / "l3_context_cells.csv"
    if l3.exists():
        sub = pd.concat([sub, pd.read_csv(l3).query("model == @model")])
    fines = {**E_FINE, "E12": 12, "E18": 18, "E24": 24, "E30": 30}
    base = sub[sub.cell == "E0"].set_index("context")
    rng = np.random.default_rng(20261019)
    out = []
    for cell, f in sorted(fines.items(), key=lambda kv: kv[1]):
        c = sub[sub.cell == cell].set_index("context")
        if len(c) == 0:
            continue
        ctx = np.array(sorted(set(c.index) & set(base.index)))
        def ratio(cs):
            num = np.sum(c.loc[cs, "overtake_rate"] * c.loc[cs, "agent_steps"]) / np.sum(c.loc[cs, "agent_steps"])
            den = np.sum(base.loc[cs, "overtake_rate"] * base.loc[cs, "agent_steps"]) / np.sum(base.loc[cs, "agent_steps"])
            return num / den
        draws = [ratio(rng.choice(ctx, len(ctx))) for _ in range(4000)]
        out.append(((f / 6) / g, ratio(ctx), np.percentile(draws, 2.5), np.percentile(draws, 97.5), cell in ("E12", "E18", "E24", "E30")))
    return out


def fig5_spine():
    """Observed against predicted stopping fine (calibration plot with a 1:1 line, the standard observed-versus-predicted
    form; 9 Oct 2026). x: the expected fine at which over-taking should stop according to a gain; y: the expected fine at
    which it did stop. Language models: predicted from the one-round gain g1 (filled) and from the whole-game gain g*
    (open); vertical whiskers span the last fine that did not deter and the first that did; horizontal whiskers are 95%
    bootstrap intervals of the gain over the 10 populations. Programmed cheaters (R3, grey): predicted from g*."""
    ph = json.loads((NOTES / "claude_l2_v1" / "l2_posthoc.json").read_text())
    l3 = json.loads((NOTES / "claude_l3_v1" / "l3_summary.json").read_text())["models"]
    R3 = json.loads((NOTES / "claude_r3_v1/cells.json").read_text())
    grid = [0, 1, 2, 4, 8, 12, 18, 24, 30, 36]
    rng = np.random.default_rng(20261019)
    def ratio_ci(num, den):
        num, den = np.asarray(num, float), np.asarray(den, float)
        idx = rng.integers(0, len(num), size=(4000, len(num)))
        dr = num[idx].sum(1) / den[idx].sum(1)
        return num.sum() / den.sum(), np.percentile(dr, 2.5), np.percentile(dr, 97.5)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    lim = (-2.6, 6.2)
    ax.plot(lim, lim, color=INK2, lw=0.7, ls="--", zorder=1, label="Observed equals predicted")
    rp = [(c_["g_star"], c_["e_star"]) for c_ in R3 if c_["testable"]]
    ax.scatter([p_[0] for p_ in rp], [p_[1] for p_ in rp], s=14, color=LIGHT, edgecolor=INK2, lw=0.4, zorder=2,
               label="Programmed cheaters, whole-game gain")
    for m in ("gpt-oss_120b-cloud", "nemotron-3-super_cloud"):
        lab, col, mk = LLM[m]
        wg = ph[m]["whole_game_gain"]
        g1 = ratio_ci(wg["g1_excess_by_context"], wg["g1_steps_by_context"])
        gs_ = ratio_ci(wg["net_gain_by_context"], wg["overtake_steps_by_context"])
        F = l3[m]["F_star"]; obs, obs_lo = F / 6, grid[grid.index(F) - 1] / 6
        for (v, lo, hi), filled in ((g1, True), (gs_, False)):
            ax.errorbar(v, obs, xerr=[[v - lo], [hi - v]], yerr=[[obs - obs_lo], [0]], fmt=mk, color=col, ms=4.2, lw=0.8,
                        capsize=1.5, mfc=col if filled else "white", zorder=3)
        ax.annotate("", xy=(g1[0], obs), xytext=(gs_[0], obs), zorder=1,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.5, ls=(0, (2, 2))))
        ax.plot([], [], mk, color=col, ms=4.2, label=lab)
    ax.plot([], [], "o", color=INK2, mfc=INK2, ms=4, label="Predicted from one-round gain")
    ax.plot([], [], "o", color=INK2, mfc="white", ms=4, label="Predicted from whole-game gain")
    ax.set_xlim(*lim); ax.set_ylim(-0.3, 6.2)
    ax.set_xlabel("Predicted stopping fine (expected fine, t)")
    ax.set_ylabel("Observed stopping fine (expected fine, t)")
    ax.legend(loc="lower right", fontsize=5.6, frameon=True, framealpha=0.95, edgecolor="none", handlelength=1.4,
              labelspacing=0.45)
    fig.tight_layout()
    save(fig, "fig5_spine")


# ------------------------------------------------------------------ Appendix figure A1: post hoc behaviour of the LLM agents (L2)
def figA1_llm_behaviour():
    ph = json.loads((NOTES / "claude_l2_v1" / "l2_posthoc.json").read_text())
    ms = ("gpt-oss_120b-cloud", "nemotron-3-super_cloud")
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTW, 2.3))
    labels = ["fine < gain", "no fine"] * len(ms)
    series = []
    for key, name, col in (("not_checked", "Over-take not checked", LIGHT), ("caught", "Over-take caught", INK2)):
        v, lo, hi = [], [], []
        for m in ms:
            for cond in ("fine_below_gain", "checked_no_fine"):
                r = ph[m]["after_overtake_next_round"][cond][key]
                k = round(r["rate"] * r["n"])
                l, h = wilson(k, r["n"])
                v.append(100 * r["rate"]); lo.append(100 * l); hi.append(100 * h)
        series.append((name, col, np.array(v), (np.array(lo), np.array(hi))))
    grouped(a, labels, series, ylim=(0, 100))
    for k, m in enumerate(ms):
        a.text(2 * k + 0.5, -0.2, LLM[m][0].split(" (")[0], transform=a.get_xaxis_transform(), ha="center", va="top",
               fontsize=7)
    a.set_ylabel("Over-takes again next round (%)")
    a.legend(loc="upper right", ncol=2, fontsize=6, handlelength=1.0, columnspacing=0.8)
    title(a, "a", "Being caught does not change the next round")
    bins = np.arange(0, 6.51, 0.5)
    for m in ms:
        lab, col, _ = LLM[m]
        x = ph[m]["overtake_size"]["catches_t"]
        b.hist(x, bins=bins, histtype="step", color=col, lw=1.2, density=True,
               label=f"{lab.split(' (')[0]}: {100 * ph[m]['overtake_size']['share_at_max']:.0f}% at 6 t")
    b.set_xlabel("Catch on an over-take round (t)")
    b.set_ylabel("Density")
    b.legend(loc="upper left", fontsize=6.4)
    title(b, "b", "Over-takes are all-out (flat fine)")
    fig.tight_layout(w_pad=1.5)
    save(fig, "figA1_llm_behaviour")


# ------------------------------------------------------------------ Table 3: positioning against prior work (from novelty/README.md)
def table3_positioning():
    Y, N, P = r"\checkmark", "--", r"(\checkmark)"
    cols = ["LLM agents", "Several agents", "Renewable resource", "Random audits", "3+ fine levels", "Memory",
            "Predicted threshold"]
    rows = [
        (r"Enforcement economics$^{a}$", [N, P, N, Y, Y, Y, N]),
        (r"AI control$^{b}$", [Y, N, N, P, N, N, N]),
        (r"Makins et al.\ 2026", [Y, Y, N, N, N, N, N]),
        (r"Gans \& Holden 2026", [N, N, N, Y, N, N, N]),
        (r"GovSim (Piatti et al.\ 2024)", [Y, Y, Y, N, N, N, N]),
        (r"Bracale Syrnikov et al.\ 2026", [Y, Y, Y, N, N, N, N]),
        (r"Ye \& Steinhardt 2026", [Y, Y, P, N, N, P, N]),
        (r"Okamoto et al.\ 2026", [Y, N, N, P, N, N, N]),
        (r"\textbf{This work}", [Y, Y, Y, Y, Y, Y, Y]),
    ]
    head = " & ".join([""] + [r"\rotatebox{60}{" + c + "}" for c in cols]) + r" \\"
    body = "\n".join(" & ".join([r] + v) + r" \\" for r, v in rows[:-1])
    tex = "\n".join([r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", head, r"\midrule", body, r"\midrule",
                     " & ".join([rows[-1][0]] + rows[-1][1]) + r" \\", r"\bottomrule", r"\end{tabular}"])
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "table3_positioning.tex").write_text(tex + "\n")


# ------------------------------------------------------------------ Table 4: the settings of every study, side by side (consistency)
def table4_settings():
    rows = [
        ("R1, R2", "1--3", "Fishery, Forest", "none (honest); R2: reviewer's model wrong", "--", "--", "half capacity$^a$", "--"),
        ("S1, S1b", "3", "Forest, Fishery", "programmed: under-report", "one level per game$^b$", "fine, exclusion or none", "half capacity$^a$", "--"),
        ("S2, S3", "4", "Fishery", "programmed: over-take", "one level per game$^b$", r"fine ($s \le 1$)", "half capacity", "whole-game $g^*$"),
        ("S4, S5", "3--4", "Fishery", "programmed: over-take, adapting", "one level per game$^b$", "memory$^c$, fine", "half capacity", "whole-game $g^*$"),
        ("R3", "4", "Fishery (6 settings)", "programmed: over-take", "one level per game$^b$", "fine", "half capacity", r"whole-game $g^*$, predicted"),
        ("T1", "5", "all three", "programmed: under-report by half", "fixed", "memory, no fine", r"half capacity$^d$", "--"),
        ("C1", "1, 5", "River", "A: honest; B: under-report", "fixed", "B: memory", r"quality $<30$$^e$", "--"),
        ("R4", "3", "River (4 settings)", "programmed: under-report by half", "fixed", "memory, or audits without memory", "half capacity", "--"),
        ("L2", "3, 6", "Fishery", "4 LLMs: over-take", "every round", "fine 0--36 t, or memory$^c$", "half capacity", "one-round $g_1$"),
        ("L3", "6", "Fishery", "gpt-oss, Nemotron: over-take", "every round", "fine 12--30 t", "half capacity", r"one-round $g_1$, predicted"),
    ]
    head = r"Study & Claims & Games & Who misbehaves, and how & How it is chosen & Consequence of a catch & Harm line & Gain used \\"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    tex = "\n".join([r"\begin{tabular}{llllllll}", r"\toprule", head, r"\midrule", body, r"\bottomrule", r"\end{tabular}"])
    (TAB / "table4_settings.tex").write_text(tex + "\n")


if __name__ == "__main__":
    for f in (fig2_reviewer, fig3_audits, fig4_llm, lambda: fig4_llm("main"), lambda: fig4_llm("controls"), fig5_spine, figA1_llm_behaviour, table2_regimes, table3_positioning, table4_settings):
        f()
        print("done", f.__name__)
