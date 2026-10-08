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
REGIME = {"none": ("No audits", INK2), "fine": ("Fine", ORANGE), "memory": ("Memory", AQUA),
          "memory_cap": ("Memory + extra checks", VIOLET), "memory_cut": ("Memory + tighter shared cut", YELLOW),
          "fine+memory_cut": ("Fine + memory + tighter cut", BLUE)}

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


def title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=7.6, fontweight="bold", pad=4)


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
    M = pd.read_csv(NOTES / "claude_r2_v1/A_memory.csv")
    S3 = json.loads((NOTES / "claude_s3_partA_v1/s3_summary.json").read_text())
    T = pd.read_csv(NOTES / "claude_s5_v1/s5_cells.csv")
    C1 = pd.read_csv(NOTES / "claude_c1_v1/c1_closed_loop.csv")
    FINE_C, MEM_C, BOTH_C = ORANGE, AQUA, BLUE
    fig = plt.figure(figsize=(TEXTW, 4.1))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.75, wspace=0.45)
    a = fig.add_subplot(gs[0, :2]); b = fig.add_subplot(gs[0, 2])
    c = fig.add_subplot(gs[1, 0]); d = fig.add_subplot(gs[1, 1]); e = fig.add_subplot(gs[1, 2])

    # (a) memory across nine settings
    order = ["fishery_s4_r0.9", "fishery_s4_r0.7", "fishery_s4_r0.5", "fishery_s2_r0.5", "fishery_s2_r0.7",
             "harvest_s4_m1.0", "harvest_s4_m0.85", "harvest_s2_m0.85", "harvest_s2_m1.0"]
    def lab(cell):
        g, s_, p_ = cell.split("_")
        return f"{s_[1]}g\n{p_[1:]}"
    def harm(cell, mode):
        g = M[(M.cell == cell) & (M["mode"] == mode) & (M.liar == "fixed")].iloc[0]
        return 100 * (g.target_breaking_rate if g.target == "msy" else g.unsafe_fixed)
    grouped(a, [lab(x) for x in order], [("Trust reports", INK2, [harm(x, "trust") for x in order], None),
                                         ("Audit, no memory", LIGHT, [harm(x, "memoryless") for x in order], None),
                                         ("Audit + memory", MEM_C, [harm(x, "memory") for x in order], None)], ylim=(0, 100))
    a.set_ylim(0, 118); a.set_yticks([0, 25, 50, 75, 100])
    a.set_ylabel("Harm (% of steps)$^*$"); a.legend(loc="upper right", ncol=3, fontsize=6.4, borderaxespad=0.1)
    a.axvline(4.5, color=INK2, lw=0.5)
    for xc, gname in ((2.0, "Fishery"), (6.5, "Forest")):
        a.text(xc, -0.30, gname, transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=6.8, fontweight="bold")
    title(a, "a", "Memory makes audits work (nine settings, under-reporters)")

    # (b) predicted break-even against observed threshold (R3 settings + S3 audit-rate / detection variants)
    R3 = json.loads((NOTES / "claude_r3_v1/cells.json").read_text())
    pts = [(c["g_star"], c["e_star"]) for c in R3 if c["testable"]]
    g3 = S3["AB"]["A_allow"]["predicted_e_threshold"]
    s3pts = [(g3, S3["AB"][k]["e_star"]) for k in S3["AB"] if k.startswith(("A_q", "B_q"))]
    lo_, hi_ = 0.12, 2.0
    xx = np.geomspace(lo_, hi_, 50)
    b.fill_between(xx, 0.8 * xx, 1.2 * xx, color=GRID, lw=0, label="±1 grid step")
    b.plot(xx, xx, color=INK2, lw=0.7, ls="--")
    b.scatter([p[0] for p in s3pts], [p[1] for p in s3pts], s=16, facecolor="white", edgecolor=FINE_C, lw=0.9, zorder=3,
              label="S3: audit rates, misses")
    b.scatter([p[0] for p in pts], [p[1] for p in pts], s=18, color=FINE_C, zorder=4, label="R3: five settings")
    b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(lo_, hi_); b.set_ylim(lo_, hi_)
    b.set_xticks([0.2, 0.5, 1]); b.set_xticklabels(["0.2", "0.5", "1"]); b.set_yticks([0.2, 0.5, 1]); b.set_yticklabels(["0.2", "0.5", "1"])
    b.minorticks_off()
    b.set_xlabel("Predicted break-even $g^*$"); b.set_ylabel("Observed threshold $e^*$")
    b.legend(fontsize=5.5, loc="upper left", borderaxespad=0.1, handletextpad=0.2)
    title(b, "b", "Fines deter at break-even")

    # (c) gain against audit rate, by audit rule
    T0 = T[T.tier == "T0"]
    rules = [("fine", "Fine", FINE_C, "-"), ("memory", "Memory", MEM_C, "-"), ("fine+memory_cut", "Fine + memory", BOTH_C, "-"),
             ("memory_cap", "Memory + extra checks", VIOLET, "-"), ("memory_cut", "Memory + tighter cut", YELLOW, "-")]
    for key, lab_, col, ls in rules:
        g = T0[T0.regime == key].sort_values("q")
        c.fill_between(g.q, g.cheater_gain_lo, g.cheater_gain_hi, color=col, alpha=0.14, lw=0)
        c.plot(g.q, g.cheater_gain, color=col, ls=ls, marker="o", ms=2.4, lw=1.0, label=lab_)
    none_gain = float(T0[T0.regime == "none"].cheater_gain.iloc[0])
    c.axhline(none_gain, color=INK2, lw=0.6, ls=":"); c.axhline(0, color=INK2, lw=0.5)
    c.text(0.0205, none_gain + 4, "no audits", fontsize=6.0, color=INK2)
    c.set_xscale("log"); c.set_xticks([0.02, 0.05, 0.1, 1 / 6]); c.set_xticklabels(["0.02", "0.05", "0.10", "1/6"]); c.minorticks_off()
    c.set_xlabel("Audit rate"); c.set_ylabel("Cheaters' gain (group of 4)")
    hc, lc = c.get_legend_handles_labels()
    fig.legend(hc, lc, loc="lower left", bbox_to_anchor=(0.0, -0.06), ncol=3, fontsize=6.3, title="Audit rule (panel c):",
               title_fontsize=6.3, alignment="left", handlelength=1.4, columnspacing=1.0)
    title(c, "c", "Which audit rule deters")

    # (d) predictable against random audits
    SC = S3["C"]
    comply = SC["allow"]["comply_honest_per_agent"]
    keys = [("Nobody\ncheats", None, LIGHT, None), ("No\naudits", "allow", INK2, None), ("Random\naudits", "bern_F24", FINE_C, None),
            ("Predict.\naudits", "periodic6_F24", FINE_C, "////")]
    for i, (lab_, key, col, hat) in enumerate(keys):
        v = comply if key is None else comply + SC[key]["honest_drop"]["estimate"]
        err = None if key is None else [[v - (comply + SC[key]["honest_drop"]["ci"][0])], [(comply + SC[key]["honest_drop"]["ci"][1]) - v]]
        d.bar(i, v, width=0.68, color="white" if hat else col, edgecolor=col if hat else "white", hatch=hat, lw=0.8,
              yerr=err, capsize=1.2, error_kw=dict(lw=0.6, ecolor=INK))
    d.set_xticks(range(4)); d.set_xticklabels([k[0] for k in keys], fontsize=5.4); d.grid(axis="x", visible=False)
    d.set_ylim(0, 120); d.set_ylabel("Harvest per honest agent")
    d.text(3, 38, "12/64\ncollapse", ha="center", fontsize=5.8, color=INK2)
    title(d, "d", "Audit timing (fine 24)")

    # (e) aiming audits (T1): harm left as a share of no-audit harm, three games
    T1 = json.loads((NOTES / "claude_t1_v1/t1_summary.json").read_text())
    cov = json.loads((NOTES / "claude_t1_v1/t1_posthoc_distinct_liars.json").read_text())
    arms = [("random", "Random", MEM_C, None), ("report", "Largest report", MEM_C, "////"), ("signal", "Signal (Forest)", "#0b6b4a", None)]
    games = [("fishery", "Fishery"), ("harvest", "Forest"), ("river", "River")]
    for gi, (g, gl) in enumerate(games):
        present = [a for a in arms if a[0] in T1[g]["harm_relative_to_trust"]]
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
    fig.legend(he, le, loc="lower right", bbox_to_anchor=(0.99, -0.06), ncol=3, fontsize=6.3, title="Audit aimed at (panel e):",
               title_fontsize=6.3, alignment="left", handlelength=1.4, columnspacing=1.0)
    title(e, "e", "Audit targeting")
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
LLM = {"gpt-oss_120b-cloud": ("gpt-oss-120b (OpenAI)", BLUE, "o"),
       "nemotron-3-super_cloud": ("Nemotron 3 Super (NVIDIA)", ORANGE, "s"),
       "gemma4_31b-cloud": ("Gemma 4 31B (Google)", AQUA, "D"),
       "mistral-large-3_675b-cloud": ("Mistral Large 3 (added)", VIOLET, "^")}
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


def fig4_llm():
    d = NOTES / "claude_l2_v1"
    pc = pd.read_csv(d / "l2_context_cells.csv")
    summ = {r["model"]: r for r in json.loads((d / "l2_summary.json").read_text())["models"]}
    em_ok = "EM" in set(pc.cell)
    fig, axs = plt.subplots(2, 2, figsize=(TEXTW, 4.9))
    (a, b), (c, dd) = axs
    x = np.arange(len(E_CELLS))
    off = np.linspace(-0.24, 0.24, len(LLM))
    for k, (m, (lab, col, mk)) in enumerate(LLM.items()):
        sub = pc[pc.model == m]
        for ax, colname, w, scale in ((a, "overtake_rate", "agent_steps", 100), (b, "honest", None, 1)):
            pts = [_boot_pooled(sub[sub.cell == e], colname, w) for e in E_CELLS]
            est, lo, hi = (np.array(t) * scale for t in zip(*pts))
            ax.errorbar(x + off[k], est, yerr=[est - lo, hi - est], fmt=mk, color=col, ms=3.6, lw=0.8, capsize=1.2,
                        label=lab, mfc=col if m != "mistral-large-3_675b-cloud" else "white")
    for ax in (a, b):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{E_FINE[e]}\n({E_FINE[e] / 6:.2g})" for e in E_CELLS])
        ax.set_xlabel("Fine F, t (expected fine e = F/6, t)", fontsize=7)
        ax.grid(axis="x", visible=False)
        ax.axvspan(4.5, 5.5, color=GRID, alpha=0.7, zorder=0, lw=0)
    gs = [summ[m]["g_tonnes"] for m in ("gpt-oss_120b-cloud", "nemotron-3-super_cloud") if summ.get(m)]
    a.text(5, 72, f"e $\\geq$ g\n(g = {min(gs):.1f}–\n{max(gs):.1f} t)", ha="center", va="bottom", fontsize=6, color=INK2)
    a.set_ylim(-3, 103)
    a.set_ylabel("Over-taking (% of agent-steps)")
    title(a, "a", "Over-taking stops once e exceeds the gain g")
    b.set_ylabel("Catch per rule-following fisher (t)")
    b.set_ylim(0, 40)
    title(b, "b", "Harm to the rule-followers")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6.6, handletextpad=0.2,
               columnspacing=1.0)
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
    title(c, "c", "Same pattern under other wordings")
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
    title(dd, "d", "What an over-take is (post hoc)")
    fig.subplots_adjust(left=0.09, right=0.93, top=0.89, bottom=0.13, hspace=0.62, wspace=0.42)
    save(fig, "fig4_llm_agents")


if __name__ == "__main__":
    for f in (fig2_reviewer, fig3_audits, fig4_llm, table2_regimes):
        f()
        print("done", f.__name__)
