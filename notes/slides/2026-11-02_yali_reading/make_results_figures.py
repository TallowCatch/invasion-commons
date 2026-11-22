"""Heatmaps of my own measured results for the 2 Nov 2026 Yali deck (Leibo et al. 2017, Fig. 4 style).

Every number is read from a saved summary; nothing is re-run and no LLM is called.

- L4 (Fishery, live LLM fishers, gpt-oss-120b): notes/claude_audit_20261005/runs/claude_l4_v1/l4_summary_firstplay.json
  (first plays only; see the data note in studies/L4_llm_identification/results.md). Columns used per cell:
  overtake_rate[0] and below_half[0] (point estimates). Cell -> (audit chance q, fine F) from that results table.
- R1a (Willis et al.'s CPR, LLM-written strategies under cultural evolution):
  notes/claude_audit_20261005/runs/claude_r1a_v1/r1a_conditions.csv, columns n, q, mult, exempt,
  collective_frequency, efficiency_transfer.

  python3 notes/slides/2026-11-02_yali_reading/make_results_figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "notes/claude_audit_20261005/runs"
OUT = Path(__file__).resolve().parent / "figures"

for f in ("Calibri.ttf", "Calibrib.ttf", "Calibrii.ttf"):
    p = Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts") / f
    if p.exists():
        fm.fontManager.addfont(str(p))
SANS = "Calibri" if any(f.name == "Calibri" for f in fm.fontManager.ttflist) else "Arial"

BG, NAVY, INK, MUTED, EMPTY = "#F5F4F2", "#062746", "#1C191A", "#55524E", "#E9E7E3"
CMAP = LinearSegmentedColormap.from_list("navy", ["#E1E8EF", "#93A7BC", NAVY])
plt.rcParams.update({"font.family": SANS, "font.size": 12, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "figure.facecolor": BG, "savefig.facecolor": BG})


def heat(ax, grid, notes, rows, cols, title, xlabel, fmt):
    """grid: values in [0, 1] or nan (not run); notes: small text under each value."""
    m = np.ma.masked_invalid(grid)
    ax.set_facecolor(BG)
    ax.imshow(m, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    for i, j in zip(*np.where(np.isnan(grid))):  # cells with no run: hatched, page-coloured
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=BG, edgecolor="#D3CFC8", hatch="///",
                                   linewidth=0))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "not run", ha="center", va="center", fontsize=10, color="#8F8F8F", style="italic")
                continue
            c = "white" if v > 0.45 else INK
            ax.text(j, i - (0.12 if notes[i][j] else 0), fmt(v), ha="center", va="center", fontsize=15,
                    fontweight="bold", color=c)
            if notes[i][j]:
                ax.text(j, i + 0.25, notes[i][j], ha="center", va="center", fontsize=9.5, color=c)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color=BG, linewidth=3)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=8)
    ax.set_title(title, fontfamily="Georgia", fontweight="bold", fontsize=15, color=NAVY, loc="left", pad=10)


def l4():
    s = json.loads((RUNS / "claude_l4_v1/l4_summary_firstplay.json").read_text())["models"]["gpt-oss:120b-cloud"]["cells"]
    # cell -> (row, expected fine e = q * F, fine F); rows: 0 none, 1 q=1/3, 2 q=1/6, 3 q=1/12
    cells = {"N0": (0, 0, None), "Q3_6": (1, 2, 6), "Q3_9": (1, 3, 9), "E0": (2, 0, 0), "E12": (2, 2, 12),
             "E18": (2, 3, 18), "E24": (2, 4, 24), "E30": (2, 5, 30), "Q12_24": (3, 2, 24), "Q12_36": (3, 3, 36)}
    es = [0, 2, 3, 4, 5]
    rows = ["Audits never\nmentioned", "1 in 3 checked\n(≈ 2 of 6 fishers)", "1 in 6 checked\n(≈ 1 of 6)",
            "1 in 12 checked\n(≈ 1 every 2 rounds)"]
    over, harm = np.full((4, 5), np.nan), np.full((4, 5), np.nan)
    notes = [[""] * 5 for _ in rows]
    for name, (r, e, F) in cells.items():
        c = s[name]
        over[r, es.index(e)], harm[r, es.index(e)] = c["overtake_rate"][0], c["below_half"][0]
        notes[r][es.index(e)] = "no audits" if F is None else f"fine {F} t"
    cols = ["0", "2 t", "3 t", "4 t", "5 t"]
    xl = "Expected fine for one over-take = chance of being checked × fine"
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.7), gridspec_kw=dict(wspace=0.08))
    heat(axs[0], over, notes, rows, cols, "Over-taking (share of LLM decisions)", xl, lambda v: f"{100 * v:.1f}%")
    heat(axs[1], harm, notes, rows, cols, "Rounds with the stock below half", xl, lambda v: f"{100 * v:.1f}%")
    axs[1].set_yticklabels([])
    fig.savefig(OUT / "my_l4_heatmap.png", dpi=200, bbox_inches="tight", pad_inches=0.08)


def r1a():
    rows = list(csv.DictReader(open(RUNS / "claude_r1a_v1/r1a_conditions.csv")))
    mults = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    keys = [(64, 0.1), (64, 0.25), (4, 0.1), (4, 0.25)]
    labels = ["64 agents,\n10% checked", "64 agents,\n25% checked", "4 agents,\n10% checked", "4 agents,\n25% checked"]
    coll, eff = np.full((4, 7), np.nan), np.full((4, 7), np.nan)
    for r in rows:
        n, q, m = int(r["n"]), float(r["q"]), float(r["mult"])
        if r["exempt"] != "True":
            continue  # the secondary "fine the final round too" arm
        for i, (kn, kq) in enumerate(keys):
            if n == kn and (q == 0.0 or q == kq):
                j = 0 if q == 0.0 else 1 + mults.index(m)
                coll[i, j], eff[i, j] = float(r["collective_frequency"]), float(r["efficiency_transfer"])
    cols = ["none", "0.5×", "0.75×", "1×", "1.25×", "1.5×", "2×"]
    xl = "Expected fine, as a multiple of the predicted balance fine"
    blank = [[""] * 7 for _ in keys]
    fig, axs = plt.subplots(2, 1, figsize=(6.6, 5.6), gridspec_kw=dict(hspace=0.42))
    heat(axs[0], coll, blank, labels, cols, "Share of Collective strategies", "", lambda v: f"{v:.2f}")
    heat(axs[1], eff, blank, labels, cols, "Welfare efficiency", xl, lambda v: f"{v:.2f}")
    for ax in axs:
        for t in ax.texts:
            t.set_fontsize(12.5)
        ax.tick_params(labelsize=11.5)
    fig.savefig(OUT / "my_r1a_heatmap.png", dpi=200, bbox_inches="tight", pad_inches=0.08)


if __name__ == "__main__":
    l4()
    r1a()
