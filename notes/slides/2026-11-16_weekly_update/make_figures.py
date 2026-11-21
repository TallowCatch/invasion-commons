"""Figures for the 16 Nov 2026 weekly update (spec.json here; skill: weekly-supervisor-update).

  .venv-socialjax/bin/python notes/slides/2026-11-16_weekly_update/make_figures.py

Needs:
- the phase 2 runs in results/runs/socialjax_phase2/ (rsync from CREATE);
- notes/socialjax/phase2_ippo/summary.json;
- the three paper PDFs in notes/icml27_lit_review/papers/Read properly/ (git-ignored; from arXiv 2504.18530,
  2501.16173 and 2610.06748). The abstract images are git-ignored too.

harvest_setup.png and schelling_no_overseer.png are made by notes/slides/2026-11-16_yali_update/make_figures.py.
"""
import glob
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.socialjax.make_week2_figures import AQUA, BLUE, INK2, ORANGE, VIOLET, YELLOW  # noqa: E402

OUT = Path("notes/slides/2026-11-16_weekly_update/figures")
RUNS = "results/runs/socialjax_phase2/"
PAPERS = Path("notes/icml27_lit_review/papers/Read properly")


def page1(pdf):
    """First page of a PDF as a white-backed RGB image (sips renders page 1)."""
    with tempfile.TemporaryDirectory() as d:
        png = Path(d) / "p.png"
        subprocess.run(["sips", "-s", "format", "png", "-Z", "2000", str(pdf), "--out", str(png)], check=True,
                       capture_output=True)
        im = Image.open(png).convert("RGBA")
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[3])
        return bg


def stack(parts, centre=True, pad=30):
    w = max(p.width for p in parts)
    h = sum(p.height for p in parts) + 10 * (len(parts) - 1)
    out = Image.new("RGB", (w + 2 * pad, h + 2 * pad), "white")
    y = pad
    for p in parts:
        out.paste(p, (pad + ((w - p.width) // 2 if centre else 0), y))
        y += p.height + 10
    return out


def frac(im, x0, y0, x1, y1):
    w, h = im.size
    return im.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))


def abstracts():
    e = page1(PAPERS / "Oversight - Engels 2025 - Scaling Laws for Scalable Oversight.pdf")
    stack([frac(e, 0.16, 0.10, 0.84, 0.605)]).save(OUT / "abstract_engels.png")
    w = page1(PAPERS / "Setting - Willis 2025 - Will Systems of LLM Agents Cooperate.pdf")
    stack([frac(w, 0.08, 0.105, 0.92, 0.245), frac(w, 0.08, 0.265, 0.49, 0.585)]).save(OUT / "abstract_willis.png")
    b = page1(PAPERS / "Oversight - Wang 2026 - BazaarBench.pdf")  # title, author names, abstract (no affiliations/links)
    stack([frac(b, 0.16, 0.09, 0.70, 0.175), frac(b, 0.17, 0.19, 0.68, 0.225), frac(b, 0.22, 0.34, 0.77, 0.795)],
          centre=False).save(OUT / "abstract_bazaar.png")


def penalty_bars():
    """Apples eaten at each expected penalty, one bar per number of checks. Bars are seed means; whiskers the seed range."""
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "legend.fontsize": 12})
    t = json.load(open("notes/socialjax/phase2_ippo/summary.json"))["arms"]
    blues = ("#c6dbef", "#6baed6", "#2171b5", "#08306b")  # 1, 2, 4, 7 checks: light to dark
    fig, ax = plt.subplots(figsize=(9, 5.0))
    n = t["A_none"]["apples_eaten"]
    ax.bar(0, n["mean"], 0.55, color="#9e9e9e", edgecolor="white")
    ax.errorbar(0, n["mean"], [[n["mean"] - n["lo"]], [n["hi"] - n["mean"]]], color="#333333", lw=1.2, capsize=0)
    w = 0.2
    for i, (k, col) in enumerate(zip((1, 2, 4, 7), blues)):
        for j, c in enumerate((0.5, 1.0, 2.0)):
            r = t[f"B_random_k{k}_c{c}"]["apples_eaten"]
            x = 1 + j + (i - 1.5) * w
            ax.bar(x, r["mean"], w, color=col, edgecolor="white", linewidth=0.8,
                   label=f"{k} of the 7 agents checked each step" if j == 0 else None)
            ax.errorbar(x, r["mean"], [[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], color="#333333", lw=1.2, capsize=0)
    for x, txt in ((0, "the orchard died\nin every run"), (1, "the orchard died\nin every run"),
                   (2, "the orchard survived\nin every run"), (3, "the orchard survived\nin every run")):
        y = 95 if x < 2 else 560
        ax.text(x, y, txt, ha="center", va="bottom" if x < 2 else "bottom", fontsize=11, color="#555555", style="italic")
    ax.set_xticks(range(4), ["no overseer", "0.5", "1", "2"])
    ax.set_ylim(0, 620)
    ax.set(xlabel="average penalty for each over-harvest (apples)", ylabel="apples collected per game")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.92))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "penalty_threshold.png", dpi=200)
    plt.close(fig)


def targeting_curve():
    import numpy as np
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "legend.fontsize": 12.5})
    arms = [("A_none", INK2, "no overseer"), ("C_freeze50_k2", ORANGE, "random checks, freeze"),
            ("B_random_k2_c1.0", BLUE, "random checks"), ("D_history_k2_c1.0", AQUA, "repeat offenders first"),
            ("D_location_k2_c1.0", YELLOW, "by location")]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for arm, col, name in arms:
        cs = np.array([np.load(f)["apples_eaten"] for f in sorted(glob.glob(f"{RUNS}{arm}_seed*/metrics.npz"))])
        x = np.arange(cs.shape[1]) * 256_000 / 1e6
        ax.plot(x, cs.mean(0), color=col, lw=2.6, label=name)
        ax.fill_between(x, cs.min(0), cs.max(0), color=col, alpha=0.15, lw=0)
    ax.set(xlabel="training steps (millions)", ylabel="apples collected per game")
    ax.legend(loc="center right", frameon=False, bbox_to_anchor=(1.0, 0.45))
    fig.tight_layout()
    fig.savefig(OUT / "targeting_apples.png", dpi=200)
    plt.close(fig)


def bots_chart():
    """What a bot that holds back and a greedy bot each collect, by how many of the 7 hold back (week 2, no overseer)."""
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "legend.fontsize": 12})
    d = json.load(open("notes/socialjax/week2_bots/summary.json"))["schelling"]["none"]
    ls = sorted(int(x) for x in d)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for key, col, lab in (("R_c", BLUE, "a bot that holds back"), ("R_d", ORANGE, "a greedy bot")):
        pts = [(l, d[str(l)][key]) for l in ls if d[str(l)][key] == d[str(l)][key]]  # drop NaN (no such bot)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=col, lw=2.4, ms=6, label=lab)
    ax.set_xticks(range(8))
    ax.set(xlabel="how many of the 7 bots hold back", ylabel="apples collected per bot")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "bots_dilemma.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    abstracts()
    penalty_bars()
    targeting_curve()
    bots_chart()
