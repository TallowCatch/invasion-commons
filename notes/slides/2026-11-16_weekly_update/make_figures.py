"""Figures for the 16 Nov 2026 weekly update (spec.json here; skill: weekly-supervisor-update).

  .venv-socialjax/bin/python notes/slides/2026-11-16_weekly_update/make_figures.py

Needs:
- the phase 2 runs in results/runs/socialjax_phase2/ (rsync from CREATE);
- notes/socialjax/phase2_ippo/summary.json;
- the three paper PDFs in notes/icml27_lit_review/papers/Read properly/ (git-ignored; from arXiv 2504.18530,
  2501.16173 and 2610.06748). The abstract images are git-ignored too.

harvest_setup.png is made by notes/slides/2026-11-16_yali_update/make_figures.py.
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
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "legend.fontsize": 12})
    t = json.load(open("notes/socialjax/phase2_ippo/summary.json"))["arms"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    n = t["A_none"]["apples_eaten"]
    ax.bar(0, n["mean"], 0.5, color=INK2, yerr=[[n["mean"] - n["lo"]], [n["hi"] - n["mean"]]], capsize=3)
    w = 0.19
    for i, (k, col) in enumerate(zip((1, 2, 4, 7), (YELLOW, BLUE, VIOLET, AQUA))):
        for j, c in enumerate((0.5, 1.0, 2.0)):
            r = t[f"B_random_k{k}_c{c}"]["apples_eaten"]
            ax.bar(1 + j + (i - 1.5) * w, r["mean"], w, color=col, capsize=2,
                   yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                   label=f"{k} of 7 checked each step" if j == 0 else None)
    ax.axhline(500, color=INK2, ls="--", lw=1.3)
    ax.text(-0.35, 510, "7 well-behaved scripted bots: 500", color=INK2, fontsize=11.5, va="bottom")
    for x in (0, 1):
        ax.text(x, 95, "orchard\ndies", ha="center", color=ORANGE, fontsize=11.5, weight="bold")
    ax.set_xticks(range(4), ["no overseer", "3.5", "7", "14"])
    ax.set_ylim(0, 590)
    ax.set(xlabel="expected penalty per over-harvest (apples)", ylabel="apples eaten per game")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), frameon=False)
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
    ax.set(xlabel="training steps (millions)", ylabel="apples eaten per game")
    ax.legend(loc="center right", frameon=False, bbox_to_anchor=(1.0, 0.45))
    fig.tight_layout()
    fig.savefig(OUT / "targeting_apples.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    abstracts()
    penalty_bars()
    targeting_curve()
