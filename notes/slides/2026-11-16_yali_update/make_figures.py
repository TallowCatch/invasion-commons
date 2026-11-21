"""Slide figures for the 16 Nov 2026 progress update (spec.json in this folder).

  .venv-socialjax/bin/python notes/slides/2026-11-16_yali_update/make_figures.py
Needs the phase 2 runs in results/runs/socialjax_phase2/ (rsync from CREATE) and notes/socialjax/*/ outputs.
"""
import glob
import json
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.socialjax.make_week2_figures import AQUA, BLUE, INK, INK2, ORANGE, YELLOW  # noqa: E402
from overseer.socialjax import make_harvest_env  # noqa: E402

OUT = Path("notes/slides/2026-11-16_yali_update/figures")
RUNS = "results/runs/socialjax_phase2/"


def harvest_setup():
    env = make_harvest_env()
    _, state = env.reset(jax.random.PRNGKey(3))
    im = np.asarray(env.render(state)).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=220)
    ax.imshow(im)
    ax.axis("off")
    for x, y in [(240, 336), (528, 336)]:  # the two agents in the middle row
        ax.add_patch(Circle((x, y), 30, fill=False, ec="#062746", lw=2.6))
    ax.annotate("checked this step\n(2 of 7 agents)", xy=(556, 356), xytext=(560, 430), fontsize=11, color="#062746",
                weight="bold", arrowprops=dict(arrowstyle="-", color="#062746", lw=1.4))
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "harvest_setup.png", facecolor="white")
    plt.close(fig)


def schelling_crop():
    im = Image.open("notes/socialjax/week2_bots/fig_rule_and_schelling.png").convert("RGB")
    w, h = im.size
    im.crop((round(0.25 * w) + 22, round(0.085 * h), round(0.5 * w), h)).save(OUT / "schelling_no_overseer.png")


def heatmap():
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
    t = json.load(open("notes/socialjax/phase2_ippo/summary.json"))["arms"]
    ks, levels = (1, 2, 4, 7), (0.5, 1.0, 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (key, cmap, fmt, lab) in zip(axes, [
            ("apples_eaten", "Blues", "{:.0f}", "apples eaten per episode (no overseer: 68)"),
            ("orchard_alive", "Greens", "{:.2f}", "orchard alive at the end (no overseer: 0.00)")]):
        g = np.array([[t[f"B_random_k{k}_c{c}"][key]["mean"] for c in levels] for k in ks])
        vmax = 1 if key == "orchard_alive" else g.max()
        ax.imshow(g, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        for (a, b), v in np.ndenumerate(g):
            ax.text(b, a, fmt.format(v), ha="center", va="center", fontsize=14, weight="bold",
                    color="white" if v > 0.6 * vmax else INK)
        ax.set_xticks(range(3), ["3.5", "7", "14"])
        ax.set_yticks(range(4), ks)
        ax.grid(False)
        ax.set(xlabel="expected penalty per over-harvest (k × penalty, apples)",
               ylabel="agents checked per step (k of 7)", title=lab)
    fig.tight_layout()
    fig.savefig(OUT / "heatmap_slide.png", dpi=200)
    plt.close(fig)


def curves():
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "legend.fontsize": 11.5})
    arms = [("A_none", INK2, "no overseer"), ("C_freeze50_k2", ORANGE, "random, freeze"),
            ("B_random_k2_c1.0", BLUE, "random, penalty"), ("D_history_k2_c1.0", AQUA, "history, penalty"),
            ("D_location_k2_c1.0", YELLOW, "location, penalty")]
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 7.4), sharex=True)
    for ax, key, lab in zip(axes, ("apples_eaten", "orchard_alive"),
                            ("apples eaten\nper episode", "orchard alive\n(share of games)")):
        for arm, col, name in arms:
            cs = np.array([np.load(f)[key] for f in sorted(glob.glob(f"{RUNS}{arm}_seed*/metrics.npz"))])
            x = np.arange(cs.shape[1]) * 256_000 / 1e6
            ax.plot(x, cs.mean(0), color=col, lw=2.4, label=name)
            ax.fill_between(x, cs.min(0), cs.max(0), color=col, alpha=0.15, lw=0)
        ax.set_ylabel(lab)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("training steps (millions)")
    axes[0].legend(loc="center right", frameon=False, bbox_to_anchor=(1.0, 0.42))
    fig.tight_layout()
    fig.savefig(OUT / "stage1_curves_tall.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    harvest_setup()
    schelling_crop()
    heatmap()
    curves()
