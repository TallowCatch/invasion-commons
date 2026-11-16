"""Phase 2 analysis (notes/socialjax/phase2_ippo/protocol.md): tables and figures from the CREATE runs.

  rsync the runs first (params excluded):
    rsync -az --exclude params.msgpack k26108758@hpc.create.kcl.ac.uk:/scratch/users/k26108758/ippo_overseer/ \
        results/runs/socialjax_phase2/
  .venv-socialjax/bin/python -m experiments.socialjax.analyze_phase2 --runs results/runs/socialjax_phase2
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.socialjax.make_week2_figures import AQUA, BLUE, INK, INK2, ORANGE, VIOLET, YELLOW  # noqa: E402

OUT = Path("notes/socialjax/phase2_ippo")
MEASURES = ("apples_eaten", "orchard_alive", "apples_end", "violations", "detected", "missed", "penalty_apples",
            "checks")
KS, LEVELS = (1, 2, 4, 7), (0.5, 1.0, 2.0)


def load(runs):
    out = {}
    for d in sorted(Path(runs).iterdir()):
        if (d / "summary.json").exists() and (d / "metrics.npz").exists() and not d.name.startswith("pilot"):
            m = re.match(r"(.+)_seed(\d+)$", d.name)
            out[d.name] = dict(arm=m.group(1), seed=int(m.group(2)), summary=json.loads((d / "summary.json").read_text()),
                               curves={k: v for k, v in np.load(d / "metrics.npz").items()})
    return out


def arm_table(runs):
    by_arm = defaultdict(list)
    for r in runs.values():
        by_arm[r["arm"]].append(r)
    table = {}
    for arm, rs in sorted(by_arm.items()):
        row = dict(seeds=len(rs))
        for k in MEASURES:
            v = [r["summary"]["last_10_updates"][k] for r in rs]
            row[k] = dict(mean=float(np.mean(v)), lo=float(np.min(v)), hi=float(np.max(v)))
        table[arm] = row
    return table


def fig_grid(table):
    base = table.get("A_none")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, (key, cmap, fmt, lab) in zip(axes, [("apples_eaten", "Blues", "{:.0f}", "apples eaten per episode"),
                                                ("orchard_alive", "Greens", "{:.2f}", "orchard alive at the end")]):
        grid = np.full((len(KS), len(LEVELS)), np.nan)
        for i, k in enumerate(KS):
            for j, c in enumerate(LEVELS):
                row = table.get(f"B_random_k{k}_c{c}")
                if row:
                    grid[i, j] = row[key]["mean"]
        vmax = 1.0 if key == "orchard_alive" else np.nanmax(grid)
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        for (a, b), v in np.ndenumerate(grid):
            if not np.isnan(v):
                ax.text(b, a, fmt.format(v), ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.6 * vmax else INK)
        ax.set_xticks(range(len(LEVELS)), [f"{c:g} apple{"s" if c != 1 else ""}\n(k x P = {7 * c:g})" for c in LEVELS])
        ax.set_yticks(range(len(KS)), KS)
        ax.grid(False)
        ax.set(xlabel="expected penalty per over-harvest (apples) = k/7 x penalty",
               ylabel="agents checked per step (k of 7)")
        base_txt = f"; no overseer: {fmt.format(base[key]['mean'])}" if base else ""
        ax.set_title(f"{'(a)' if key == 'apples_eaten' else '(b)'} {lab}{base_txt}", fontsize=8.5)
        fig.colorbar(im, ax=ax, fraction=0.05)
    fig.tight_layout()
    fig.savefig(OUT / "fig_checks_by_penalty.png", dpi=200)
    plt.close(fig)


def curve_panel(ax, runs, arms, key, title):
    for arm, col, lab in arms:
        cs = [r["curves"][key] for r in runs.values() if r["arm"] == arm]
        if not cs:
            continue
        n = min(len(c) for c in cs)
        cs = np.array([c[:n] for c in cs])
        x = np.arange(n) * 256_000 / 1e6
        ax.plot(x, cs.mean(0), color=col, lw=1.6, label=f"{lab} ({len(cs)} seeds)")
        ax.fill_between(x, cs.min(0), cs.max(0), color=col, alpha=0.15, lw=0)
    ax.set(title=title, xlabel="training steps (millions)")


def fig_curves(runs):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    stage1 = [("A_none", INK2, "no overseer"), ("B_random_k2_c1.0", BLUE, "random k=2, penalty 3.5"),
              ("C_freeze50_k2", ORANGE, "random k=2, freeze 50"), ("D_history_k2_c1.0", AQUA, "history k=2, penalty 3.5"),
              ("D_location_k2_c1.0", YELLOW, "location k=2, penalty 3.5")]
    stage2 = [("R_none", INK2, "still no overseer")] + [(f"R_random_k{k}_c1.0", col, f"random k={k}, expected penalty 1 apple")
                                                         for k, col in zip(KS, (YELLOW, BLUE, VIOLET, AQUA))] + \
             [("R_freeze50_k2", ORANGE, "random k=2, freeze 50"), ("R_history_k2_c1.0", "#1a6b4c", "history k=2")]
    stage3 = [(f"S_off_after_B_random_k{k}_c1.0", col, f"after random k={k}") for k, col in
              zip(KS, (YELLOW, BLUE, VIOLET, AQUA))] + [("S_off_after_C_freeze50_k2", ORANGE, "after freeze"),
                                                        ("S_off_after_D_history_k2_c1.0", "#1a6b4c", "after history")]
    for col_i, (arms, name) in enumerate([(stage1, "Stage 1: from scratch"),
                                          (stage2, "Stage 2: overseer switched on for over-harvesters"),
                                          (stage3, "Stage 3: overseer switched off, agents keep learning")]):
        curve_panel(axes[0, col_i], runs, arms, "apples_eaten", f"({'abc'[col_i]}) {name}")
        curve_panel(axes[1, col_i], runs, arms, "orchard_alive", "")
        axes[0, col_i].set_ylabel("apples eaten per episode")
        axes[1, col_i].set_ylabel("orchard alive at the end")
        axes[1, col_i].set_ylim(-0.03, 1.03)
        axes[0, col_i].legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "fig_learning_curves.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load(a.runs)
    table = arm_table(runs)
    (OUT / "summary.json").write_text(json.dumps(dict(runs=len(runs), arms=table), indent=1))
    fig_grid(table)
    fig_curves(runs)
    print(f"{len(runs)} runs")
    print(f"{'arm':38s} {'n':>2s} {'apples':>8s} {'alive':>6s} {'viol':>6s} {'caught':>7s} {'penalty':>8s}")
    for arm, r in table.items():
        print(f"{arm:38s} {r['seeds']:2d} {r['apples_eaten']['mean']:8.1f} {r['orchard_alive']['mean']:6.2f} "
              f"{r['violations']['mean']:6.1f} {r['detected']['mean']:7.1f} {r['penalty_apples']['mean']:8.1f}")


if __name__ == "__main__":
    main()
