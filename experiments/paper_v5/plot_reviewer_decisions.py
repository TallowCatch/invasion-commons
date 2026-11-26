"""Plot fresh-cohort decision quality with context-clustered uncertainty."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("joint", "local_bounded", "local_optimistic")
LABELS = {"joint": "Joint", "local_bounded": "Bounded local",
          "local_optimistic": "Optimistic local"}
COLORS = {"joint": "#21618c", "local_bounded": "#9c4a23",
          "local_optimistic": "#21796a"}
STYLES = {"joint": ("o", "-"), "local_bounded": ("s", "--"),
          "local_optimistic": ("^", ":")}


def estimate(frame: pd.DataFrame, *, replicates: int = 4000,
             seed: int = 20260924) -> tuple[float, float, float]:
    frame = frame.sort_values("context")
    numerator = frame.safe_rejected.to_numpy(dtype=float)
    denominator = frame.safe_resolved.to_numpy(dtype=float)
    if not len(frame) or denominator.sum() <= 0:
        raise ValueError("No resolved-safe proposals for the plot")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(frame), size=(replicates, len(frame)))
    den = denominator[draws].sum(axis=1)
    if np.any(den <= 0):
        raise ValueError("A resample has no resolved-safe proposals")
    boot = numerator[draws].sum(axis=1) / den
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(numerator.sum() / denominator.sum()), float(low), float(high)


def plot(input_dir: Path, output_dir: Path) -> dict:
    context = pd.read_csv(input_dir / "analysis" / "context_decision_counts.csv")
    primary = pd.read_csv(input_dir / "analysis" / "primary_decision_quality.csv")
    expected = {(game, budget, method)
                for game in ("fishery", "harvest") for budget in (0, 3, 6)
                for method in METHODS}
    actual = set(map(tuple, primary[["game", "inspection_budget", "method"]].itertuples(
        index=False, name=None)))
    if actual != expected:
        raise ValueError("Confirmation decision table has missing or extra cells")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "savefig.dpi": 360})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), sharey=True,
                             constrained_layout=True)
    report = []
    for ax, game in zip(axes, ("fishery", "harvest"), strict=True):
        for method in METHODS:
            samples = []
            for budget in (0, 3, 6):
                cell = context[(context.game.eq(game)) &
                               (context.inspection_budget.eq(budget)) &
                               (context.method.eq(method))]
                if len(cell) != 64:
                    raise ValueError(f"Expected 64 independent contexts for {game}/{budget}/{method}")
                value, low, high = estimate(cell, seed=20260924 + budget)
                samples.append((value, low, high))
                check = primary[(primary.game.eq(game)) &
                                (primary.inspection_budget.eq(budget)) &
                                (primary.method.eq(method))].iloc[0]
                if not np.isclose(value, check.safe_reject_rate, atol=1e-12):
                    raise ValueError("Figure estimate disagrees with the primary table")
                report.append(dict(game=game, budget=budget, method=method,
                                   safe_restricted=value, ci_low=low, ci_high=high,
                                   safe_n=int(check.safe_resolved),
                                   risky_approved=int(check.harmful_accepted),
                                   risky_n=int(check.risky_resolved),
                                   unresolved_n=int(check.unresolved)))
            values = np.array(samples)
            marker, linestyle = STYLES[method]
            ax.plot((0, 3, 6), values[:, 0], marker=marker, linestyle=linestyle,
                    linewidth=1.55, markersize=4, color=COLORS[method],
                    label=LABELS[method], zorder=3)
            ax.fill_between((0, 3, 6), values[:, 1], values[:, 2],
                            color=COLORS[method], alpha=.11, linewidth=0, zorder=1)
        ax.set_title("Fishery: shared stock" if game == "fishery" else
                     "Harvest: coupled patches", loc="left", fontweight="bold", pad=10)
        ax.set_xticks((0, 3, 6))
        ax.set_xlabel("Requests inspected (of 6)")
        ax.set_xlim(-.2, 6.2)
        ax.set_ylim(-.035, 1.08)
        ax.set_yticks((0, .25, .5, .75, 1))
        ax.grid(axis="y", linewidth=.45, color="#d4d8dc", zorder=0)
        if game == "fishery":
            annotation = "At 6: risky approved J/B/O = 0/0/63 of 63"
            annotation_y = .94
        else:
            annotation = "At 6: risky approved = 0/1,368 each; 923 unresolved"
            annotation_y = .09
        ax.text(.02, annotation_y, annotation, transform=ax.transAxes,
                fontsize=6.0, color="#3d4650", va="top")
    axes[0].set_ylabel("Resolved-safe proposals scaled down")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, -.065), fontsize=7)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "fig08_reviewer_decisions"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=.06)
    plt.close(fig)
    pd.DataFrame(report).to_csv(output_dir / "fig08_reviewer_decisions_data.csv", index=False)
    return {"points": len(report), "files": [str(stem.with_suffix(f".{ext}"))
                                               for ext in ("pdf", "svg", "png")]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(plot(args.input_dir, args.output_dir))
