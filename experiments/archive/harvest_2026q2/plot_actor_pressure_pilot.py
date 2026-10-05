"""Plot the bounded actor-search pressure pilot from context-level summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REQUIRED_COLUMNS = {
    "context", "actor_budget", "mode", "inspection_budget", "train_score",
    "heldout_entrant_payoff", "resolved_risky", "harmful_accepted",
    "resolved_safe", "safe_restricted", "retained_safe_fraction",
    "global_unsafe_rate", "mean_patch_health", "total_welfare", "request_inspections",
    "component_evaluations",
}
COUNT_COLUMNS = ("resolved_risky", "harmful_accepted", "resolved_safe", "safe_restricted")
NUMERIC_COLUMNS = REQUIRED_COLUMNS - {"context", "mode"}
MODES = ("joint", "local_bounded", "local_optimistic")
INSPECTION_BUDGETS = (0, 3, 6)
COLORS = {
    "none": "#5B6570",
    "joint": "#0072B2",
    "local_bounded": "#009E73",
    "local_optimistic": "#D55E00",
}
MARKERS = {"joint": "o", "local_bounded": "s", "local_optimistic": "^"}
LABELS = {
    "none": "No review",
    "joint": "Joint",
    "local_bounded": "Local bounded",
    "local_optimistic": "Local optimistic",
}
INK = "#252B30"
MUTED = "#68747D"
GRID = "#DCE2E4"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.7,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def read_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Summary CSV is empty")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame[["context", "actor_budget", "mode", "inspection_budget"]].isna().any().any():
        raise ValueError("Context, actor budget, mode, and inspection budget must be present")
    if frame.duplicated(["context", "actor_budget", "mode", "inspection_budget"]).any():
        raise ValueError("Duplicate context/actor/mode/inspection-budget rows")
    if not frame["mode"].isin(("none", *MODES)).all():
        raise ValueError("Unexpected reviewer mode")
    if not frame["inspection_budget"].isin(INSPECTION_BUDGETS).all():
        raise ValueError("Inspection budgets must be 0, 3, or 6")
    if not np.isfinite(frame["actor_budget"]).all() or (frame["actor_budget"] < 0).any():
        raise ValueError("Actor budgets must be finite and nonnegative")
    if not np.isfinite(frame["inspection_budget"]).all():
        raise ValueError("Inspection budgets must be finite")
    if frame[list(COUNT_COLUMNS)].isna().any().any():
        raise ValueError("Decision counts must be present, including zero counts")
    counts = frame[list(COUNT_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("Decision counts must be finite nonnegative integers")
    if ((frame["harmful_accepted"] > frame["resolved_risky"]) |
            (frame["safe_restricted"] > frame["resolved_safe"])).any():
        raise ValueError("Decision numerators cannot exceed their denominators")
    return frame


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def finish_axis(ax: plt.Axes, budgets: list[float], *, xlabel: bool = False) -> None:
    ax.set_xticks(range(len(budgets)), [f"{budget:g}" for budget in budgets])
    ax.set_xlim(-0.5, len(budgets) - 0.5)
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel("Actor search budget")


def plot_training_heldout(frame: pd.DataFrame, output_dir: Path) -> None:
    baseline = frame.loc[frame["mode"].eq("none")].copy()
    if baseline.empty:
        raise ValueError("Figure 1 requires mode='none' rows")
    # A no-reviewer condition can be repeated at several inspection budgets.
    for column in ("train_score", "heldout_entrant_payoff"):
        repeated = baseline.groupby(["context", "actor_budget"])[column].nunique(dropna=False)
        if (repeated > 1).any():
            raise ValueError(f"Conflicting none-row values for {column}")
    baseline = baseline.sort_values("inspection_budget").drop_duplicates(
        ["context", "actor_budget"]
    )
    budgets = sorted(frame["actor_budget"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
    specifications = (
        ("train_score", "Training score", "#0072B2"),
        ("heldout_entrant_payoff", "Held-out entrant payoff", "#D55E00"),
    )
    for ax, (column, label, color) in zip(axes, specifications):
        wide = baseline.pivot(index="context", columns="actor_budget", values=column)
        for _, values in wide.iterrows():
            valid = values.reindex(budgets).notna().to_numpy()
            if valid.any():
                x = np.arange(len(budgets))[valid]
                y = values.reindex(budgets).to_numpy(dtype=float)[valid]
                ax.plot(x, y, color=MUTED, alpha=0.33, linewidth=0.8, marker="o",
                        markersize=2.5, zorder=2)
        means = baseline.groupby("actor_budget")[column].mean().reindex(budgets)
        ax.plot(range(len(budgets)), means, color=color, linewidth=2.0, marker="o",
                markersize=4.5, label="Context mean", zorder=3)
        ax.set_title(label)
        ax.set_ylabel(label)
        finish_axis(ax, budgets, xlabel=True)
    fig.suptitle("Training and held-out outcomes", fontsize=10)
    fig.text(0.5, -0.025, "Thin lines connect the same context across budgets; bold lines are context means.",
             ha="center", color=MUTED, fontsize=7.2)
    save_figure(fig, output_dir, "actor_pressure_training_heldout")


def plot_decision_tradeoff(frame: pd.DataFrame, output_dir: Path) -> None:
    reviewer = frame.loc[frame["mode"].isin(MODES)].copy()
    if reviewer.empty:
        raise ValueError("Figure 2 requires reviewer-mode rows")
    budgets = sorted(frame["actor_budget"].unique())
    specifications = (
        ("harmful_accepted", "resolved_risky", "Harmful-approval rate"),
        ("safe_restricted", "resolved_safe", "Safe-restriction rate"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.82, bottom=0.18,
                        wspace=0.13, hspace=0.24)
    offsets = {"joint": -0.17, "local_bounded": 0.0, "local_optimistic": 0.17}
    for row, (numerator, denominator, label) in enumerate(specifications):
        for col, inspection_budget in enumerate(INSPECTION_BUDGETS):
            ax = axes[row, col]
            subset = reviewer.loc[reviewer["inspection_budget"].eq(inspection_budget)]
            baseline = frame.loc[frame["mode"].eq("none")].groupby("actor_budget")[[numerator, denominator]].sum()
            baseline_rates = [baseline.loc[budget, numerator] / baseline.loc[budget, denominator]
                              if budget in baseline.index and baseline.loc[budget, denominator] else np.nan
                              for budget in budgets]
            ax.plot(range(len(budgets)), baseline_rates, color=COLORS["none"],
                    linewidth=1.2, linestyle="--", zorder=1)
            if subset.empty:
                ax.text(0.5, 0.5, "No rows", transform=ax.transAxes,
                        ha="center", va="center", color=MUTED, fontsize=8)
            for mode in MODES:
                mode_rows = subset.loc[subset["mode"].eq(mode)]
                x_line: list[float] = []
                y_line: list[float] = []
                for index, actor_budget in enumerate(budgets):
                    cell = mode_rows.loc[mode_rows["actor_budget"].eq(actor_budget)]
                    if cell.empty:
                        continue
                    x = index + offsets[mode]
                    numerator_sum = int(cell[numerator].sum())
                    denominator_sum = int(cell[denominator].sum())
                    defined = cell.loc[cell[denominator].gt(0)]
                    ax.scatter(np.full(len(defined), x),
                               defined[numerator] / defined[denominator],
                               s=12, marker=MARKERS[mode], color=COLORS[mode], alpha=0.35,
                               linewidths=0, zorder=2)
                    if denominator_sum:
                        rate = numerator_sum / denominator_sum
                        x_line.append(x)
                        y_line.append(rate)
                        ax.scatter(x, rate, s=30, marker=MARKERS[mode], color=COLORS[mode],
                                   edgecolor="white", linewidth=0.5, zorder=4)
                if x_line:
                    ax.plot(x_line, y_line, color=COLORS[mode], linewidth=1.1,
                            alpha=0.85, zorder=3)
            ax.set_ylim(-0.05, 1.08)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            if row == 0:
                ax.set_title(f"Inspect {inspection_budget} of 6 requests")
            if col == 0:
                ax.set_ylabel(label)
            finish_axis(ax, budgets, xlabel=row == 1)
    handles = [Line2D([0], [0], color=COLORS["none"], linestyle="--", linewidth=1.2,
                      label=LABELS["none"])]
    handles += [Line2D([0], [0], color=COLORS[mode], marker=MARKERS[mode],
                       linewidth=1.2, label=LABELS[mode]) for mode in MODES]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.905),
               frameon=False)
    fig.suptitle("Reviewer decisions on the same frozen requests", y=0.985, fontsize=10)
    fig.text(0.5, 0.035,
             "Small marks: context rates; large marks: pooled count ratios. "
             "Undefined rates are omitted; counts are in the summary table. No confidence intervals.",
             ha="center", color=MUTED, fontsize=7)
    save_figure(fig, output_dir, "actor_pressure_decision_tradeoff")


def plot_resources_welfare(frame: pd.DataFrame, output_dir: Path) -> None:
    reviewer = frame.loc[frame["mode"].isin(MODES) & frame["inspection_budget"].eq(6)]
    if reviewer.empty:
        raise ValueError("Figure 3 requires reviewer-mode rows at inspection budget 6")
    budgets = sorted(frame["actor_budget"].unique())
    specifications = (
        ("total_welfare", "Total welfare"),
        ("global_unsafe_rate", "Global unsafe rate"),
        ("mean_patch_health", "Mean patch health"),
        ("component_evaluations", "Component evaluations"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.15,
                        wspace=0.25, hspace=0.35)
    offsets = {"joint": -0.15, "local_bounded": 0.0, "local_optimistic": 0.15}
    for ax, (column, label) in zip(axes.flat, specifications):
        if column != "component_evaluations":
            no_review = frame.loc[frame["mode"].eq("none")].groupby("actor_budget")[column].mean().reindex(budgets)
            ax.plot(range(len(budgets)), no_review, color=COLORS["none"],
                    linestyle="--", linewidth=1.3, zorder=1)
        for mode in MODES:
            mode_rows = reviewer.loc[reviewer["mode"].eq(mode)]
            means = mode_rows.groupby("actor_budget")[column].mean().reindex(budgets)
            x = np.arange(len(budgets), dtype=float) + offsets[mode]
            ax.plot(x, means, color=COLORS[mode], marker=MARKERS[mode],
                    markersize=4, linewidth=1.5, label=LABELS[mode], zorder=3)
            for index, budget in enumerate(budgets):
                values = mode_rows.loc[mode_rows["actor_budget"].eq(budget), column]
                values = values[np.isfinite(values)]
                ax.scatter(np.full(len(values), x[index]), values, s=13,
                           color=COLORS[mode], alpha=0.38, linewidths=0, zorder=2)
        ax.set_title(label)
        ax.set_ylabel(label)
        finish_axis(ax, budgets, xlabel=True)
    handles = [Line2D([0], [0], color=COLORS["none"], linestyle="--", linewidth=1.2,
                      label=LABELS["none"])]
    handles += [Line2D([0], [0], color=COLORS[mode], marker=MARKERS[mode],
                       linewidth=1.4, label=LABELS[mode]) for mode in MODES]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.91), frameon=False)
    fig.suptitle("Long-run outcomes and reviewer work", y=0.985, fontsize=10)
    fig.text(0.5, 0.035, "Reviewers inspect all 6 requests. Small marks are contexts; lines are means. "
             "No-review outcome baseline is dashed.",
             ha="center", color=MUTED, fontsize=7)
    save_figure(fig, output_dir, "actor_pressure_resources_welfare")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path,
                        default=Path("OUTPUT/analysis/context_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("OUTPUT/analysis/figures"))
    args = parser.parse_args()
    frame = read_summary(args.summary_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_training_heldout(frame, args.output_dir)
    plot_decision_tradeoff(frame, args.output_dir)
    plot_resources_welfare(frame, args.output_dir)
    source = Path(__file__)
    (args.output_dir / "figure_manifest.json").write_text(json.dumps({
        "summary_csv": str(args.summary_csv.resolve()),
        "summary_sha256": hashlib.sha256(args.summary_csv.read_bytes()).hexdigest(),
        "plot_script": str(source.resolve()),
        "plot_script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "context_count": int(frame["context"].nunique()),
        "actor_budgets": sorted(int(value) for value in frame["actor_budget"].unique()),
        "figures": [f"actor_pressure_{name}.{suffix}"
                    for name in ("training_heldout", "decision_tradeoff", "resources_welfare")
                    for suffix in ("pdf", "svg", "png")],
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
