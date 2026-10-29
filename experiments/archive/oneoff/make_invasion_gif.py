import argparse
import os
import re
import textwrap

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join("results", ".mplconfig")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = os.path.join("results", ".cache")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an animated dashboard GIF from invasion generation metrics."
    )
    parser.add_argument(
        "--input",
        default="results/runs/invasion/invasion_bench_smoke_generations.csv",
        help="Generation CSV from run_invasion.",
    )
    parser.add_argument(
        "--output",
        default="results/runs/showcase/invasion_train_vs_heldout_dynamics.gif",
        help="Output GIF path.",
    )
    parser.add_argument("--title", default="Fishery Commons Invasion Dynamics")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def _regime_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c.startswith("test_") and c.endswith("_collapse_rate") and c != "test_collapse_rate":
            cols.append(c)
    return cols


def _pretty_regime(col: str) -> str:
    name = re.sub(r"^test_", "", col)
    name = re.sub(r"_collapse_rate$", "", name)
    return name.replace("_", " ")


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible support for older generation CSVs.
    """
    out = df.copy()
    if "train_mean_stock" not in out.columns and "mean_stock" in out.columns:
        out["train_mean_stock"] = out["mean_stock"]
    if "test_mean_stock" not in out.columns and "mean_stock" in out.columns:
        out["test_mean_stock"] = out["mean_stock"]
    if "train_collapse_rate" not in out.columns and "collapse_rate" in out.columns:
        out["train_collapse_rate"] = out["collapse_rate"]
    if "test_collapse_rate" not in out.columns and "collapse_rate" in out.columns:
        out["test_collapse_rate"] = out["collapse_rate"]
    required = [
        "generation",
        "train_mean_stock",
        "test_mean_stock",
        "train_collapse_rate",
        "test_collapse_rate",
        "mean_high_harvest_frac",
        "population_diversity",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    return out


def _phase_label(collapse_rate: float) -> str:
    if collapse_rate < 0.25:
        return "stable"
    if collapse_rate < 0.6:
        return "stress"
    if collapse_rate < 0.9:
        return "critical"
    return "collapse-dominant"


def _render_interpretation_panel(
    ax: plt.Axes,
    generation: int,
    train_collapse: float,
    test_collapse: float,
    top_regime_name: str | None,
    top_regime_collapse: float | None,
) -> None:
    ax.clear()
    ax.set_axis_off()
    ax.set_title("Interpretation Guide", fontsize=11, pad=10)

    generalization_gap = test_collapse - train_collapse
    gap_label = "worse held-out robustness" if generalization_gap > 0.05 else "similar train/test robustness"
    base_lines = [
        f"generation: {generation}",
        f"train phase: {_phase_label(train_collapse)} ({train_collapse:.3f})",
        f"test phase: {_phase_label(test_collapse)} ({test_collapse:.3f})",
        f"generalization gap: {generalization_gap:+.3f} ({gap_label})",
        "",
        "How to read panels:",
        "1) Stock Trajectory: higher is better.",
        "2) Collapse Rates: lower is better.",
        "3) Held-Out Regimes: identifies brittle regimes.",
        "4) Strategy Pressure: high-harvest + low diversity means exploiter dominance.",
    ]
    if top_regime_name is not None and top_regime_collapse is not None:
        base_lines.extend(
            [
                "",
                f"Current hardest regime:",
                f"{top_regime_name} ({top_regime_collapse:.3f})",
            ]
        )
    lines: list[str] = []
    for line in base_lines:
        if not line:
            lines.append("")
            continue
        wrapped = textwrap.wrap(line, width=42)
        lines.extend(wrapped if wrapped else [""])

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.6,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f7f7f7", "edgecolor": "#bdbdbd"},
    )


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input).sort_values("generation").reset_index(drop=True)
    df = _ensure_required_columns(df)
    if df.empty:
        raise ValueError(f"No rows found in {args.input}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    generations = df["generation"].to_numpy()
    regime_cols = _regime_columns(df)
    regime_names = [_pretty_regime(c) for c in regime_cols]
    regime_labels = ["\n".join(textwrap.wrap(n, width=14)) for n in regime_names]

    fig = plt.figure(figsize=(15.5, 8.0))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.05, 1.05, 0.78],
        height_ratios=[1.0, 1.0],
        wspace=0.28,
        hspace=0.34,
    )
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.08, top=0.90)
    fig.suptitle(args.title, fontsize=13, fontweight="bold", y=0.965)

    ax_stock = fig.add_subplot(gs[0, 0])
    ax_col = fig.add_subplot(gs[0, 1])
    ax_reg = fig.add_subplot(gs[1, 0])
    ax_strat = fig.add_subplot(gs[1, 1])
    ax_note = fig.add_subplot(gs[:, 2])
    ax_div = ax_strat.twinx()

    x_max = int(df["generation"].max())

    def animate(frame_idx: int):
        row = df.iloc[frame_idx]
        x = generations[: frame_idx + 1]

        # Panel 1: stock
        ax_stock.clear()
        ax_stock.plot(x, df["train_mean_stock"][: frame_idx + 1], label="train mean stock", color="#1f77b4")
        ax_stock.plot(x, df["test_mean_stock"][: frame_idx + 1], label="test mean stock", color="#2ca02c")
        ax_stock.scatter([row["generation"]], [row["train_mean_stock"]], color="#1f77b4", zorder=3, s=24)
        ax_stock.scatter([row["generation"]], [row["test_mean_stock"]], color="#d62728", zorder=3)
        ax_stock.set_xlim(0, max(1, x_max))
        ax_stock.set_ylim(0, max(1.0, float(df[["train_mean_stock", "test_mean_stock"]].max().max()) * 1.05))
        ax_stock.set_title("Stock Trajectory")
        ax_stock.set_xlabel("Generation")
        ax_stock.set_ylabel("Mean Stock")
        ax_stock.grid(alpha=0.25)
        ax_stock.legend(loc="upper right", fontsize=8)

        # Panel 2: collapse
        ax_col.clear()
        ax_col.plot(x, df["train_collapse_rate"][: frame_idx + 1], label="train collapse", color="#ff7f0e")
        ax_col.plot(x, df["test_collapse_rate"][: frame_idx + 1], label="test collapse", color="#9467bd")
        ax_col.scatter([row["generation"]], [row["test_collapse_rate"]], color="#d62728", zorder=3)
        ax_col.set_xlim(0, max(1, x_max))
        ax_col.set_ylim(0, 1.05)
        ax_col.set_title("Collapse Rates")
        ax_col.set_xlabel("Generation")
        ax_col.set_ylabel("Rate")
        ax_col.grid(alpha=0.25)
        ax_col.legend(loc="upper left", fontsize=8)

        # Panel 3: per-regime collapse bars at current generation
        ax_reg.clear()
        top_regime_name = None
        top_regime_value = None
        if regime_cols:
            values = [float(row[c]) for c in regime_cols]
            y_pos = np.arange(len(regime_cols))
            ax_reg.barh(y_pos, values, color="#8c564b")
            ax_reg.set_yticks(y_pos)
            ax_reg.set_yticklabels(regime_labels, fontsize=8)
            ax_reg.set_xlim(0, 1.05)
            ax_reg.set_title("Held-Out Regime Collapse (Current Gen)")
            ax_reg.set_xlabel("Collapse Rate")
            ax_reg.grid(alpha=0.2, axis="x")
            if values:
                top_idx = int(np.argmax(values))
                top_regime_name = regime_names[top_idx]
                top_regime_value = float(values[top_idx])
            if len(values) > 1:
                span = float(max(values) - min(values))
                if min(values) >= 0.999:
                    ax_reg.text(
                        0.02,
                        0.02,
                        "All held-out regimes are saturated at collapse=1.0",
                        transform=ax_reg.transAxes,
                        fontsize=8,
                        color="#a94442",
                        va="bottom",
                    )
                elif max(values) <= 0.001:
                    ax_reg.text(
                        0.02,
                        0.02,
                        "All held-out regimes are stable at collapse=0.0",
                        transform=ax_reg.transAxes,
                        fontsize=8,
                        color="#2b7a0b",
                        va="bottom",
                    )
                elif span < 1e-3:
                    ax_reg.text(
                        0.02,
                        0.02,
                        "Held-out regimes are nearly indistinguishable here",
                        transform=ax_reg.transAxes,
                        fontsize=8,
                        color="#555555",
                        va="bottom",
                    )
        else:
            ax_reg.text(0.5, 0.5, "No per-regime columns", ha="center", va="center")
            ax_reg.set_axis_off()

        # Panel 4: strategy pressure proxies
        ax_strat.clear()
        ax_div.clear()
        ax_strat.plot(
            x,
            df["mean_high_harvest_frac"][: frame_idx + 1],
            label="mean high harvest frac",
            color="#17becf",
            linewidth=2.0,
        )
        ax_strat.scatter([row["generation"]], [row["mean_high_harvest_frac"]], color="#17becf", zorder=3, s=24)
        ax_strat.set_xlim(0, max(1, x_max))
        ax_strat.set_ylim(0, 1.05)
        ax_strat.set_title("Strategy Pressure + Diversity")
        ax_strat.set_xlabel("Generation")
        ax_strat.set_ylabel("High Harvest Fraction", color="#17becf")
        ax_strat.tick_params(axis="y", labelcolor="#17becf")
        ax_strat.grid(alpha=0.25)

        ax_div.plot(
            x,
            df["population_diversity"][: frame_idx + 1],
            label="population diversity",
            color="#bcbd22",
            linewidth=2.0,
        )
        ax_div.scatter([row["generation"]], [row["population_diversity"]], color="#bcbd22", zorder=3, s=24)
        ax_div.set_ylabel("Population Diversity", color="#bcbd22")
        ax_div.tick_params(axis="y", labelcolor="#bcbd22")
        div_max = float(max(1.0, df["population_diversity"].max() * 1.05))
        ax_div.set_ylim(0, div_max)

        l1, lab1 = ax_strat.get_legend_handles_labels()
        l2, lab2 = ax_div.get_legend_handles_labels()
        ax_strat.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=8)

        _render_interpretation_panel(
            ax=ax_note,
            generation=int(row["generation"]),
            train_collapse=float(row["train_collapse_rate"]),
            test_collapse=float(row["test_collapse_rate"]),
            top_regime_name=top_regime_name,
            top_regime_collapse=top_regime_value,
        )

        fig.suptitle(
            f"{args.title} | generation={int(row['generation'])} | test collapse={row['test_collapse_rate']:.3f}",
            fontsize=13,
            fontweight="bold",
        )

    anim = FuncAnimation(fig, animate, frames=len(df), interval=max(100, int(1000 / max(1, args.fps))), repeat=True)
    writer = PillowWriter(fps=max(1, args.fps))
    anim.save(args.output, writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved GIF: {args.output}")


if __name__ == "__main__":
    main()
