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
        description="Create side-by-side governance comparison GIF (none vs monitoring+sanctions)."
    )
    parser.add_argument(
        "--none-input",
        default="results/runs/invasion/invasion_visual_none_generations.csv",
        help="Generation CSV for no-governance condition.",
    )
    parser.add_argument(
        "--governed-input",
        default="results/runs/invasion/invasion_visual_monitoring_sanctions_generations.csv",
        help="Generation CSV for monitoring+sanctions condition.",
    )
    parser.add_argument(
        "--output",
        default="results/runs/showcase/governance_none_vs_sanctions_comparison.gif",
        help="Output GIF path.",
    )
    parser.add_argument("--title", default="Governance Defense Under Strategy Invasion")
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
    out = df.copy()
    if "train_mean_stock" not in out.columns and "mean_stock" in out.columns:
        out["train_mean_stock"] = out["mean_stock"]
    if "test_mean_stock" not in out.columns and "mean_stock" in out.columns:
        out["test_mean_stock"] = out["mean_stock"]
    if "train_collapse_rate" not in out.columns and "collapse_rate" in out.columns:
        out["train_collapse_rate"] = out["collapse_rate"]
    if "test_collapse_rate" not in out.columns and "collapse_rate" in out.columns:
        out["test_collapse_rate"] = out["collapse_rate"]
    required = ["generation", "train_mean_stock", "test_mean_stock", "train_collapse_rate", "test_collapse_rate"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    return out


def _draw_condition_top(
    ax_stock: plt.Axes,
    ax_collapse: plt.Axes,
    x: np.ndarray,
    df: pd.DataFrame,
    row: pd.Series,
    x_max: int,
    title: str,
    show_stock_ylabel: bool,
) -> None:
    ax_stock.clear()
    ax_collapse.clear()

    ax_stock.plot(x, df["train_mean_stock"][: len(x)], color="#1f77b4", linestyle="--", label="train stock")
    ax_stock.plot(x, df["test_mean_stock"][: len(x)], color="#2ca02c", label="held-out stock")
    ax_stock.scatter([row["generation"]], [row["test_mean_stock"]], color="#d62728", s=18, zorder=3)
    ax_stock.set_xlim(0, max(1, x_max))
    ax_stock.set_ylabel("Mean Stock" if show_stock_ylabel else "")
    ax_stock.set_xlabel("Generation")
    ax_stock.grid(alpha=0.25)
    ax_stock.set_title(title)

    ax_collapse.plot(x, df["train_collapse_rate"][: len(x)], color="#ff7f0e", linestyle="--", label="train collapse")
    ax_collapse.plot(x, df["test_collapse_rate"][: len(x)], color="#9467bd", label="held-out collapse")
    ax_collapse.scatter([row["generation"]], [row["test_collapse_rate"]], color="#d62728", s=18, zorder=3)
    ax_collapse.set_ylim(0.0, 1.05)
    ax_collapse.set_ylabel("")

    h1, l1 = ax_stock.get_legend_handles_labels()
    h2, l2 = ax_collapse.get_legend_handles_labels()
    ax_stock.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.5)


def _draw_regime_bars(
    ax: plt.Axes,
    row: pd.Series,
    regime_cols: list[str],
    regime_labels: list[str],
    title: str,
) -> float:
    ax.clear()
    if not regime_cols:
        ax.text(0.5, 0.5, "No per-regime columns", ha="center", va="center")
        ax.set_axis_off()
        return float(row["test_collapse_rate"])

    values = [float(row[c]) for c in regime_cols]
    y_pos = np.arange(len(regime_cols))
    ax.barh(y_pos, values, color="#8c564b")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(regime_labels, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_title(title)
    ax.set_xlabel("Held-Out Regime Collapse")
    ax.grid(alpha=0.2, axis="x")
    return float(np.mean(values))


def main() -> None:
    args = parse_args()

    none_df = _ensure_required_columns(pd.read_csv(args.none_input).sort_values("generation").reset_index(drop=True))
    gov_df = _ensure_required_columns(pd.read_csv(args.governed_input).sort_values("generation").reset_index(drop=True))
    if none_df.empty or gov_df.empty:
        raise ValueError("One of the input CSVs is empty.")

    shared_generations = sorted(set(none_df["generation"]).intersection(set(gov_df["generation"])))
    if not shared_generations:
        raise ValueError("No overlapping generations between none/governed inputs.")

    none_df = none_df[none_df["generation"].isin(shared_generations)].reset_index(drop=True)
    gov_df = gov_df[gov_df["generation"].isin(shared_generations)].reset_index(drop=True)
    if len(none_df) != len(gov_df):
        raise ValueError("Aligned generation counts mismatch after intersection.")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    regime_cols_none = _regime_columns(none_df)
    regime_cols_gov = _regime_columns(gov_df)
    regime_cols = [c for c in regime_cols_none if c in regime_cols_gov]
    regime_names = [_pretty_regime(c) for c in regime_cols]
    regime_labels = ["\n".join(textwrap.wrap(n, width=16)) for n in regime_names]

    x_max = int(max(shared_generations))

    fig = plt.figure(figsize=(16.5, 8.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 0.78], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.34)
    fig.subplots_adjust(left=0.045, right=0.975, bottom=0.08, top=0.90)
    fig.suptitle(args.title, fontsize=13, fontweight="bold", y=0.965)

    ax_none_top = fig.add_subplot(gs[0, 0])
    ax_gov_top = fig.add_subplot(gs[0, 1])
    ax_none_reg = fig.add_subplot(gs[1, 0])
    ax_gov_reg = fig.add_subplot(gs[1, 1])
    ax_note = fig.add_subplot(gs[:, 2])

    ax_none_top_c = ax_none_top.twinx()
    ax_gov_top_c = ax_gov_top.twinx()

    def animate(frame_idx: int):
        row_none = none_df.iloc[frame_idx]
        row_gov = gov_df.iloc[frame_idx]
        x = np.array(shared_generations[: frame_idx + 1], dtype=float)

        _draw_condition_top(
            ax_stock=ax_none_top,
            ax_collapse=ax_none_top_c,
            x=x,
            df=none_df,
            row=row_none,
            x_max=x_max,
            title="No Governance",
            show_stock_ylabel=True,
        )
        _draw_condition_top(
            ax_stock=ax_gov_top,
            ax_collapse=ax_gov_top_c,
            x=x,
            df=gov_df,
            row=row_gov,
            x_max=x_max,
            title="Monitoring + Sanctions",
            show_stock_ylabel=False,
        )
        mean_none = _draw_regime_bars(
            ax=ax_none_reg,
            row=row_none,
            regime_cols=regime_cols,
            regime_labels=regime_labels,
            title="No Governance: Per-Regime Collapse",
        )
        mean_gov = _draw_regime_bars(
            ax=ax_gov_reg,
            row=row_gov,
            regime_cols=regime_cols,
            regime_labels=regime_labels,
            title="Monitoring+Sanctions: Per-Regime Collapse",
        )

        ax_note.clear()
        ax_note.set_axis_off()
        delta = float(row_none["test_collapse_rate"] - row_gov["test_collapse_rate"])
        stock_delta = float(row_gov["test_mean_stock"] - row_none["test_mean_stock"])
        lines = [
            "Interpretation",
            "",
            f"generation: {int(row_none['generation'])}",
            f"held-out collapse (none): {row_none['test_collapse_rate']:.3f}",
            f"held-out collapse (gov): {row_gov['test_collapse_rate']:.3f}",
            f"collapse gap (none-gov): {delta:+.3f}",
            "",
            f"held-out stock (none): {row_none['test_mean_stock']:.1f}",
            f"held-out stock (gov): {row_gov['test_mean_stock']:.1f}",
            f"stock gain (gov-none): {stock_delta:+.1f}",
            "",
            f"per-regime mean collapse gap: {mean_none - mean_gov:+.3f}",
            "",
            "Reading rule:",
            "- lower collapse and higher stock under governance",
            "  means stronger invasion resistance.",
        ]
        ax_note.text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=ax_note.transAxes,
            va="top",
            ha="left",
            fontsize=8.7,
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f7f7f7", "edgecolor": "#bdbdbd"},
        )

        fig.suptitle(
            f"{args.title} | generation={int(row_none['generation'])} | collapse gap={delta:+.3f}",
            fontsize=13,
            fontweight="bold",
            y=0.965,
        )

    anim = FuncAnimation(
        fig,
        animate,
        frames=len(shared_generations),
        interval=max(100, int(1000 / max(1, args.fps))),
        repeat=True,
    )
    anim.save(args.output, writer=PillowWriter(fps=max(1, args.fps)), dpi=args.dpi)
    plt.close(fig)
    print(f"Saved GIF: {args.output}")


if __name__ == "__main__":
    main()
