from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path("results/runs/showcase/curated")
PAPER_DIR = Path("paper/paper_v5_scalable_oversight_commons")
FIG_DIR = PAPER_DIR / "figures"
TABLE_DIR = PAPER_DIR / "tables"


CONDITION_ORDER = ["none", "bottom_up_only", "top_down_only", "hybrid"]
CONDITION_LABELS = {
    "none": "None",
    "bottom_up_only": "Local",
    "top_down_only": "Global",
    "hybrid": "Hybrid",
}
MODEL_LABELS = {
    "qwen": "Qwen 2.5 3B",
    "llama": "Llama 3.2 3B",
}
SCENARIO_LABELS = {
    "community_irrigation": "Moderate coupling",
    "forest_co_management": "High coupling",
}
ACTOR_LABELS = {
    "low_actor": "Low",
    "medium_actor": "Medium",
    "high_actor": "High",
}
OVERSEER_LABELS = {
    "strong_overseer": "Strong",
    "limited_overseer": "Limited",
    "weak_overseer": "Weak",
}
INK = "#1f2933"
MUTED = "#667085"
GRID = "#d0d5dd"
PAPER = "#fbfaf7"

# Okabe-Ito inspired palette with distinct grayscale values.
CONDITION_COLORS = {
    "none": "#7a8088",
    "bottom_up_only": "#009e73",
    "top_down_only": "#d55e00",
    "hybrid": "#0072b2",
}
CONDITION_MARKERS = {
    "none": "o",
    "bottom_up_only": "s",
    "top_down_only": "^",
    "hybrid": "D",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.75,
            "axes.labelsize": 8.4,
            "axes.titlesize": 9.0,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.4,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def _save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for base in [FIG_DIR, OUT_DIR]:
        fig.savefig(base / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(base / f"{stem}.svg", bbox_inches="tight")
        fig.savefig(base / f"{stem}.png", bbox_inches="tight")


def _write_table(stem: str, text: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / f"{stem}.tex").write_text(text, encoding="utf-8")
    (OUT_DIR / f"{stem}.tex").write_text(text, encoding="utf-8")


def _box(
    ax,
    xy: tuple[float, float],
    text: str,
    *,
    width: float = 2.2,
    height: float = 0.86,
    face: str = "#eef2f7",
    edge: str = INK,
    fontsize: float = 8.0,
    align: str = "center",
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.035",
        linewidth=0.85,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha=align,
        va="center",
        fontsize=fontsize,
        linespacing=1.16,
        color=INK,
    )


def _arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.85,
            color=INK,
            shrinkA=2,
            shrinkB=2,
        )
    )


def make_evidence_chain_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    ax.set_axis_off()
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 3.75)

    ax.text(0.05, 3.55, "Study logic", fontsize=10.0, fontweight="bold", va="top", color=INK)
    ax.text(
        0.05,
        3.28,
        "The substrate stays fixed while strategy source and oversight difficulty become progressively harder.",
        fontsize=7.6,
        color=MUTED,
        va="top",
    )

    stages = [
        ("1", "Fishery support", "central signals\nquotas, sanctions,\nclosures", "#f8fafc"),
        ("2", "Harvest core", "local oversight\nglobal signal\nhybrid oversight", "#eaf3fb"),
        ("3", "Capability pressure", "mutation entrants\nsearch entrants\ncapability gap", "#fff4e8"),
        ("4", "Safety readout", "local predicate\nglobal predicate\ncompositional fail", "#eaf8ef"),
        ("5", "LLM bridge", "structured strategies\nvalidation\nsame pipeline", "#f1edfb"),
    ]
    x0 = 0.22
    w = 1.18
    gap = 0.22
    y = 1.95
    for idx, (num, title, body, face) in enumerate(stages):
        x = x0 + idx * (w + gap)
        _box(ax, (x, y), f"{title}\n{body}", width=w, height=1.00, face=face, fontsize=6.6)
        ax.text(
            x + 0.08,
            y + 0.83,
            num,
            fontsize=7.3,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle,pad=0.16", facecolor=INK, edgecolor=INK, linewidth=0.0),
        )
        if idx < len(stages) - 1:
            _arrow(ax, (x + w + 0.02, y + 0.50), (x + w + gap - 0.03, y + 0.50))

    ax.text(0.22, 1.36, "Variables varied in the scalable-oversight pilot", fontsize=8.2, fontweight="bold", color=INK)
    variable_specs = [
        ("Actor capability", "entrant generation strength", 0.22),
        ("Overseer capability", "recall, delay, capacity, cost", 2.56),
        ("Architecture", "none, local, global signal, hybrid", 4.90),
    ]
    for title, body, x in variable_specs:
        _box(ax, (x, 0.78), f"{title}\n{body}", width=1.92, height=0.46, face=PAPER, fontsize=6.9)

    ax.text(0.22, 0.42, "Primary readouts", fontsize=8.2, fontweight="bold", color=INK)
    ax.text(
        1.38,
        0.42,
        "global unsafe rate, local-pass/global-fail rate, patch health, welfare, intervention burden",
        fontsize=7.4,
        color=INK,
        va="center",
    )

    _save_figure(fig, "fig01_evidence_chain")
    plt.close(fig)


def make_method_schematic() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_axis_off()
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 3.9)

    ax.text(0.05, 3.68, "Benchmark mechanism", fontsize=10.0, fontweight="bold", va="top", color=INK)
    ax.text(
        0.05,
        3.42,
        "The same episode is evaluated through local action checks and global resource-state checks.",
        fontsize=7.6,
        color=MUTED,
        va="top",
    )

    _box(ax, (0.25, 2.46), "Strategy artifacts\nstructured policies", width=1.28, height=0.58, face="#f8fafc", fontsize=6.9)
    _box(ax, (1.86, 2.46), "Entrant process\nmutation or search", width=1.28, height=0.58, face="#f8fafc", fontsize=6.9)
    _box(ax, (3.47, 2.35), "Harvest commons\njoint harvesting\npatch regeneration", width=1.42, height=0.80, face="#eaf3fb", fontsize=6.9)
    _box(ax, (5.34, 2.46), "Next state\nresource health\nwelfare", width=1.28, height=0.58, face="#eaf8ef", fontsize=6.9)
    _arrow(ax, (1.53, 2.75), (1.83, 2.75))
    _arrow(ax, (3.14, 2.75), (3.44, 2.75))
    _arrow(ax, (4.89, 2.75), (5.31, 2.75))

    _box(ax, (0.65, 1.39), "Oversight architecture\nlocal / global signal / hybrid", width=2.25, height=0.58, face=PAPER, fontsize=6.9)
    _box(ax, (4.15, 1.39), "Overseer limits\nrecall, delay, capacity, cost", width=2.25, height=0.58, face=PAPER, fontsize=6.9)
    _arrow(ax, (2.92, 1.68), (4.12, 1.68))
    _arrow(ax, (5.25, 1.98), (4.45, 2.33))

    ax.plot([0.55, 6.65], [0.98, 0.98], color=GRID, linewidth=0.75)
    ax.text(0.55, 0.72, "Local check", fontsize=8.0, fontweight="bold", color=INK)
    ax.text(1.42, 0.72, r"$a_{i,t} \leq a_{\mathrm{sust}}+\epsilon$", fontsize=8.2, color=INK)
    ax.text(3.78, 0.72, "Global check", fontsize=8.0, fontweight="bold", color=INK)
    ax.text(4.72, 0.72, r"$\bar{p}_{t+1}\geq\tau_p,\ f_{t+1}<\tau_f$", fontsize=8.2, color=INK)
    ax.text(
        0.55,
        0.32,
        "Local-pass/global-fail records steps where every local request passes but the aggregate state is unsafe.",
        fontsize=7.5,
        color=INK,
    )
    _save_figure(fig, "fig02_method_schematic")
    plt.close(fig)


def _load_stagea_table() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "harvest_oversight_gap_stageA_table.csv")


def _load_stagea_ranking() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "harvest_oversight_gap_stageA_ranking.csv")


def make_capability_gap_figure() -> None:
    df = _load_stagea_table().copy()
    df["capability_gap"] = pd.to_numeric(df["capability_gap"], errors="coerce")
    runs_path = Path("results/runs/harvest_invasion/curated/harvest_oversight_gap_stageA_runs.csv")
    run_grouped = None
    if runs_path.exists():
        runs = pd.read_csv(runs_path)
        runs["capability_gap"] = pd.to_numeric(runs["capability_gap"], errors="coerce")
        run_grouped = (
            runs.groupby(["capability_gap", "condition"], as_index=False)
            .agg(
                unsafe=("test_global_unsafe_rate_mean", "mean"),
                unsafe_sem=("test_global_unsafe_rate_mean", "sem"),
                local_global=("test_local_pass_global_fail_rate_mean", "mean"),
                local_global_sem=("test_local_pass_global_fail_rate_mean", "sem"),
                patch=("test_mean_patch_health_mean", "mean"),
                patch_sem=("test_mean_patch_health_mean", "sem"),
            )
            .sort_values(["capability_gap", "condition"])
        )
    grouped = (
        df.groupby(["capability_gap", "condition"], as_index=False)
        .agg(
            unsafe=("test_global_unsafe_rate_mean_mean", "mean"),
            unsafe_low=("test_global_unsafe_rate_mean_mean", "min"),
            unsafe_high=("test_global_unsafe_rate_mean_mean", "max"),
            local_global=("test_local_pass_global_fail_rate_mean_mean", "mean"),
            local_global_low=("test_local_pass_global_fail_rate_mean_mean", "min"),
            local_global_high=("test_local_pass_global_fail_rate_mean_mean", "max"),
            patch=("test_mean_patch_health_mean_mean", "mean"),
            patch_low=("test_mean_patch_health_mean_mean", "min"),
            patch_high=("test_mean_patch_health_mean_mean", "max"),
        )
        .sort_values(["capability_gap", "condition"])
    )
    panels = [
        ("unsafe", "Global unsafe rate", "(a) System-level failure"),
        ("local_global", "Local-pass/global-fail rate", "(b) Local checks missing global failure"),
        ("patch", "Mean patch health", "(c) Resource health"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.68), sharex=True)
    for ax, (metric, ylabel, title) in zip(axes, panels):
        ax.axvspan(0.0, 2.18, color="#f2eadf", alpha=0.52, zorder=0)
        for condition in CONDITION_ORDER:
            sub = grouped[grouped["condition"] == condition]
            x = sub["capability_gap"].to_numpy(dtype=float)
            y = sub[metric].to_numpy(dtype=float)
            y_low = sub[f"{metric}_low"].to_numpy(dtype=float)
            y_high = sub[f"{metric}_high"].to_numpy(dtype=float)
            ax.fill_between(
                x,
                y_low,
                y_high,
                color=CONDITION_COLORS[condition],
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                x,
                y,
                marker=CONDITION_MARKERS[condition],
                markersize=3.2,
                linewidth=1.35,
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition],
                zorder=2,
            )
            if run_grouped is not None:
                run_sub = run_grouped[run_grouped["condition"] == condition].sort_values("capability_gap")
                if not run_sub.empty:
                    ax.errorbar(
                        run_sub["capability_gap"].to_numpy(dtype=float),
                        run_sub[metric].to_numpy(dtype=float),
                        yerr=(1.96 * run_sub[f"{metric}_sem"].fillna(0.0).to_numpy(dtype=float)),
                        fmt="none",
                        ecolor=CONDITION_COLORS[condition],
                        elinewidth=0.65,
                        capsize=1.8,
                        capthick=0.65,
                        alpha=0.72,
                        zorder=3,
                    )
            if metric == "patch":
                label_y = y[-1]
                ax.text(
                    2.10,
                    label_y,
                    CONDITION_LABELS[condition],
                    color=CONDITION_COLORS[condition],
                    fontsize=6.6,
                    va="center",
                    ha="left",
                )
        ax.set_title(title, loc="left", pad=4, fontweight="bold")
        ax.set_xlabel(r"Capability gap $\Delta c$")
        ax.set_ylabel(ylabel)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.grid(axis="y", color=GRID, alpha=0.42, linewidth=0.5)
        ax.set_xlim(-2.2, 2.45)
    axes[0].text(0.95, axes[0].get_ylim()[1] * 0.91, "actor advantage", fontsize=6.8, ha="center", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.055))
    fig.text(
        0.99,
        0.035,
        "Bands show stress-cell range; thin bars show approximate 95% run-level CI.",
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save_figure(fig, "fig03_capability_gap")
    plt.close(fig)


def make_winner_map() -> None:
    ranking = _load_stagea_ranking()
    winners = ranking[ranking["rank"] == 1].copy()
    actor_order = ["low_actor", "medium_actor", "high_actor"]
    overseer_order = ["strong_overseer", "limited_overseer", "weak_overseer"]
    condition_order = ["none", "bottom_up_only", "top_down_only", "hybrid"]
    values = {condition: idx for idx, condition in enumerate(condition_order)}
    abbrev = {"none": "N", "bottom_up_only": "L", "top_down_only": "G", "hybrid": "H"}
    cmap = ListedColormap([CONDITION_COLORS[c] for c in condition_order])
    norm = BoundaryNorm(np.arange(-0.5, len(condition_order) + 0.5), cmap.N)

    scenarios = ["community_irrigation", "forest_co_management"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), constrained_layout=True)
    for ax, scenario in zip(axes, scenarios):
        data = np.full((len(actor_order), len(overseer_order)), np.nan)
        labels = [["" for _ in overseer_order] for _ in actor_order]
        margins = [["" for _ in overseer_order] for _ in actor_order]
        sdf = winners[winners["scenario_preset"] == scenario]
        for _, row in sdf.iterrows():
            i = actor_order.index(row["actor_capability_level"])
            j = overseer_order.index(row["overseer_capability_level"])
            condition = row["condition"]
            data[i, j] = values[condition]
            labels[i][j] = abbrev[condition]
            cell = ranking[
                (ranking["scenario_preset"] == scenario)
                & (ranking["actor_capability_level"] == row["actor_capability_level"])
                & (ranking["overseer_capability_level"] == row["overseer_capability_level"])
            ].sort_values("rank")
            if len(cell) >= 2:
                other = cell[cell["condition"] != condition]
                best_other_patch = float(other["test_mean_patch_health_mean_mean"].max())
                margin = float(row["test_mean_patch_health_mean_mean"]) - best_other_patch
                margins[i][j] = f"{margin:+.2f}"
        ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(SCENARIO_LABELS[scenario], fontweight="bold")
        ax.set_xticks(np.arange(len(overseer_order)), [OVERSEER_LABELS[o] for o in overseer_order])
        ax.set_yticks(np.arange(len(actor_order)), [ACTOR_LABELS[a] for a in actor_order])
        ax.set_xlabel("Overseer capability")
        ax.set_ylabel("Actor capability")
        for i in range(len(actor_order)):
            for j in range(len(overseer_order)):
                ax.text(j, i - 0.07, labels[i][j], ha="center", va="center", fontsize=11, fontweight="bold", color="white")
                ax.text(j, i + 0.19, margins[i][j], ha="center", va="center", fontsize=6.6, color="white")
        ax.set_xticks(np.arange(-0.5, len(overseer_order), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(actor_order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
    legend_handles = [
        Rectangle((0, 0), 1, 1, color=CONDITION_COLORS[c], label=f"{abbrev[c]} = {CONDITION_LABELS[c]}")
        for c in condition_order
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    _save_figure(fig, "fig04_winner_map")
    plt.close(fig)


def make_case_trace_figure() -> None:
    trace = pd.read_csv(OUT_DIR / "harvest_oversight_gap_stageA_oversight_case_trace.csv")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.35), sharex=True, gridspec_kw={"height_ratios": [1.12, 1]})
    x = trace["step"]
    fail_steps = trace["local_pass_global_fail"].astype(bool)

    for ax in axes:
        ax.fill_between(x, 0, 1, where=fail_steps, color="#b23a48", alpha=0.10, transform=ax.get_xaxis_transform(), step="mid")

    axes[0].plot(x, trace["mean_patch_health_after"], color=INK, linewidth=1.35)
    axes[0].axhline(10.0, color="#b23a48", linestyle="--", linewidth=1.0, label="Global threshold")
    axes[0].scatter(x[fail_steps], trace.loc[fail_steps, "mean_patch_health_after"], color="#b23a48", s=16, zorder=3)
    axes[0].set_ylabel("Patch health")
    axes[0].set_title("(a) Aggregate resource state crosses the unsafe threshold", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color=GRID, alpha=0.42, linewidth=0.5)
    axes[0].text(68, 10.10, "global threshold", color="#b23a48", fontsize=6.8, va="bottom")
    axes[0].text(7, 9.15, "red markers: local-pass/global-fail steps", color="#b23a48", fontsize=6.8)

    axes[1].plot(x, trace["max_requested_frac"], color="#4aa889", linewidth=1.35, label="Largest local request")
    axes[1].axhline(0.40, color="#4aa889", linestyle="--", linewidth=1.0, alpha=0.75, label="Local threshold")
    axes[1].set_ylim(0, max(0.9, float(trace["max_requested_frac"].max()) + 0.05))
    axes[1].set_ylabel("Requested fraction")
    axes[1].set_xlabel("Episode step")
    axes[1].set_title("(b) The largest local request remains at or near the local acceptance boundary", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color=GRID, alpha=0.42, linewidth=0.5)
    axes[1].text(55, 0.42, "local threshold", color="#4aa889", fontsize=6.8, va="bottom")
    axes[1].text(52, 0.70, "largest local request", color="#4aa889", fontsize=6.8, va="bottom")
    fig.tight_layout()
    _save_figure(fig, "fig05_case_trace")
    plt.close(fig)


def _load_llm_maps(use_samples: bool = False) -> pd.DataFrame:
    rows = []
    suffix = "map_samples.csv" if use_samples else "map_summary.csv"
    paths = {
        ("qwen", "ideal"): OUT_DIR / f"harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_{suffix}",
        ("qwen", "constrained"): OUT_DIR / f"harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_constrained_{suffix}",
        ("llama", "ideal"): OUT_DIR / f"harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_{suffix}",
        ("llama", "constrained"): OUT_DIR / f"harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_constrained_{suffix}",
    }
    for (model, regime), path in paths.items():
        df = pd.read_csv(path)
        df["model_short"] = model
        df["regime_short"] = regime
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def make_llm_bridge_figure() -> None:
    df = _load_llm_maps(use_samples=True)
    grouped = (
        df.groupby(["model_short", "scenario_preset", "regime_short", "condition", "exploitative_share"], as_index=False)
        .agg(
            mean_patch_health=("mean_patch_health", "mean"),
            mean_patch_health_sem=("mean_patch_health", "sem"),
        )
    )
    line_styles = {"ideal": "-", "constrained": "--"}

    columns = [
        ("qwen", "community_irrigation"),
        ("qwen", "forest_co_management"),
        ("llama", "community_irrigation"),
        ("llama", "forest_co_management"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.95), sharex=True, sharey=True)
    for col, (model, scenario) in enumerate(columns):
        sub = grouped[(grouped["model_short"] == model) & (grouped["scenario_preset"] == scenario)]
        for condition in CONDITION_ORDER:
            for regime in ["ideal", "constrained"]:
                vals = (
                    sub[(sub["condition"] == condition) & (sub["regime_short"] == regime)]
                    .sort_values("exploitative_share")
                    .copy()
                )
                marker = CONDITION_MARKERS[condition]
                axes[col].plot(
                    vals["exploitative_share"],
                    vals["mean_patch_health"],
                    color=CONDITION_COLORS[condition],
                    linestyle=line_styles[regime],
                    marker=marker,
                    markersize=2.7,
                    linewidth=1.05,
                    alpha=0.92,
                )
                axes[col].errorbar(
                    vals["exploitative_share"],
                    vals["mean_patch_health"],
                    yerr=1.96 * vals["mean_patch_health_sem"].fillna(0.0),
                    fmt="none",
                    ecolor=CONDITION_COLORS[condition],
                    elinewidth=0.50,
                    capsize=1.4,
                    capthick=0.50,
                    alpha=0.50,
                )
        title = f"{MODEL_LABELS[model]}\n{SCENARIO_LABELS[scenario]}"
        axes[col].set_title(title, pad=4, fontweight="bold", fontsize=7.9)
        axes[col].set_xticks([0.0, 0.5, 1.0])
        axes[col].grid(axis="y", color=GRID, alpha=0.40, linewidth=0.45)
        axes[col].set_ylim(2.0, 18.0)
    axes[0].set_ylabel("Patch health")
    fig.supxlabel("Exploitative share", y=0.12, fontsize=8.2, color=INK)
    condition_handles = [
        Line2D([0], [0], color=CONDITION_COLORS[c], linewidth=1.6, label=CONDITION_LABELS[c])
        for c in CONDITION_ORDER
    ]
    regime_handles = [
        Line2D([0], [0], color=INK, linestyle="-", linewidth=1.4, label="Ideal"),
        Line2D([0], [0], color=INK, linestyle="--", linewidth=1.4, label="Constrained"),
    ]
    fig.legend(
        handles=condition_handles + regime_handles,
        ncol=6,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.4,
        handlelength=2.4,
        handletextpad=0.5,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    _save_figure(fig, "fig06_llm_bridge")
    plt.close(fig)


def write_llm_tables() -> None:
    bank_rows = []
    bank_paths = {
        "Qwen 2.5 3B": (
            OUT_DIR / "harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_bank.csv",
            OUT_DIR / "harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_summary.csv",
        ),
        "Llama 3.2 3B": (
            OUT_DIR / "harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_bank.csv",
            OUT_DIR / "harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_summary.csv",
        ),
    }
    for model_label, (bank_path, summary_path) in bank_paths.items():
        bank = pd.read_csv(bank_path)
        summary = pd.read_csv(summary_path)
        piv = bank.groupby("bank_attitude")[["low_harvest_frac", "mid_harvest_frac", "high_harvest_frac", "cap_compliance_margin"]].mean()
        row = {
            "Model": model_label,
            "Accepted": "32+32",
            "Attempts": f"{int(summary['attempts'].sum())}",
            "Parse failures": f"{int(summary['parse_failures'].sum())}",
            "Coop high harvest": f"{piv.loc['cooperative', 'high_harvest_frac']:.2f}",
            "Exploit high harvest": f"{piv.loc['exploitative', 'high_harvest_frac']:.2f}",
            "Coop cap margin": f"{piv.loc['cooperative', 'cap_compliance_margin']:.2f}",
            "Exploit cap margin": f"{piv.loc['exploitative', 'cap_compliance_margin']:.2f}",
        }
        bank_rows.append(row)
    bank_df = pd.DataFrame(bank_rows)

    bank_lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{LLM strategy-bank validity and separation. Each model generated 32 cooperative and 32 exploitative structured Harvest strategies.}",
        "    \\label{tab:llm-bank-validity}",
        "    \\resizebox{\\linewidth}{!}{%",
        "    \\begin{tabular}{lrrrrrrr}",
        "        \\toprule",
        "        Model & Accepted & Attempts & Failures & Coop high & Exploit high & Coop cap & Exploit cap \\\\",
        "        \\midrule",
    ]
    for _, row in bank_df.iterrows():
        bank_lines.append(
            f"        {row['Model']} & {row['Accepted']} & {row['Attempts']} & {row['Parse failures']} & "
            f"{row['Coop high harvest']} & {row['Exploit high harvest']} & {row['Coop cap margin']} & {row['Exploit cap margin']} \\\\"
        )
    bank_lines.extend(["        \\bottomrule", "    \\end{tabular}", "    }", "\\end{table}", ""])
    _write_table("table_llm_bank_validity", "\n".join(bank_lines))

    map_df = _load_llm_maps(use_samples=True)
    local = map_df[map_df["condition"] == "bottom_up_only"][
        [
            "model_short",
            "regime_short",
            "scenario_preset",
            "exploitative_share",
            "population_id",
            "mean_patch_health",
            "garden_failure_rate",
            "mean_welfare",
            "mean_governance_budget_spent",
        ]
    ].rename(
        columns={
            "mean_patch_health": "local_patch_health",
            "garden_failure_rate": "local_failure_rate",
            "mean_welfare": "local_welfare",
            "mean_governance_budget_spent": "local_burden",
        }
    )
    summary_rows = []
    for condition in ["top_down_only", "hybrid"]:
        sub = map_df[map_df["condition"] == condition].merge(
            local,
            on=["model_short", "regime_short", "scenario_preset", "exploitative_share", "population_id"],
            how="inner",
        )
        constrained = sub[sub["regime_short"] == "constrained"]
        worst_cell = (
            map_df[(map_df["condition"] == condition) & (map_df["exploitative_share"] == 1.0)]
            .groupby(["model_short", "regime_short", "scenario_preset"], as_index=False)["garden_failure_rate"]
            .mean()["garden_failure_rate"]
            .max()
        )
        summary_rows.append(
            {
                "Architecture": CONDITION_LABELS[condition],
                "Patch gain vs local": float((sub["mean_patch_health"] - sub["local_patch_health"]).mean()),
                "Failure reduction vs local": float((sub["local_failure_rate"] - sub["garden_failure_rate"]).mean()),
                "Welfare delta vs local": float((sub["mean_welfare"] - sub["local_welfare"]).mean()),
                "Constrained burden": float(constrained["mean_governance_budget_spent"].mean()),
                "Worst-cell failure at share 1.0": float(worst_cell),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    outcome_lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{LLM bridge protection summary relative to local oversight. Values average over both models, both stress settings, ideal and constrained oversight, and exploitative shares of 0.0, 0.5, and 1.0.}",
        "    \\label{tab:llm-bridge-outcomes}",
        "    \\begin{tabular}{lrrrrr}",
        "        \\toprule",
        "        Architecture & $\\Delta$ patch & $\\Delta$ failure & $\\Delta$ welfare & Constrained burden & Worst-cell failure \\\\",
        "        \\midrule",
    ]
    for _, row in summary_df.iterrows():
        outcome_lines.append(
            f"        {row['Architecture']} & "
            f"{row['Patch gain vs local']:.2f} & "
            f"{row['Failure reduction vs local']:.3f} & "
            f"{row['Welfare delta vs local']:.2f} & "
            f"{row['Constrained burden']:.2f} & "
            f"{row['Worst-cell failure at share 1.0']:.3f} \\\\"
        )
    outcome_lines.extend(["        \\bottomrule", "    \\end{tabular}", "\\end{table}", ""])
    _write_table("table_llm_bridge_outcomes", "\n".join(outcome_lines))

    detailed = (
        _load_llm_maps()
        .groupby(["model_short", "regime_short", "condition"], as_index=False)
        .agg(
            patch=("mean_patch_health", "mean"),
            fail=("garden_failure_rate", "mean"),
            welfare=("mean_welfare", "mean"),
            burden=("mean_governance_budget_spent", "mean"),
        )
    )
    detailed_lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Detailed LLM-bridge outcomes, averaged over two stress settings and exploitative-share levels.}",
        "    \\label{tab:llm-bridge-outcomes-detailed}",
        "    \\begin{tabular}{llrrrr}",
        "        \\toprule",
        "        Model/regime & Architecture & Patch health & Failure & Welfare & Burden \\\\",
        "        \\midrule",
    ]
    for model in ["qwen", "llama"]:
        for regime in ["ideal", "constrained"]:
            sub = detailed[(detailed["model_short"] == model) & (detailed["regime_short"] == regime)].set_index("condition").reindex(CONDITION_ORDER)
            first = True
            label = f"{MODEL_LABELS[model]}, {regime}"
            for condition, row in sub.iterrows():
                model_cell = label if first else ""
                detailed_lines.append(
                    f"        {model_cell} & {CONDITION_LABELS[condition]} & {row['patch']:.2f} & "
                    f"{row['fail']:.3f} & {row['welfare']:.2f} & {row['burden']:.2f} \\\\"
                )
                first = False
            detailed_lines.append("        \\addlinespace")
    detailed_lines.extend(["        \\bottomrule", "    \\end{tabular}", "\\end{table}", ""])
    _write_table("table_llm_bridge_outcomes_detailed", "\n".join(detailed_lines))


def write_stagea_tables() -> None:
    for src_name, dst_name in [
        ("harvest_oversight_gap_stageA_condition_means.tex", "table_stagea_condition_means.tex"),
        ("harvest_oversight_gap_stageA_actor_capability_validation_extended.tex", "table_actor_capability_validation.tex"),
    ]:
        src = OUT_DIR / src_name
        dst = TABLE_DIR / dst_name
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding="utf-8")
        text = text.replace("Stage A mean outcomes", "Main-pilot mean outcomes")
        dst.write_text(text, encoding="utf-8")

    sens = pd.read_csv(OUT_DIR / "harvest_oversight_gap_stageA_threshold_sensitivity.csv")
    sens = sens[sens["local_safety_margin"].isin([0.00, 0.05, 0.10])].copy()
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Episode-level threshold sensitivity for the extracted local-pass/global-fail case.}",
        "    \\label{tab:threshold-sensitivity}",
        "    \\begin{tabular}{rrrr}",
        "        \\toprule",
        "        Local margin & Global threshold & Fail rate & Fail steps \\\\",
        "        \\midrule",
    ]
    for _, row in sens.iterrows():
        lines.append(
            f"        {row['local_safety_margin']:.2f} & "
            f"{row['global_min_mean_patch_health']:.1f} & "
            f"{row['local_pass_global_fail_rate']:.3f} & "
            f"{int(row['local_pass_global_fail_steps'])} \\\\"
        )
    lines.extend(["        \\bottomrule", "    \\end{tabular}", "\\end{table}", ""])
    _write_table("table_threshold_sensitivity", "\n".join(lines))


def _load_threshold_replay_grid() -> pd.DataFrame:
    path = OUT_DIR / "harvest_oversight_gap_threshold_replay_full_grid_recovered.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing full threshold replay grid: {path}. "
            "Download the recovered GitHub artifact before building paper figures."
        )
    return pd.read_csv(path)


def _threshold_cell_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            [
                "scenario_preset",
                "local_safety_margin",
                "global_min_mean_patch_health",
                "actor_capability_level",
                "overseer_capability_level",
                "condition",
            ],
            as_index=False,
        )
        .agg(
            unsafe=("test_global_unsafe_rate_mean", "mean"),
            lpgf=("test_local_pass_global_fail_rate_mean", "mean"),
            patch=("test_mean_patch_health_mean", "mean"),
            garden=("test_garden_failure_mean", "mean"),
            welfare=("test_mean_welfare_mean", "mean"),
            burden=("test_governance_budget_spent_mean", "mean"),
        )
    )


def make_threshold_robustness_figure() -> None:
    cell = _threshold_cell_summary(_load_threshold_replay_grid())
    wins = []
    for keys, group in cell.groupby(
        [
            "scenario_preset",
            "local_safety_margin",
            "global_min_mean_patch_health",
            "actor_capability_level",
            "overseer_capability_level",
        ]
    ):
        best = group.sort_values(["patch", "lpgf"], ascending=[False, True]).iloc[0]
        wins.append(
            {
                "scenario_preset": keys[0],
                "condition": best["condition"],
            }
        )
    win_df = pd.DataFrame(wins)
    win_counts = (
        win_df.groupby(["scenario_preset", "condition"], as_index=False)
        .size()
        .rename(columns={"size": "wins"})
    )
    summary = (
        cell.groupby(["scenario_preset", "condition"], as_index=False)
        .agg(
            unsafe=("unsafe", "mean"),
            lpgf=("lpgf", "mean"),
            patch=("patch", "mean"),
        )
        .merge(win_counts, on=["scenario_preset", "condition"], how="left")
        .fillna({"wins": 0})
    )

    scenarios = ["community_irrigation", "forest_co_management"]
    x = np.arange(len(scenarios))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(CONDITION_ORDER))
    panels = [
        ("wins", "Patch-health winning cells", "(a) Patch-health winners", 225),
        ("unsafe", "Global unsafe rate", "(b) System-level failure", None),
        ("lpgf", "Local-pass/global-fail rate", "(c) Compositional failure", None),
        ("patch", "Mean patch health", "(d) Resource health", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.35), sharex=True)
    axes = axes.ravel()
    for ax, (metric, ylabel, title, ref_line) in zip(axes, panels):
        for idx, condition in enumerate(CONDITION_ORDER):
            vals = []
            for scenario in scenarios:
                row = summary[(summary["scenario_preset"] == scenario) & (summary["condition"] == condition)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else 0.0)
            bars = ax.bar(
                x + offsets[idx],
                vals,
                width=width,
                color=CONDITION_COLORS[condition],
                edgecolor=INK,
                linewidth=0.35,
                label=CONDITION_LABELS[condition],
            )
            if metric == "wins":
                for bar, value in zip(bars, vals):
                    if value > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            value + 4,
                            f"{int(value)}",
                            ha="center",
                            va="bottom",
                            fontsize=6.4,
                            color=INK,
                        )
        if ref_line is not None:
            ax.axhline(ref_line, color=GRID, linewidth=0.6, linestyle="--")
            ax.set_ylim(0, ref_line * 1.18)
            ax.text(
                0.98,
                0.92,
                f"{ref_line} cells per setting",
                fontsize=6.2,
                color=MUTED,
                ha="right",
                va="top",
                transform=ax.transAxes,
            )
        ax.set_title(title, loc="left", fontweight="bold", pad=4)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios])
        ax.grid(axis="y", color=GRID, alpha=0.40, linewidth=0.45)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save_figure(fig, "fig07_threshold_robustness")
    plt.close(fig)


def write_threshold_robustness_table() -> None:
    cell = _threshold_cell_summary(_load_threshold_replay_grid())
    wins = []
    for keys, group in cell.groupby(
        [
            "scenario_preset",
            "local_safety_margin",
            "global_min_mean_patch_health",
            "actor_capability_level",
            "overseer_capability_level",
        ]
    ):
        best = group.sort_values(["patch", "lpgf"], ascending=[False, True]).iloc[0]
        wins.append({"scenario_preset": keys[0], "condition": best["condition"]})
    win_counts = (
        pd.DataFrame(wins)
        .groupby(["scenario_preset", "condition"], as_index=False)
        .size()
        .rename(columns={"size": "cell_wins"})
    )
    summary = (
        cell.groupby(["scenario_preset", "condition"], as_index=False)
        .agg(
            unsafe=("unsafe", "mean"),
            lpgf=("lpgf", "mean"),
            positive_lpgf=("lpgf", lambda s: int((s > 0).sum())),
            cells=("lpgf", "size"),
            patch=("patch", "mean"),
            garden=("garden", "mean"),
        )
        .merge(win_counts, on=["scenario_preset", "condition"], how="left")
        .fillna({"cell_wins": 0})
    )
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Full threshold robustness sweep. The grid varies five local safety margins, five global patch-health thresholds, three actor-capability levels, and three overseer-capability levels. Cell wins are patch-health wins out of 225 threshold-capability cells per stress setting. Positive LPGF cells are cells with nonzero local-pass/global-fail rate.}",
        "    \\label{tab:threshold-robustness-sweep}",
        "    \\resizebox{\\linewidth}{!}{%",
        "    \\begin{tabular}{llrrrrrr}",
        "        \\toprule",
        "        Stress setting & Architecture & Mean unsafe & Mean LPGF & Positive LPGF cells & Cell wins & Mean patch & Garden fail \\\\",
        "        \\midrule",
    ]
    for scenario in ["community_irrigation", "forest_co_management"]:
        for condition in CONDITION_ORDER:
            row = summary[(summary["scenario_preset"] == scenario) & (summary["condition"] == condition)].iloc[0]
            lines.append(
                f"        {SCENARIO_LABELS[scenario]} & {CONDITION_LABELS[condition]} & "
                f"{row['unsafe']:.3f} & "
                f"{row['lpgf']:.3f} & "
                f"{int(row['positive_lpgf'])}/{int(row['cells'])} & "
                f"{int(row['cell_wins'])} & "
                f"{row['patch']:.2f} & "
                f"{row['garden']:.3f} \\\\"
            )
        if scenario == "community_irrigation":
            lines.append("        \\addlinespace")
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "    }",
            "\\end{table}",
            "",
        ]
    )
    _write_table("table_threshold_robustness_sweep", "\n".join(lines))


def write_stress_setting_table() -> None:
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Stress-setting parameters used in the main pilot. Partner weights are ordered as cooperative, balanced, adversarial.}",
        "    \\label{tab:stress-settings}",
        "    \\resizebox{\\linewidth}{!}{%",
        "    \\begin{tabular}{llrrrrll}",
        "        \\toprule",
        "        Setting & Internal preset & Patch init & Regen. & Weather & Spillover & Partner weights & Communication / credit \\\\",
        "        \\midrule",
        "        Moderate coupling & community\\_irrigation & 14.0 & 0.64 & 0.30 & 0.15 & 0.34 / 0.33 / 0.33 & on / off \\\\",
        "        High coupling & forest\\_co\\_management & 12.5 & 0.56 & 0.42 & 0.22 & 0.10 / 0.30 / 0.60 & on / on \\\\",
        "        \\bottomrule",
        "    \\end{tabular}",
        "    }",
        "\\end{table}",
        "",
    ]
    _write_table("table_stress_settings", "\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    _setup_style()
    make_evidence_chain_figure()
    make_method_schematic()
    make_capability_gap_figure()
    make_winner_map()
    make_case_trace_figure()
    make_llm_bridge_figure()
    make_threshold_robustness_figure()
    write_stagea_tables()
    write_stress_setting_table()
    write_llm_tables()
    write_threshold_robustness_table()


if __name__ == "__main__":
    main()
