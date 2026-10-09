from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


LEVEL_ORDER = [
    "strong_overseer",
    "recall_limited_only",
    "delay_limited_only",
    "capacity_limited_only",
    "cost_limited_only",
    "limited_overseer",
    "weak_overseer",
]

LEVEL_LABELS = {
    "strong_overseer": "Strong",
    "recall_limited_only": "Recall limit",
    "delay_limited_only": "Delay limit",
    "capacity_limited_only": "Capacity limit",
    "cost_limited_only": "Cost limit",
    "limited_overseer": "Bundled limited",
    "weak_overseer": "Bundled weak",
}

CONDITION_LABELS = {
    "top_down_only": "Global",
    "hybrid": "Hybrid",
}

METRIC_COLUMNS = {
    "unsafe": "test_global_unsafe_rate_mean_mean",
    "local_global_fail": "test_local_pass_global_fail_rate_mean_mean",
    "patch_health": "test_mean_patch_health_mean_mean",
    "welfare": "test_mean_welfare_mean_mean",
    "burden": "test_governance_budget_spent_mean_mean",
    "missed_target": "test_missed_target_rate_mean_mean",
    "delayed_count": "test_delayed_intervention_count_mean_mean",
}

CONDITION_COLORS = {
    "top_down_only": "#d55e00",
    "hybrid": "#0072b2",
}

PAPER = "#fbfaf7"
INK = "#1f2933"
GRID = "#d0d5dd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize reduced overseer-limit ablation outputs.")
    parser.add_argument(
        "--summary-csv",
        default="results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced.csv",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced",
    )
    parser.add_argument("--scenario", default=None, help="Select a scenario; never pool scenarios implicitly.")
    parser.add_argument("--actor", default=None, help="Select an actor setting for this slice plot.")
    return parser.parse_args()


def _load_summary(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"condition", "overseer_capability_level"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df[df["condition"].isin(CONDITION_LABELS)].copy()
    df["condition_label"] = df["condition"].map(CONDITION_LABELS)
    df["level_label"] = df["overseer_capability_level"].map(LEVEL_LABELS)
    df["level_rank"] = df["overseer_capability_level"].map({level: idx for idx, level in enumerate(LEVEL_ORDER)})
    df = df.sort_values(["level_rank", "condition_label"]).reset_index(drop=True)
    return df


def _compact_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in LEVEL_ORDER:
        sub = df[df["overseer_capability_level"] == level].set_index("condition")
        if sub.empty:
            continue
        row = {"Overseer limit": LEVEL_LABELS[level]}
        for condition in ["top_down_only", "hybrid"]:
            prefix = "G" if condition == "top_down_only" else "H"
            row[f"{prefix} unsafe"] = sub.loc[condition, METRIC_COLUMNS["unsafe"]]
            row[f"{prefix} LPGF"] = sub.loc[condition, METRIC_COLUMNS["local_global_fail"]]
            row[f"{prefix} patch"] = sub.loc[condition, METRIC_COLUMNS["patch_health"]]
            row[f"{prefix} welfare"] = sub.loc[condition, METRIC_COLUMNS["welfare"]]
            row[f"{prefix} burden"] = sub.loc[condition, METRIC_COLUMNS["burden"]]
            row[f"{prefix} missed"] = sub.loc[condition, METRIC_COLUMNS["missed_target"]]
            row[f"{prefix} delayed"] = sub.loc[condition, METRIC_COLUMNS["delayed_count"]]
        rows.append(row)
    return pd.DataFrame(rows)


def _mechanism_summary(df: pd.DataFrame) -> str:
    strong = df[df["overseer_capability_level"] == "strong_overseer"].set_index("condition")
    effect_rows = []
    for level in LEVEL_ORDER[1:]:
        sub = df[df["overseer_capability_level"] == level].set_index("condition")
        if sub.empty:
            continue
        unsafe_shift = (sub[METRIC_COLUMNS["unsafe"]] - strong[METRIC_COLUMNS["unsafe"]]).mean()
        lpgf_shift = (sub[METRIC_COLUMNS["local_global_fail"]] - strong[METRIC_COLUMNS["local_global_fail"]]).mean()
        patch_shift = (sub[METRIC_COLUMNS["patch_health"]] - strong[METRIC_COLUMNS["patch_health"]]).mean()
        welfare_shift = (sub[METRIC_COLUMNS["welfare"]] - strong[METRIC_COLUMNS["welfare"]]).mean()
        burden_shift = (sub[METRIC_COLUMNS["burden"]] - strong[METRIC_COLUMNS["burden"]]).mean()
        effect_rows.append(
            {
                "level": level,
                "unsafe_shift": unsafe_shift,
                "lpgf_shift": lpgf_shift,
                "patch_shift": patch_shift,
                "welfare_shift": welfare_shift,
                "burden_shift": burden_shift,
            }
        )
    effects = pd.DataFrame(effect_rows)
    if effects.empty:
        return "Reduced overseer-limit ablation did not produce comparable rows."
    worst = effects.sort_values(["patch_shift", "unsafe_shift"], ascending=[True, False]).iloc[0]
    worst_label = LEVEL_LABELS[str(worst["level"])]
    return (
        "Relative to the strong overseer baseline, "
        f"{worst_label.lower()} has the largest observed mean patch-health reduction across the protected architectures. "
        f"Its mean patch-health shift is {worst['patch_shift']:.2f}, its unsafe-rate shift is {worst['unsafe_shift']:.3f}, "
        f"and its local-pass/global-fail shift is {worst['lpgf_shift']:.3f}. "
        "This is a reduced ablation over one stress setting and one actor-capability level, so it is a mechanism check rather than a full robustness result."
    )


def _write_markdown(table: pd.DataFrame, mechanism_text: str, path: Path) -> None:
    lines = [
        "# Reduced Overseer-Limit Ablation",
        "",
        "This ablation covers the single scenario and generation treatment selected from the input, with the global/hybrid packages. It does not pool scenarios.",
        "",
        mechanism_text,
        "",
        "| Overseer limit | G unsafe | H unsafe | G LPGF | H LPGF | G patch | H patch | G welfare | H welfare | G burden | H burden | G missed | H missed | G delayed | H delayed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['Overseer limit']} | "
            f"{row['G unsafe']:.3f} | {row['H unsafe']:.3f} | "
            f"{row['G LPGF']:.3f} | {row['H LPGF']:.3f} | "
            f"{row['G patch']:.2f} | {row['H patch']:.2f} | "
            f"{row['G welfare']:.2f} | {row['H welfare']:.2f} | "
            f"{row['G burden']:.2f} | {row['H burden']:.2f} | "
            f"{row['G missed']:.3f} | {row['H missed']:.3f} | "
            f"{row['G delayed']:.2f} | {row['H delayed']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Overseer-limit ablation for the selected scenario and generation treatment. G denotes the global-signal package and H the hybrid package.}",
        "    \\label{tab:reduced-overseer-ablation}",
        "    \\resizebox{\\linewidth}{!}{%",
        "    \\begin{tabular}{lrrrrrrrrrrrrrr}",
        "        \\toprule",
        "        Overseer limit & G unsafe & H unsafe & G LPGF & H LPGF & G patch & H patch & G welfare & H welfare & G burden & H burden & G missed & H missed & G delayed & H delayed \\\\",
        "        \\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"        {row['Overseer limit']} & "
            f"{row['G unsafe']:.3f} & {row['H unsafe']:.3f} & "
            f"{row['G LPGF']:.3f} & {row['H LPGF']:.3f} & "
            f"{row['G patch']:.2f} & {row['H patch']:.2f} & "
            f"{row['G welfare']:.2f} & {row['H welfare']:.2f} & "
            f"{row['G burden']:.2f} & {row['H burden']:.2f} & "
            f"{row['G missed']:.3f} & {row['H missed']:.3f} & "
            f"{row['G delayed']:.2f} & {row['H delayed']:.2f} \\\\"
        )
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "    }",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


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
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def _write_figure(df: pd.DataFrame, output_prefix: Path) -> None:
    _setup_style()
    panels = [
        ("test_global_unsafe_rate_mean_mean", "test_global_unsafe_rate_mean_sem", "Unsafe rate", "(a) Global unsafe"),
        (
            "test_local_pass_global_fail_rate_mean_mean",
            "test_local_pass_global_fail_rate_mean_sem",
            "LPGF rate",
            "(b) Local-pass/global-fail",
        ),
        ("test_mean_patch_health_mean_mean", "test_mean_patch_health_mean_sem", "Patch health", "(c) Resource health"),
        (
            "test_governance_budget_spent_mean_mean",
            "test_governance_budget_spent_mean_sem",
            "Burden",
            "(d) Oversight burden",
        ),
    ]
    x = range(len(LEVEL_ORDER))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.3), sharex=True)
    axes = axes.ravel()
    for ax, (metric, sem_col, ylabel, title) in zip(axes, panels):
        for condition in ["top_down_only", "hybrid"]:
            sub = (
                df[df["condition"] == condition]
                .set_index("overseer_capability_level")
                .reindex(LEVEL_ORDER)
                .reset_index()
            )
            y = sub[metric]
            yerr = 1.96 * sub[sem_col].fillna(0.0) if sem_col in sub else None
            ax.errorbar(
                list(x),
                y,
                yerr=yerr,
                marker="o" if condition == "top_down_only" else "s",
                markersize=3.0,
                linewidth=1.1,
                elinewidth=0.6,
                capsize=1.8,
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition],
            )
        ax.set_title(title, loc="left", fontweight="bold", pad=4)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, alpha=0.42, linewidth=0.45)
    labels = [LEVEL_LABELS[level].replace(" limit", "").replace("Bundled ", "B. ") for level in LEVEL_ORDER]
    for ax in axes:
        ax.set_xticks(list(x), labels, rotation=28, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for suffix in [".pdf", ".svg", ".png"]:
        fig.savefig(output_prefix.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = _load_summary(args.summary_csv)
    for column, selection in [("scenario_preset", args.scenario), ("actor_capability_level", args.actor)]:
        if selection is not None:
            if column not in df:
                raise ValueError(f"Cannot select {column}: absent from summary")
            df = df[df[column].eq(selection)]
        if column in df and df[column].nunique() > 1:
            raise ValueError(f"Select one {column} for this slice analyzer; refusing to pool different treatments")
    if df.empty:
        raise ValueError("No matching ablation rows")
    table = _compact_table(df)
    mechanism_text = _mechanism_summary(df)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_prefix.with_name(output_prefix.name + "_compact.csv"), index=False)
    _write_markdown(table, mechanism_text, output_prefix.with_name(output_prefix.name + "_summary.md"))
    _write_latex(table, output_prefix.with_name(output_prefix.name + "_table.tex"))
    _write_figure(df, output_prefix)
    print(output_prefix.with_name(output_prefix.name + "_compact.csv"))
    print(output_prefix.with_name(output_prefix.name + "_summary.md"))
    print(output_prefix.with_name(output_prefix.name + "_table.tex"))
    print(output_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
