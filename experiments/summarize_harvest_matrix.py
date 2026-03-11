import argparse

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Harvest Commons matrix outputs.")
    parser.add_argument("--table-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--delta-csv", default=None)
    parser.add_argument("--ranking-csv", default=None)
    return parser.parse_args()


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[h]) for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _best_condition(group: pd.DataFrame) -> pd.Series:
    ranked = group.sort_values(
        [
            "mean_patch_health_mean",
            "mean_welfare_mean",
            "mean_neighborhood_overharvest_mean",
            "payoff_gini_mean",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.table_csv)

    scenario_rows = []
    for (tier, social_mix), gdf in df.groupby(["tier", "social_mix"], sort=True):
        best = _best_condition(gdf)
        top = gdf[gdf["condition"] == "top_down_only"]
        hybrid = gdf[gdf["condition"] == "hybrid"]
        bottom = gdf[gdf["condition"] == "bottom_up_only"]
        none = gdf[gdf["condition"] == "none"]

        row = {
            "tier": tier,
            "social_mix": social_mix,
            "best_condition": str(best["condition"]),
            "best_mean_patch_health": float(best["mean_patch_health_mean"]),
            "best_mean_welfare": float(best["mean_welfare_mean"]),
            "best_mean_neighborhood_overharvest": float(best["mean_neighborhood_overharvest_mean"]),
        }
        if not top.empty and not hybrid.empty:
            row.update(
                {
                    "hybrid_minus_top_patch_health": float(
                        hybrid["mean_patch_health_mean"].iloc[0] - top["mean_patch_health_mean"].iloc[0]
                    ),
                    "hybrid_minus_top_welfare": float(
                        hybrid["mean_welfare_mean"].iloc[0] - top["mean_welfare_mean"].iloc[0]
                    ),
                    "hybrid_minus_top_local_aggression": float(
                        hybrid["mean_max_local_aggression_mean"].iloc[0]
                        - top["mean_max_local_aggression_mean"].iloc[0]
                    ),
                    "hybrid_minus_top_neighborhood_overharvest": float(
                        hybrid["mean_neighborhood_overharvest_mean"].iloc[0]
                        - top["mean_neighborhood_overharvest_mean"].iloc[0]
                    ),
                }
            )
        if not bottom.empty and not none.empty:
            row.update(
                {
                    "bottom_up_minus_none_patch_health": float(
                        bottom["mean_patch_health_mean"].iloc[0] - none["mean_patch_health_mean"].iloc[0]
                    ),
                    "bottom_up_minus_none_welfare": float(
                        bottom["mean_welfare_mean"].iloc[0] - none["mean_welfare_mean"].iloc[0]
                    ),
                }
            )
        row.setdefault("hybrid_minus_top_patch_health", 0.0)
        row.setdefault("hybrid_minus_top_welfare", 0.0)
        row.setdefault("hybrid_minus_top_local_aggression", 0.0)
        row.setdefault("hybrid_minus_top_neighborhood_overharvest", 0.0)
        row.setdefault("bottom_up_minus_none_patch_health", 0.0)
        row.setdefault("bottom_up_minus_none_welfare", 0.0)
        scenario_rows.append(row)

    scenario_df = pd.DataFrame(scenario_rows)
    ranking_df = df.sort_values(
        ["tier", "social_mix", "mean_patch_health_mean", "mean_welfare_mean"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    delta_df = scenario_df[
        [
            "tier",
            "social_mix",
            "hybrid_minus_top_patch_health",
            "hybrid_minus_top_welfare",
            "hybrid_minus_top_local_aggression",
            "hybrid_minus_top_neighborhood_overharvest",
        ]
    ].copy()

    best_counts = scenario_df["best_condition"].value_counts().sort_index()

    lines = [
        "# Harvest Commons matrix summary",
        "",
        "## Best condition frequency",
        "",
    ]
    for condition, count in best_counts.items():
        lines.append(f"- `{condition}` wins {int(count)} scenario(s).")

    lines.extend(
        [
            "",
            "## Mean hybrid minus top-down deltas",
            "",
            f"- Mean patch-health delta: `{delta_df['hybrid_minus_top_patch_health'].mean():.4f}`",
            f"- Mean welfare delta: `{delta_df['hybrid_minus_top_welfare'].mean():.4f}`",
            f"- Mean local-aggression delta: `{delta_df['hybrid_minus_top_local_aggression'].mean():.4f}`",
            f"- Mean neighborhood-overharvest delta: `{delta_df['hybrid_minus_top_neighborhood_overharvest'].mean():.4f}`",
            "",
            "## Scenario table",
            "",
            _markdown_table(scenario_df),
            "",
        ]
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    if args.delta_csv:
        delta_df.to_csv(args.delta_csv, index=False)
    if args.ranking_csv:
        ranking_df.to_csv(args.ranking_csv, index=False)

    print(f"Saved: {args.output}")
    if args.delta_csv:
        print(f"Saved: {args.delta_csv}")
    if args.ranking_csv:
        print(f"Saved: {args.ranking_csv}")


if __name__ == "__main__":
    main()
