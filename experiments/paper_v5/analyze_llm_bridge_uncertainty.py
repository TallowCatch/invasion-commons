from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "mean_patch_health",
    "garden_failure_rate",
    "mean_welfare",
    "mean_governance_budget_spent",
    "mean_neighborhood_overharvest",
    "mean_exploitative_action_share",
]


CONDITION_LABELS = {
    "none": "No oversight",
    "bottom_up_only": "Local oversight",
    "top_down_only": "Global signal",
    "hybrid": "Hybrid oversight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute uncertainty summaries for existing Harvest LLM bridge map samples."
    )
    parser.add_argument(
        "--input-glob",
        default="results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_*_map_samples.csv",
    )
    parser.add_argument(
        "--constrained-input-glob",
        default="results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_*_constrained_map_samples.csv",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_uncertainty",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def _read_samples(patterns: list[str]) -> pd.DataFrame:
    frames = []
    for pattern in patterns:
        for path in sorted(Path().glob(pattern)):
            df = pd.read_csv(path)
            df["source_file"] = str(path)
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No LLM map sample files matched: {patterns}")
    return pd.concat(frames, ignore_index=True)


def _bootstrap_ci(values: np.ndarray, *, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1 or n_boot <= 0:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _summarise(df: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group_cols = [
        "bank_model_label",
        "governance_friction_regime",
        "scenario_preset",
        "condition",
        "exploitative_share",
    ]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["condition_label"] = CONDITION_LABELS.get(str(row["condition"]), str(row["condition"]))
        row["n_populations"] = int(group["population_id"].nunique()) if "population_id" in group else int(len(group))
        for metric in METRICS:
            if metric not in group:
                continue
            values = group[metric].astype(float).to_numpy()
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            sem = float(std / np.sqrt(np.isfinite(values).sum())) if np.isfinite(values).sum() > 1 else 0.0
            ci_low, ci_high = _bootstrap_ci(values, rng=rng, n_boot=n_boot)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_sem"] = sem
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def _condition_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    contrast_rows = []
    index_cols = ["bank_model_label", "governance_friction_regime", "scenario_preset", "exploitative_share"]
    for keys, group in summary.groupby(index_cols, dropna=False):
        by_condition = group.set_index("condition")
        for left, right in [
            ("hybrid", "bottom_up_only"),
            ("top_down_only", "bottom_up_only"),
            ("hybrid", "top_down_only"),
            ("bottom_up_only", "none"),
            ("top_down_only", "none"),
            ("hybrid", "none"),
        ]:
            if left not in by_condition.index or right not in by_condition.index:
                continue
            row = dict(zip(index_cols, keys))
            row["contrast"] = f"{left}_minus_{right}"
            for metric in METRICS:
                col = f"{metric}_mean"
                if col in by_condition:
                    row[f"delta_{metric}"] = float(by_condition.loc[left, col] - by_condition.loc[right, col])
            contrast_rows.append(row)
    return pd.DataFrame(contrast_rows)


def _write_note(summary: pd.DataFrame, contrasts: pd.DataFrame, path: Path) -> None:
    lines = [
        "# LLM Bridge Uncertainty",
        "",
        "This analysis uses existing sampled-population rows from the LLM bridge governance maps. It does not regenerate strategies and does not run live LLM agents.",
        "",
        f"Rows summarized: {len(summary)} condition cells.",
        "",
        "## Main caution",
        "",
        "Uncertainty is over sampled populations from fixed strategy banks. It is not uncertainty over model families, prompt templates, or live model behavior.",
        "",
        "## Large-effect checks",
        "",
    ]
    if contrasts.empty:
        lines.append("No pairwise condition contrasts were available.")
    else:
        subset = contrasts[
            contrasts["contrast"].isin(
                ["hybrid_minus_bottom_up_only", "top_down_only_minus_bottom_up_only", "hybrid_minus_none"]
            )
        ].copy()
        if subset.empty:
            subset = contrasts
        lines.append("| Model | Regime | Scenario | Exploitative share | Contrast | Patch health delta | Failure delta | Welfare delta |")
        lines.append("| --- | --- | --- | ---: | --- | ---: | ---: | ---: |")
        for _, row in subset.iterrows():
            lines.append(
                f"| {row['bank_model_label']} | {row['governance_friction_regime']} | {row['scenario_preset']} | "
                f"{row['exploitative_share']:.2f} | {row['contrast']} | "
                f"{row.get('delta_mean_patch_health', float('nan')):.2f} | "
                f"{row.get('delta_garden_failure_rate', float('nan')):.3f} | "
                f"{row.get('delta_mean_welfare', float('nan')):.2f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    samples = _read_samples([args.input_glob, args.constrained_input_glob])
    summary = _summarise(samples, n_boot=args.bootstrap_samples, seed=args.seed)
    contrasts = _condition_contrasts(summary)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
    contrasts.to_csv(prefix.with_name(prefix.name + "_contrasts.csv"), index=False)
    _write_note(summary, contrasts, prefix.with_name(prefix.name + ".md"))
    print(f"Saved: {prefix.with_name(prefix.name + '_summary.csv')}")
    print(f"Saved: {prefix.with_name(prefix.name + '_contrasts.csv')}")
    print(f"Saved: {prefix.with_name(prefix.name + '.md')}")


if __name__ == "__main__":
    main()
