"""Paired, context-clustered long-run contrasts for the reviewer confirmation run."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("total_welfare", "mean_patch_health", "unsafe_fixed_horizon")
COMPARATORS = ("local_bounded", "local_optimistic")


def analyze(episodes: Path, output_dir: Path, *, draws: int = 4000, seed: int = 20260924) -> pd.DataFrame:
    frame = pd.read_csv(episodes)
    selected = frame.loc[
        (frame["inspection_budget"] == 6)
        & frame["mode"].isin(("joint", *COMPARATORS))
    ].copy()
    keys = ["game", "policy_group", "regime", "context", "weather", "mode"]
    if selected.duplicated(keys).any():
        raise ValueError("Duplicate episode within a paired context and weather seed")
    paired = selected.pivot(index=keys[:-1], columns="mode", values=list(METRICS))
    if paired.isna().any().any():
        raise ValueError("Missing paired mode or outcome")

    context_keys = keys[:-2]
    context_deltas = []
    for game, game_frame in paired.groupby(level="game"):
        contexts = game_frame.index.droplevel("weather").unique()
        if len(contexts) != 64:
            raise ValueError(f"Expected 64 independent {game} contexts; found {len(contexts)}")
        for comparator in COMPARATORS:
            for metric in METRICS:
                differences = game_frame[(metric, "joint")] - game_frame[(metric, comparator)]
                by_context = differences.groupby(level=context_keys).mean()
                for key, difference in by_context.items():
                    context_deltas.append({
                        "game": game, "policy_group": key[1], "regime": key[2],
                        "context": key[3], "comparison": f"joint_minus_{comparator}",
                        "metric": metric, "difference": float(difference),
                    })

    details = pd.DataFrame(context_deltas).sort_values(
        ["game", "comparison", "metric", "context"]
    ).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows = []
    for (game, comparison, metric), group in details.groupby(["game", "comparison", "metric"]):
        values = group["difference"].to_numpy()
        boot = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
        rows.append({
            "game": game, "comparison": comparison, "metric": metric,
            "independent_contexts": len(values), "mean_difference": values.mean(),
            "ci_95_low": np.quantile(boot, 0.025),
            "ci_95_high": np.quantile(boot, 0.975),
            "bootstrap_draws": draws, "bootstrap_seed": seed,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "longrun_context_differences.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "longrun_paired_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(analyze(args.episodes, args.output_dir).to_string(index=False))
