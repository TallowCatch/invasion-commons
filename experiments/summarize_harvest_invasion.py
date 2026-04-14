import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SUMMARY_METRICS = [
    "train_garden_failure_mean",
    "test_garden_failure_mean",
    "train_garden_failure_last",
    "test_garden_failure_last",
    "test_mean_patch_health_mean",
    "test_mean_welfare_mean",
    "test_mean_credit_transferred_mean",
    "test_mean_aggressive_request_fraction_mean",
    "test_mean_max_local_aggression_mean",
    "test_mean_neighborhood_overharvest_mean",
    "test_mean_capped_action_fraction_mean",
    "test_mean_targeted_agent_fraction_mean",
    "test_missed_target_rate_mean",
    "test_targeted_share_mean",
    "test_delayed_intervention_count_mean",
    "test_governance_budget_spent_mean",
    "test_mean_prevented_harvest_mean",
    "test_mean_patch_variance_mean",
    "test_mean_requested_harvest_mean",
    "test_mean_realized_harvest_mean",
    "time_to_garden_failure",
    "first_generation_test_failure_ge_0_8",
    "per_regime_health_survival_over_generations_mean",
    "population_diversity_mean",
    "llm_json_fraction",
    "llm_fallback_fraction",
    "direct_json_fraction",
    "repaired_json_fraction",
    "effective_llm_fraction",
    "unrepaired_fallback_fraction",
]

PAIR_METRICS = [
    "test_garden_failure_mean",
    "test_mean_patch_health_mean",
    "test_mean_welfare_mean",
    "test_mean_credit_transferred_mean",
    "test_mean_max_local_aggression_mean",
    "test_mean_neighborhood_overharvest_mean",
    "test_mean_capped_action_fraction_mean",
    "test_mean_targeted_agent_fraction_mean",
    "test_missed_target_rate_mean",
    "test_targeted_share_mean",
    "test_delayed_intervention_count_mean",
    "test_governance_budget_spent_mean",
    "test_mean_prevented_harvest_mean",
]

AGGRESSION_INCIDENT_METRICS = [
    "mean_welfare",
    "mean_prevented_harvest",
    "mean_realized_harvest",
    "targeted_step_fraction",
]

TARGETING_INCIDENT_METRICS = [
    "mean_welfare",
    "mean_prevented_harvest",
    "mean_realized_harvest",
    "mean_local_patch_health",
]

GOVERNANCE_ORDER = {
    "none": 0,
    "bottom_up_only": 1,
    "top_down_only": 2,
    "hybrid": 3,
}

INJECTOR_ORDER = {
    "random": 0,
    "mutation": 1,
    "adversarial_heuristic": 2,
    "search_mutation": 3,
}

OPTIONAL_CONTEXT_COLS = [
    "scenario_preset",
    "governance_friction_regime",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Harvest invasion matrix outputs.")
    parser.add_argument("--runs-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--agent-history-csv", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def _ci95(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    if arr.size == 1:
        return (mean, mean, mean)
    se = float(arr.std(ddof=1) / np.sqrt(arr.size))
    margin = 1.96 * se
    return (mean, mean - margin, mean + margin)


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def _context_group_cols(df: pd.DataFrame, base_cols: list[str]) -> list[str]:
    return base_cols + [col for col in OPTIONAL_CONTEXT_COLS if col in df.columns]


def _aggregate_with_ci(per_run_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = _context_group_cols(
        per_run_df,
        ["tier", "partner_mix", "injector_mode_requested", "adversarial_pressure", "condition"],
    )
    rows = []
    regime_cols = sorted([c for c in per_run_df.columns if c.startswith("per_regime_health_survival_over_generations__")])
    parse_error_cols = sorted([c for c in per_run_df.columns if c.startswith("llm_parse_error_count__")])
    metric_keys = [key for key in SUMMARY_METRICS if key in per_run_df.columns] + regime_cols + parse_error_cols
    for keys, gdf in per_run_df.groupby(group_cols, sort=True):
        row = {
            "tier": keys[group_cols.index("tier")],
            "partner_mix": keys[group_cols.index("partner_mix")],
            "injector_mode_requested": keys[group_cols.index("injector_mode_requested")],
            "adversarial_pressure": keys[group_cols.index("adversarial_pressure")],
            "condition": keys[group_cols.index("condition")],
            "n_runs": int(len(gdf)),
            "llm_provider": str(gdf["llm_provider"].iloc[0]) if "llm_provider" in gdf.columns else "none",
            "llm_model": str(gdf["llm_model"].iloc[0]) if "llm_model" in gdf.columns else "",
        }
        for col in OPTIONAL_CONTEXT_COLS:
            if col in gdf.columns:
                row[col] = str(gdf[col].iloc[0])
        for key in metric_keys:
            mean, lo, hi = _ci95(gdf[key].tolist())
            row[f"{key}_mean"] = round(mean, 6)
            row[f"{key}_ci95_low"] = round(lo, 6)
            row[f"{key}_ci95_high"] = round(hi, 6)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            [col for col in OPTIONAL_CONTEXT_COLS if col in out.columns]
            + [
                "tier",
                "partner_mix",
                "injector_mode_requested",
                "adversarial_pressure",
                "test_garden_failure_mean_mean",
                "test_mean_patch_health_mean_mean",
                "test_mean_welfare_mean_mean",
            ],
            ascending=[True] * len([col for col in OPTIONAL_CONTEXT_COLS if col in out.columns]) + [True, True, True, True, True, False, False],
        ).reset_index(drop=True)
    return out


def _build_integrity_df(per_run_df: pd.DataFrame) -> pd.DataFrame:
    if "llm_json_fraction" not in per_run_df.columns:
        return pd.DataFrame()
    llm_df = per_run_df[per_run_df["injector_mode_requested"] == "llm_json"].copy()
    if llm_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["tier", "partner_mix", "adversarial_pressure", "condition"]
    parse_error_cols = sorted([c for c in llm_df.columns if c.startswith("llm_parse_error_count__")])

    def _mean_or_zero(gdf: pd.DataFrame, column: str) -> float:
        if column not in gdf.columns:
            return 0.0
        return float(gdf[column].mean())

    for keys, gdf in llm_df.groupby(group_cols, sort=True):
        effective_llm_fraction_mean = round(
            _mean_or_zero(gdf, "effective_llm_fraction")
            if "effective_llm_fraction" in gdf.columns
            else float(gdf["llm_json_fraction"].mean()),
            6,
        )
        unrepaired_fallback_fraction_mean = round(
            _mean_or_zero(gdf, "unrepaired_fallback_fraction")
            if "unrepaired_fallback_fraction" in gdf.columns
            else float(gdf["llm_fallback_fraction"].mean()),
            6,
        )
        row = {
            "tier": keys[0],
            "partner_mix": keys[1],
            "adversarial_pressure": keys[2],
            "condition": keys[3],
            "n_runs": int(len(gdf)),
            "llm_provider": str(gdf["llm_provider"].iloc[0]) if "llm_provider" in gdf.columns else "none",
            "llm_model": str(gdf["llm_model"].iloc[0]) if "llm_model" in gdf.columns else "",
            "llm_json_fraction_mean": round(float(gdf["llm_json_fraction"].mean()), 6),
            "llm_fallback_fraction_mean": round(float(gdf["llm_fallback_fraction"].mean()), 6),
            "direct_json_fraction_mean": round(_mean_or_zero(gdf, "direct_json_fraction"), 6),
            "repaired_json_fraction_mean": round(_mean_or_zero(gdf, "repaired_json_fraction"), 6),
            "effective_llm_fraction_mean": effective_llm_fraction_mean,
            "unrepaired_fallback_fraction_mean": unrepaired_fallback_fraction_mean,
            "effective_llm_gate_pass": effective_llm_fraction_mean >= 0.90,
            "fallback_gate_pass": unrepaired_fallback_fraction_mean <= 0.05,
        }
        row["usable_for_evidence"] = bool(row["effective_llm_gate_pass"] and row["fallback_gate_pass"])
        for col in parse_error_cols:
            row[f"{col}_mean"] = round(float(gdf[col].mean()), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_ranking_table(table_df: pd.DataFrame) -> pd.DataFrame:
    if table_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = _context_group_cols(
        table_df,
        ["tier", "partner_mix", "injector_mode_requested", "adversarial_pressure"],
    )
    for keys, gdf in table_df.groupby(group_cols, sort=True):
        ranked = gdf.sort_values(
            ["test_garden_failure_mean_mean", "test_mean_patch_health_mean_mean", "test_mean_welfare_mean_mean"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            rows.append(
                {
                    "tier": keys[group_cols.index("tier")],
                    "partner_mix": keys[group_cols.index("partner_mix")],
                    "injector_mode_requested": keys[group_cols.index("injector_mode_requested")],
                    "adversarial_pressure": keys[group_cols.index("adversarial_pressure")],
                    "condition": row["condition"],
                    "rank": rank,
                    "test_garden_failure_mean_mean": row["test_garden_failure_mean_mean"],
                    "test_mean_patch_health_mean_mean": row["test_mean_patch_health_mean_mean"],
                    "test_mean_welfare_mean_mean": row["test_mean_welfare_mean_mean"],
                }
            )
            for col in OPTIONAL_CONTEXT_COLS:
                if col in row.index:
                    rows[-1][col] = row[col]
    return pd.DataFrame(rows)


def _contrast_pairs(conditions: list[str]) -> list[tuple[str, str]]:
    preferred = [
        ("hybrid", "top_down_only"),
        ("hybrid", "bottom_up_only"),
        ("top_down_only", "bottom_up_only"),
        ("hybrid", "none"),
        ("top_down_only", "none"),
        ("bottom_up_only", "none"),
    ]
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    condition_set = set(conditions)
    for pair in preferred:
        if pair[0] in condition_set and pair[1] in condition_set:
            out.append(pair)
            seen.add(pair)
    ordered = sorted(conditions, key=lambda x: (GOVERNANCE_ORDER.get(x, 999), x))
    for i in range(len(ordered) - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            pair = (ordered[i], ordered[j])
            if pair not in seen:
                out.append(pair)
                seen.add(pair)
    return out


def _build_condition_delta_df(per_run_df: pd.DataFrame, pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    if per_run_df.empty:
        return pd.DataFrame()
    join_cols = _context_group_cols(
        per_run_df,
        ["tier", "partner_mix", "injector_mode_requested", "adversarial_pressure", "run_id"],
    )
    conditions = sorted(per_run_df["condition"].dropna().astype(str).unique().tolist(), key=lambda x: (GOVERNANCE_ORDER.get(x, 999), x))
    rows = []
    for left_condition, right_condition in (pairs or _contrast_pairs(conditions)):
        left_df = per_run_df[per_run_df["condition"] == left_condition].copy()
        right_df = per_run_df[per_run_df["condition"] == right_condition].copy()
        if left_df.empty or right_df.empty:
            continue
        merged = left_df.merge(right_df, on=join_cols, suffixes=("_left", "_right"))
        for _, row in merged.iterrows():
            out = {key: row[key] for key in join_cols}
            out["left_condition"] = left_condition
            out["right_condition"] = right_condition
            out["contrast_name"] = f"{left_condition}_minus_{right_condition}"
            for metric in PAIR_METRICS:
                out[f"delta__{metric}"] = float(row[f"{metric}_left"] - row[f"{metric}_right"])
            rows.append(out)
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_samples: int) -> tuple[float, float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(values.mean())
    if values.size == 1:
        return (point, point, point)
    samples = []
    for _ in range(n_samples):
        idx = rng.integers(0, values.size, size=values.size)
        samples.append(float(values[idx].mean()))
    lo, hi = np.quantile(np.asarray(samples, dtype=float), [0.025, 0.975])
    return (point, float(lo), float(hi))


def _build_condition_ci_df(delta_df: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    group_cols = _context_group_cols(
        delta_df,
        [
            "tier",
            "partner_mix",
            "injector_mode_requested",
            "adversarial_pressure",
            "left_condition",
            "right_condition",
            "contrast_name",
        ],
    )
    metric_cols = [c for c in delta_df.columns if c.startswith("delta__")]
    for keys, gdf in delta_df.groupby(group_cols, sort=True):
        row = {
            "tier": keys[group_cols.index("tier")],
            "partner_mix": keys[group_cols.index("partner_mix")],
            "injector_mode_requested": keys[group_cols.index("injector_mode_requested")],
            "adversarial_pressure": keys[group_cols.index("adversarial_pressure")],
            "left_condition": keys[group_cols.index("left_condition")],
            "right_condition": keys[group_cols.index("right_condition")],
            "contrast_name": keys[group_cols.index("contrast_name")],
            "n_pairs": int(len(gdf)),
        }
        for col in OPTIONAL_CONTEXT_COLS:
            if col in gdf.columns:
                row[col] = str(gdf[col].iloc[0])
        for metric in metric_cols:
            mean, lo, hi = _bootstrap_ci(gdf[metric].to_numpy(dtype=float), rng=rng, n_samples=n_samples)
            row[f"{metric}_mean"] = round(mean, 6)
            row[f"{metric}_ci95_low"] = round(lo, 6)
            row[f"{metric}_ci95_high"] = round(hi, 6)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_paired_delta_df(per_run_df: pd.DataFrame) -> pd.DataFrame:
    generic = _build_condition_delta_df(per_run_df, pairs=[("hybrid", "top_down_only")])
    if generic.empty:
        return pd.DataFrame()
    base_cols = ["tier", "partner_mix", "injector_mode_requested", "adversarial_pressure", "run_id"]
    out = generic[_context_group_cols(generic, base_cols)].copy()
    for metric in PAIR_METRICS:
        out[f"hybrid_minus_top__{metric}"] = generic[f"delta__{metric}"]
    return out


def _build_paired_ci_df(delta_df: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    group_cols = _context_group_cols(
        delta_df,
        ["tier", "partner_mix", "injector_mode_requested", "adversarial_pressure"],
    )
    for keys, gdf in delta_df.groupby(group_cols, sort=True):
        row = {
            "tier": keys[group_cols.index("tier")],
            "partner_mix": keys[group_cols.index("partner_mix")],
            "injector_mode_requested": keys[group_cols.index("injector_mode_requested")],
            "adversarial_pressure": keys[group_cols.index("adversarial_pressure")],
            "n_pairs": int(len(gdf)),
        }
        for col in OPTIONAL_CONTEXT_COLS:
            if col in gdf.columns:
                row[col] = str(gdf[col].iloc[0])
        for metric in [c for c in delta_df.columns if c.startswith("hybrid_minus_top__")]:
            mean, lo, hi = _bootstrap_ci(gdf[metric].to_numpy(dtype=float), rng=rng, n_samples=n_samples)
            row[f"{metric}_mean"] = round(mean, 6)
            row[f"{metric}_ci95_low"] = round(lo, 6)
            row[f"{metric}_ci95_high"] = round(hi, 6)
        rows.append(row)
    return pd.DataFrame(rows)


def _infer_agent_history_path(runs_csv: str) -> str:
    path = Path(runs_csv)
    if not path.name.endswith("_runs.csv"):
        return ""
    return str(path.with_name(path.name.replace("_runs.csv", "_agent_history.csv")))


def _load_agent_history(agent_history_csv: str | None) -> pd.DataFrame:
    path = agent_history_csv or ""
    if not path:
        return pd.DataFrame()
    path_obj = Path(path)
    if not path_obj.exists():
        return pd.DataFrame()
    return pd.read_csv(path_obj)


def _assign_aggression_groups(agent_df: pd.DataFrame) -> pd.DataFrame:
    if agent_df.empty:
        return pd.DataFrame()
    working = agent_df.copy()
    working["aggression_group"] = ""
    group_cols = _context_group_cols(agent_df, [
        "tier",
        "partner_mix",
        "injector_mode_requested",
        "adversarial_pressure",
        "condition",
        "run_id",
        "generation",
        "phase",
        "regime",
        "seed",
    ])
    value_col = (
        "aggressive_request_fraction"
        if "aggressive_request_fraction" in working.columns and not working["aggressive_request_fraction"].isna().all()
        else "mean_requested_harvest"
    )
    labels = ["low_aggression", "mid_aggression", "high_aggression"]
    for _, idx in working.groupby(group_cols, sort=False).groups.items():
        ranks = working.loc[idx, value_col].rank(method="first")
        bins = pd.qcut(ranks, q=3, labels=labels)
        working.loc[idx, "aggression_group"] = bins.astype(str)
    return working


def _build_agent_group_means(agent_df: pd.DataFrame, label_col: str, metric_cols: list[str]) -> pd.DataFrame:
    if agent_df.empty:
        return pd.DataFrame()
    group_cols = _context_group_cols(agent_df, [
        "tier",
        "partner_mix",
        "injector_mode_requested",
        "adversarial_pressure",
        "condition",
        "run_id",
        "generation",
        "phase",
        "regime",
        "seed",
        label_col,
    ])
    aggregations = {col: "mean" for col in metric_cols if col in agent_df.columns}
    return agent_df.groupby(group_cols, sort=True).agg(aggregations).reset_index()


def _build_agent_group_deltas(group_means_df: pd.DataFrame, label_col: str, metric_cols: list[str]) -> pd.DataFrame:
    if group_means_df.empty:
        return pd.DataFrame()
    join_cols = _context_group_cols(group_means_df, [
        "tier",
        "partner_mix",
        "injector_mode_requested",
        "adversarial_pressure",
        "run_id",
        "generation",
        "phase",
        "regime",
        "seed",
        label_col,
    ])
    conditions = sorted(group_means_df["condition"].dropna().astype(str).unique().tolist(), key=lambda x: (GOVERNANCE_ORDER.get(x, 999), x))
    rows = []
    for left_condition, right_condition in _contrast_pairs(conditions):
        left_df = group_means_df[group_means_df["condition"] == left_condition].copy()
        right_df = group_means_df[group_means_df["condition"] == right_condition].copy()
        if left_df.empty or right_df.empty:
            continue
        merged = left_df.merge(right_df, on=join_cols, suffixes=("_left", "_right"))
        for _, row in merged.iterrows():
            out = {key: row[key] for key in join_cols}
            out["left_condition"] = left_condition
            out["right_condition"] = right_condition
            out["contrast_name"] = f"{left_condition}_minus_{right_condition}"
            for metric in metric_cols:
                if f"{metric}_left" not in row.index or f"{metric}_right" not in row.index:
                    continue
                out[f"delta__{metric}"] = float(row[f"{metric}_left"] - row[f"{metric}_right"])
            rows.append(out)
    return pd.DataFrame(rows)


def _aggregate_agent_group_deltas(delta_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    metric_cols = [c for c in delta_df.columns if c.startswith("delta__")]
    rows = []
    group_cols = _context_group_cols(delta_df, [
        "tier",
        "partner_mix",
        "injector_mode_requested",
        "adversarial_pressure",
        "left_condition",
        "right_condition",
        "contrast_name",
        label_col,
    ])
    for keys, gdf in delta_df.groupby(group_cols, sort=True):
        row = {
            "tier": keys[group_cols.index("tier")],
            "partner_mix": keys[group_cols.index("partner_mix")],
            "injector_mode_requested": keys[group_cols.index("injector_mode_requested")],
            "adversarial_pressure": keys[group_cols.index("adversarial_pressure")],
            "left_condition": keys[group_cols.index("left_condition")],
            "right_condition": keys[group_cols.index("right_condition")],
            "contrast_name": keys[group_cols.index("contrast_name")],
            label_col: keys[group_cols.index(label_col)],
            "n_episode_pairs": int(len(gdf)),
        }
        for col in OPTIONAL_CONTEXT_COLS:
            if col in gdf.columns:
                row[col] = str(gdf[col].iloc[0])
        for metric in metric_cols:
            mean, lo, hi = _ci95(gdf[metric].tolist())
            row[f"{metric}_mean"] = round(mean, 6)
            row[f"{metric}_ci95_low"] = round(lo, 6)
            row[f"{metric}_ci95_high"] = round(hi, 6)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_welfare_incidence_outputs(agent_history_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if agent_history_df.empty:
        return {}
    eval_df = agent_history_df[agent_history_df["phase"] == "test"].copy() if "phase" in agent_history_df.columns else agent_history_df.copy()
    if eval_df.empty:
        return {}

    aggression_df = _assign_aggression_groups(eval_df)
    aggression_means = _build_agent_group_means(aggression_df, "aggression_group", AGGRESSION_INCIDENT_METRICS)
    aggression_deltas = _build_agent_group_deltas(aggression_means, "aggression_group", AGGRESSION_INCIDENT_METRICS)
    aggression_summary = _aggregate_agent_group_deltas(aggression_deltas, "aggression_group")

    targeting_df = eval_df.copy()
    targeted_series = (
        targeting_df["targeted_step_fraction"].fillna(0.0)
        if "targeted_step_fraction" in targeting_df.columns
        else pd.Series(0.0, index=targeting_df.index)
    )
    targeting_df["target_group"] = np.where(targeted_series > 0.0, "targeted", "untargeted")
    targeting_means = _build_agent_group_means(targeting_df, "target_group", TARGETING_INCIDENT_METRICS)
    targeting_deltas = _build_agent_group_deltas(targeting_means, "target_group", TARGETING_INCIDENT_METRICS)
    targeting_summary = _aggregate_agent_group_deltas(targeting_deltas, "target_group")

    return {
        "aggression_groups": aggression_means,
        "aggression_deltas": aggression_deltas,
        "aggression_summary": aggression_summary,
        "targeting_groups": targeting_means,
        "targeting_deltas": targeting_deltas,
        "targeting_summary": targeting_summary,
    }


def _build_capability_ladder_df(contrast_ci_df: pd.DataFrame) -> pd.DataFrame:
    if contrast_ci_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = _context_group_cols(
        contrast_ci_df,
        ["tier", "partner_mix", "adversarial_pressure", "left_condition", "right_condition", "contrast_name"],
    )
    for keys, gdf in contrast_ci_df.groupby(group_cols, sort=True):
        ordered = gdf.copy()
        ordered["injector_rank"] = ordered["injector_mode_requested"].map(lambda x: INJECTOR_ORDER.get(x, 999))
        ordered = ordered.sort_values("injector_rank").reset_index(drop=True)

        def _first_break(metric_col: str, predicate) -> str:
            for _, row in ordered.iterrows():
                if metric_col in row.index and predicate(float(row[metric_col])):
                    return str(row["injector_mode_requested"])
            return "none"

        rows.append(
            {
                "tier": keys[group_cols.index("tier")],
                "partner_mix": keys[group_cols.index("partner_mix")],
                "adversarial_pressure": keys[group_cols.index("adversarial_pressure")],
                "left_condition": keys[group_cols.index("left_condition")],
                "right_condition": keys[group_cols.index("right_condition")],
                "contrast_name": keys[group_cols.index("contrast_name")],
                "first_ecological_break_injector": _first_break(
                    "delta__test_mean_patch_health_mean_mean",
                    lambda x: x <= 0.0,
                ),
                "first_control_break_injector": _first_break(
                    "delta__test_mean_neighborhood_overharvest_mean_mean",
                    lambda x: x > 0.0,
                ),
                "first_costly_robustness_injector": _first_break(
                    "delta__test_mean_welfare_mean_mean",
                    lambda x: x < -0.15,
                ),
            }
        )
        for col in OPTIONAL_CONTEXT_COLS:
            if col in gdf.columns:
                rows[-1][col] = str(gdf[col].iloc[0])
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    per_run_df = pd.read_csv(args.runs_csv)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    agent_history_csv = args.agent_history_csv or _infer_agent_history_path(args.runs_csv)
    agent_history_df = _load_agent_history(agent_history_csv)

    table_df = _aggregate_with_ci(per_run_df)
    ranking_df = _build_ranking_table(table_df)
    delta_df = _build_paired_delta_df(per_run_df)
    ci_df = _build_paired_ci_df(delta_df, n_samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    integrity_df = _build_integrity_df(per_run_df)
    contrast_delta_df = _build_condition_delta_df(per_run_df)
    contrast_ci_df = _build_condition_ci_df(contrast_delta_df, n_samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    capability_ladder_df = _build_capability_ladder_df(contrast_ci_df)
    incidence_outputs = _build_welfare_incidence_outputs(agent_history_df)

    best_counts = pd.Series(dtype=int)
    if not ranking_df.empty:
        best_counts = ranking_df[ranking_df["rank"] == 1]["condition"].value_counts().sort_index()

    lines = [
        "# Harvest invasion summary",
        "",
        "## Best condition frequency",
        "",
    ]
    for condition, count in best_counts.items():
        lines.append(f"- `{condition}` wins {int(count)} cell(s).")

    if not delta_df.empty:
        lines.extend(
            [
                "",
                "## Mean paired hybrid minus top-down deltas",
                "",
                f"- Mean failure delta: `{delta_df['hybrid_minus_top__test_garden_failure_mean'].mean():.4f}`",
                f"- Mean patch-health delta: `{delta_df['hybrid_minus_top__test_mean_patch_health_mean'].mean():.4f}`",
                f"- Mean welfare delta: `{delta_df['hybrid_minus_top__test_mean_welfare_mean'].mean():.4f}`",
                f"- Mean local-aggression delta: `{delta_df['hybrid_minus_top__test_mean_max_local_aggression_mean'].mean():.4f}`",
                f"- Mean neighborhood-overharvest delta: `{delta_df['hybrid_minus_top__test_mean_neighborhood_overharvest_mean'].mean():.4f}`",
            ]
        )
    if not contrast_ci_df.empty:
        lines.extend(["", "## Contrast preview", "", _markdown_table(contrast_ci_df.head(12))])
    if not integrity_df.empty:
        usable_count = int(integrity_df["usable_for_evidence"].sum())
        lines.extend(
            [
                "",
                "## LLM integrity",
                "",
                f"- Usable cells: `{usable_count} / {len(integrity_df)}`",
                f"- Mean effective LLM fraction: `{integrity_df['effective_llm_fraction_mean'].mean():.4f}`",
                f"- Mean unrepaired fallback fraction: `{integrity_df['unrepaired_fallback_fraction_mean'].mean():.4f}`",
                "",
                _markdown_table(integrity_df.head(24)),
            ]
        )
    if "aggression_summary" in incidence_outputs and not incidence_outputs["aggression_summary"].empty:
        lines.extend(["", "## Welfare incidence by aggression tercile", "", _markdown_table(incidence_outputs["aggression_summary"].head(12))])
    if "targeting_summary" in incidence_outputs and not incidence_outputs["targeting_summary"].empty:
        lines.extend(["", "## Welfare incidence by targeting", "", _markdown_table(incidence_outputs["targeting_summary"].head(12))])
    if not capability_ladder_df.empty:
        lines.extend(["", "## Capability ladder preview", "", _markdown_table(capability_ladder_df.head(12))])
    lines.extend(["", "## Table preview", "", _markdown_table(table_df.head(24)) if not table_df.empty else "No rows.", ""])

    table_csv = str(output_prefix.with_name(output_prefix.name + "_table.csv"))
    ranking_csv = str(output_prefix.with_name(output_prefix.name + "_ranking.csv"))
    delta_csv = str(output_prefix.with_name(output_prefix.name + "_delta.csv"))
    ci_csv = str(output_prefix.with_name(output_prefix.name + "_ci.csv"))
    integrity_csv = str(output_prefix.with_name(output_prefix.name + "_integrity.csv"))
    contrast_delta_csv = str(output_prefix.with_name(output_prefix.name + "_contrast_delta.csv"))
    contrast_ci_csv = str(output_prefix.with_name(output_prefix.name + "_contrast_ci.csv"))
    capability_ladder_csv = str(output_prefix.with_name(output_prefix.name + "_capability_ladder.csv"))
    summary_md = str(output_prefix.with_name(output_prefix.name + "_summary.md"))

    table_df.to_csv(table_csv, index=False)
    ranking_df.to_csv(ranking_csv, index=False)
    delta_df.to_csv(delta_csv, index=False)
    ci_df.to_csv(ci_csv, index=False)
    integrity_df.to_csv(integrity_csv, index=False)
    contrast_delta_df.to_csv(contrast_delta_csv, index=False)
    contrast_ci_df.to_csv(contrast_ci_csv, index=False)
    capability_ladder_df.to_csv(capability_ladder_csv, index=False)

    for key, df in incidence_outputs.items():
        df.to_csv(str(output_prefix.with_name(output_prefix.name + f"_{key}.csv")), index=False)

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print(f"Saved: {table_csv}")
    print(f"Saved: {ranking_csv}")
    print(f"Saved: {delta_csv}")
    print(f"Saved: {ci_csv}")
    print(f"Saved: {integrity_csv}")
    print(f"Saved: {contrast_delta_csv}")
    print(f"Saved: {contrast_ci_csv}")
    print(f"Saved: {capability_ladder_csv}")
    print(f"Saved: {summary_md}")


if __name__ == "__main__":
    main()
