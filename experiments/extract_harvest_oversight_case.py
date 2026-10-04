from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.harvest_oversight_reporting import condition_label
from experiments.harvest_oversight_reporting import scenario_label
from experiments.run_harvest_invasion import _safe_git_hash
from fishery_sim.harvest import HarvestStrategySpec
from fishery_sim.harvest import run_harvest_episode
from fishery_sim.harvest_benchmarks import (
    get_harvest_governance_friction_regime,
    get_harvest_overseer_capability_preset,
    get_harvest_regime_pack,
    make_harvest_cfg_for_scenario,
    make_harvest_cfg_for_tier,
)
from fishery_sim.harvest_evolution import _apply_cfg_overrides
from fishery_sim.harvest_evolution import _make_condition_setup


_STRATEGY_FIELDS = [
    "low_patch_threshold",
    "high_patch_threshold",
    "low_harvest_frac",
    "mid_harvest_frac",
    "high_harvest_frac",
    "restraint_low",
    "restraint_high",
    "credit_request_low",
    "credit_request_high",
    "credit_offer_threshold",
    "credit_offer_amount",
    "neighbor_reciprocity_weight",
    "credit_response_weight",
    "cap_compliance_margin",
]

_DEFAULT_GOVERNMENT_PARAMS = {
    "trigger": 16.0,
    "strict_cap_frac": 0.18,
    "relaxed_cap_frac": 0.35,
    "soft_trigger": 18.0,
    "deterioration_threshold": 0.35,
    "activation_warmup": 3,
    "aggressive_request_threshold": 0.75,
    "aggressive_agent_fraction_trigger": 0.34,
    "local_neighborhood_trigger": 0.67,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a concrete Harvest oversight failure trace.")
    parser.add_argument("--runs-csv", required=True)
    parser.add_argument("--generation-history-csv", default=None)
    parser.add_argument("--strategy-history-csv", default=None)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-candidates", type=int, default=25)
    parser.add_argument("--cfg-seed-start", type=int, default=0)
    parser.add_argument("--run-seed-stride", type=int, default=1000)
    return parser.parse_args()


def _infer_history_path(runs_csv: str, suffix: str) -> Path:
    path = Path(runs_csv)
    stem = path.name
    if stem.endswith("_runs.csv"):
        return path.with_name(stem.removesuffix("_runs.csv") + suffix)
    return path.with_name(path.stem + suffix)


def _clean_text(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    if text.lower() == "nan":
        return default
    return text


def _clean_int(value: object, default: int) -> int:
    if value is None or pd.isna(value):
        return int(default)
    return int(float(value))


def _strategy_from_row(row: pd.Series) -> HarvestStrategySpec:
    missing = [field for field in _STRATEGY_FIELDS if field not in row.index]
    if missing:
        raise ValueError(f"Strategy row is missing fields: {missing}")
    return HarvestStrategySpec(
        strategy_id=str(row["strategy_id"]),
        origin=_clean_text(row.get("origin"), default="restored"),
        rationale=_clean_text(row.get("rationale")),
        llm_parse_status=_clean_text(row.get("llm_parse_status")),
        llm_parse_error_type=_clean_text(row.get("llm_parse_error_type")),
        **{field: float(row[field]) for field in _STRATEGY_FIELDS},
    )


def _candidate_generations(generation_df: pd.DataFrame, max_candidates: int) -> pd.DataFrame:
    if generation_df.empty or "test_local_pass_global_fail_rate" not in generation_df.columns:
        return pd.DataFrame()
    candidates = generation_df.copy()
    candidates["test_local_pass_global_fail_rate"] = pd.to_numeric(
        candidates["test_local_pass_global_fail_rate"],
        errors="coerce",
    ).fillna(0.0)
    if "test_global_unsafe_rate" in candidates.columns:
        candidates["test_global_unsafe_rate"] = pd.to_numeric(candidates["test_global_unsafe_rate"], errors="coerce").fillna(0.0)
    else:
        candidates["test_global_unsafe_rate"] = 0.0
    candidates = candidates[candidates["test_local_pass_global_fail_rate"] > 0.0].copy()
    if candidates.empty:
        return candidates
    return candidates.sort_values(
        ["test_local_pass_global_fail_rate", "test_global_unsafe_rate"],
        ascending=[False, False],
    ).head(max_candidates)


def _matching_strategy_rows(strategy_df: pd.DataFrame, candidate: pd.Series, *, restore_order: bool = True) -> pd.DataFrame:
    filters = [
        ("run_id", candidate.get("run_id")),
        ("generation", candidate.get("generation")),
        ("condition", candidate.get("condition")),
        ("tier", candidate.get("tier")),
        ("partner_mix", candidate.get("partner_mix")),
    ]
    out = strategy_df.copy()
    for column, value in filters:
        if column in out.columns and column in candidate.index and not pd.isna(value):
            out = out[out[column].astype(str) == str(value)]
    for column in ["scenario_preset", "actor_capability_level", "overseer_capability_level"]:
        value = _clean_text(candidate.get(column))
        if column in out.columns and value:
            out = out[out[column].fillna("").astype(str) == value]
    if restore_order and not out.empty:
        generation = int(candidate["generation"])
        if "agent_index" in out:
            out = out.sort_values("agent_index")
        elif generation == 0:
            order = sorted(out.strategy_id, key=lambda sid: int(sid.rsplit("_s", 1)[1]))
            out = out.set_index("strategy_id", drop=False).loc[order].reset_index(drop=True)
        else:
            previous = candidate.copy()
            previous["generation"] = generation - 1
            prior = _matching_strategy_rows(strategy_df, previous, restore_order=False)
            if prior.empty or "rank" not in prior:
                raise ValueError("Cannot restore spatial order without previous-generation fitness ranks or agent_index")
            survivors = [sid for sid in prior.sort_values("rank").strategy_id if sid in set(out.strategy_id)]
            entrants = [sid for sid in out.strategy_id if sid not in set(survivors)]
            if any(not sid.startswith(f"g{generation}_s") for sid in entrants):
                raise ValueError("Unrecognized entrant IDs; spatial order needs original agent_index")
            order = survivors + sorted(entrants, key=lambda sid: int(sid.rsplit("_s", 1)[1]))
            out = out.set_index("strategy_id", drop=False).loc[order].reset_index(drop=True)
    return out


def _government_params(candidate: pd.Series) -> dict[str, float | int | bool]:
    params: dict[str, float | int | bool] = dict(_DEFAULT_GOVERNMENT_PARAMS)
    regime = _clean_text(candidate.get("governance_friction_regime"), default="ideal")
    if regime:
        params.update(get_harvest_governance_friction_regime(regime))
    overseer = _clean_text(candidate.get("overseer_capability_level"))
    if overseer:
        params.update(
            {
                key: value
                for key, value in get_harvest_overseer_capability_preset(overseer).items()
                if key
                in {
                    "detection_recall",
                    "enforcement_delay_rounds",
                    "max_target_share",
                    "governance_budget_cost",
                }
            }
        )
    return params


def _base_cfg(candidate: pd.Series, population_size: int, seed: int):
    scenario = _clean_text(candidate.get("scenario_preset"))
    if scenario:
        return make_harvest_cfg_for_scenario(scenario, n_agents=population_size, seed=seed)
    return make_harvest_cfg_for_tier(_clean_text(candidate.get("tier"), default="medium_h1"), n_agents=population_size, seed=seed)


def _candidate_test_seeds(candidate: pd.Series, runs_df: pd.DataFrame, cfg_seed_start: int, run_seed_stride: int) -> Iterable[int]:
    run_id = _clean_int(candidate.get("run_id"), default=0)
    generation = _clean_int(candidate.get("generation"), default=0)
    n_seeds = 32
    if not runs_df.empty and "run_id" in runs_df.columns:
        matching = runs_df[runs_df["run_id"].astype(str) == str(run_id)]
        if not matching.empty and "test_seeds_per_generation" in matching.columns:
            n_seeds = _clean_int(matching["test_seeds_per_generation"].iloc[0], default=n_seeds)
    base_seed = cfg_seed_start + run_id * run_seed_stride
    first = base_seed + 10_000_000 + generation * n_seeds
    return range(first, first + n_seeds)


def _write_no_case(prefix: Path, reason: str) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    trace_path = prefix.with_name(prefix.name + "_trace.csv")
    pd.DataFrame(
        columns=[
            "step",
            "mean_requested_harvest",
            "max_requested_frac",
            "mean_requested_frac",
            "local_safe_action_fraction",
            "mean_patch_health_after",
            "failed_patch_fraction_after",
            "global_unsafe",
            "local_pass_global_fail",
            "condition",
            "scenario_label",
        ]
    ).to_csv(trace_path, index=False)
    summary_path = prefix.with_name(prefix.name + "_summary.md")
    summary_path.write_text(f"# Harvest Oversight Case\n\nNo case found.\n\nReason: {reason}\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6.0, 2.4), constrained_layout=True)
    ax.axis("off")
    ax.text(0.5, 0.5, "No local-pass/global-fail case found", ha="center", va="center", fontsize=10)
    fig.savefig(prefix.with_name(prefix.name + "_trace.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_trace(trace_df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.8), constrained_layout=True)
    ax.plot(trace_df["step"], trace_df["mean_patch_health_after"], color="#355C7D", linewidth=1.8, label="Mean patch health")
    fail_steps = trace_df[trace_df["local_pass_global_fail"] == 1]
    if not fail_steps.empty:
        ax.scatter(
            fail_steps["step"],
            fail_steps["mean_patch_health_after"],
            color="#D95F02",
            s=42,
            zorder=3,
            label="Local-pass / global-fail step",
        )
    ax.axhline(10.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6, label="Global safety threshold")
    ax.set_xlabel("Episode step")
    ax.set_ylabel("Mean patch health")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def extract_case_from_frames(
    runs_df: pd.DataFrame,
    generation_df: pd.DataFrame,
    strategy_df: pd.DataFrame,
    output_prefix: str | Path,
    *,
    max_candidates: int = 25,
    cfg_seed_start: int = 0,
    run_seed_stride: int = 1000,
) -> bool:
    prefix = Path(output_prefix)
    candidates = _candidate_generations(generation_df, max_candidates=max_candidates)
    if candidates.empty:
        _write_no_case(prefix, "No generation has a positive test_local_pass_global_fail_rate.")
        return False

    for _, candidate in candidates.iterrows():
        strategy_rows = _matching_strategy_rows(strategy_df, candidate)
        if strategy_rows.empty:
            continue
        population = [_strategy_from_row(row) for _, row in strategy_rows.iterrows()]
        base_cfg = _base_cfg(candidate, population_size=len(population), seed=0)
        regimes = get_harvest_regime_pack(_clean_text(candidate.get("tier"), default="medium_h1"))
        condition = _clean_text(candidate.get("condition"), default="none")
        government_params = _government_params(candidate)
        for seed in _candidate_test_seeds(candidate, runs_df, cfg_seed_start=cfg_seed_start, run_seed_stride=run_seed_stride):
            for regime in regimes:
                cfg = _apply_cfg_overrides(copy.deepcopy(base_cfg), regime.get("overrides", {}))
                cfg.seed = int(seed)
                cfg.n_agents = len(population)
                episode_cfg, governor = _make_condition_setup(cfg, condition=condition, government_params=government_params)
                out = run_harvest_episode(episode_cfg, [spec.to_agent() for spec in population], governor=governor, record_trace=True)
                trace_df = pd.DataFrame(out.get("episode_trace_rows", []))
                if trace_df.empty or float(trace_df["local_pass_global_fail"].sum()) <= 0.0:
                    continue
                scenario = _clean_text(candidate.get("scenario_preset"))
                trace_df["condition"] = condition
                trace_df["condition_label"] = condition_label(condition)
                trace_df["scenario_preset"] = scenario
                trace_df["scenario_label"] = scenario_label(scenario) if scenario else ""
                trace_df["actor_capability_level"] = _clean_text(candidate.get("actor_capability_level"))
                trace_df["overseer_capability_level"] = _clean_text(candidate.get("overseer_capability_level"))
                trace_df["capability_gap"] = candidate.get("capability_gap", "")
                trace_df["regime_name"] = regime.get("name", "default")
                trace_df["seed"] = int(seed)
                trace_df["run_id"] = candidate.get("run_id", "")
                trace_df["generation"] = candidate.get("generation", "")
                prefix.parent.mkdir(parents=True, exist_ok=True)
                trace_path = prefix.with_name(prefix.name + "_trace.csv")
                trace_df.to_csv(trace_path, index=False)
                _plot_trace(trace_df, prefix.with_name(prefix.name + "_trace.png"))
                summary = [
                    "# Harvest Oversight Case",
                    "",
                    f"- Git commit: `{_safe_git_hash()}`",
                    f"- Condition: {condition_label(condition)}",
                    f"- Scenario: {scenario_label(scenario) if scenario else 'Default Harvest'}",
                    f"- Actor capability: {_clean_text(candidate.get('actor_capability_level'))}",
                    f"- Overseer capability: {_clean_text(candidate.get('overseer_capability_level'))}",
                    f"- Capability gap: {candidate.get('capability_gap', '')}",
                    f"- Run/generation: {candidate.get('run_id', '')}/{candidate.get('generation', '')}",
                    f"- Seed/regime: {seed}/{regime.get('name', 'default')}",
                    f"- Local-pass/global-fail steps: {int(trace_df['local_pass_global_fail'].sum())}",
                    "",
                    "The trace highlights an episode in which local safety checks pass at one or more steps while the aggregate Harvest state becomes globally unsafe.",
                ]
                prefix.with_name(prefix.name + "_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
                return True

    _write_no_case(prefix, "Positive aggregate rates were present, but no matching episode was reproduced from the saved strategies and seeds.")
    return False


def main() -> None:
    args = parse_args()
    runs_path = Path(args.runs_csv)
    generation_path = Path(args.generation_history_csv) if args.generation_history_csv else _infer_history_path(args.runs_csv, "_generation_history.csv")
    strategy_path = Path(args.strategy_history_csv) if args.strategy_history_csv else _infer_history_path(args.runs_csv, "_strategy_history.csv")
    runs_df = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()
    generation_df = pd.read_csv(generation_path)
    strategy_df = pd.read_csv(strategy_path)
    found = extract_case_from_frames(
        runs_df,
        generation_df,
        strategy_df,
        args.output_prefix,
        max_candidates=args.max_candidates,
        cfg_seed_start=args.cfg_seed_start,
        run_seed_stride=args.run_seed_stride,
    )
    print("Found local-pass/global-fail case." if found else "No local-pass/global-fail case found.")


if __name__ == "__main__":
    main()
