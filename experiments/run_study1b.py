import argparse
import copy
import json
import os

import pandas as pd

from fishery_sim.benchmarks import get_pressure_levels
from fishery_sim.benchmarks import load_benchmark_pack_file
from fishery_sim.benchmarks import get_benchmark_pack
from fishery_sim.config import load_config
from fishery_sim.evolution import make_strategy_injector
from fishery_sim.evolution import run_evolutionary_invasion
from experiments.run_governance_ablation import _aggregate_with_ci
from experiments.run_governance_ablation import _build_llm_client
from experiments.run_governance_ablation import _build_survival_curves
from experiments.run_governance_ablation import _origin_fraction
from experiments.run_governance_ablation import _summary_from_generation_df
from experiments.run_governance_ablation import _to_markdown_table
from experiments.run_governance_ablation import _write_manifest

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study 1b matrix sweep: governance x benchmark pack x partner mix x pressure."
    )
    parser.add_argument("--config", default="experiments/configs/base.yaml")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--seeds-per-generation", type=int, default=32)
    parser.add_argument("--test-seeds-per-generation", type=int, default=None)
    parser.add_argument("--replacement-fraction", type=float, default=0.3)
    parser.add_argument("--collapse-penalty", type=float, default=50.0)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--rng-seed-start", type=int, default=0)
    parser.add_argument("--cfg-seed-start", type=int, default=0)
    parser.add_argument("--run-seed-stride", type=int, default=10_000)
    parser.add_argument(
        "--injector-mode",
        choices=["mutation", "random", "adversarial_heuristic", "search_mutation", "llm_json"],
        default="mutation",
    )
    parser.add_argument("--llm-policy-replay-file", default=None)
    parser.add_argument("--llm-provider", choices=["openai", "ollama"], default="ollama")
    parser.add_argument("--llm-model", default="qwen2.5:3b-instruct")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-timeout-s", type=float, default=45.0)
    parser.add_argument("--llm-temperature", type=float, default=0.8)
    parser.add_argument("--benchmark-packs", default="easy_v1,medium_v1")
    parser.add_argument(
        "--partner-mixes",
        default="cooperative_heavy,balanced,adversarial_heavy",
    )
    parser.add_argument("--pressure-levels", default="study1b")
    parser.add_argument("--benchmark-pack-file", default=None)
    parser.add_argument("--benchmark-pack-file-name", default=None)
    parser.add_argument(
        "--conditions",
        default=None,
        help="Optional comma-separated subset of governance conditions to run. Defaults to all Study 1b conditions.",
    )
    parser.add_argument("--train-regen-rate", type=float, default=None)
    parser.add_argument("--train-obs-noise-std", type=float, default=None)
    parser.add_argument("--monitoring-prob", type=float, default=0.9)
    parser.add_argument("--quota-fraction", type=float, default=0.07)
    parser.add_argument("--sanction-base-fine-rate", type=float, default=2.0)
    parser.add_argument("--sanction-fine-growth", type=float, default=0.8)
    parser.add_argument("--adaptive-quota-min-scale", type=float, default=0.35)
    parser.add_argument("--adaptive-quota-sensitivity", type=float, default=0.9)
    parser.add_argument("--temporary-closure-trigger", type=float, default=15.0)
    parser.add_argument("--temporary-closure-quota-fraction", type=float, default=0.01)
    parser.add_argument("--output-prefix", default="results/runs/ablation/study1b")
    parser.add_argument("--experiment-tag", default="study1b")
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _parse_csv_arg(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _resolve_pressure_levels(text: str) -> list[float]:
    key = text.strip()
    if not key:
        return [0.1, 0.3, 0.5, 0.7]
    if "," in key or key.replace(".", "", 1).isdigit():
        return [float(x) for x in _parse_csv_arg(key)]
    return get_pressure_levels(key)


def _study1b_conditions(args: argparse.Namespace) -> dict[str, dict[str, float | str]]:
    conditions = {
        "none": {
            "monitoring_prob": 0.0,
            "quota_fraction": 0.0,
            "base_fine_rate": 0.0,
            "fine_growth": 0.0,
            "governance_variant": "static",
        },
        "monitoring": {
            "monitoring_prob": args.monitoring_prob,
            "quota_fraction": args.quota_fraction,
            "base_fine_rate": 0.0,
            "fine_growth": 0.0,
            "governance_variant": "static",
        },
        "monitoring_sanctions": {
            "monitoring_prob": args.monitoring_prob,
            "quota_fraction": args.quota_fraction,
            "base_fine_rate": args.sanction_base_fine_rate,
            "fine_growth": args.sanction_fine_growth,
            "governance_variant": "static",
        },
        "adaptive_quota": {
            "monitoring_prob": args.monitoring_prob,
            "quota_fraction": args.quota_fraction,
            "base_fine_rate": args.sanction_base_fine_rate,
            "fine_growth": args.sanction_fine_growth,
            "governance_variant": "adaptive_quota",
        },
        "temporary_closure": {
            "monitoring_prob": 1.0,
            "quota_fraction": max(args.quota_fraction, args.temporary_closure_quota_fraction),
            "base_fine_rate": args.sanction_base_fine_rate,
            "fine_growth": args.sanction_fine_growth,
            "governance_variant": "temporary_closure",
        },
    }
    if not args.conditions:
        return conditions
    selected = _parse_csv_arg(args.conditions)
    missing = [name for name in selected if name not in conditions]
    if missing:
        raise ValueError(f"Unknown Study 1b conditions: {missing}")
    return {name: conditions[name] for name in selected}


def _build_ranking_table(per_run_df: pd.DataFrame) -> pd.DataFrame:
    if per_run_df.empty:
        return pd.DataFrame()
    group_cols = ["benchmark_pack", "partner_mix", "adversarial_pressure", "run_id"]
    rows = []
    for keys, gdf in per_run_df.groupby(group_cols, sort=True):
        ranked = gdf.sort_values(
            ["test_collapse_mean", "test_mean_welfare_mean", "test_mean_stock_mean"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            rows.append(
                {
                    "benchmark_pack": keys[0],
                    "partner_mix": keys[1],
                    "adversarial_pressure": keys[2],
                    "run_id": keys[3],
                    "condition": row["condition"],
                    "rank": rank,
                    "test_collapse_mean": row["test_collapse_mean"],
                    "test_mean_welfare_mean": row["test_mean_welfare_mean"],
                    "test_mean_stock_mean": row["test_mean_stock_mean"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    llm_client = _build_llm_client(args)
    injector = make_strategy_injector(injector_mode=args.injector_mode, llm_client=llm_client)

    benchmark_pack_names = _parse_csv_arg(args.benchmark_packs)
    partner_mixes = _parse_csv_arg(args.partner_mixes)
    pressure_levels = _resolve_pressure_levels(args.pressure_levels)
    conditions = _study1b_conditions(args)
    train_overrides = {
        "regen_rate": args.train_regen_rate,
        "obs_noise_std": args.train_obs_noise_std,
    }

    resolved_pack_map: dict[str, list[dict]] = {}
    if args.benchmark_pack_file:
        pack = load_benchmark_pack_file(path=args.benchmark_pack_file, pack_name=args.benchmark_pack_file_name)
        only_name = args.benchmark_pack_file_name or "custom_pack"
        resolved_pack_map[only_name] = pack
        benchmark_pack_names = [only_name]
    else:
        for pack_name in benchmark_pack_names:
            resolved_pack_map[pack_name] = get_benchmark_pack(pack_name)

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_runs = (
        len(benchmark_pack_names)
        * len(partner_mixes)
        * len(pressure_levels)
        * len(conditions)
        * args.n_runs
        * args.generations
    )
    progress_bar = None
    if not args.no_progress and tqdm is not None:
        progress_bar = tqdm(total=total_runs, desc="Study1b matrix")
    elif not args.no_progress:
        print(f"Progress: 0/{total_runs} generation-steps")

    def _progress(done: int, total: int) -> None:
        del done, total
        if progress_bar is not None:
            progress_bar.update(1)
            return
        _progress.state += 1
        if _progress.state == total_runs or _progress.state % max(1, total_runs // 20) == 0:
            print(f"Progress: {_progress.state}/{total_runs} generation-steps")

    _progress.state = 0  # type: ignore[attr-defined]

    per_run_rows: list[dict] = []
    generation_histories: list[pd.DataFrame] = []

    llm_provider = args.llm_provider if args.injector_mode == "llm_json" else "none"
    llm_model = args.llm_model if args.injector_mode == "llm_json" else ""

    try:
        for benchmark_pack_name in benchmark_pack_names:
            test_regimes = resolved_pack_map[benchmark_pack_name]
            for partner_mix in partner_mixes:
                for pressure in pressure_levels:
                    common_row_meta = {
                        "experiment_tag": args.experiment_tag,
                        "injector_mode_requested": args.injector_mode,
                        "llm_provider": llm_provider,
                        "llm_model": llm_model,
                        "benchmark_pack": benchmark_pack_name,
                        "generations": int(args.generations),
                        "population_size": int(args.population_size),
                        "seeds_per_generation": int(args.seeds_per_generation),
                        "test_seeds_per_generation": int(
                            args.test_seeds_per_generation
                            if args.test_seeds_per_generation is not None
                            else args.seeds_per_generation
                        ),
                        "replacement_fraction": float(args.replacement_fraction),
                        "collapse_penalty": float(args.collapse_penalty),
                        "partner_mix": partner_mix,
                        "adversarial_pressure": float(pressure),
                        "n_runs_planned": int(args.n_runs),
                    }
                    for condition_name, knobs in conditions.items():
                        for run_id in range(args.n_runs):
                            cfg = copy.deepcopy(base_cfg)
                            cfg.monitoring_prob = float(knobs["monitoring_prob"])
                            cfg.quota_fraction = float(knobs["quota_fraction"])
                            cfg.base_fine_rate = float(knobs["base_fine_rate"])
                            cfg.fine_growth = float(knobs["fine_growth"])
                            cfg.governance_variant = str(knobs["governance_variant"])
                            cfg.adaptive_quota_min_scale = args.adaptive_quota_min_scale
                            cfg.adaptive_quota_sensitivity = args.adaptive_quota_sensitivity
                            cfg.temporary_closure_trigger = args.temporary_closure_trigger
                            cfg.temporary_closure_quota_fraction = args.temporary_closure_quota_fraction
                            cfg.seed = args.cfg_seed_start + run_id * args.run_seed_stride

                            run_rng_seed = args.rng_seed_start + run_id * args.run_seed_stride
                            generation_df, strategy_df = run_evolutionary_invasion(
                                base_cfg=cfg,
                                generations=args.generations,
                                population_size=args.population_size,
                                seeds_per_generation=args.seeds_per_generation,
                                test_seeds_per_generation=args.test_seeds_per_generation,
                                replacement_fraction=args.replacement_fraction,
                                collapse_penalty=args.collapse_penalty,
                                adversarial_pressure=pressure,
                                rng_seed=run_rng_seed,
                                train_overrides=train_overrides,
                                test_regimes=test_regimes,
                                partner_mix_preset=partner_mix,
                                injector=injector,
                                progress_callback=None if args.no_progress else _progress,
                            )
                            generation_df = generation_df.assign(
                                condition=condition_name,
                                run_id=run_id,
                                benchmark_pack=benchmark_pack_name,
                                partner_mix=partner_mix,
                                adversarial_pressure=float(pressure),
                                governance_variant=cfg.governance_variant,
                            )
                            strategy_df = strategy_df.assign(
                                condition=condition_name,
                                run_id=run_id,
                                benchmark_pack=benchmark_pack_name,
                                partner_mix=partner_mix,
                                adversarial_pressure=float(pressure),
                                governance_variant=cfg.governance_variant,
                            )
                            generation_histories.append(generation_df)
                            summary_row = _summary_from_generation_df(condition_name, run_id, generation_df)
                            summary_row["llm_json_fraction"] = _origin_fraction(strategy_df, "llm_json")
                            summary_row["llm_fallback_fraction"] = _origin_fraction(strategy_df, "llm_fallback_mutation")
                            summary_row["benchmark_pack"] = benchmark_pack_name
                            summary_row["partner_mix"] = partner_mix
                            summary_row["adversarial_pressure"] = float(pressure)
                            summary_row["governance_variant"] = cfg.governance_variant
                            summary_row.update(common_row_meta)
                            per_run_rows.append(summary_row)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    per_run_df = pd.DataFrame(per_run_rows)
    table_group_cols = [
        "benchmark_pack",
        "partner_mix",
        "adversarial_pressure",
        "condition",
    ]
    table_rows = []
    for keys, cdf in per_run_df.groupby(table_group_cols, sort=True):
        summary = _aggregate_with_ci(cdf.rename(columns={"condition": "condition"}))
        if summary.empty:
            continue
        row = summary.iloc[0].to_dict()
        row["benchmark_pack"] = keys[0]
        row["partner_mix"] = keys[1]
        row["adversarial_pressure"] = keys[2]
        row["condition"] = keys[3]
        table_rows.append(row)
    table_df = pd.DataFrame(table_rows)
    history_df = pd.concat(generation_histories, ignore_index=True) if generation_histories else pd.DataFrame()
    survival_curves_df = _build_survival_curves(history_df)
    ranking_df = _build_ranking_table(per_run_df)

    runs_csv = f"{args.output_prefix}_runs.csv"
    table_csv = f"{args.output_prefix}_table.csv"
    table_md = f"{args.output_prefix}_table.md"
    history_csv = f"{args.output_prefix}_generation_history.csv"
    survival_curves_csv = f"{args.output_prefix}_survival_curves.csv"
    ranking_csv = f"{args.output_prefix}_ranking.csv"

    per_run_df.to_csv(runs_csv, index=False)
    table_df.to_csv(table_csv, index=False)
    history_df.to_csv(history_csv, index=False)
    survival_curves_df.to_csv(survival_curves_csv, index=False)
    ranking_df.to_csv(ranking_csv, index=False)
    with open(table_md, "w", encoding="utf-8") as f:
        f.write(_to_markdown_table(table_df))
        f.write("\n")

    print(f"Saved: {runs_csv}")
    print(f"Saved: {table_csv}")
    print(f"Saved: {history_csv}")
    print(f"Saved: {survival_curves_csv}")
    print(f"Saved: {ranking_csv}")
    if args.manifest_out:
        outputs = {
            "runs_csv": runs_csv,
            "table_csv": table_csv,
            "table_md": table_md,
            "generation_history_csv": history_csv,
            "survival_curves_csv": survival_curves_csv,
            "ranking_csv": ranking_csv,
        }
        _write_manifest(
            path=args.manifest_out,
            args=args,
            test_regimes=[{"name": name} for name in benchmark_pack_names],
            outputs=outputs,
        )
        print(f"Saved: {args.manifest_out}")


if __name__ == "__main__":
    main()
