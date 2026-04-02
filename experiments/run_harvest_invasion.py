import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

from fishery_sim.harvest_benchmarks import get_harvest_regime_pack, make_harvest_cfg_for_tier
from fishery_sim.harvest_evolution import make_harvest_strategy_injector
from fishery_sim.harvest_evolution import run_harvest_invasion
from fishery_sim.llm_adapter import (
    FileReplayPolicyLLMClient,
    OllamaPolicyLLMClient,
    OpenAIResponsesPolicyLLMClient,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolutionary invasion experiment for Harvest Commons.")
    parser.add_argument("--tier", choices=["easy_h1", "medium_h1", "hard_h1"], default="medium_h1")
    parser.add_argument("--condition", choices=["none", "top_down_only", "bottom_up_only", "hybrid"], default="hybrid")
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument("--seeds-per-generation", type=int, default=32)
    parser.add_argument("--test-seeds-per-generation", type=int, default=32)
    parser.add_argument("--replacement-fraction", type=float, default=0.2)
    parser.add_argument("--adversarial-pressure", type=float, default=0.3)
    parser.add_argument("--partner-mix", choices=["cooperative_heavy", "balanced", "adversarial_heavy"], default="balanced")
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--injector-mode", choices=["random", "mutation", "adversarial_heuristic", "search_mutation", "llm_json"], default="mutation")
    parser.add_argument("--llm-policy-replay-file", default=None)
    parser.add_argument("--llm-provider", choices=["openai", "ollama"], default="ollama")
    parser.add_argument("--llm-model", default="qwen2.5:3b-instruct")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-timeout-s", type=float, default=120.0)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--output-prefix", default="results/runs/harvest_invasion/harvest_invasion")
    parser.add_argument("--experiment-tag", default="harvest_invasion")
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--government-trigger", type=float, default=16.0)
    parser.add_argument("--strict-cap-frac", type=float, default=0.18)
    parser.add_argument("--relaxed-cap-frac", type=float, default=0.35)
    parser.add_argument("--soft-trigger", type=float, default=18.0)
    parser.add_argument("--deterioration-threshold", type=float, default=0.35)
    parser.add_argument("--activation-warmup", type=int, default=3)
    parser.add_argument("--aggressive-request-threshold", type=float, default=0.75)
    parser.add_argument("--aggressive-agent-fraction-trigger", type=float, default=0.34)
    parser.add_argument("--local-neighborhood-trigger", type=float, default=0.67)
    return parser.parse_args()


def _safe_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _write_manifest(
    path: str,
    args: argparse.Namespace,
    generation_path: str,
    strategy_path: str,
    agent_history_path: str,
    test_regimes: list[dict],
) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        "script": "experiments/run_harvest_invasion.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _safe_git_hash(),
        "experiment_tag": args.experiment_tag,
        "output_prefix": args.output_prefix,
        "outputs": {
            "generations_csv": generation_path,
            "strategies_csv": strategy_path,
            "agent_history_csv": agent_history_path,
        },
        "params": vars(args),
        "benchmark": {
            "tier": args.tier,
            "resolved_regime_count": len(test_regimes),
            "resolved_regime_names": [reg["name"] for reg in test_regimes],
        },
        "injector": {
            "mode": args.injector_mode,
            "provider": args.llm_provider if args.injector_mode == "llm_json" else "none",
            "model": args.llm_model if args.injector_mode == "llm_json" else "",
            "replay_file": args.llm_policy_replay_file or "",
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    args = parse_args()
    cfg = make_harvest_cfg_for_tier(args.tier, n_agents=args.population_size)
    llm_client = None
    if args.llm_policy_replay_file:
        llm_client = FileReplayPolicyLLMClient(path=args.llm_policy_replay_file)
    elif args.injector_mode == "llm_json":
        if args.llm_provider == "ollama":
            llm_client = OllamaPolicyLLMClient(
                model=args.llm_model,
                base_url=args.llm_base_url,
                timeout_s=args.llm_timeout_s,
                temperature=args.llm_temperature,
            )
        elif args.llm_provider == "openai":
            api_key = os.environ.get(args.llm_api_key_env)
            if not api_key:
                raise ValueError(
                    f"{args.llm_api_key_env} is required for OpenAI injection. "
                    "Set the env var or switch to --llm-provider ollama / replay file."
                )
            llm_client = OpenAIResponsesPolicyLLMClient(
                model=args.llm_model,
                api_key=api_key,
                base_url=args.llm_base_url,
                timeout_s=args.llm_timeout_s,
                temperature=args.llm_temperature,
            )
        else:
            raise ValueError(f"Unsupported llm provider: {args.llm_provider}")
    injector = make_harvest_strategy_injector(args.injector_mode, llm_client=llm_client)
    test_regimes = get_harvest_regime_pack(args.tier)
    government_params = {
        "trigger": args.government_trigger,
        "strict_cap_frac": args.strict_cap_frac,
        "relaxed_cap_frac": args.relaxed_cap_frac,
        "soft_trigger": args.soft_trigger,
        "deterioration_threshold": args.deterioration_threshold,
        "activation_warmup": args.activation_warmup,
        "aggressive_request_threshold": args.aggressive_request_threshold,
        "aggressive_agent_fraction_trigger": args.aggressive_agent_fraction_trigger,
        "local_neighborhood_trigger": args.local_neighborhood_trigger,
    }

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    progress_bar = None
    progress_callback = None
    if not args.no_progress and tqdm is not None:
        progress_bar = tqdm(total=args.generations, desc="Harvest invasion generations")

        def _callback(done: int, total: int) -> None:
            del total
            progress_bar.n = int(done)
            progress_bar.refresh()

        progress_callback = _callback
    elif not args.no_progress:
        print(f"Progress: 0/{args.generations} generations")

        def _callback(done: int, total: int) -> None:
            if done == total or done % max(1, total // 10) == 0:
                print(f"Progress: {done}/{total} generations")

        progress_callback = _callback

    try:
        generation_df, strategy_df, agent_history_df = run_harvest_invasion(
            base_cfg=cfg,
            condition=args.condition,
            generations=args.generations,
            population_size=args.population_size,
            seeds_per_generation=args.seeds_per_generation,
            test_seeds_per_generation=args.test_seeds_per_generation,
            replacement_fraction=args.replacement_fraction,
            adversarial_pressure=args.adversarial_pressure,
            rng_seed=args.rng_seed,
            partner_mix_preset=args.partner_mix,
            injector=injector,
            test_regimes=test_regimes,
            government_params=government_params,
            progress_callback=progress_callback,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    generation_df["experiment_tag"] = args.experiment_tag
    strategy_df["experiment_tag"] = args.experiment_tag
    generation_df["tier"] = args.tier
    strategy_df["tier"] = args.tier
    generation_df["condition"] = args.condition
    strategy_df["condition"] = args.condition
    generation_df["partner_mix"] = args.partner_mix
    strategy_df["partner_mix"] = args.partner_mix
    generation_df["injector_mode_requested"] = args.injector_mode
    strategy_df["injector_mode_requested"] = args.injector_mode
    generation_df["llm_provider"] = args.llm_provider if args.injector_mode == "llm_json" else "none"
    strategy_df["llm_provider"] = args.llm_provider if args.injector_mode == "llm_json" else "none"
    generation_df["llm_model"] = args.llm_model if args.injector_mode == "llm_json" else ""
    strategy_df["llm_model"] = args.llm_model if args.injector_mode == "llm_json" else ""
    agent_history_df["experiment_tag"] = args.experiment_tag
    agent_history_df["tier"] = args.tier
    agent_history_df["condition"] = args.condition
    agent_history_df["partner_mix"] = args.partner_mix
    agent_history_df["injector_mode_requested"] = args.injector_mode
    agent_history_df["llm_provider"] = args.llm_provider if args.injector_mode == "llm_json" else "none"
    agent_history_df["llm_model"] = args.llm_model if args.injector_mode == "llm_json" else ""

    generation_path = f"{args.output_prefix}_generations.csv"
    strategy_path = f"{args.output_prefix}_strategies.csv"
    agent_history_path = f"{args.output_prefix}_agent_history.csv"
    generation_df.to_csv(generation_path, index=False)
    strategy_df.to_csv(strategy_path, index=False)
    agent_history_df.to_csv(agent_history_path, index=False)
    print(f"Saved: {generation_path}")
    print(f"Saved: {strategy_path}")
    print(f"Saved: {agent_history_path}")

    if args.manifest_out:
        _write_manifest(args.manifest_out, args, generation_path, strategy_path, agent_history_path, test_regimes)
        print(f"Saved: {args.manifest_out}")
    if args.injector_mode == "llm_json" and not strategy_df.empty:
        print(f"llm_json_fraction: {float((strategy_df['origin'] == 'llm_json').mean()):.4f}")
        print(f"llm_fallback_fraction: {float((strategy_df['origin'] == 'llm_fallback_mutation').mean()):.4f}")
        if "repaired_json_fraction" in generation_df.columns:
            print(f"repaired_json_fraction: {float(generation_df['repaired_json_fraction'].mean()):.4f}")
        if "effective_llm_fraction" in generation_df.columns:
            print(f"effective_llm_fraction: {float(generation_df['effective_llm_fraction'].mean()):.4f}")
        if "unrepaired_fallback_fraction" in generation_df.columns:
            print(f"unrepaired_fallback_fraction: {float(generation_df['unrepaired_fallback_fraction'].mean()):.4f}")


if __name__ == "__main__":
    main()
