import argparse
import os

from fishery_sim.benchmarks import get_benchmark_pack, load_benchmark_pack_file
from fishery_sim.config import load_config
from fishery_sim.evolution import make_strategy_injector, run_evolutionary_invasion
from fishery_sim.llm_adapter import (
    FileReplayPolicyLLMClient,
    OllamaPolicyLLMClient,
    OpenAIResponsesPolicyLLMClient,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evolutionary strategy-invasion experiment for Fishery Commons."
    )
    parser.add_argument("--config", default="experiments/configs/base.yaml")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--seeds-per-generation", type=int, default=64)
    parser.add_argument("--replacement-fraction", type=float, default=0.3)
    parser.add_argument("--adversarial-pressure", type=float, default=0.7)
    parser.add_argument("--collapse-penalty", type=float, default=50.0)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--injector-mode", choices=["mutation", "llm_json"], default="mutation")
    parser.add_argument("--llm-policy-replay-file", default=None)
    parser.add_argument("--llm-provider", choices=["openai", "ollama"], default="ollama")
    parser.add_argument("--llm-model", default="qwen2.5:3b-instruct")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-timeout-s", type=float, default=45.0)
    parser.add_argument("--llm-temperature", type=float, default=0.8)
    parser.add_argument("--output-prefix", default="results/runs/invasion/invasion")

    # Regime split controls.
    parser.add_argument("--train-regen-rate", type=float, default=None)
    parser.add_argument("--train-obs-noise-std", type=float, default=None)
    parser.add_argument("--test-regen-rate", type=float, default=None)
    parser.add_argument("--test-obs-noise-std", type=float, default=None)
    parser.add_argument("--test-seeds-per-generation", type=int, default=None)
    parser.add_argument("--benchmark-pack", default=None, help="Built-in held-out pack name, e.g. harsh_v1")
    parser.add_argument("--benchmark-pack-file", default=None, help="YAML file with custom regimes")
    parser.add_argument("--benchmark-pack-file-name", default=None, help="Pack key inside custom YAML file")

    # Optional governance overrides for defense tests.
    parser.add_argument("--monitoring-prob", type=float, default=None)
    parser.add_argument("--quota-fraction", type=float, default=None)
    parser.add_argument("--base-fine-rate", type=float, default=None)
    parser.add_argument("--fine-growth", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.monitoring_prob is not None:
        cfg.monitoring_prob = args.monitoring_prob
    if args.quota_fraction is not None:
        cfg.quota_fraction = args.quota_fraction
    if args.base_fine_rate is not None:
        cfg.base_fine_rate = args.base_fine_rate
    if args.fine_growth is not None:
        cfg.fine_growth = args.fine_growth

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
    injector = make_strategy_injector(injector_mode=args.injector_mode, llm_client=llm_client)

    train_overrides = {
        "regen_rate": args.train_regen_rate,
        "obs_noise_std": args.train_obs_noise_std,
    }
    test_overrides = {
        "regen_rate": args.test_regen_rate,
        "obs_noise_std": args.test_obs_noise_std,
    }
    test_regimes = None
    if args.benchmark_pack_file:
        test_regimes = load_benchmark_pack_file(
            path=args.benchmark_pack_file,
            pack_name=args.benchmark_pack_file_name,
        )
    elif args.benchmark_pack:
        test_regimes = get_benchmark_pack(args.benchmark_pack)

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    generation_df, strategy_df = run_evolutionary_invasion(
        base_cfg=cfg,
        generations=args.generations,
        population_size=args.population_size,
        seeds_per_generation=args.seeds_per_generation,
        test_seeds_per_generation=args.test_seeds_per_generation,
        replacement_fraction=args.replacement_fraction,
        collapse_penalty=args.collapse_penalty,
        adversarial_pressure=args.adversarial_pressure,
        rng_seed=args.rng_seed,
        train_overrides=train_overrides,
        test_overrides=test_overrides,
        test_regimes=test_regimes,
        injector=injector,
    )

    generation_path = f"{args.output_prefix}_generations.csv"
    strategy_path = f"{args.output_prefix}_strategies.csv"
    generation_df.to_csv(generation_path, index=False)
    strategy_df.to_csv(strategy_path, index=False)

    first = generation_df.iloc[0]
    last = generation_df.iloc[-1]
    print(f"Saved: {generation_path}")
    print(f"Saved: {strategy_path}")
    print(
        "train collapse_rate: "
        f"{first['train_collapse_rate']:.3f} -> {last['train_collapse_rate']:.3f} | "
        "test collapse_rate: "
        f"{first['test_collapse_rate']:.3f} -> {last['test_collapse_rate']:.3f}"
    )
    print(
        "train mean_stock: "
        f"{first['train_mean_stock']:.2f} -> {last['train_mean_stock']:.2f} | "
        "test mean_stock: "
        f"{first['test_mean_stock']:.2f} -> {last['test_mean_stock']:.2f}"
    )
    if "test_regime_count" in generation_df.columns:
        print(f"test regimes used: {int(generation_df['test_regime_count'].iloc[0])}")


if __name__ == "__main__":
    main()
