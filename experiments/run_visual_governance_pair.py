import argparse
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched-seed none vs monitoring+sanctions invasion visuals and render comparison GIFs."
    )
    parser.add_argument("--config", default="experiments/configs/base.yaml")
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--seeds-per-generation", type=int, default=32)
    parser.add_argument("--test-seeds-per-generation", type=int, default=32)
    parser.add_argument("--replacement-fraction", type=float, default=0.2)
    parser.add_argument("--adversarial-pressure", type=float, default=0.2)
    parser.add_argument("--rng-seed", type=int, default=11)
    parser.add_argument("--train-regen-rate", type=float, default=1.6)
    parser.add_argument("--train-obs-noise-std", type=float, default=8.0)
    parser.add_argument("--benchmark-pack", default="heldout_v1")

    parser.add_argument("--monitoring-prob", type=float, default=0.9)
    parser.add_argument("--quota-fraction", type=float, default=0.07)
    parser.add_argument("--base-fine-rate", type=float, default=2.0)
    parser.add_argument("--fine-growth", type=float, default=0.8)

    parser.add_argument("--none-prefix", default="results/runs/invasion/invasion_visual_none")
    parser.add_argument(
        "--governed-prefix",
        default="results/runs/invasion/invasion_visual_monitoring_sanctions",
    )
    parser.add_argument("--none-gif", default="results/runs/showcase/invasion_none_baseline.gif")
    parser.add_argument(
        "--governed-gif",
        default="results/runs/showcase/invasion_monitoring_sanctions.gif",
    )
    parser.add_argument(
        "--comparison-gif",
        default="results/runs/showcase/invasion_none_vs_monitoring_sanctions.gif",
    )
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=130)
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    common = [
        "--config",
        args.config,
        "--generations",
        str(args.generations),
        "--population-size",
        str(args.population_size),
        "--seeds-per-generation",
        str(args.seeds_per_generation),
        "--test-seeds-per-generation",
        str(args.test_seeds_per_generation),
        "--replacement-fraction",
        str(args.replacement_fraction),
        "--adversarial-pressure",
        str(args.adversarial_pressure),
        "--rng-seed",
        str(args.rng_seed),
        "--injector-mode",
        "mutation",
        "--train-regen-rate",
        str(args.train_regen_rate),
        "--train-obs-noise-std",
        str(args.train_obs_noise_std),
        "--benchmark-pack",
        args.benchmark_pack,
    ]

    _run(
        ["python", "-m", "experiments.run_invasion"]
        + common
        + ["--output-prefix", args.none_prefix]
    )

    _run(
        ["python", "-m", "experiments.run_invasion"]
        + common
        + [
            "--monitoring-prob",
            str(args.monitoring_prob),
            "--quota-fraction",
            str(args.quota_fraction),
            "--base-fine-rate",
            str(args.base_fine_rate),
            "--fine-growth",
            str(args.fine_growth),
            "--output-prefix",
            args.governed_prefix,
        ]
    )

    _run(
        [
            "python",
            "-m",
            "experiments.make_invasion_gif",
            "--input",
            f"{args.none_prefix}_generations.csv",
            "--title",
            "Invasion Dynamics | No Governance",
            "--output",
            args.none_gif,
            "--fps",
            str(args.fps),
            "--dpi",
            str(args.dpi),
        ]
    )
    _run(
        [
            "python",
            "-m",
            "experiments.make_invasion_gif",
            "--input",
            f"{args.governed_prefix}_generations.csv",
            "--title",
            "Invasion Dynamics | Monitoring + Sanctions",
            "--output",
            args.governed_gif,
            "--fps",
            str(args.fps),
            "--dpi",
            str(args.dpi),
        ]
    )
    _run(
        [
            "python",
            "-m",
            "experiments.make_governance_comparison_gif",
            "--none-input",
            f"{args.none_prefix}_generations.csv",
            "--governed-input",
            f"{args.governed_prefix}_generations.csv",
            "--output",
            args.comparison_gif,
            "--fps",
            str(args.fps),
            "--dpi",
            str(args.dpi),
        ]
    )

    print("Saved visual pair assets:")
    print(f"- {args.none_gif}")
    print(f"- {args.governed_gif}")
    print(f"- {args.comparison_gif}")


if __name__ == "__main__":
    main()
