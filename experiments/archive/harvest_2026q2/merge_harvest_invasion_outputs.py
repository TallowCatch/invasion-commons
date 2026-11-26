import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded Harvest invasion matrix CSV outputs.")
    parser.add_argument("--input-dir", required=True, help="Directory containing shard artifact folders/files.")
    parser.add_argument("--output-prefix", required=True, help="Prefix for merged output CSVs.")
    return parser.parse_args()


def _collect_csvs(root: Path, suffix: str) -> list[Path]:
    return sorted(path for path in root.rglob(f"*{suffix}") if path.is_file())


def _merge(files: list[Path]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    runs_files = _collect_csvs(input_dir, "_runs.csv")
    gen_files = _collect_csvs(input_dir, "_generation_history.csv")
    strat_files = _collect_csvs(input_dir, "_strategy_history.csv")
    agent_files = _collect_csvs(input_dir, "_agent_history.csv")

    if not runs_files and not gen_files and not strat_files and not agent_files:
        raise FileNotFoundError(f"No Harvest invasion shard CSV files found under {input_dir}")

    runs_df = _merge(runs_files)
    gen_df = _merge(gen_files)
    strat_df = _merge(strat_files)
    agent_df = _merge(agent_files)

    if not runs_df.empty:
        sort_cols = [col for col in ["tier", "partner_mix", "condition", "injector_mode_requested", "adversarial_pressure", "run_id"] if col in runs_df.columns]
        if sort_cols:
            runs_df = runs_df.sort_values(sort_cols).reset_index(drop=True)
        runs_df.to_csv(output_prefix.with_name(output_prefix.name + "_runs.csv"), index=False)

    if not gen_df.empty:
        sort_cols = [col for col in ["tier", "partner_mix", "condition", "injector_mode_requested", "adversarial_pressure", "run_id", "generation"] if col in gen_df.columns]
        if sort_cols:
            gen_df = gen_df.sort_values(sort_cols).reset_index(drop=True)
        gen_df.to_csv(output_prefix.with_name(output_prefix.name + "_generation_history.csv"), index=False)

    if not strat_df.empty:
        sort_cols = [col for col in ["tier", "partner_mix", "condition", "injector_mode_requested", "adversarial_pressure", "run_id", "generation", "strategy_id"] if col in strat_df.columns]
        if sort_cols:
            strat_df = strat_df.sort_values(sort_cols).reset_index(drop=True)
        strat_df.to_csv(output_prefix.with_name(output_prefix.name + "_strategy_history.csv"), index=False)

    if not agent_df.empty:
        sort_cols = [
            col
            for col in [
                "tier",
                "partner_mix",
                "condition",
                "injector_mode_requested",
                "adversarial_pressure",
                "run_id",
                "generation",
                "phase",
                "regime",
                "seed",
                "agent_index",
            ]
            if col in agent_df.columns
        ]
        if sort_cols:
            agent_df = agent_df.sort_values(sort_cols).reset_index(drop=True)
        agent_df.to_csv(output_prefix.with_name(output_prefix.name + "_agent_history.csv"), index=False)

    print(
        f"Merged {len(runs_files)} run shard(s), {len(gen_files)} generation shard(s), "
        f"{len(strat_files)} strategy shard(s), {len(agent_files)} agent shard(s)."
    )


if __name__ == "__main__":
    main()
