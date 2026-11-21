import argparse
import os
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize matched mutation vs live LLM injector invasion runs."
    )
    parser.add_argument(
        "--mutation-prefix",
        default="results/runs/invasion/invasion_step4_match_mutation",
        help="Prefix used for *_generations.csv and *_strategies.csv.",
    )
    parser.add_argument(
        "--llm-prefix",
        default="results/runs/invasion/invasion_step4_match_ollama_live",
        help="Prefix used for *_generations.csv and *_strategies.csv.",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/runs/invasion/invasion_step4_match_comparison",
        help="Prefix for summary CSV and Markdown outputs.",
    )
    parser.add_argument("--collapse-threshold", type=float, default=0.8)
    return parser.parse_args()


def _read_pair(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    g_path = f"{prefix}_generations.csv"
    s_path = f"{prefix}_strategies.csv"
    if not os.path.exists(g_path):
        raise FileNotFoundError(f"Missing generations file: {g_path}")
    if not os.path.exists(s_path):
        raise FileNotFoundError(f"Missing strategies file: {s_path}")
    return pd.read_csv(g_path), pd.read_csv(s_path)


def _first_generation_ge(series: pd.Series, threshold: float) -> int | None:
    idx = np.where(series.to_numpy(dtype=float) >= float(threshold))[0]
    return int(idx[0]) if idx.size else None


def _per_regime_survival_mean(df: pd.DataFrame) -> dict[str, float]:
    regime_cols = [
        c
        for c in df.columns
        if c.startswith("test_")
        and c.endswith("_collapse_rate")
        and c != "test_collapse_rate"
    ]
    out: dict[str, float] = {}
    for c in regime_cols:
        regime = c[len("test_") : -len("_collapse_rate")]
        out[regime] = float((1.0 - df[c].astype(float)).mean())
    return out


def _flatten_regime_dict(prefix: str, d: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_survival_{k}": v for k, v in sorted(d.items())}


def _round_or_none(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return round(float(v), 4)
    return v


def _summarize_run(name: str, g_df: pd.DataFrame, s_df: pd.DataFrame, collapse_threshold: float) -> dict[str, Any]:
    regime_survival = _per_regime_survival_mean(g_df)
    out: dict[str, Any] = {
        "run": name,
        "generations": int(len(g_df)),
        "train_collapse_first": float(g_df["train_collapse_rate"].iloc[0]),
        "train_collapse_last": float(g_df["train_collapse_rate"].iloc[-1]),
        "train_collapse_mean": float(g_df["train_collapse_rate"].mean()),
        "test_collapse_first": float(g_df["test_collapse_rate"].iloc[0]),
        "test_collapse_last": float(g_df["test_collapse_rate"].iloc[-1]),
        "test_collapse_mean": float(g_df["test_collapse_rate"].mean()),
        "train_mean_stock_mean": float(g_df["train_mean_stock"].mean()),
        "test_mean_stock_mean": float(g_df["test_mean_stock"].mean()),
        "test_mean_stock_last": float(g_df["test_mean_stock"].iloc[-1]),
        "time_to_train_collapse_ge_threshold": _first_generation_ge(
            g_df["train_collapse_rate"], threshold=collapse_threshold
        ),
        "time_to_test_collapse_ge_threshold": _first_generation_ge(
            g_df["test_collapse_rate"], threshold=collapse_threshold
        ),
        "first_generation_test_collapse_ge_0_8": _first_generation_ge(
            g_df["test_collapse_rate"], threshold=0.8
        ),
        "overall_per_regime_survival_mean": float(np.mean(list(regime_survival.values())))
        if regime_survival
        else float("nan"),
        "llm_fallback_fraction": float((s_df["origin"] == "llm_fallback_mutation").mean()),
        "llm_json_fraction": float((s_df["origin"] == "llm_json").mean()),
        "mutation_fraction": float((s_df["origin"] == "mutation").mean()),
    }
    out.update(_flatten_regime_dict("per_regime", regime_survival))
    return {k: _round_or_none(v) for k, v in out.items()}


def _build_delta_row(mut_row: dict[str, Any], llm_row: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {"run": "delta_llm_minus_mutation"}
    all_keys = sorted(set(mut_row.keys()) | set(llm_row.keys()))
    skip = {"run"}
    for key in all_keys:
        if key in skip:
            continue
        a = mut_row.get(key)
        b = llm_row.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta[key] = round(float(b) - float(a), 4)
        else:
            delta[key] = None
    return delta


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    mut_g, mut_s = _read_pair(args.mutation_prefix)
    llm_g, llm_s = _read_pair(args.llm_prefix)

    mut_row = _summarize_run("mutation", mut_g, mut_s, collapse_threshold=args.collapse_threshold)
    llm_row = _summarize_run("llm_json_ollama_live", llm_g, llm_s, collapse_threshold=args.collapse_threshold)
    delta_row = _build_delta_row(mut_row, llm_row)

    summary_df = pd.DataFrame([mut_row, llm_row, delta_row])

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    csv_path = f"{args.output_prefix}.csv"
    md_path = f"{args.output_prefix}.md"
    summary_df.to_csv(csv_path, index=False)

    key_cols = [
        "run",
        "generations",
        "test_collapse_first",
        "test_collapse_last",
        "test_collapse_mean",
        "first_generation_test_collapse_ge_0_8",
        "overall_per_regime_survival_mean",
        "llm_json_fraction",
        "llm_fallback_fraction",
    ]
    key_df = summary_df[key_cols].copy()
    lines = []
    lines.append("# Injector Comparison (Mutation vs Live Ollama)")
    lines.append("")
    lines.append(
        f"Matched settings; only injector mode differs. Collapse threshold for time-to-collapse: {args.collapse_threshold}."
    )
    lines.append("")
    lines.append("## Key Metrics")
    lines.append("")
    lines.append(_markdown_table(key_df))
    lines.append("")
    lines.append("## Full Metrics")
    lines.append("")
    lines.append(_markdown_table(summary_df))
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
