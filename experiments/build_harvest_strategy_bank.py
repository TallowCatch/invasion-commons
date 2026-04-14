from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from fishery_sim.harvest_benchmarks import make_harvest_cfg_for_tier
from fishery_sim.harvest_evolution import (
    parse_harvest_policy_response,
    harvest_policy_json_to_strategy_spec,
)
from fishery_sim.harvest_llm_population import (
    ATTITUDE_CHOICES,
    build_harvest_bank_prompt,
    harvest_policy_signature,
    sanitize_bank_label,
    strategy_spec_to_bank_row,
)
from fishery_sim.llm_adapter import (
    FileReplayPolicyLLMClient,
    OllamaPolicyLLMClient,
    OpenAIResponsesPolicyLLMClient,
    PolicyLLMClient,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLM-generated Harvest strategy banks.")
    parser.add_argument("--providers", default="openai,ollama")
    parser.add_argument("--models", default="gpt-5-mini,qwen2.5:3b-instruct")
    parser.add_argument("--attitudes", default="cooperative,exploitative")
    parser.add_argument("--target-per-pair", type=int, default=128)
    parser.add_argument("--max-attempts-per-pair", type=int, default=1024)
    parser.add_argument("--reference-tier", default="medium_h1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--llm-policy-replay-file", default=None)
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--ollama-base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def _parse_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _make_client(provider: str, model: str, args: argparse.Namespace) -> PolicyLLMClient:
    if args.llm_policy_replay_file:
        return FileReplayPolicyLLMClient(path=str(args.llm_policy_replay_file))
    if provider == "openai":
        api_key = os.environ.get(str(args.openai_api_key_env))
        if not api_key:
            raise ValueError(f"{args.openai_api_key_env} is required for provider=openai")
        return OpenAIResponsesPolicyLLMClient(
            model=model,
            api_key=api_key,
            base_url=args.openai_base_url,
            timeout_s=float(args.timeout_s),
            temperature=float(args.temperature),
        )
    if provider == "ollama":
        return OllamaPolicyLLMClient(
            model=model,
            base_url=args.ollama_base_url,
            timeout_s=float(args.timeout_s),
            temperature=float(args.temperature),
        )
    raise ValueError(f"Unsupported provider: {provider}")


def main() -> None:
    args = parse_args()
    providers = _parse_csv(args.providers)
    models = _parse_csv(args.models)
    attitudes = _parse_csv(args.attitudes)
    if len(providers) != len(models):
        raise ValueError("--providers and --models must have the same number of entries")
    if any(attitude not in ATTITUDE_CHOICES for attitude in attitudes):
        raise ValueError(f"--attitudes must be from {ATTITUDE_CHOICES}")

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    cfg = make_harvest_cfg_for_tier(args.reference_tier, n_agents=6, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    all_rows: list[dict] = []
    summary_rows: list[dict] = []

    for provider, model in zip(providers, models, strict=True):
        model_label = f"{sanitize_bank_label(provider)}__{sanitize_bank_label(model)}"
        client = _make_client(provider, model, args)
        for attitude in attitudes:
            accepted = 0
            attempts = 0
            parse_failures = 0
            seen_signatures: set[tuple[float, ...]] = set()
            while accepted < int(args.target_per_pair) and attempts < int(args.max_attempts_per_pair):
                attempts += 1
                prompt_nonce = int(rng.integers(0, 1_000_000_000))
                prompt = build_harvest_bank_prompt(attitude=attitude, patch_max=cfg.patch_max, prompt_nonce=prompt_nonce)
                try:
                    raw_response = client.complete(prompt)
                    policy, parse_status, parse_error_type = parse_harvest_policy_response(raw_response, patch_max=cfg.patch_max)
                except Exception:
                    parse_failures += 1
                    continue

                spec = harvest_policy_json_to_strategy_spec(
                    policy=policy,
                    strategy_id=f"{model_label}_{attitude}_{accepted}",
                    origin="llm_bank",
                    llm_parse_status=parse_status,
                    llm_parse_error_type=parse_error_type,
                )
                signature = harvest_policy_signature(spec)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                accepted += 1
                all_rows.append(
                    strategy_spec_to_bank_row(
                        spec,
                        bank_model_label=model_label,
                        bank_provider=provider,
                        bank_model_name=model,
                        bank_attitude=attitude,
                        prompt_nonce=prompt_nonce,
                    )
                )

            summary_rows.append(
                {
                    "bank_model_label": model_label,
                    "bank_provider": provider,
                    "bank_model_name": model,
                    "bank_attitude": attitude,
                    "target_per_pair": int(args.target_per_pair),
                    "accepted_unique": accepted,
                    "attempts": attempts,
                    "parse_failures": parse_failures,
                    "acceptance_rate": round(float(accepted / max(1, attempts)), 6),
                    "target_reached": bool(accepted >= int(args.target_per_pair)),
                }
            )

    bank_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)
    bank_csv = str(output_prefix.with_name(output_prefix.name + "_bank.csv"))
    summary_csv = str(output_prefix.with_name(output_prefix.name + "_summary.csv"))
    bank_df.to_csv(bank_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved: {bank_csv}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
