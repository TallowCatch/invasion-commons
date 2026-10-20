import argparse
import json

from fishery_sim.llm_adapter import (
    FileReplayPolicyLLMClient,
    OllamaPolicyLLMClient,
    OpenAIResponsesPolicyLLMClient,
    extract_json_object,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick check for LLM backend connectivity and JSON policy output format."
    )
    parser.add_argument("--provider", choices=["ollama", "openai", "replay"], default="ollama")
    parser.add_argument("--model", default="qwen2.5:3b-instruct")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--replay-file", default="experiments/configs/llm_policy_replay.jsonl")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.provider == "replay":
        client = FileReplayPolicyLLMClient(args.replay_file)
    elif args.provider == "ollama":
        client = OllamaPolicyLLMClient(model=args.model, base_url=args.base_url)
    elif args.provider == "openai":
        import os

        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(f"{args.api_key_env} is missing.")
        client = OpenAIResponsesPolicyLLMClient(model=args.model, api_key=api_key, base_url=args.base_url)
    else:
        raise ValueError(f"Unsupported provider: {args.provider}")

    prompt = (
        "Return JSON only with keys: rationale, low_stock_threshold, high_stock_threshold, "
        "low_harvest_frac, mid_harvest_frac, high_harvest_frac."
    )
    raw = client.complete(prompt)
    parsed = extract_json_object(raw)
    print("Backend OK.")
    print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    main()

