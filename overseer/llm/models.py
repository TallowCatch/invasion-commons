"""The inspector ladder (notes/llm_overseer/protocol.md, section 3).

Local tier: Qwen2.5-Instruct through Ollama (default 4-bit quantisation, Q4_K_M), one family at four sizes.
Cloud tier: gpt-oss through Ollama's cloud via the local daemon (OpenAI's open-weight reasoning models). The family
changes at the top of the ladder, so size and family are confounded there.
hf_id is the matching HuggingFace model for the optional vLLM path on CREATE (bf16, so not identical to Ollama's
4-bit weights).
"""
from __future__ import annotations

LADDER = [
    # tag, ollama name, billions of parameters, family, tier, HuggingFace id
    ("qwen2.5-0.5b", "qwen2.5:0.5b-instruct", 0.5, "qwen2.5", "local", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen2.5-1.5b", "qwen2.5:1.5b-instruct", 1.5, "qwen2.5", "local", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen2.5-3b", "qwen2.5:3b-instruct", 3.1, "qwen2.5", "local", "Qwen/Qwen2.5-3B-Instruct"),
    ("qwen2.5-7b", "qwen2.5:7b-instruct", 7.6, "qwen2.5", "local", "Qwen/Qwen2.5-7B-Instruct"),
    # X5 capability curve (notes/formulation/x5_capability/): larger Qwen2.5 judges, run with Ollama on CREATE GPUs
    ("qwen2.5-14b", "qwen2.5:14b-instruct", 14.7, "qwen2.5", "create", "Qwen/Qwen2.5-14B-Instruct"),
    ("qwen2.5-32b", "qwen2.5:32b-instruct", 32.5, "qwen2.5", "create", "Qwen/Qwen2.5-32B-Instruct"),
    ("qwen2.5-72b", "qwen2.5:72b-instruct", 72.7, "qwen2.5", "create", "Qwen/Qwen2.5-72B-Instruct"),
    ("gpt-oss-20b", "gpt-oss:20b-cloud", 21.0, "gpt-oss", "cloud", "openai/gpt-oss-20b"),
    ("gpt-oss-120b", "gpt-oss:120b-cloud", 117.0, "gpt-oss", "cloud", "openai/gpt-oss-120b"),
]
BY_TAG = {row[0]: dict(tag=row[0], ollama=row[1], params_b=row[2], family=row[3], tier=row[4], hf_id=row[5])
          for row in LADDER}
# vLLM ladder for CREATE (optional; verified to exist on HuggingFace on 22 Nov 2026)
HF_QWEN25 = [f"Qwen/Qwen2.5-{s}-Instruct" for s in ("0.5B", "1.5B", "3B", "7B", "14B", "32B")]
