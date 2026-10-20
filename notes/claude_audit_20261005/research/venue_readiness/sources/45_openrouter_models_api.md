# OpenRouter models API (prices per token, converted to USD per 1M)
- URL: https://openrouter.ai/api/v1/models
- Accessed: 2026-10-12 (downloaded directly with curl; 458 models; parsed with python)
- Source type: official marketplace API
- Credibility 4 / Recency 5 / Bias 2 (pass-through of provider list prices; individual routed providers may differ)

## Values (input / output USD per 1M tokens)
anthropic/claude-haiku-5.5 0.10 / 0.50; anthropic/claude-sonnet-5.5 2.0 / 10.0; anthropic/claude-opus-5.5 4.0 / 20.0; anthropic/claude-fable-5.1 10.0 / 50.0;
openai/gpt-6-luna 0.10 / 0.50; openai/gpt-6.1-sol 2.0 / 10.0; openai/gpt-6-astra 10.0 / 50.0; openai/gpt-5.6-luna 0.20 / 1.20; openai/gpt-5.6-terra 2.0 / 12.0; openai/gpt-5.5 5.0 / 30.0; openai/gpt-5.4-mini 0.75 / 4.50; openai/gpt-5-mini 0.25 / 2.0; openai/gpt-5-nano 0.05 / 0.40; openai/gpt-oss-120b 0.037 / 0.17;
google/gemini-3.8-flash 0.75 / 3.75; google/gemini-3.5-flash-lite 0.30 / 2.50; google/gemini-3.1-pro-preview 2.0 / 12.0; google/gemini-2.5-flash-lite 0.10 / 0.40;
deepseek/deepseek-v4.1-flash 0.30 / 1.20; deepseek/deepseek-v4-pro-0813 1.32 / 3.96; deepseek/deepseek-v4-flash 0.03 / 1.28; deepseek/deepseek-v3.2 0.259 / 0.80;
qwen/qwen3.8-flash 0.15 / 0.47; qwen/qwen3.8-max-0902 2.0 / 6.0; qwen/qwen3.7-flash 0.03 / 0.13; qwen/qwen3.5-397b-a17b 0.55 / 3.5.
Batch variants (":batch") are listed at about half price.
