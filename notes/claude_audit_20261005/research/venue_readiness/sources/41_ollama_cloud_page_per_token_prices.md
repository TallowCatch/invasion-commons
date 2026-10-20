# Ollama cloud page: per-token prices, concurrency and queue
- URL: https://ollama.com/cloud
- Accessed: 2026-10-12
- Source type: official vendor page
- Credibility 5 / Recency 5 / Bias 3
- Extraction note: page text was retrieved through a fetch tool that passes the page through a small summarising model; quoted strings below are as returned by that tool and were not re-checked against the raw HTML unless stated.

## Quotes
- "Pay as you go with usage credits. Prices are per million tokens."
- Table (USD per 1M tokens; input / cached input / output): gpt-oss:120b 0.15 / 0.014 / 0.60; gpt-oss:20b 0.07 / 0.035 / 0.30; nemotron-3-super 0.015 / 0.015 / 0.60; nemotron-3-ultra 0.10 / 0.10 / 3.00; nemotron-3-nano 0.06 / - / 0.24; gemma4 0.14 / 0.05 / 0.40; mistral-large-3 0.50 / - / 1.50; mistral-large-4 0.68 / 0.07 / 2.09; deepseek-v4.1-flash 0.30 / 0.006 / 1.20 (off-peak 0.15 / 0.003 / 0.60); deepseek-v4-pro 1.32 / 0.044 / 3.96 (off-peak 0.66 / 0.022 / 1.98); glm-5.3 and glm-5.2 1.40 / 0.26 / 4.40; glm-5.3-flash 0.15 / 0.03 / 0.50; kimi-k3 3.00 / 0.30 / 15.00; kimi-k2.7-code 0.95 / 0.19 / 4.00; kimi-k2.6 0.95 / 0.16 / 4.00; minimax-m3 0.60 / 0.12 / 2.40; minimax-m2.7 0.30 / 0.06 / 1.20.
- Off-peak: "Off-peak pricing apply outside 12:00 and 18:00 UTC on weekdays and all day on weekends." (listed only for the two DeepSeek rows)
- "Usage is measured in tokens at each model's rates."
- "Free includes 1 concurrent request, Pro 3, and Max and Team 10."
- "Requests beyond your plan's concurrency limit are queued and processed as soon as a slot is available." "Queued requests are held up to a fixed limit" and "if the queue is full, the request will be rejected until one of your concurrency slots opens."
- "included usage resets monthly on the same day of the month your subscription started." Rollover: "No. Instead, your included amount refreshes at each monthly reset."
- "On paid plans, Ollama sends an email reminder at 90% of your included monthly usage."
- "Running models on your own hardware is always unlimited."
