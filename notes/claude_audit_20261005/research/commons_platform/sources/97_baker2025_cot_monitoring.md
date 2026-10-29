# Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation

- **Authors:** Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y. Guan, Aleksander Madry, Wojciech Zaremba, Jakub Pachocki, David Farhi (OpenAI)
- **Year:** 2025
- **Venue / identifier:** arXiv:2503.11926
- **URL:** https://arxiv.org/abs/2503.11926 (HTML at https://arxiv.org/html/2503.11926)
- **Source type:** Preprint (industry lab)
- **Date accessed:** 14 Oct 2026
- **Depth read:** Abstract and opening of the introduction.

## Verbatim quotes

1. "CoT monitoring can be far more effective than monitoring agent actions and outputs alone"
2. "a LLM weaker than o3-mini, namely GPT-4o, can effectively monitor a stronger model"
3. "agents learn obfuscated reward hacking"
4. "pay a monitorability tax"

## Relevance to the commons overseer

Two points for an LLM overseer: an overseer that reads the agents' reasoning can catch more than one that sees only actions, and a weaker model can monitor a stronger one. The warning: optimising agents against the monitor teaches them to hide intent. In the commons, compare an overseer that sees actions only, actions plus messages, and actions plus reasoning, and never train or select agents against the same overseer that is used for evaluation.
