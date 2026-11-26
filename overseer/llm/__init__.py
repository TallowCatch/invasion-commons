"""An LLM as the overseer's inspector (notes/llm_overseer/protocol.md).

Running an LLM inside 1e8-step RL is infeasible, so each model is measured offline as an inspector: given one
agent's record, does it say "violation" when there was one (1 - miss rate alpha) and "honest" when there was not
(1 - false-alarm rate beta), and how many tokens does one check cost? The measured alpha and beta then go into
Harvest RL through the trainer's --miss-rate / --false-alarm flags.

Modules (imports are kept separate so each step needs only its own dependencies):
- harvest_records: Harvest judging dataset from scripted bots (needs JAX + SocialJax);
- bazaar_records: BazaarBench judging set with the GPT-5 labels (standard library only);
- prompts: the rule text, the prompt for each example, and verdict parsing;
- judge: batch inference with vLLM (GPU) or transformers (CPU smoke test);
- metrics: alpha, beta, confusion tables, tokens per check, and the C1 prediction.
"""
