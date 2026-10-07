# Protocol L2 (DRAFT): do language-model agents follow the same audit logic?

**Status: draft, not frozen.** It is frozen, with a timestamp, once the
decisions below have been made. Nothing has been run.

L2 is the last experiment for the paper (claim 6). It is the main study that
the L1 pilot proposed (`studies/L1_llm_actor_pilot/results.md`). It adds
nothing outside claim 6.

## Decisions needed before freezing (Ameer, with Yali and Edward)

| Decision | Options | Why it matters |
| --- | --- | --- |
| Compute | One month of Ollama Pro (about $20), or about $6–10 through an API, or KCL GPUs | L2 needs about 20 M tokens, beyond the free plan that L1 used |
| Second model family | gemma4 on Ollama Cloud, or Llama 3.3 70B through OpenRouter | Claim 6 needs at least two families. gpt-oss:120b is the first. |
| Who runs it | The run uses Ameer's Ollama account on his Mac, as L1 did. A key or app sign-in is needed on the machine that runs it. | No key is configured in this environment |

## Question

LLM fishers in Fishery face random audits and fines. Do they:
1. stop over-taking once the expected fine exceeds the gain from over-taking
   (P1);
2. keep over-taking under memory without fines (P4)?

## Design

Fishery, MSY target, 20 steps. Four of the six fishers are LLM agents; two
follow fixed rules, as in L1. The interface is L1's v3 (tonnes throughout;
the catch may be 0–6 t). Audit rate q = 1/6.

| Cell | Rules wording | Fine F (t) | Expected fine e = F/6 (t) |
| --- | --- | --- | --- |
| E0–E36 | explicit ("you can choose to catch more than your allowance") | 0, 1, 2, 4, 8, 36 | 0 to 6 |
| EM | explicit | 0, with memory (a caught agent's allowance is reduced as in S4) | 0 |
| S0, S36 | silent (L1's control wording) | 0 and 36 | 0 and 6 |

- **Size:** 10 contexts per cell, 9 cells, 2 models, so 180 episodes.
- **Cost:** about 113,000 tokens per episode, so about 20 M tokens.
- **Controls carried over from L1:** a shuffled answer order in the
  comprehension check, and 2 paraphrases of the rules (in E0 and E36 only).

## Hypotheses (to be fixed when frozen)

- **L2-H1 (P1 for LLMs).** Over-taking is lower at e ≥ g than at e < g,
  where g is the gain per over-take step measured in E0.
- **L2-H2.** Over-taking falls as e rises.
- **L2-H3 (P4).** Under EM, over-taking stays above S0, but harm is lower
  than in E0.
- **L2-H4 (replication of L1).** S0 shows almost no over-taking (under 5%).

## Gates and stop rules (to be fixed when frozen)

- A pilot gate per model: 1 context per cell; at least 95% valid answers.
- No prompt changes after the pilot.
- One full run per model.
- A token cap per model, set from the budget decision.
