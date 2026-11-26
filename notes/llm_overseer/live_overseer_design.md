# Design note: a live LLM overseer that chooses whom to check (22 Nov 2026; no code, nothing run)

**Why.** The ladder (`protocol.md`) measures an LLM only as an *inspector*: it is handed a record and says yes or no. The overseer's other job is *choosing* whom to check (claim C3: targeting beats random only with a good signal). In Harvest, the targeting rules so far were hand-written: random, repeat offenders first, and "where apples are scarce", which was worse than random (finding 7). The question here is whether an LLM can choose better than these rules from short summaries, and at what token cost.

**Why not inside training.** One IPPO run is 1e8 steps; even one LLM call every 50 steps across 256 environments would be millions of calls. So the live overseer is evaluated against **frozen trained policies**, as an evaluation, not a training signal.

## Setting

- **Agents:** frozen IPPO policies from the existing runs (params on CREATE, `/scratch/users/k26108758/ippo_overseer/<run>/params.msgpack`). Two populations:
  - agents trained with no overseer (`A_none_seed*`): they over-harvest about 45% of their eats, so there is a lot to catch;
  - agents trained under random k = 2 at c = 1 (`B_random_k2_c1.0_seed*`): mostly honest, with occasional over-harvests, so the overseer has to find needles.
  - Because the policies are frozen, they cannot react to the LLM's choices. This measures *catching*, not deterrence.
- **Episode:** Harvest Open, 7 agents, 1,000 steps, the existing wrapper (`HarvestOverseer`), with the checks chosen from outside.
- **Every N steps** (N = 10, 50), the LLM gets a short summary of each agent and returns the k agents to check for the next N steps (k = 1, 2).

## What the LLM sees (short summaries)

Per agent, one line, built from information a real overseer would have:
- apples taken in the last N steps, and how many of those were taken where few apples were left (the summary rendering of the ladder, compressed to counts);
- how often it has been checked and caught so far (the repeat-offender score);
- its region of the map (for later decentralised variants).

About 7 lines × 20–30 tokens, plus the rule and instructions: about 400 tokens in, under 50 out (a list of agent numbers), so 20–100 calls per episode at N = 50–10.

Variants of the information (the C4 dial):
1. counts only;
2. counts plus caught history;
3. raw: the agent's last N moves with the local map at each eat (expensive).

## Compared against

Each at the same k: random; round robin; repeat offenders first (the existing `history` chooser); "where apples are scarce" (`location`); and an oracle that checks the agents with the most violations pending (an upper bound on catching).

## Measures

- **Catch rate:** caught violations ÷ all violations, per episode;
- **η:** (LLM − random) ÷ (oracle − random);
- **tokens per episode and per caught violation;**
- whether the LLM's choices are better than random once the inspector's measured α, β are applied to the checks it chose (pair each chooser with a ladder inspector: a "choose with a big model, inspect with a small one" split is a natural cost saver).

## Predictions (untested)

- Against the no-overseer population, almost everyone offends, so every chooser is close to random (as in BazaarBench's exploit cell, finding 17).
- Against the trained population, an LLM given counts plus history beats random by 5–20 points of catch rate at k = 1, and roughly matches repeat offenders first; the raw variant adds little per token.
- Small models (0.5–1.5B) choose no better than random, often repeating the same agent.

## What it would take

- A rollout loop that runs JAX policies for N steps, pauses, calls Ollama, then continues (outside `jit`; about 1 second of JAX per 50 steps on a CPU).
- Loading the IPPO params and network from `experiments/socialjax/train_ippo_overseer.py` without changing it.
- Budget: at N = 50, 20 calls per episode × 20 episodes × 2 populations × 2 k × 3 information variants = 4,800 calls per model, about 2.4M tokens. That is fine locally, but above the cloud cap, so the cloud tier would need N = 100 or fewer episodes.
