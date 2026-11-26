# `overseer`: an oversight layer that plugs into any commons game

Design v0, 14 October 2026. Background: `notes/claude_audit_20261005/research/commons_platform/2026-10-14_decision.md` §5.

## Purpose

One implementation of the overseer (reviewer), with the same behaviour whichever game it runs on:
- the text Fishery, Forest and River;
- Willis et al.'s `emergent_llm` CPR;
- SocialJax Commons Harvest (RL agents);
- OpenSocialJax (LLM agents).

The overseer is the research object. Games are reached only through an **adapter**. Every oversight protocol is a composition of small parts, so studies change one part at a time.

## The oversight cycle

The overseer acts once per **window**: one round in the text games, K environment steps in a grid.

```
           ┌────────────── window w ──────────────┐
observe ──► set allowances ──► agents act ──► ledger ──► choose audits ──► detect ──► sanction ──► update memory
 (stock,      (AllowancePolicy)               (per-agent   (AuditPolicy,     (Detector)  (Sanction)    (Memory)
  requests,                                    take, report, under Budget)
  messages)                                    position)
```

### Data passed between the layer and the game

- `WindowObs`: window index, resource state (for example stock, per-patch health), per-agent requests (if the game has them), per-agent positions (spatial games), messages (if a talk channel exists), and the game's `ResourceModel` (below).
- `Allowances`: a per-agent cap for this window (or `None` = no cap), plus optional text for the agents (rules or announcements).
- `Ledger`: per-agent amount taken, amount reported (if reporting exists), and allowance, for the window.
- `AuditPlan`: which agents are audited (boolean per agent), and the budget used.
- `Findings`: per audited agent, whether over-taking was detected, and its size.
- `Sanctions`: per-agent score change (fine), allowance changes for future windows (memory), exclusion flags.

### Components (each a small class with one method)

| Component | Implementations, v0 first | Literature |
|---|---|---|
| `AllowancePolicy` | `NoCap`; `SharedCut` (largest λ in {1, ¾, ½, ¼, 0} with residual ≥ target; the current reviewer); `TargetedCut` (S5's `memory_cut`: compliant agents' cut ignores known cheaters); `LLMAllowance` (an LLM chooses λ; L9a) | current paper; [25, 27] |
| `AuditPolicy` | `UniformRandom(q)` (current); `FixedCount(k)`; `Targeted(score)` (top-k by suspicion; C2); `Proximity(radius)` (spatial; S4); `Adaptive` (per-agent credence; C8) | [80, 83, 95] |
| `Budget` | audits per window or per episode; tracks % of agent-rounds and cost c per audit | [80, 94] |
| `Detector` | `Perfect` (current); `Noisy(miss, false_alarm)`; `LLMMonitor` (reads ledger, reports and messages; C2, C5, C9) | [84, 85] |
| `Sanction` | `Fine(F)` (current); `Memory` ("share" and "excess" rules, current); `Exclude`; `Leniency` (reporter's fine waived; L7); `Graduated` | Becker; Ostrom; Motta–Polo |
| `Memory` | per-agent record of caught excess, read by `AllowancePolicy` | current S3/S4 |
| `Brain` | `Rule` (deterministic composition) or `LLM` (one call per window returns allowances and/or audit targets as JSON; it is validated, and any invalid output falls back to the rule) | L9; [80] |

An `Overseer` is a `Brain` plus the components above. It has two calls:

```python
decision = overseer.begin_window(obs)          # -> Allowances (+ rule text)
outcome  = overseer.end_window(obs, ledger)    # -> AuditPlan, Findings, Sanctions; updates Memory and Budget
```

### `ResourceModel`, supplied by the adapter

- `safe_total(state) -> float`: the largest total take that keeps the game's target. Fishery: residual ≥ K/2 under logistic regrowth. Forest: the chance constraint. River: the quality line.
- `harm(state_after) -> bool`: the paper's harm line.
- `first_best_per_window(state) -> float`: used for efficiency.

The overseer never imports a game. Games never import overseer internals, only `Overseer`, the data classes and `metrics`.

### Determinism

- All randomness comes from `numpy.random.default_rng(stable_seed(...))` keyed by (study seed, context, window, agent), as now.
- **"One uniform per agent and window"** is kept, so audits at a smaller q are a subset of those at a larger q (L4).
- An LLM brain logs every prompt and reply, with seeds, like the current `calls.jsonl`.

### `metrics` (shared by all studies)

Implements W1's definitions:
- efficiency (two welfare counts);
- break-even audit cost;
- oversight efficiency (OE, OE_net);
- safety per round and per episode;
- usefulness;
- harm gap recovered;
- audit budget %.

## Adapters (in order of building)

1. **`adapters/fishery_text`.** Wraps the existing Fishery (`fishery_sim` and the L2/L4 loop). **Admission test:** replay every stored L4 episode (`results/runs/l2_results_branch/claude_l4_v1`). Feed the logged requests and catches to the layer, and assert that allowances, scale, audited, caught and fines match the log **exactly**. Also re-run S3 Part A with programmed agents through the layer, and assert identical summary numbers.
2. **`adapters/harvest_ring`** (Forest) and **`adapters/river`.** Same admission idea, against L6 and C1/R4 logs.
3. **`adapters/emergent_llm`.** The Willis CPR: C/D actions mapped to take = S/(2n) or S/n. Allowance = the C take. Audits detect D. Fines change payoffs only; the stock update must use the harvest, not the net payoff. The last round is reported separately (decision report §4 trap).
4. **`adapters/socialjax`.** A JAX wrapper holding the overseer state (per-agent apple counter per window, allowance, fines). Rule brain only inside `jit`. Allowance added to observations. Individual rewards forced; the ×7 scaling noted.
5. **`adapters/opensocialjax`.** A hook into the Python episode loop: a score-changing callback, an overseer block in the prompt, hidden rules off (`reveal_rules`), and high-level moves every K steps.

## Not in v0

Talk channels, LLM brains, and spatial audits. They arrive with the studies that need them: L7, L9, C2, S4. The interfaces above already have fields for them (messages, positions), so adding them does not change the API.

## Package layout

```
overseer/
  __init__.py       # Overseer, data classes
  core.py           # WindowObs, Allowances, Ledger, AuditPlan, Findings, Sanctions, Overseer
  allowance.py      # NoCap, SharedCut, TargetedCut
  audit.py          # UniformRandom, FixedCount, Targeted
  detect.py         # Perfect, Noisy
  sanction.py       # Fine, Memory, Exclude, Leniency
  budget.py
  metrics.py
  seeding.py        # stable_seed (moved from the runners, same function)
  adapters/
    fishery_text.py
tests/test_overseer_*.py
```
