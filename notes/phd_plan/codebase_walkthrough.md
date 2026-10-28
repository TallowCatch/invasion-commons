# My codebase, explained so I can talk about it

Written 28 October 2026, from the code on branch `claude/r1-s1-oversight-experiments`.

## 30-second version

> The repo has four layers. **Games** are shared resources that regrow. **Agents** play them: scripted bots, LLMs or RL. An **overseer** sets allowances, checks a few agents and punishes cheaters. **Experiments** run one study each, with a protocol written before the code. The overseer is a separate package that plugs into any game through an adapter. Right now it runs on my own text games and on Richard Willis's code. SocialJax Harvest is next.

## The map

| Folder | What it is | Say it like this |
|---|---|---|
| `fishery_sim/` | My games and agents (27 files) | "the game engines" |
| `overseer/` | The overseer package (14 files) | "the oversight layer — the thing I'm actually studying" |
| `external/` | Other people's code, pinned: `emergent_llm` (Richard's), `GovSim` | "I run on the field's own code, unchanged" |
| `experiments/oversight/` | One `run_X.py` and one `analyze_X.py` per study | "every study is a script plus an analysis" |
| `notes/claude_audit_20261005/studies/` | One folder per study: `protocol.md` (frozen first), then `results.md` | "pre-registered, with every failed prediction reported" |
| `tests/` | 43 test files, about 490 tests | "everything is tested, including exact replay of every stored LLM game" |
| `paper/paper_v6/` | The current, paused draft. v2–v5 are older. | — |
| `results/` | 2.5 GB of raw runs, not in git | — |
| GitHub branches `l2-results`, `r1a-results`, `r1b-results` | Results of the remote runs | "the LLM runs ran on GitHub Actions, so my laptop could close" |

## 1. The games (`fishery_sim/`)

| Paper name | File | What it is |
|---|---|---|
| **Fishery** | `env.py` | One shared fish stock that regrows logistically. Six agents each take up to 6 t per round. |
| **Forest** | `harvest.py` | Six plots in a ring. Over-cutting your own plot damages your neighbours'. Weather noise. |
| **River** | `two_reagent.py` | Three agents discharge X and three discharge Y. Harm comes from the two together. |

⚠️ **Naming trap.** The file called `harvest.py` is **my own Harvest game**, which the paper calls **Forest**. It is **not** SocialJax's Commons Harvest. Say "Forest" when talking. Say "SocialJax Harvest" for the real one.

Older parts from Act 1 (February–September), still there:
- `evolution.py` and `harvest_evolution.py`: strategies that do badly get replaced, so better ones "invade". This is where the repo's name comes from.
- `fishery_rl.py` and `harvest_rl.py`: RL agents (PPO) for both games. These were checks, not the main result.
- `llm_adapter.py` and `harvest_llm_population.py`: LLMs write strategies as JSON (the Qwen and Llama banks).
- `cleanup_oversight.py`: a sidecar for SocialJax Clean Up. It exists, but never produced a usable test.

## 2. The agents

- **Scripted bots** (`agents.py`, plus the agent classes in `harvest.py`): threshold rules. Some are greedy: they ask for a lot, lie about it, or over-take. In some studies a search picks each cheater's best level of cheating. These are like Melting Pot's background bots.
- **LLM players** (`llm_actor.py` and `experiments/oversight/run_l2_llm_agents.py`). This is how one game works:
  1. Each model gets the rules: the lake, the 1-in-6 chance of a check, and the fine. It then answers a 3-question quiz on them.
  2. **Each round** it sees the stock and its own last 3 rounds, and replies with a request in JSON.
  3. The reviewer gives it an allowance, and the model replies with its actual catch.
  4. Four of the six fishers are LLMs, and two follow fixed rules.
  5. The models run through Ollama Cloud, and every prompt and reply is logged.
- **RL agents:** PPO exists (`s5_ppo.py`, and the `*_rl.py` files above), but no October result depends on it. This is the gap the SocialJax work fills.

## 3. The overseer

**The old way, used by the October studies:** the reviewer logic lives inside each game.
- `calibrated_oversight.py`: the reviewer picks one shared cut (full, ¾, ½, ¼ or nothing). It takes the largest cut that still keeps the stock at half capacity.
- `oversight_protocol.py` and `budgeted_oversight.py`: the September "fair reviewer" tests.

**The new way:** the `overseer/` package, built 14 October. It is one overseer for every game. Each **window** (a round, or K steps in a grid game) it does one cycle:

```
see the state → set allowances → agents act → record who took what → choose whom to check → detect → punish → remember
```

Each step is a small, swappable part:

| Part | File | Options |
|---|---|---|
| Allowance | `allowance.py` | no cap; a shared cut; a targeted cut |
| Who to check | `audit.py` | random with chance q; a fixed number; targeted |
| What a check finds | `detect.py` | perfect; noisy (misses some) |
| Punishment | `sanction.py` | fine; memory (lower future allowances); exclusion; a leniency stub |
| Budget | `budget.py` | counts checks used, as a % of agent-rounds |
| Scores | `metrics.py` | efficiency, oversight efficiency, break-even check cost, safety, usefulness |

**How it connects to a game:** through an **adapter** in `overseer/adapters/`. The overseer never imports a game.
- `fishery_text.py`: my text Fishery.
- `emergent_llm.py`: Richard's game, plus "cultural evolution" (strategies that earn more get copied). This ran R1a and R1b.
- **Not built yet:** a SocialJax Harvest adapter, a JAX wrapper that counts each agent's apples per window. This is step 3 of the gameplan.

**Why trust it:** it replays every stored LLM game exactly (L2 402/402, L3 80/80, L4 263/263, L5 10/10), and it re-runs an old study byte for byte. See `ADMISSION.md`.

## 4. Experiments (`experiments/oversight/`)

There is one pair of files per study, for example `run_s3_threshold_timing_memory.py` and `analyze_s3.py`. Some groupings:
- **Scripted-agent studies:** R1–R5, S1–S5, C1 and T1. They run locally in seconds and cost nothing.
- **LLM studies:** L1–L6. `l4_ci.py` runs them on GitHub Actions in three parallel "lanes".
- **Checks on my own games (Phase 0):** G1 (Schelling diagrams), G2 (setting sweeps) and W1 (welfare and cost re-scoring).
- **Richard's code:** R1a (`run_r1a_willis_audits.py`) and R1b (`run_r1b_generate.py` and `run_r1b_parts.py`).
- **Figures:** `make_paper_exhibits.py` draws every paper figure from saved tables only.

Other folders: `common/` holds shared code; `paper_v5/` holds the September paper's scripts; `archive/` holds Act 1.

Run anything from the repo root as `python -m experiments.oversight.<name>`.

## 5. How a study goes from idea to result

1. Write `studies/<ID>/protocol.md`: the question, predictions, sample size and stop rule. Commit it **before** any code.
2. Write `run_<id>.py` and its tests.
3. Run it: locally if it's scripted, on GitHub Actions if it uses LLMs or Richard's code. Results land on a results branch.
4. `analyze_<id>.py` uses paired bootstrap intervals over populations.
5. Write `results.md`. If it changes what the paper must say, add an F-numbered entry to `FINDINGS_AND_IMPLICATIONS.md`. Update `PROJECT_HISTORY.md`.
6. Any change after seeing results goes in as a dated amendment, never as an edit to the protocol.

## 6. Honest caveats, so nobody catches me out

- **The top-level README is out of date.** It still describes the September paper (v5) and calls Harvest the main game.
- **Most of the code is history.** Act 1's evolution, RL and Clean Up code is still there. The live parts are `overseer/`, `experiments/oversight/` and the three game files.
- **No SocialJax Harvest code exists yet.** The SocialJax code in the repo is the Clean Up sidecar from September.
- **The LLM players were told they could over-take.** The prompt says so explicitly (`rules_text`).
- **The workflows that ran the LLM studies and R1a/R1b live on other branches** (`r1a-workflow`, `r1b-workflows`), not on this one.

## 7. Questions people might ask

- **"Is the overseer specific to your game?"** No. It's one package with an adapter per game, and it already runs on two different codebases.
- **"How do you know the numbers are right?"** Exact replay of every stored game, tests on every study, and protocols frozen before the code.
- **"Where are the RL agents?"** PPO exists for my games but wasn't central. Proper RL against the overseer is the next step, in SocialJax.
- **"Can I rerun it?"** Scripted studies run in seconds with `python -m ...`. LLM studies need an Ollama key and run on GitHub Actions.
