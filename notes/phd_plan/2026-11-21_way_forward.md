# Way forward: one overseer, one way of judging it, several testbeds (21 Nov 2026)

**The aim in one sentence.** Show that an overseer that checks only a few agents at a time works across different games, say in advance when it will and won't work, and judge it the same way everywhere (the bounded evaluation).

The general definitions are in `2026-11-21_general_formulation.md`. Results so far are in `../OVERSEER_TRACKER.md`.

## 1. What each benchmark is for

| | SocialJax (spatial games) | BazaarBench (LLM marketplace) |
|---|---|---|
| Agents | RL learners that change their behaviour in response to the overseer | LLM traders, the kind of agents being deployed now |
| Why use it | cheap to run thousands of settings; several games with different kinds of cheating; agents really adapt | real misbehaviour (selling items you don't have, lying about condition, leaking personal data); checking has a real token cost |
| What it can't do | real costs; LLM agents (OpenSocialJax could add those later) | in the released data the traders never react to an overseer (that needs a live version) |
| Role | the main testbed: does the overseer work against agents that adapt? | the reality check: do the same rules and numbers hold for LLM agents and real costs? |

## 2. The bounded evaluation: one scorecard for every testbed

Every overseer is reported the same way. Each piece comes from a paper.

| Piece | What we measure | Where it comes from |
|---|---|---|
| **Bottom** | no overseer | AI Control's "don't use the risky model / no protocol" reference points (Greenblatt et al. 2024) |
| **Top** | check every agent every step with a perfect inspector. In BazaarBench offline, the full GPT-5 judge run on everyone, which the authors already paid for | same |
| **Score** | η, the share of the gap from bottom to top that this overseer recovers | our W1 "oversight efficiency"; AI Control's safety/usefulness |
| **Cost** | checks used, and history read per check; tokens in BazaarBench | Monitored MDPs (Parisi et al. 2024): monitoring has a cost; BazaarBench's judge cost |
| **Harm to honest agents** | wrongful penalties and honest work lost (the "usefulness" side) | AI Control's usefulness; the false-alarm prediction C2 |
| **Weaker overseer** | η as one thing gets worse: fewer checks (k), a shorter look (w), mistakes (misses α, false alarms β), a smaller model | Engels et al. 2025: oversight against the capability gap |
| **Worst case** | η against agents that can see or predict the checks and are trained or told to dodge them | AI Control's red team; Wang et al. 2019's train-the-attacker-then-the-defender loop |

**The key link between the two testbeds:** a cheap overseer is simply one that makes more mistakes. In BazaarBench we can measure how often a cheap judge (a small open model) misses or wrongly flags, against the GPT-5 labels. That gives real α and β at a real token price. We then plug those α and β into Harvest and see whether the overseer still holds when agents adapt. That turns "a smaller, less capable overseer" into one number that carries across testbeds.

## 3. The spatial side: which SocialJax games, and why

Pick games by **how easy the cheating is to see in one check**, because the general claims predict that this decides whether a few checks can work (claim C4; Gans & Holden 2026).

| Game | What counts as cheating | Visible in one check? | What it tests |
|---|---|---|---|
| Harvest (done) | taking an apple from a thin patch | yes | claims C1–C3 |
| Coins (`coin_game`) | picking up another agent's coin | yes | the same claims in a game with no shared resource to collapse: does the average-penalty rule still decide? |
| Clean Up (`cleanup`) | not cleaning the river (free-riding builds up over time) | no, only over many steps | claim C4, where it should fail: does it need a longer look (more cost), and does that make it work again? |

Doing these three, with the same overseer code and the same scorecard, is the spatial version of what DMPO did across traffic, power and pandemic networks (Du et al. 2022; Ma et al. 2024).

## 4. The BazaarBench side, in two steps

1. **Offline replay (now, no model cost).**
   - Download the judge labels for a few markets (about 100–150 MB each).
   - Replay an overseer that checks k of the 20 tested traders each simulated day.
   - Compare:
     - random checks;
     - repeat offenders first (Ye & Steinhardt 2026 found that tracking each agent's reliability is what makes enforcement robust);
     - cheap record checks (selling unowned items, overstating condition, double-selling) against expensive judge checks (messages, personal data, fake reputation).
   - Measure the share of failures caught, how early they're caught (before or after harm reaches a buyer), and the cost.
2. **Live version (later).**
   - Put the overseer inside the simulator with consequences the traders can see.
   - Tell some traders about the audits.
   - Compare budgets of 1 and 10 units.
   - This needs model budget, or open models on CREATE; the paper's judge alone cost about $6,500 for 48 runs.

## 5. Literature behind each part

| Part | Papers |
|---|---|
| What a social dilemma is, and Harvest | Leibo et al. 2017; SocialJax (Guo et al. 2026) |
| A general problem class plus many testbeds | Du et al. 2022 (DMPO); Ma, Li, Du et al. 2024 (Nature Machine Intelligence) |
| Monitoring with a cost, as a general framework | Parisi et al. 2024 (Monitored MDPs); inspection games in game theory |
| Bottom, top and worst case | Greenblatt et al. 2024 (AI Control) |
| A weaker overseer | Engels et al. 2025 |
| When random audits deter, and when hidden evidence breaks them | Gans & Holden 2026 |
| Keeping checks unpredictable | Zhu et al. 2026 (Blindfold); Fishery T1 |
| Many agents splitting an attack | Makins et al. 2026 |
| Where to put a limited audit budget; switching an overseer on for misbehaving learners | Casheekar et al. 2026 |
| Switching the overseer off | Baumann et al. 2020; our R1b (F16) and stage 3 (F18) |
| Reliability tracking and escalating penalties | Ye & Steinhardt 2026 |
| Red-team training loop | Wang et al. 2019 (double oracle) |
| LLM agents in a marketplace | Wang, Shi, …, Du, Bibi 2026 (BazaarBench) |

## 6. Steps

**By Monday 24 Nov**
- **Mistakes test in Harvest** (running) → do misses and false alarms behave as the general claims say?
- **Finish stages 2 and 3** (rerunning).
- **BazaarBench offline replay** on 2 markets (needs Ameer's OK for about 250–300 MB).
- **Ameer:** read Parisi, Gans & Holden, and Ye & Steinhardt; correct the general formulation.

**Next 2 weeks**
- **Coins** with the same overseer and scorecard (the code needs a cheating rule for Coins; small).
- **BazaarBench cheap judge.** A small open model on CREATE judges a sample of the same evidence. That measures its α and β against GPT-5, which then go into Harvest.
- **Harvest worst case.** Agents that can see who was checked last step; random against repeat offenders against a randomised mix.

**After that**
- **Clean Up**, the hard-to-see case, with longer looks.
- **A learned overseer:** an RL policy that picks whom to check, trained in turns against the agents.
- **BazaarBench live**, with a token budget.
