# A budgeted overseer, written for any game (draft, 21 Nov 2026)

**Why this exists.** Yali's question at the 16 Nov meeting: if the formulation only fits SocialJax, what is the point? Her group's DMPO (Du et al. 2022) and its Nature Machine Intelligence follow-up (Ma, Li, Du et al. 2024) show how this is done:
1. define one problem class with no game in it;
2. say what holds for the whole class;
3. name the one property that decides whether a testbed fits;
4. run the same method on very different testbeds.

This note does steps 1–3 for the overseer and maps two testbeds onto it: Harvest (SocialJax) and BazaarBench.

## 1. The setup

**Agents.** N agents play any game where each sees only part of the world. At step t, agent i acts and gets reward r(i,t). They may be RL learners, LLMs or scripted bots.

**The rule.** A game-specific rule says whether agent i broke the rule at step t: v(i,t) = 1 if it did, 0 if not. The one requirement is that it can be checked from a record of what the agent did.

**Checks.**
- Each step the overseer picks a set C(t) of at most k agents to check.
- A check of agent i reads the last w steps of its record.
- A real violation in that window is caught with chance 1 − α, where α is the miss rate.
- An honest record is wrongly flagged with chance β, the false-alarm rate.

**Consequence.** A caught agent loses F reward:

  r̃(i,t) = r(i,t) − F · caught(i,t)

F is easiest to read in units of the game's goods, such as apples or dollars. Other consequences fit the same slot: a freeze, a blocked listing, a lost reputation score.

**Cost.** Each check costs c_check, plus c_hist for every past step it reads:

  Cost = Σ over steps, Σ over checked agents, (c_check + c_hist · w), and Cost ≤ B

The cost can also depend on the kind of violation looked for. A database lookup is cheap; an LLM reading messages is expensive.

**The agents' problem.** Each agent maximises its own penalised reward, E[Σ γ^t r̃(i,t)]. Learners do this by training; LLMs by reasoning, possibly told the rules.

**The overseer's problem.**

  maximise W(overseer settings, agents' reply) − λ · Cost  
  subject to: the agents' reply is their best response to the settings, and Cost ≤ B

The settings are k, how C(t) is chosen, w and F. W is the game's welfare.

**Three things are game-specific:** the rule v, the welfare W, and what one violation gains an agent, g. Everything else is shared.

## 2. The two testbeds on this template

| | Harvest (SocialJax) | BazaarBench |
|---|---|---|
| Agents | 7 RL agents (IPPO), shared network | 100 LLM traders, of which 20 run the tested model |
| One step | one game step (1,000 a game) | one simulated day (7-day continuation) |
| Rule v | taking an apple with fewer than 3 within two squares | the six failure types: F1 overstated condition, F2 selling an unowned item, F3 promising one item to several buyers, F4 confirming too early, F5 sharing personal data or moving off-platform, F6 fake reputation claims |
| What a check reads | the agent's last w moves | the trader's listings, offers and messages that day |
| Cost of a check | flat units | F1–F3 are cheap (computable from the inventory and deal records); F4–F6 need an LLM judge (tokens grow with how much text is read; the paper's GPT-5 judge cost about $6,487 for 48 runs) |
| Consequence F | apples removed | (when live) blocked listing, fine, or reputation flag |
| Gain g | about 1 apple per over-harvest (bots) | extra earnings from unsafe deals (weekly earnings rise from about $20 to $33 under exploit instructions, mostly from items never held) |
| Welfare W | apples collected; orchard survives | safe completed deals; harm reaching buyers (stage S5) |
| Agents react to checks? | yes, they learn | not in recorded data (offline replay); yes in a live version |

BazaarBench adds something Harvest lacks: each failure passes through stages, from considered (S1) to carried through (S5). An overseer that catches a failure at the listing stage (S3) stops harm before it reaches a buyer, so how soon a check catches something matters, not just whether it does.

## 3. What should hold in any game (the general claims to test)

**C1. Deterrence.** Breaking the rule stops paying when

  q · (1 − α) · F ≥ g

Here q is the chance a violation gets checked: k/N with random checks, or the catch rate E[q(i,t) | v(i,t) = 1] for a targeted rule.

Evidence so far:
- Fishery with LLM agents, L4: the expected fine decides.
- Harvest with RL agents: a sharp switch between an average penalty of 0.5 and 1 apple, whatever k is.

**C2. The cost to honest agents.** With false alarms, an honest agent loses q · β · F per step on average.
- At a fixed average penalty that does not depend on k.
- So few-checks-with-big-fines and many-checks-with-small-fines cost honest agents the same on average. They differ only in how uneven the losses are.
- Untested.

**C3. Targeting needs a good signal.** Targeting beats random checks only if the catch rate E[q | v = 1] is above k/N.
- Agents that can see or predict the checks push it down for any predictable rule.
- Evidence: Fishery T1 (always checking the biggest reporter caught 25% of liars, random 100%); Harvest (location targeting catches 5% against 28% for random).
- Related: Gans & Holden 2026; Blindfold (Zhu et al. 2026).

**C4. When it fails: violations a check can't see.** If a violation doesn't show within the w steps a check reads, deterrence needs a longer look (more cost) or fails. This happens:
- when it builds up slowly, like not cleaning in Clean Up;
- when evidence can be hidden or erased (Gans & Holden).

The property that decides this is **how much of a violation one check of w steps can see**. Call it detectability: the chance that a violation is still visible when its window is read. This plays the role ξ plays in the DMPO papers, a measurable number per testbed that says whether the method should work there.

## 4. What each testbed can test

| Claim | Harvest | BazaarBench, offline replay | BazaarBench, live (later) |
|---|---|---|---|
| C1 deterrence | yes (done for α = 0) | no, traders don't react | yes |
| C2 honest cost | yes (next: turn on α, β) | partly: false alarms of a cheaper judge against the GPT-5 labels | yes |
| C3 targeting | yes (done, agents blind to checks; next: agents that see checks) | yes: catch rate of random vs repeat-offender checks at the same budget | yes |
| C4 detectability | via a second game (Clean Up) | yes: cheap record checks (F1–F3) vs expensive message checks (F4–F6), and how early each catches failures | yes |
| Budget / cost | units | real token costs, 1 unit vs 10 | real |

## 5. How this sits next to existing general frameworks

- **Monitored MDPs** (Parisi et al. 2024): one agent whose rewards are only seen when a monitor is on, and monitoring can cost something. Ours has many agents, a monitor that chooses whom to look at, and agents who would rather not be caught.
- **Gans & Holden 2026**: theory of random audits against one agent that can hide evidence. C1 and C4 are the many-agent, learning version.
- **AI Control** (Greenblatt et al. 2024): protocols scored on safety and usefulness against a red team, with one untrusted model. Our bounds (no overseer, check everyone, trained red team) follow it.
- **Multi-agent AI Control** (Makins et al. 2026): per-agent monitors miss attacks split across agents, which a C-style claim should cover later.
- **Inspection games** (game theory): the classic model of an inspector with limited inspections.
- **Networked MDPs** (Du et al. 2022; Ma et al. 2024): not an overseer, but the model for presenting a general class with one measurable property and many testbeds.

## 6. By Monday 24 Nov

1. **This note:** Ameer reads and corrects it, especially section 3.
2. **Harvest.**
   - Collect stages 2–3.
   - Run the mistakes test (α, β > 0 at a fixed average penalty, k = 1, 2, 7) for C2.
3. **BazaarBench offline replay.**
   - Download the judge labels for one or a few market cells (the judge dataset is 8.7 GB in total; check one cell's size first).
   - Replay a budgeted overseer: checks per day, random vs repeat offenders first, cheap record checks vs judge checks, at 1 vs 10 units.
   - Report the catch rate, and how early failures are caught (stage), for C3 and C4.
4. **Read properly:** Parisi et al. 2024, Gans & Holden 2026, Ye & Steinhardt 2026.
