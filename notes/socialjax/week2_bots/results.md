# Week 2 with scripted bots: results

**Setup.** Run on 15 Nov 2026 under `protocol.md` (with Amendment 1).
- Run: `results/runs/socialjax_week2_bots/week2.json`.
- Summary: `summary.json`.
- Figures: `fig_rule_and_schelling.png`, `fig_checks_by_freeze.png`, `fig_penalty.png`.
- Code: `experiments/socialjax/run_week2_bots.py`, `make_week2_figures.py`.

The bots are scripted and do not learn, so this is a preview to design the learning-agent runs, not a paper result. Intervals are 95% bootstrap intervals over seeds.

## Part A: the rule is m = 3

Seven restrained bots, no overseer, 10 seeds each (fig. (a)):

| m | Apples eaten | Orchard alive (of 10) |
|---|---|---|
| 1 | 94 | 10 |
| 2 | 190 | 10 |
| **3** | **500** | 10 |
| 4 | 393 | 10 |
| 5 | 377 | 10 |
| 6 | 85 | 10 |
| All greedy, for reference | 67 | 0 |

- **m = 3 wins by the decision rule.**
- It matches Harvest's own regrowth: an empty cell regrows fastest (2.5% per step) with 3 or more apples nearby.
- A looser rule (m = 1–2) thins the patches.
- A stricter one (m ≥ 4) leaves apples uneaten. At m = 6 the bots hardly eat at all.

## Part B: it is a dilemma, and one greedy agent wrecks it

Schelling diagrams at m = 3, 10 seeds per point. R is net apples per agent.

| Restrained agents (of 7) | No overseer: R_c / R_d | 2 random checks, freeze 50 | 2 random checks, 5-apple penalty |
|---|---|---|---|
| 0 | – / 9.6 | – / 10.1 | – / 2.1 |
| 1 | 10.8 / 9.5 | 18.0 / 9.7 | 10.8 / 1.6 |
| 3 | 10.3 / 10.0 | 17.7 / 10.1 | 10.3 / −1.0 |
| 5 | 8.7 / 16.4 | 17.4 / 16.9 | 8.7 / −5.0 |
| 6 | 9.2 / 30.1 | 24.8 / 33.9 | 9.2 / −8.9 |
| 7 | 71.5 / – | 71.5 / – | 71.0 / – |

**With no overseer it is a social dilemma, by the greed condition of Leibo et al. / Hughes et al.:**
- everyone restraining beats everyone greedy: 71.5 against 9.6;
- a lone restrained agent is worse off than when all restrain: 10.8 against 71.5;
- greed holds: when most restrain, the greedy agent earns more than the restrained ones (30.1 against 9.2 at 6 restrained);
- fear does not hold: with few restrained agents, both kinds earn about the same.

**Harvest Open is fragile.** The orchard survives only when all 7 restrain. With even one greedy bot it dies in 10/10 seeds, which matches the SocialJax paper.

**What the overseer does:**
- **A 5-apple penalty with 2 random checks** makes greed stop paying at every mix: the greedy agent nets less than the restrained ones. These bots ignore it, but learning agents would see it in their reward.
- **A 50-step freeze with 2 random checks** raises the restrained agents' share (the greedy one is held still part of the time). It does not save the orchard at k = 2, and greed still pays at 6 restrained (33.9 against 24.8).

**A finding the protocol did not expect (Amendment 1):** over 999 steps a lone greedy bot (30 apples) earns far less than it would by restraining when everyone else does (71.5). It strips the orchard and starves itself, so the switch to greed never pays over a whole episode. Greed pays only against the restrained agents in the same group. Learning agents discount the future (γ = 0.99 is roughly a 100-step horizon), so the short-term gain is what they will see. That is the reason they learn to over-harvest, and it is what phase 2 tests.

## Part C1: checks × freeze

5 seeds per cell; fig. (a)–(h).

**One greedy among six restrained:**
- **The freeze has to be long:** 50 steps or more. Nothing saves the orchard at 0–25 steps, except 1–2 seeds out of 5 at a 25-step freeze.
- **Random checks** need k = 6 at a 50-step freeze (k = 4–5 save it in 3–4 of 5), k = 4 at a 100-step freeze, or k = 2 at a 200-step freeze, to save it in 5/5.
- **Checking by catch history** saves it with **k = 2 at a 50-step freeze**, and k = 1 saves it in 3–4 of 5. Against a fixed cheater, targeting needs a third of the checks that random needs.

**Seven sneaky bots:**
- Any k ≥ 2 random checks, or k ≥ 1 by history, keeps the orchard alive whatever the freeze.
- This is built into the bot: it cheats only after going unchecked for 20 steps, so being watched is what stops it, not the consequence.
- With k = 1 random checks the gaps are long enough that it cheats and the orchard dies. History checks with k = 1 work as round-robin, since nobody is caught.

## Part C2: the penalty at which greed stops paying

Fig. (a), (c): the greedy agent against the restrained agents in the same group (1 greedy + 6 restrained), 10 seeds.

| Agents checked per step (k) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Break-even penalty per catch (apples) | 7.6 | 3.6 | 2.4 | 1.7 | 1.3 | 1.1 | 1.0 |
| k × break-even | 7.6 | 7.2 | 7.2 | 6.8 | 6.5 | 6.8 | 6.8 |

- **Fewer checks need a proportionally bigger penalty:** k × penalty stays at about 7 apples. With a greedy agent that cannot dodge, deterrence depends on the expected penalty per over-harvest (chance of a check × penalty), not on either alone.
- This is the same pattern as the Fishery LLM study (L4: "the expected fine decides").
- Against "everyone restrains" (fig. (b)) greed never pays, as explained above.

## What this means for the next step (IPPO on CREATE)

1. **Rule and settings:** use m = 3, individual rewards, window 1.
2. **Consequence:** a reward penalty is what a learning agent responds to. Sweep k × penalty around k × penalty ≈ 7 apples (49 reward), which is the line where greed stops paying against fixed greedy behaviour.
3. **Freeze vs penalty:** compare them. A freeze also protects the orchard directly but needs 50+ steps.
4. **Targeting:** expect it to help against agents that over-harvest openly (at a 50-step freeze, history needs k = 2 where random needs k = 6) and to be exploitable by agents that learn when they are unwatched. That is the phase 4 red-team test.
5. **Caveat:** these bots never respond to penalties and see the whole map; the learning agents will do neither.
