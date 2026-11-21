# Week 2 with scripted bots: the rule's threshold, Schelling diagrams, checks × consequence

Written 15 Nov 2026 before the runs. Code: `experiments/socialjax/run_week2_bots.py`, figures in `make_week2_figures.py`.

Setting: SocialJax Harvest Open, 7 agents, individual rewards, 999 steps. The overseer wrapper is `overseer/socialjax/`, each check sees only the current step (window 1), and the inspector is perfect. "Apples" means apples eaten (one apple pays 7 reward). A penalty is given in apples.

Bots are scripted and do not learn. A greedy bot never reacts to a penalty; a freeze holds it still.

## Part A: choose m (the rule)

- **What runs:** 7 restrained bots, no overseer, m = 1–6, 10 seeds each.
- **Decision rule, fixed now:** pick the m with the most apples eaten among those where the orchard is alive at the end in 10/10 seeds. On a tie (within 2%), take the smaller m, which is the less restrictive rule.
- **Reference:** all-greedy (which does not depend on m).

## Part B: Schelling diagrams at the chosen m

- **What runs:** l = 0–7 restrained (cooperators), the rest greedy (defectors), 10 seeds each.
- **Plotted:** R_c(l) and R_d(l), each agent's net apples (eaten minus penalties) averaged over agents of that kind, with 95% bootstrap intervals over seeds.
- **Conditions:**
  1. no overseer;
  2. k = 2 random checks with a 50-step freeze;
  3. k = 2 random checks with a 5-apple penalty and no freeze.
- **It is a dilemma when** R_c(7) > R_d(0), R_c(7) > R_c(0+) (a lone cooperator is worse off than when all cooperate), and fear or greed holds (R_d(l) > R_c(l+1) for some l).

## Part C: checks × consequence (random checks; history targeting as a second panel)

- **C1, outcomes under a freeze:**
  - k = 0–7 × freeze = 0, 10, 25, 50, 100, 200 steps;
  - populations: 1 greedy + 6 restrained, and 7 sneaky;
  - 5 seeds each;
  - plotted: share of seeds with the orchard alive, and apples eaten.
- **C2, does over-harvesting pay under a penalty (no freeze):**
  - k = 0–7 for 1 greedy + 6 restrained, and for 7 restrained, 10 seeds each;
  - behaviour does not depend on the penalty, so each penalty in {0, 1, 2, 5, 10, 20} apples per catch is applied to the same episodes;
  - plotted: the greedy agent's net apples minus a restrained agent's apples when all 7 restrain (R_d(6) − R_c(7)), and the break-even penalty per catch at each k.

This is a preview for designing the learning-agent runs, not a result for the paper.

## Amendment 1 (15 Nov, after parts A–C1, before any C2 figure)

- **What happened:** with no overseer a lone greedy bot eats 30 apples, while every agent eats 71.5 when all restrain. Over 999 steps it strips the orchard and starves itself, so R_d(6) − R_c(7) is negative at every k and penalty, and the planned break-even penalty does not exist.
- **What changes:** C2 also reports the comparison in Leibo et al.'s greed condition, the greedy agent against the restrained agents in the same group, R_d(6) − R_c(6). It reports the break-even penalty on that comparison.
- **What stays:** the original R_d(6) − R_c(7) is still reported. C2 reruns with the extra column; nothing else changes.
