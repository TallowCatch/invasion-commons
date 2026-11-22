# Clean Up with central and decentralised overseers: scripted-bot results

**Setup.** Run on 22 Nov 2026 under `protocol.md`, which was written and committed before the sweep.
- Raw runs (git-ignored): `results/runs/socialjax_decentralised/cleanup.json`.
- Figure: `fig_decentralised.png`.
- Code: `experiments/socialjax/run_decentralised_bots.py --game cleanup`.

The sweep took 6 min on a laptop CPU.

The bots are scripted and do not learn, so this is a preview, not a paper result. Each cell is 10 seeds.

**The rule:** don't eat an apple while pollution is above 0.2 unless you cleaned in the last W = 100 steps.

**The settings:**
- A caught agent is frozen for 50 steps.
- A check of w steps of record costs 1 + 0.01 w units.
- The main sweep uses w = 150 steps.
- "Caught" pools violations and catches over seeds.

## The short version

1. **Clean Up is a dilemma, and as fragile as Harvest.**
   - 7 honest bots eat 908 apples a game, and pollution averages 0.34.
   - One free-rider among 6 honest bots drops this to 374. The free-rider eats the most (85, against 48 for each honest bot), and pollution rises to 0.51, above the 0.4 at which apples stop growing.
   - 7 sneaky bots with nobody watching (they free-ride from step 20): 197 apples, pollution 0.77.
2. **A check has to read the rule's whole window.** With proof only, at 2 random checks per step:
   - looks of 1 or 50 steps catch 0% of free-rides;
   - 100 steps: 32%;
   - 150 or 200 steps: 100%.

   This is the "hard to see in one check" case (claim C4): the cost of a look rises with its length, from 1.01 to 3 units per check.
3. **Presuming guilt on a short look catches free-riders but wrecks the commons.**
   - With a 10-step look and "flag any eating with no cleaning in view": 98% of free-rides caught.
   - But 45 flags a game land on honest agents. Freezing them stops their cleaning, pollution rises to 0.55, and 7 honest bots eat 196 apples instead of 908.
   - Even 12 wrong flags a game (a 50-step look) cut it to 233.
   - **Clean Up sits at a tipping point,** so a few wrongly frozen cleaners are enough.
4. **The overseer can stop free-riders eating, not make them clean.**
   - With a 150-step look, the free-rider's apples fall from 85 to 14.
   - But total apples stay about the same (374 → 343), because the river is still as dirty (0.51 → 0.50).
5. **Shirking within the rule.**
   - Sneaky and migrator bots free-ride in short spells whenever they have gone 20 steps unchecked. They almost never break the rule, which needs 100 steps without cleaning, but they skip cleaning.
   - Under 1 random check per step, 7 sneaky bots eat 680 apples against 908 for 7 honest bots, with 0 violations.
   - The rule doesn't see this harm.
6. **Decentralisation matters less here, because a 150-step look catches every violation anyway.**
   - What it changes is how much shirking migrators get away with: under 1 random check per step, 517 apples split vs 680 central. Under repeat-offenders-first checks: 693 vs 908.

## How long a look has to be

1 greedy + 6 honest, 2 random checks per step, one overseer. Fig. (a)–(c).

| Look (steps of record) | Free-rides caught | Free-rides per game | Wrong flags per game | Apples (all 7) | Cost per check |
|---|---|---|---|---|---|
| no overseer | 0% | 73 | 0 | 374 | – |
| 1, proof only | 0% | 73 | 0 | 374 | 1.01 |
| 50, proof only | 0% | 73 | 0 | 374 | 1.5 |
| 100, proof only | 32% | 14 | 0 | 334 | 2 |
| 150, proof only | 100% | 11 | 0 | 343 | 2.5 |
| 200, proof only | 100% | 11 | 0 | 343 | 3 |
| 10, presume guilt | 98% | 24 | 35 | 130 | 1.1 |
| 50, presume guilt | 100% | 9.9 | 8.4 | 171 | 1.5 |
| 100, presume guilt | 100% | 11 | 0 | 343 | 2 |

With 7 honest bots:
- Presuming guilt on a 10-step look wrongly flags 45 honest agents a game and cuts apples from 908 to 196.
- On a 50-step look: 12 flags and 233 apples.
- On a 100-step look: 0 flags and 908 apples. A look of W steps sees an honest agent's last cleaning, because the honest bots clean every 70–100 steps.

Without presuming guilt, honest bots are never flagged.

## Who eats, with one free-rider

1 greedy + 6 honest, 150-step look. Apples per agent per game, and mean pollution. Fig. (d).

| Overseer | Free-rider | Honest (mean of 6) | All 7 | Pollution |
|---|---|---|---|---|
| none | 85 | 48 | 374 | 0.51 |
| central random, k = 1 | 18 | 51 | 328 | 0.51 |
| central random, k = 2 | 14 | 55 | 343 | 0.50 |
| central, repeat offenders first, k = 2 | 9 | 53 | 326 | 0.50 |
| 2 strips random, k = 2 | 15 | 55 | 346 | 0.51 |
| 2 strips, share suspicion, k = 2 | 10 | 60 | 372 | 0.49 |

Every condition here catches 99–100% of free-rides; unused checks with 2 strips are 7–17%.

## Shirking within the rule: sneaky bots and migrators

Apples eaten per game (7 honest: 908). Violations stayed between 0 and 1 a game in every overseer condition. Fig. (e)–(f).

| Overseer | 7 sneaky, k = 1 | 7 sneaky, k = 2 | 7 migrators, k = 1 | 7 migrators, k = 2 |
|---|---|---|---|---|
| none | 197 | – | 197 | – |
| central, random | 680 | 870 | 680 | 870 |
| 2 strips, random | 704 | 899 | 517 | 750 |
| central, repeat offenders first | 908 | 908 | 908 | 908 |
| 2 strips, independent | 886 | 908 | 693 | 757 |
| 2 strips, share suspicion + coverage | 891 | 908 | 730 | 706 |

- Repeat-offenders-first checks with nobody caught become round-robin. With 7 agents and 1 check a step, everyone is checked every 7 steps, so a sneaky bot never reaches 20 unchecked steps and never shirks.
- Split overseers lose that guarantee for migrators, who shirk in the strip that has not checked them.

## Predictions against results

| Prediction | Result |
|---|---|
| U1: 0% caught for w < 100, about 29% at w = 100, nearly all at w ≥ 150 | **Yes.** 0%, 32%, 100% |
| U2: presume at w = 10 catches every free-ride, puts more than 100 wrong flags a game on honest agents, and cuts apples below no overseer; fewer but above 0 at w = 100 | **Partly.** 98% caught, and apples collapse (908 → 196 with 7 honest). But it is 45 wrong flags, not over 100: a few wrong freezes are enough to tip the river. At w = 100 there were 0 wrong flags |
| U3: the overseer more than halves the free-rider's apples; total apples rise by less than 25% | **Yes.** 85 → 14; the total fell slightly (374 → 343) |
| U4: sneaky and migrators break the rule about 0 times; they still harm through skipped cleaning | **Yes.** 0–1 violations; 680 apples vs 908 for 7 honest (central random k = 1) |
| U5: decentralised random about equals central random; independent targeting catches less; sharing recovers it | **Not testable as planned.** A 150-step look catches 99–100% in every condition. The difference shows instead in migrators' shirking (517 vs 680 apples) |
| U6: no wrong flags without presuming | **Yes** |

## What this means

1. **Clean Up gives the general formulation its "detectability" property (C4) with a measurable knob.**
   - The rule's window is W; a check of w steps proves nothing when w < W.
   - The choices are to pay for longer looks or to presume guilt and punish honest cleaners.
   - In a commons at a tipping point, presuming guilt is worse than no overseer.
2. **A "clean once per W steps" rule misses intermittent shirking.** A learned agent can stay within it and still free-ride on most of the cleaning. A rule on the share of time spent cleaning, or a check of the cleaning *rate*, would be the next design to test. That is a rule-design question, separate from the overseer.
3. **Learning agents (GPU list):**
   - central random with a 150-step look at average penalties of 1 and 2 apples;
   - a 50-step look (cannot prove anything: a check that sees too little);
   - a 50-step look presuming guilt;
   - 2 strips (random, and repeat offenders first with sharing).
