# X6, X7, X10: what was built, and the predictions (written 26 Nov 2026, before any learner run)

The predictions themselves are in `../05_formulation.md` (§5.6 X10, §5.7 X6, §5.8 X7). This file records the
implementation choices made while building the code, and one extra X6 arm added before submission, with its reason.

## X6: headcount-proportional regional budgets

**Option.** `--region-alloc proportional` (`DecConfig.region_alloc`). Each step the k looks are split across the
M regions by headcount: region m gets floor(k N_m / N), and the looks left over go to the regions with the largest
remainders (ties at random). The total is exactly k and no region gets more looks than agents. The default
`fixed` is the old split and is unchanged bit for bit (`tests/test_x6_x7.py` checks it against the code at
48ad9a7, step for step).

**A problem found while testing.** Largest-remainder rounding does not give p = k/N in every state, which is what
Proposition 5.12 assumes. With 7 agents, k = 2 and 4 regions, the shares k N_m / N are all below 1 unless a region
holds 4 or more agents. So the two most crowded regions get one look each and the others get none. Example: with
3, 2, 1, 1 agents, the chances of a look are 1/3, 1/2, 0 and 0. Two agents alone in their strips are never looked
at.

Measured along scripted-bot episodes (Harvest, k = 2, 4 × 1,000 steps per row, `check_prob` per agent-step):

| bots | regions | allocation | agent-steps with p = 0 | mean p | spread of p (sd) |
|---|---|---|---|---|---|
| 7 greedy | 4 | equal (fixed) | 0% | 0.230 | 0.123 |
| 7 greedy | 4 | largest remainder | 13% | 0.286 | 0.135 |
| 7 greedy | 4 | unbiased rounding | 0% | 0.286 | 0.000 |
| 7 greedy | 2 | equal (fixed) | 0% | 0.275 | 0.148 |
| 7 greedy | 2 | largest remainder | 2% | 0.286 | 0.092 |
| 7 greedy | 2 | unbiased rounding | 0% | 0.286 | 0.000 |

Largest remainder fixes the wasted looks (mean p rises from 0.23 to k/N = 0.286), but with 4 regions the look
chance is as uneven as under equal budgets, only uneven the other way: lone agents are now the unwatched ones.
Over-harvesting happens at sparse apple patches, often away from other agents, so learners could use exactly
these states.

**Extra arm (added before any run).** `--region-alloc proportional_random`: unbiased (systematic) rounding. Region
m gets floor or ceil of k N_m / N, the ceil with chance equal to the remainder, at one shared random offset, so the
total is still exactly k, no region gets more than its headcount, and every agent's chance of a look is exactly
k/N in every state. Runs `DPU_R4_random_k2_c1.0_seed1–3` and `DPU_R2_…`, alongside the planned `DP_` arms.

**Predictions.**
- `DPU_R4`, `DPU_R2` (the proposition's own condition): survive at about the central random level, about 460
  apples (`B_random_k2_c1.0`). This is the test of Proposition 5.12.
- `DP_R2`: about the central level too (p = 0 in only 2% of agent-steps).
- `DP_R4`: well above equal budgets (`DH_R4_random_k2_c1.0`, about 76 apples), because no look is wasted. It may
  fall short of the central level if learners find the lone-agent states. If `DPU_R4` survives and `DP_R4` does
  not, that supports the proposition (what matters is p in every state), not the reverse.
- Falsifier of Proposition 5.12: `DPU_R4` loses about as much as equal budgets.

## X7: public catches

**Option.** `--public-catches` (`DecConfig.public_catches`). Each agent's observation gets 2N extra constant planes:
for every agent, "caught last step" (0 or 1) and min(steps since last caught, 100) / 100. They are in the same
order for every observer with the observer's own first (agent i, i+1, …), which a shared network needs. Added after
the game's channels exactly as `--reveal-checks` adds its own. Off by default; with it off, observations, rewards
and logs are identical (tested). Looks stay hidden (no `--reveal-checks`), so this is the ordinary mode.

**Cap of 100 steps, not 20.** `--reveal-checks` caps "steps since last checked" at 20. Catches are rarer than
checks, and the repeat-offenders-first chooser's suspicion decays by 0.95 per step. Its score is suspicion plus
0.001 × steps since last checked, so an agent caught 50 steps ago (suspicion 0.95^50 ≈ 0.08) still outranks an
agent never caught that has gone unchecked for under 80 steps. With a 20-step cap agents could not tell who the
current suspects are, which is what the prediction is about. 100 steps covers most of the chooser's memory
(0.95^100 ≈ 0.006).

**Predictions** (from §5.8): repeat offenders first with public catches falls well below its hidden-catch result
(`D_history_k2_c1.0`, 474 apples); random with public catches stays near 460 (`B_random_k2_c1.0`).

## X10: Clean Up with W = 25

Scripted bots first (`bots.md`), then 3 learner runs `DS_W25_random_w50_k2_c1.0_seed1–3` (k = 2, c = 1, look
window 50 ≥ W, random chooser, cost per step seen 0.01). Prediction (§5.6): if never-violating bots keep the
orchard in at least 90% of games under W = 25, learners at ρ ≥ ρ* keep it too.
