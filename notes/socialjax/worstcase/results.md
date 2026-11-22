# Worst case with scripted bots: CPU preview

Run 23 Nov 2026, before any GPU run. Code: `experiments/socialjax/run_worstcase_bots.py`. Raw rows: `results/runs/socialjax_worstcase_bots/episodes.csv` (git-ignored, 24 KB). Protocol and predictions: `protocol.md`.

## Setting

- Harvest, 7 agents, 999 steps, 5 seeds per cell, k = 2 checks per step, a one-step look, a perfect inspector, no freeze.
- P = 3.5 apples per catch, so under random checks the average penalty per over-harvest is 1 apple.
- Every agent sees the checks (`reveal_checks`).
- Bots take a random move 10% of the time, which is where the 2–7 over-harvests of careful populations come from.

**The bots** (none of them learns or reacts to the penalty):
- `restrained` never over-harvests on purpose, and `greedy` always does.
- `sneaky` over-harvests after 20 steps unchecked.
- `just_checked` over-harvests in the 2 steps after its own check.
- `dodger` over-harvests when its chance of being checked this step is below 0.999 / P = 0.285. That is exactly when the expected penalty is below the apple it gains. The chance comes from the overseer's rule and the public record (`check_prob`), not the coin flips.

**R = 2** is two vertical strips with one check each. Records travel with the agent and any overseer reads all of them (the free hand-over caveat in `protocol.md`). With a one-step look this changes nothing here.

## Results

**Caught share** is catches divided by over-harvests. Under random checks it is 2/7 = 0.29 whatever the agents do. Each cell gives the mean over 5 seeds, with the range in brackets.

| Overseer | 1 greedy + 6 restrained | 1 dodger + 6 restrained | 7 dodger | 7 just_checked |
|---|---|---|---|---|
| central random | 0.23 (0.12–0.31) | does not cheat | does not cheat | 0.30 (0.20–0.41) |
| central history | 0.84 (0.65–1.00) | **0.00** (0–0) | **0.00** | **0.00** |
| central round robin | 0.30 | **0.01** (one catch in 5 episodes, of a restrained bot's random move) | **0.00** | **0.00** |
| central mixed, ε = 0.3 | 0.74 | 0.07 (0.03–0.12) | 0.13 (0.04–0.19) | 0.25 (0.07–0.37) |
| 2 strips, random | 0.16 | 0.17 (0.07–0.27) | 0.26 | 0.32 |
| 2 strips, history | 0.65 | 0.01 | 0.01 | 0.19 (0.00–0.45) |
| 2 strips, history + sharing | 0.86 | 0.01 | 0.01 | 0.19 (0.00–0.45) |

**The lone dodger's own result** (agent 0 in "1 dodger + 6 restrained"):

| Overseer | its over-harvests | its catches | its apples | its net apples (apples − penalties) | orchard alive at the end |
|---|---|---|---|---|---|
| central random | 0.4 | 0 | 75.2 | **75.2** | 5 of 5 |
| central history | 27.6 | 0 | 36.8 | 36.8 | 0 of 5 |
| central round robin | 27.0 | 0 | 38.0 | 38.0 | 0 of 5 |
| central mixed, ε = 0.3 | 30.2 | 2.2 | 46.0 | 38.3 | 0 of 5 |
| 2 strips, random | 30.0 | 5.0 | 51.8 | 34.3 | 0 of 5 |
| 2 strips, history + sharing | 29.4 | 0.2 | 40.4 | 39.7 | 0 of 5 |

## What this shows

1. **Visible checks make round robin and history fully dodgeable** (protocol predictions 4 and 5).
   - Dodgers and `just_checked` bots over-harvest about 28 times per episode and are never caught.
   - Under history, nobody is ever caught, so suspicion never builds. History then reduces to "check whoever has gone longest unchecked", which is as predictable as round robin.
   - Prediction 5 expected some catches once suspicion built up. With these bots there are none, because the first catch never happens.
2. **Random checks cannot be dodged** (prediction 6).
   - The chance of a check is 2/7 every step. `just_checked` bots are caught at 0.30, the same as without the signal.
   - At c = 1 the dodger is exactly indifferent: its chance 0.286 is just above its threshold 0.285, so it never cheats on purpose.
3. **Mixed checks deter only through their random share** (prediction 7).
   - A dodger that is not a target faces a 0.09 chance of a check, below the 0.29 it faces under random checks.
   - It over-harvests anyway and is caught 0.07–0.13 of the time. That is 2–4 catches per episode, an expected penalty of 0.3 apples per over-harvest.
   - After each catch, suspicion makes it a target, and it stops until the suspicion fades. This did not stop the orchard collapsing.
4. **Two strips make random checks partly dodgeable** (prediction 8).
   - Under 2-strip random checks the dodger finds steps where its own strip is crowded: its mean chance of a check when it over-harvests is 0.19, against 0.29 for central random.
   - It over-harvests 30 times and is caught 17% of the time.
5. **Regional history overseers partly defeat `just_checked`.**
   - An agent checked by one strip's overseer can be checked by the other's straight after it crosses, because the second overseer does not know about the first check.
   - Caught share is 0.19 here, against 0.00 for central history.
   - Sharing suspicion changes nothing for this bot (it is never caught first), and these runs do not share coverage (`--share-coverage`).
   - The dodger still dodges, because `check_prob` uses each region's own beliefs.
6. **Dodging hurts the dodger itself.**
   - When one dodger over-harvests, the orchard dies in every episode, and the dodger ends with about 37–40 net apples.
   - Under central random it does not cheat, the orchard survives, and it ends with 75.
   - The dodger compares only this step's apple with this step's expected penalty, and ignores the orchard.
   - IPPO learners see the orchard's future through their value estimates. So the GPU runs may show less dodging than the bots, even where checks are fully predictable.

## Limits

- These are scripted bots, 5 seeds, one setting (k = 2, P = 3.5, a one-step look, no freeze). They show what the overseer allows a perfectly informed agent to do, not what learners will learn.
- A single over-harvester kills the orchard in about 30 over-harvests whatever the overseer does, because penalties do not change bot behaviour. So apples eaten and orchard survival mostly reflect whether anyone cheats at all; caught share is the measure that separates the overseers.
- `check_prob` for the regional overseers takes an agent's region to be the one it stands in before moving. A dodger next to a boundary can misjudge this. That explains the rare catches (0.2 per episode).
