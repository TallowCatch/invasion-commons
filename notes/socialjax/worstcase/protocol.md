# Separate networks and the worst case: protocol

Written 22–23 Nov 2026, before any GPU run. Branch `claude/separate-nets-worstcase`.
- Code: `experiments/socialjax/train_ippo_overseer.py` (`--separate-nets`, `--reveal-checks`, `--mixed-eps`), `overseer/socialjax/reveal.py`, the `mixed` chooser and `check_prob` in `harvest_overseer.py` and `decentralised.py`, the `just_checked` and `dodger` bots in `harvest_bots.py`.
- Tests: `tests/test_worstcase.py`. Job lists: `experiments/socialjax/decentralised_runs.py` (`separate`, `worstcase`, `worstcase_pilot`). How to submit: `README.md` in this folder.
- Builds on stage 1 (`notes/socialjax/phase2_ippo/protocol.md`) and the decentralised runs (`notes/socialjax/DECENTRALISED_README.md`).

## The two questions

1. **Separate networks.** Stage 1 and the decentralised runs train one network shared by all 7 agents (SocialJax's IPPO default). Do the overseer's effects hold when each agent has its own network, so agents can learn different policies (for example one over-harvester and six careful agents)?
2. **The worst case.** So far agents see only their own 11×11 view and cannot tell who is checked. What happens when every agent sees the checks: who was checked last step, and when it was last checked itself? This is the transparent overseer of security games: the attacker knows the rule and the record, but not the coin flips.

## Setting

Everything as in stage 1 unless stated:
- Harvest Open, 7 agents, individual rewards (one apple pays 7), 1,000-step episodes, rule m = 3, a check sees only the current step, a perfect inspector, no freeze.
- k = 2 checks per step, penalty P = 3.5 apples per catch. Under random checks the average penalty per over-harvest is (k / N) × P = 1 apple (level c = 1), the same as stage 1's `B_random_k2_c1.0` and `D_history_k2_c1.0`.
- IPPO with SocialJax's hyperparameters, 256 envs, 1e8 steps, seeds 1–3.

**Separate networks (`--separate-nets`).**
- Each agent has its own CNN actor-critic and its own Adam state (stacked along an agent axis and vmapped), so gradient clipping and Adam never mix agents.
- Each agent's minibatches hold only its own transitions. The number of gradient steps per epoch stays at 500, each on 1/7 as many samples (512 instead of 3,584).
- Each agent learns from 1/7 of the data: 1.4e7 of its own steps in a 1e8 run. Stage 1's shared network needed about 1e7 steps to learn to harvest, so this should be enough, but it is not certain.

**Visible checks (`--reveal-checks`).** Three channels are added after the game's 15 (Harvest):
- a map of the agents checked last step, cut and rotated exactly like the game's own channels (only checked agents inside the agent's view show up);
- "I was checked last step" (0 or 1);
- "steps since I was last checked", capped at 20 and scaled to 0–1.

Episodes start with "steps since checked" = 0 for everyone. The overseer-off evaluation keeps the channels (all zero, then "steps since" rising to the cap), so the observation shape matches.

**The mixed chooser (`--chooser mixed --mixed-eps 0.3`).** Each step, with chance 0.3 the k checks are random, otherwise by history (repeat offenders first, then the longest unchecked).

**Two regions (R = 2).** Two vertical strips, one check each.
- **Caveat (found 22 Nov by another agent):** in `DecentralisedOverseer` an agent's record of its last w steps travels with it, and any overseer's check reads all of it, including steps spent in the other region. So R = 2 here means two overseers that hand records over to each other for free. It measures only the loss from checks landing in the wrong place, never evidence lost at a boundary. With w = 1 (these runs) the record is just the current step, so nothing is carried across; the caveat matters for longer looks.
- What an agent sees under `--reveal-checks` is the same with R = 1 and R = 2: checks by any overseer, and "steps since any overseer checked me".

## Runs

1e8 steps, seeds 1–3, Harvest, k = 2, P = 3.5 apples.

**Separate vs shared networks (`separate`, 15 runs).** The shared arms already exist: stage 1 (`B_random_k2_c1.0`, `D_history_k2_c1.0`) and the decentralised Harvest list (`DH_R2_random_k2_c1.0`, `DH_R2_history_k2_c1.0`, `DH_R2_share_k2_c1.0`), with the same code path. Tests check that the trainer's default path gives the same parameters as the trainer at fca047d.

| Run | Overseer | Runs |
|---|---|---|
| `SN_central_random` | central, random | 3 |
| `SN_central_history` | central, history | 3 |
| `SN_R2_random` | 2 strips, random | 3 |
| `SN_R2_history` | 2 strips, history, independent | 3 |
| `SN_R2_share` | 2 strips, history, sharing suspicion | 3 |

**Worst case (`worstcase`, 18 runs).** Separate networks and visible checks. Compared with the `SN_` runs (the same networks with checks hidden).

| Run | Overseer | Runs |
|---|---|---|
| `WC_central_random` | central, random | 3 |
| `WC_central_history` | central, history | 3 |
| `WC_central_roundrobin` | central, round robin (fully predictable) | 3 |
| `WC_central_mixed0.3` | central, mixed with ε = 0.3 | 3 |
| `WC_R2_random` | 2 strips, random | 3 |
| `WC_R2_share` | 2 strips, history, sharing | 3 |

There is no no-overseer arm with separate networks. Stage 1's arm A (shared, k = 0) is the reference for "no overseer"; the switch-off test of each run gives the same policy without the overseer.

**Pilot (`worstcase_pilot`, 2 runs of 2e7 steps)** to time separate networks with and without visible checks before the full lists.

## Measures

The same as stage 1, plus per-agent measures (each agent's apples, over-harvests, catches and net reward per episode, `*_agent` in `metrics.npz`). Each arm is summarised by the mean of its last 10 updates, as mean and range over the 3 seeds.

- **Over-harvests per episode** and **apples eaten**; **orchard alive** at the end.
- **Caught share** = catches / over-harvests. With a perfect inspector and a one-step look, random checks catch 2/7 = 0.29 of over-harvests whatever the agents do. Below 0.29 means the agents dodge the checks; above means targeting finds them.
- **Spread across agents:** each agent's share of the over-harvests (separate networks only can differ).
- The switch-off test, as in stage 1.

With 3 seeds there are no significance tests. A difference counts only if the seed ranges do not overlap.

## Predictions (written before any run)

**Separate networks.**
1. The direction of every stage 1 effect holds (random and history checks cut over-harvests compared with no overseer).
2. Agents may specialise: under history checks, over-harvesting may concentrate in one or two agents. Not certain; 3 seeds may not show it.
3. Learning is slower per agent (1/7 of the data), so the final over-harvest counts may be higher than shared at 1e8 steps.

**Worst case (checks visible), against the same arm with checks hidden.**
4. **Round robin is dodged completely.** With 7 agents and 2 checks per step, each agent is checked every 3 or 4 steps, so the 2 steps after its own check are always safe ("steps since I was checked" tells it when). Catches fall close to 0 while over-harvests stay up. The scripted `just_checked` and `dodger` bots are never caught (tested).
5. **History is dodged while nobody has been caught.** It checks the longest-unchecked agents, so an agent just checked is the safest. After a catch, that agent's suspicion puts it first for about 80 steps (the suspicion 0.95^t stays above the 1e-3 × steps-unchecked score of the others). So caught share falls below what it is with checks hidden, but stays above 0.
6. **Random cannot be dodged.** The chance of a check is 2/7 every step whatever the agent sees, and the expected penalty is exactly 1 apple per over-harvest. Visible checks change nothing beyond seed noise (caught share stays near 0.29).
7. **Mixed deters only through its random share.** An agent that is not a target is checked with chance ε × k / N = 0.3 × 2/7 = 0.086, an expected penalty of 0.3 apples, less than the apple it gains. So dodging pays until the first catch. After that, suspicion locks onto the caught agent (checked almost surely while its suspicion leads), which holds one or two dodgers in check but not seven. Deterring every agent needs ε × (k / N) × P ≥ 1, which at k = 2 and P = 3.5 means ε = 1 (pure random). Prediction: over-harvests and caught share fall between WC random and WC history.
8. **Two strips: crowding makes random checks dodgeable.** Each strip's overseer checks 1 of the m agents in its strip, so an agent's chance is 1/m. In a strip holding 4 or more agents it is at most 1/4 < 2/7, so over-harvesting pays there (expected penalty ≤ 0.875 apple). Agents see only part of the map, so whether they learn this is uncertain. Prediction: WC R2 random over-harvests more than WC central random; WC R2 share behaves like WC central history (prediction 5).

## Compute

About 1e8 steps per run. Stage 1 took 18–38 min per shared-network run on an A100/H200. Separate networks run 7 CNNs side by side (vmapped) and the visible checks add 3 channels to the game's 15 (20% more stored observation memory). The parameters are small, so 7 copies add little memory. These are guesses until the pilot is timed: about 30–60 min per run, so 15 runs ≈ 8–15 GPU-hours and 18 runs ≈ 10–20 GPU-hours.
