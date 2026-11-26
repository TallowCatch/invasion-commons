# Measuring locality, and does it predict the decentralisation gap? Protocol (22 Nov 2026)

Written **before any run**. Branch `claude/locality-scale`, from `claude/r1-s1-oversight-experiments` at fca047d.
It answers gaps 1 and 4 of `../../phd_plan/2026-11-22_devils_advocate_formulation.md`, using the definitions in
§5–6 of `../../phd_plan/2026-11-23_decentralised_overseer_formulation.md`.

**Plain terms.** An *agent-step* is one agent at one time step. A *region* is the part of the map one overseer
covers. A *violation* is one rule-breaking act (Harvest: eating an apple that leaves fewer than 3 within two
squares). Every number below is a **prediction** until the results file says otherwise.

## 0. A fact about the existing code found while reading it

`DecentralisedOverseer` (`overseer/socialjax/decentralised.py`) keeps each agent's record of the last w steps
**with the agent**. A check by any overseer reads the whole record, including steps the agent spent in another
region. So the existing code is a *full hand-over at zero message cost* (κ = ∞ in the formulation's §3.3), not the
independent overseers (κ = 0) that the P_0 formula describes. Crossing a boundary never erases evidence in it; the
only thing that splitting changes there is where the checks can go (budget density).

So this study adds, in a new file, the independent version the formula is about:
`overseer/socialjax/local_records.py`, `LocalRecordOverseer`, in which overseer m's check reads only the steps
the agent spent in region m (ρ_m in §3.2). A record of a violation in region m stays pending, and is caught only
if overseer m checks the agent while it is back in m within the window. With `records="handover"` it is the
existing behaviour, step for step (tested).

## 1. Task 1: measure locality

### Testbeds

| Name | Agents | Map | Regions measured |
|---|---|---|---|
| `harvest` | 7 | Harvest Open 16 × 22 | strips R = 2, 4; bands (`rows`) R = 2; grid R = 4 |
| `harvest_large` | 14 | Harvest Open tiled twice, 16 × 44 | strips R = 2, 4; grid R = 4 |
| `rooms` | 7 | Harvest Open with a wall down column 11 (closed: two rooms, no door) | the two rooms (= strips R = 2); strips R = 4 |
| `rooms_door` | 7 | the same wall with a one-cell door at row 11 | the two rooms; strips R = 4 |
| `rooms_large` | 14 | the 14-agent map with walls at columns 11, 22 and 33 (four closed rooms; the wall at column 22 replaces 4 apple cells of the seam patch) | the four rooms (= strips R = 4); strips R = 2 |
| `coins` | 2 | Coin Game | bands R = 2 (as the existing sweeps), strips R = 2 |
| `cleanup` | 7 | Clean Up | strips R = 2, 4; bands R = 2 |

The rooms maps are new env factories in `overseer/socialjax/rooms.py`; SocialJax's `harvest_common_open` takes a
`map_ASCII` with `W` walls, and walls block movement (harvest_open.py, the `blocked` mask before
`resolve_movement`). No existing wrapper is edited.

### Populations (scripted bots, CPU)

Per testbed: all honest, 1 greedy + the rest honest, all sneaky, all migrator (the migrator under each layout's
own overseers, 2 random checks per step, no penalty, so its moves depend on the layout). 10 seeds × 999 steps for
the 7-agent and 2-agent games, 5 seeds for 14 agents. χ and ζ are also measured for trained IPPO agents on CREATE
(section 5).

### The numbers

- **χ, crossing rate:** region changes ÷ agent-steps (t ≥ 1).
- **Evidence stays, S_w:** the share of violations after which the violator is still in the same region at every
  one of the next w − 1 steps. The formula's stand-in for it is (1 − χ)^{w−1}.
- **ζ, harm containment.** Three measures, reported side by side:
  1. *Footprint* (Harvest, rooms): of the apple cells in the 12-cell neighbourhood of the eaten apple whose
     regrowth chance falls (the empty apple cells there at that moment, plus the eaten cell's own neighbours that
     still hold apples, counted once each), the share in the violator's region. The static version (all apple
     spawn cells in the neighbourhood) is also given.
  2. *Counterfactual regrowth* (Harvest, rooms): for a sample of violations, two copies of the map after the
     step: one as it is, one with the eaten apple put back. Both run regrowth only (Harvest's rule, no agents) for
     H = 100 steps with the same random numbers. Lost apple-steps by region; ζ = the share in the violator's
     region, as a ratio of sums over violations. The eaten cell itself is left out (it is the gain, not the harm).
  3. *Counterfactual with agents* (Harvest): the same two copies, but the bots keep playing for H = 100 steps with
     the same keys. Noisier, because bots react to the restored apple; reported with a bootstrap interval.
  - **Clean Up:** regrowth is the same chance for every empty orchard cell and depends on one global dirt
    fraction (clean_up.py). So the harm of not cleaning lands on empty orchard cells in proportion to their
    number: ζ = the violator's region's share of the empty orchard cells at that step.
  - **Coins:** the harm of a theft is the other player's loss (−2 coins of value); ζ = the share of thefts at
    which the victim is in the thief's region.
- **Λ_w and ξ_loc** for w = 1, 10, 50, two ways:
  - formula: Λ_w = (1 − χ)^{w−1} · ζ;
  - direct: the mean over violations of S_w × ζ (this violation).
  - ξ_loc = 1 − Λ_w.

### Predictions (bots)

| Testbed, layout | χ (per agent-step) | ζ (footprint) | ξ_loc at w = 1 / 10 / 50 |
|---|---|---|---|
| harvest, strips R = 2 | 0.005–0.03 | ≥ 0.95 (the boundary falls in the empty columns 10–11) | ≤ 0.05 / 0.05–0.25 / 0.2–0.8 |
| harvest, strips R = 4 | 2–3 × the R = 2 value | 0.7–0.95 (boundaries cut the two top patches) | higher than R = 2 at every w |
| harvest, grid R = 4 | between strips R = 2 and R = 4 | 0.7–0.95 | as strips R = 4 |
| harvest_large, R = 2 / 4 | as harvest | 0.8–0.95 (R = 2 splits the seam patch) | |
| rooms (closed), the two rooms | **exactly 0** | **exactly 1** | **exactly 0** at every w |
| rooms_door | ≤ 0.003 | ≥ 0.98 | ≤ 0.02 / ≤ 0.05 / ≤ 0.15 |
| rooms, strips R = 4 (regions not aligned with rooms) | > 0 | < 1 | > 0: rooms only help if regions follow the walls |
| coins, R = 2 | 0.03–0.15 (a small map, coins anywhere) | 0.3–0.7 | ≥ 0.3 at w = 1, near 1 at w = 50 |
| cleanup, strips R = 2 | 0.005–0.03 | 0.3–0.7 (≈ the region's share of the orchard) | ≥ 0.3 at every w |

Also predicted: (i) S_w < (1 − χ)^{w−1} for migrators (they cross towards unwatched regions right after
cheating) and ≈ for the others; (ii) the counterfactual ζ ≤ footprint ζ, since regrowth cascades spread harm
beyond radius 2, but by less than 0.1 in Harvest Open, where patches are 3 or more cells apart.

## 2. Task 2: is the derived catch formula right?

P_C = (1 − α)[1 − (1 − q)^w] and P_0 = (1 − α) q [1 − ((1 − q)(1 − χ))^w] / [1 − (1 − q)(1 − χ)], from
§5.2. Assumptions: random checks; every agent checked with the same chance q each step; leaves its region with
chance χ each step and doesn't come back within the window; the window is cleared after a check.

**Simulations** (`experiments/socialjax/measure_locality_formula.py`):
1. *Abstract model*, NumPy, exactly the formula's model (Bernoulli(q) checks, Markov crossing), 200,000
   violations per cell, over q ∈ {1/7, 2/7, 0.05}, χ ∈ {0, 0.01, 0.05, 0.2, 0.5}, w ∈ {1, 2, 5, 10, 20, 50},
   α ∈ {0, 0.5}, with three crossing patterns: *drift* (each crossing goes to a new region; no return), *two
   regions* (the only crossing is back to the other one), *ring of four* (left or right at random).
2. *The real overseer code* (`LocalRecordOverseer`, fixed k_m = K/R checks per region) on a toy corridor game
   (`overseer/socialjax/corridor.py`): N agents on R regions, a controllable crossing bot that changes region
   with chance χ and violates with chance v each step. Same grid of χ and w, N = 8, R = 2 and 4, K = 2.

**Predictions.**
- Drift, abstract: the formula holds to within Monte Carlo error (|sim − formula| ≤ 0.005) in every cell.
- Two regions and ring: the formula **underestimates** P_0, because an agent that comes back is checked again
  with its old record still there. The gap grows with χ·w and is largest for two regions; at χ = 0.5, w = 50 the
  true P_0 is close to P_C, not to the formula's q/(q + χ − qχ). A Markov version (exact, `p0_markov`) should fit.
- Real code: the formula with q = K/N is off whenever occupancy is uneven, because a region's k_m is shared by
  however many agents are there. With balanced occupancy (agents assigned evenly, no crossing) it holds; with
  crossing it is too high for crowded regions and too low for empty ones, and overall too high (unused checks in
  empty regions). The formula with the measured per-agent check rate should fix most of this.
- At w = 1 every version gives P_0 = P_C = (1 − α)q under matched budgets.

## 3. Task 3: does locality predict the decentralisation gap (D1)?

`experiments/socialjax/measure_locality_gap.py`. Random checks, K = 2 checks per step in total (K = 4 for 14
agents, K = 1 for Coins), no penalty and a 50-step freeze on a catch (as in the existing sweeps, so catches change
welfare even though bots ignore penalties), w ∈ {1, 10, 50}. For each testbed and layout: central (R = 1),
independent local-record overseers (κ = 0), and the existing full hand-over. Populations: 1 greedy + rest honest,
and all sneaky (both don't exploit regions; migrators are reported separately, as they exploit them on purpose).
10 seeds (5 for 14 agents). Clean Up only at w = 150 (its rule needs 100 steps, so shorter looks catch nothing).

- **Gap:** Δ_dec = (share of violations caught, central) − (the same, decentralised), at the same K. The welfare
  version, η = (W − W_none)/(W_all − W_none), is also reported, with W_all = every agent checked every step,
  central.

**Predictions.**
- Local records: Δ_dec grows with ξ_loc (Spearman ρ ≥ 0.6 over the testbed × layout × w points), ≈ 0 at w = 1
  apart from budget-density losses, and largest at w = 50 on the open maps.
- Rooms (closed) with rooms as regions: Δ_dec stays near 0 at w = 50 (≤ 0.05 of violations), while Harvest Open at
  the same w loses ≥ 0.15.
- Hand-over: Δ_dec does not grow with w; what's left is the budget-density loss, which locality does not measure.
  This would mean ξ_loc predicts the gap of independent overseers only.
- Honest limit: the points share testbeds and populations, so they are not independent. With 7 testbeds, a few
  layouts and 3 windows there are about 30–40 points from 7 maps; any correlation is a lead, not a law.

## 4. Task 4: scale, in Willis et al.'s commons

`experiments/scale/` (new). The Common Pool Resource game of Willis, Zhao, Du & Leibo (2026) through the existing
adapter (`overseer/adapters/emergent_llm.py`), with the LLM-written strategies from their committed pool (no LLM
calls). N = 64 and 256 players in one game, 20 rounds, capacity 4N. Each round the overseer audits K agents
(random, or *history-first*: agents with the most past catches first, then the longest unaudited), a D is caught
by an audit (perfect detection), and fines don't feed back to the fixed strategies (as in R1a). K = N/8 per round.
- **M overseer groups,** M ∈ {1, 4, 16} (and 64 for N = 256): agents split into M fixed groups of N/M; each
  overseer audits K/M per round in its own group. So χ = 0 by construction and ζ = 1/M (one shared stock).
- **Variants:** independent; and *budget pooling*, in which each overseer reports one number per round (how many
  agents it has caught before), and the K audits are re-split in proportion.
- 20 seeds (different strategy draws and group assignments).
- **Communication** as a share of central collection: a central overseer collects N records per round (w = 1);
  independent groups send 0 numbers between overseers; budget pooling sends M numbers per round, M/N of central.
  Each overseer reads N/M records instead of N.

**Predictions.**
- Random audits: Δ_dec = 0 within ±0.01 of defections caught at every N and M (χ = 0; ζ does not enter catching).
  So **ξ_loc = 1 − 1/M is high but the gap is zero**: ζ matters for what overseers *want* (version b), not for
  what they can catch. If this holds, ξ_loc as a single number is the wrong predictor for catching, and χ alone is.
- History-first: the independent groups lose a little (≤ 0.05) because repeat defectors are spread unevenly over
  groups; budget pooling recovers most of it at M/N = 6% (N = 64, M = 4) to 25% (N = 256, M = 64) of central
  traffic.
- Version (b), derived with threshold-rational players (defect iff the expected fine q·F is below the extra
  share): each group's overseer gains only 1/M of the stock its audits protect, so it audits only when the
  audit cost is below 1/M of the level a team of overseers would accept.

## 5. Trained agents (CREATE, prepared, not run here)

`scripts/create/measure_locality_ippo.sbatch` runs `experiments/socialjax/measure_locality.py --params` on stage 1
runs: it loads `/scratch/users/k26108758/ippo_overseer/<run>/params.msgpack` with the trainer's own network
(`train_ippo_overseer.socialjax_actor_critic`, imported, not edited), plays 32 episodes under the run's own
overseer settings (`config.json`), and writes χ, S_w, ζ and Λ_w for every layout. **Prediction:** trained agents
cross more than bots (χ 2–5 × the honest bots'), because the no-overseer agents roam to strip the orchard.

## 6. What would change the plan

- If ξ_loc doesn't order the gap at all on the local-record overseer, the locality number is not the contribution
  and the formulation's §5 must be rewritten around χ and budget density.
- If closed rooms still show a large gap, it comes from budget density, and budgets that follow occupancy are the
  first thing to add.
