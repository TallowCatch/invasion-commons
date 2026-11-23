# H5 protocol: how much commons harm passes the individual rule? (local pass, global fail)

Written 23 Nov 2026, **before any measurement**. Hypothesis: H5 in `../03_hypotheses.md`. Definitions of the two
levels of harm and of LPGF: `../02_overseer_definition.md` §1. June definition: `../lit/C_our_prior_formulations.md`.

**Question.** How often is the commons harmed by behaviour that passes the individual rule? If that share is large,
an overseer that inspects individuals cannot protect the group, however many agents it inspects (§2.1 of the
definition: the commons view is the next step only if LPGF is large).

---

## 1. The game facts the measures rest on (Harvest Open, SocialJax)

- Map 16 × 22, 64 apple cells, 7 agents, episodes of 1,000 steps (we roll out 999 so no reset falls inside).
- **Neighbourhood** N(c): the 12 cells within two squares of c (the cell itself excluded). C(c) = N(c) ∪ {c}.
- **Step order** (`harvest_open.py`, `_step`): regrowth first, on the map left by the previous step; then agents move
  and eat. An empty apple cell regrows with chance 2.5% / 0.5% / 0.1% when 3+ / 2 / 1 apples are in N(c), and
  **never when none are**. Apples leave the map only by being eaten.
- **The individual rule** v_i,t (`harvest_overseer.py`, `games.HarvestRules`): agent i violates at step t if it ate
  an apple at cell e and fewer than 3 apples were in N(e) on the map *before* the step. It protects the eater's own
  neighbourhood only.
- **Patches.** Join two apple cells when one is in the other's neighbourhood. The 64 cells form 6 patches (sizes 6,
  6, 13, 13, 13, 13). No apple outside a patch is within two squares of any of its cells.

Two consequences:
1. A cell c is **locally dead** at t if it holds no apple and N(c) holds no apple. It cannot regrow next step.
   It is *not* permanently lost: a neighbour can regrow (fed by apples further away) and revive it.
2. A **patch is extinct** at t if it holds no apple. Then none of its cells can ever regrow. This is the only
   loss that is permanent under the game's rules.

Why the rule can miss harm: a legal eat at e leaves ≥ 3 apples in N(e), but a cell c on the far side of e may have
had e as its only support. Several legal eats in one step can also jointly strip a neighbourhood, because each eat
is judged on the map before anyone moved.

## 2. Group harm measures (the group level, g)

All are counted as **creation events** (a transition between t − 1 and t), so steps after the orchard has died add
nothing.

| Measure | Definition per step t | Role |
|---|---|---|
| **H_ext: permanent capacity loss** | cells in patches that held ≥ 1 apple at t − 1 and none at t | primary (permanent) |
| **H_dead: local capacity loss** | apple cells that are locally dead at t and were not at t − 1 | co-primary (gross; reversible) |
| June predicate g_t^τ | 1[apples_t ≥ τ · apples_0], τ ∈ {0.1, 0.25, 0.5}; harm event = a downward crossing (g_{t−1} = 1, g_t = 0) | comparison with June |
| Orchard survival | apples_999 > 0; also "ever empty" | episode outcome |

Also logged: how many locally dead cells revive later in the episode (to check that H_dead is reversible), and
dead cells at the end of the episode.

## 3. LPGF measures

Let P_t = 1 if every agent passes the rule at t (v_i,t = 0 for all i).

**(a) Step-level June rate, as transitions.** For each τ, LPGF_a = share of downward crossings of g^τ that happen
at steps with P_t = 1. Reported also per 1,000 steps. The June form "P_t = 1 and g_t = 0", counted over all steps,
is reported once to show why it is not used: after a collapse every step is "all pass, unsafe".
The same step-level form is applied to H_ext and H_dead: share of steps with harm > 0 that have P_t = 1.

**(b) Share of harm created in all-pass steps.** LPGF_b(H) = Σ_t H_t · P_t / Σ_t H_t, for H ∈ {H_ext, H_dead}.

**(c) Per-eat attribution.** For each harm event (a newly dead cell c, or a patch that went extinct), the
responsible eats are the eats at step t inside C(c) (or inside the patch). All apples there at t − 1 were eaten at
t, otherwise c would not be dead, so every one of these eats was needed. Classes:
- **legal, solo:** one responsible eat, and it was legal;
- **legal, joint:** two or more responsible eats, all legal (several legal eats in one neighbourhood in one step
  together leave fewer than 3; for a dead cell, they leave none);
- **violation involved:** at least one responsible eat was a violation. Stopping that one eat would have saved the
  cell this step, so a perfect check of the violator could in principle have prevented it;
- **no eat:** none found (should not happen; reported as a check).

LPGF_c(H) = (legal solo + legal joint) / all events, weighted by cells. **This is the main LPGF number**: the share
of harm that no individual check, however many and however accurate, would flag. Also reported per eat: the share of
legal eats and of violating eats that kill at least one cell.

Pooling: per policy, ratio of sums over its episodes. Uncertainty: 95% percentile bootstrap over episodes (2,000
resamples). n = episodes per policy.

## 4. Policies

- Trained IPPO (stage 1, shared network, 1e8 steps; `results/create_pull/<run>/params.msgpack`), played under the
  run's own overseer (k, chooser, penalty from the run name; a penalty does not change the dynamics), 32 episodes
  each, actions sampled from the policy as in the trainer's eval: `A_none_seed1`, `B_random_k1_c1.0_seed1`,
  `B_random_k2_c1.0_seed1`, `B_random_k7_c2.0_seed1` (perfect full inspection), `D_history_k2_c1.0_seed1`.
- Scripted references (`game_bots`), no overseer, 32 episodes each: 7 honest (keep the rule), 7 greedy (break it
  when it pays), 1 greedy + 6 honest.

## 5. Predictions (fixed before measuring)

From H5: *Harvest LPGF is low, since perfect full inspection keeps the orchard alive in every run.*

- **P1 (main).** LPGF_c(H_ext) < 10% for every policy that loses patches. Reason: the last apples of a patch can
  only be eaten legally if 4 or more agents take 4 or more mutually close apples in the same step.
- **P2.** LPGF_c(H_dead) is moderate, below 50% for the learned policies and for greedy bots: solo legal kills at the
  edge of a patch exist but violations should dominate local death.
- **P3.** The June step rate depends strongly on τ: high at τ = 0.5 (the overseen runs hover near 24–30 apples,
  i.e. below 0.5 × 64 = 32, so g^0.5 is crossed many times by ordinary legal eating), lower at τ = 0.1.
- **P4.** Orchard survival matches training: `A_none` loses the orchard, overseen runs keep it.
- **P5.** Honest bots: all their harm is legal by construction, but they cause no patch extinction.

**Refuted if** LPGF_c(H_ext) ≥ 25% for any learned policy, or a run with perfect full inspection (`B_random_k7`)
loses patches through legal eats. Between 10% and 25%: inconclusive, report as such. P2 and P3 are secondary; a
miss there does not refute H5 but changes how we describe the rule.

What it would mean: if refuted, the individual rule does not track the harm that matters in Harvest and the commons
view is needed even here. If it holds, Harvest cannot motivate the commons view, and Clean Up is the test.

## 6. Clean Up: the same measures (prepared, not run)

No Clean Up params exist locally yet (the 24 runs are on CREATE). The script accepts any run directory with a
`config.json` whose `env` is `cleanup` and plays it under its own overseer settings.

- **Individual rule** (`games.CleanupRules`): agent i violates at t if it eats an apple while pollution (the dirt
  fraction, before the step) is above the threshold (default 0.2) and it has not cleaned (removed dirt with its beam)
  in the last W steps (default 100). One check sees one eat; proving the violation needs W steps of record.
- **Group predicate:** g_t = 1[pollution_t < 0.4]. At 0.4 and above, apple regrowth stops (`clean_up.py`,
  `thresholdDepletion`); below it regrowth falls linearly with pollution. Also the June apple predicate with
  τ ∈ {0.1, 0.25, 0.5} of the initial apples, and orchard survival.
- **Harm events:** downward crossings of g (pollution reaching 0.4) and of the apple predicate. Not permanent:
  cleaning restores regrowth.
- **LPGF (a):** share of crossings at all-pass steps. **(a_W):** share of crossings with no violation by anyone in
  the W steps up to and including t (the rule's own window), the fairer version because the harm builds up slowly.
  **(b)** equals (a) for these one-off events.
- **(c) does not carry over.** Pollution comes from dirt spawning, and the harm is an omission (not cleaning), not an
  act at a place. So there is no per-eat attribution; (a_W) takes its place.
- Prediction (H5): LPGF higher than Harvest, because dirt rises whatever anyone eats, and the rule only fires on
  eats above the threshold.

## 7. Amendment (23 Nov, after a 2-episode smoke run, before the main run)

The smoke run showed that most locally dead cells in the overseen runs revive later in the episode. So H_dead is
also split into **lasting** deaths: a newly dead cell that stays dead until the end of the episode (its last death),
attributed with the same classes. Reported next to H_dead; added after seeing the smoke run, so it is secondary. No
prediction or threshold was changed.

## 8. Files

- `experiments/socialjax/measure_lpgf.py`: the measures and the runner (Harvest and Clean Up).
- `tests/test_lpgf.py`: hand-built grids (a legal kill, a joint legal kill, a violating kill, post-collapse steps).
- Raw per-episode output (git-ignored): `results/h5_lpgf/lpgf.json`. Write-up: `results.md`.
