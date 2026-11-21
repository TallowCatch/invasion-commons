# G1/G2 protocol: are Fishery, Forest and River real social dilemmas at our settings, and do the main programmed results survive a parameter sweep?

**Frozen:** 14 October 2026, before any G1/G2 code exists. Amendments are appended below and never edit this text.

**Why** (deep-research report `research/commons_platform/2026-10-14_decision.md` §2–3):
- The literature validates commons games with **Schelling diagrams** (Schelling 1973 [70]; Leibo et al. 2017 [66]; Perolat et al. 2017 [09]; Hughes et al. 2018 [10]; SocialJax [20, 21]; OpenSocialJax [22]).
- Our settings audit found 35 of 104 settings with no recorded reason and no dilemma check [L01–L12].

## Part A (G1): Schelling diagrams with programmed agents

**Agents.** Use the existing programmed policies, unchanged.
- **Fishery:** "restrained" = the useful request ranges (5–20% below threshold, 20–40% above); "greedy" = the stress ranges (45–65% / 75–95%). Thresholds U(25, 65). Setup from `claude_oversight_common.fishery_setup`.
- **Forest:** `cooperative_harvest_strategy` against `adversarial_harvest_strategy` (`harvest_setup`).
- **River:** the compositional game's honest and stress dischargers, as used in C1/R4.

**Diagram 1, the base commons (no reviewer, no audits).**
- For N agents and k = 0…N greedy agents, run P = 64 populations.
- Reuse the fresh-seed convention: seed base `G1_BASE = 26101400`, never used before.
- Horizon: the paper's programmed horizon (80).
- Plot the mean per-agent return of a restrained agent, R_c(m), and of a greedy agent, R_d(m), against m, the number of *other* agents who are restrained (Hughes et al. 2018 convention).
- Report the 95% population-bootstrap interval for each point.

**Diagram 2, the audited commons.** Same agents. In each population a share of agents, chosen at random, are "over-takers" (d = 1: take their full request), and the rest comply with their allowance.
- Reviewer: the MSY shared cut as in the paper.
- Audits: q = 1/6, with fine F ∈ {0, F\*/2, F\*, 2F\*}, where F\* is the paper's predicted break-even fine for that game.
- Here "cooperate" = comply and "defect" = over-take.

**Dilemma test.** The conditions are fixed before running and taken from Hughes et al. 2018 / Leibo et al. 2017, with m = the number of other cooperators and N − 1 = everyone else.
- **(D1)** R_c(N−1) > R_d(0): mutual cooperation is better than mutual defection.
- **(D2)** R_c(N−1) > R_c(0): mutual cooperation is better than being exploited.
- **(D3)** Fear, R_d(m) > R_c(m) for some small m (m ≤ ⌊(N−1)/2⌋), **or** greed, R_d(m) > R_c(m) for some large m (m > ⌊(N−1)/2⌋).

A game "is a social dilemma at our settings" if D1, D2 and D3 all hold, each with the bootstrap interval of the difference excluding 0.

**Predictions.**
- **P1:** all three base commons pass. A failure is a result, and the paper must say so.
- **P2:** in the audited commons, at F ≥ F\*, defection no longer pays: R_d(m) ≤ R_c(m) for every m (D3 fails). At F = 0, D3 holds.

**Part B (G1-LLM, run only once credit allows, about $3).** The OpenSocialJax role-prompt method [22]:
- the same model (gpt-oss-120b) in every seat, told by one role line to be restrained or greedy, k = 0…N greedy;
- Fishery only; 3 seeds per k with seats rotated;
- the same D1–D3 tests.

## Part B (G2): parameter sweeps (programmed agents, CPU only)

**One factor at a time**, all others at the paper's values.

| Factor | Values | Grounding |
|---|---|---|
| Fishery growth r | 0.4, 0.55, 0.7, 0.85, 1.0 | FishBase "medium"–"high" resilience bands [79] |
| Group size N (K scaled to keep K/N = 100/6) | 4, 6, 8, 10 | human CPR groups below 10 [77] |
| Horizon T | 20, 40, 80 | 20 = LLM games; 80 = programmed games |
| Forest neighbour externality | 0, 0.11, 0.22, 0.33, 0.44 | never varied before [L06] |
| River synergy κ (if the River setup exposes it) | 0.5×, 1×, 2× the paper value | — |

**At every grid point, record:**
1. D1–D3 for diagram 1.
2. **The deterrence threshold:** the smallest expected fine e (q = 1/6, F on a grid of 0.1·F\* steps up to 3F\*) at which best-responding over-takers stop. This uses the existing search over over-take level d ∈ {0, ¼, ½, ¾, 1} on 8 training populations, then 64 test populations, as in S2/S3. Compare it with the closed-form prediction for that grid point.
3. Harm (share of rounds below K/2) under no oversight, and under audits with F = 2F\*.

**Holding criteria, fixed now.**
- **(S1)** The dilemma holds at every grid point, or the failing points are reported.
- **(S2)** The observed threshold is within ±25% of the predicted one at every grid point where the dilemma holds.
- **(S3)** Audits with F = 2F\* reduce harm against no oversight at every grid point.

**Independent unit:** the population. 64 test populations per grid point, with the seed base `G2_BASE = 26101420`.

## Outputs

- **Code:** `experiments/oversight/run_g1_schelling.py`, `run_g2_sweeps.py`, `analyze_g1_g2.py`, with tests in `tests/test_g1_g2.py`.
- **Results:**
  - raw files in `results/runs/claude_g1_v1/` and `claude_g2_v1/`;
  - summaries in `notes/claude_audit_20261005/runs/claude_g1_v1/` and `claude_g2_v1/`;
  - `studies/G1_schelling_and_sweeps/results.md`.
- **Figures:** a Schelling-diagram panel figure (3 games × 2 diagrams) and a sweep-robustness figure, following `figures_guide.md`.

**Stop rule.**
- No reruns for significance.
- If a policy class cannot be expressed as restrained or greedy in a game, report it and skip that game's diagram. Do not redefine the policy class.

## Amendments

(none)

### Amendment 1 (14 October 2026, before any G1/G2 code was run; written after reading the code, nothing run)

The protocol leaves several choices open or names a value that does not exist. Each is fixed here, before any G1/G2 episode is run.

1. **F\* for Forest and River.** The paper states a break-even only for Fishery: g\* = 0.356 per over-take round in the pilot setting (`body.tex:96`; `studies/S3_threshold_timing_memory/results.md:49`), so Fishery F\* = g\*/q = 0.356 × 6 = **2.136 t**. Forest and River have no predicted break-even in the paper. For them, F\* is computed with R3's rule before any run with a fine: g\* = max over d ∈ {¼, ½, ¾, 1} with G(d) > 0 of G(d)/n(d), where G(d) is the cheaters' mean net gain over complying (per cheater) and n(d) their mean number of over-take rounds, on 8 training populations with the reviewer and no audits; F\* = g\*/q. The cheaters are the paper's greedy requesters.
2. **Diagram 2 agents.** "Same agents" is read as the paper's populations for that game (Fishery: 4 of 6 greedy requesters; Forest: 2 of 6 adversarial; River: one X and one Y stress discharger, as in C1). The k = 0…N over-takers are chosen at random, independently of request type. Returns are net of fines.
3. **Paired populations.** Population c draws, once, both a restrained and a greedy policy for every agent, and a random order; the first k agents in that order are greedy (or over-takers). Weather and audit draws depend only on (c, round). So the populations are nested across k and F, and the bootstrap resamples populations jointly across k (paired). 4,000 resamples. Return = an agent's total payoff over the horizon.
4. **Indexing.** R_c(m) is read from populations with k = N − 1 − m greedy, R_d(m) from k = N − m.
5. **River.** Payoff = discharge (C1 protocol, "Each agent earns its discharge"). Restrained / greedy = C1's normal / stress request ranges (`wide_low`). The reviewer aims at quality 50 (R4, as in the paper's grid). Harm = Q′ < 50.
6. **Harm**, at a fixed horizon (rounds after a collapse or forest failure count as harmful): Fishery, stock after fishing < K/2; Forest, mean plot health after the round < 10; River, Q′ < 50.
7. **G2 factors by game.** r, N and T are varied in Fishery only; the neighbour externality in Forest; κ in River with λ held at its paper value 1.6 (one factor at a time). For N: K = 100N/6 and every stock level (start 70, thresholds U(25, 65), collapse line 10, harm line K/2) is scaled by K/100; the number of greedy agents is round(4N/6) (3, 4, 5, 7 for N = 4, 6, 8, 10).
8. **G2 deterrence grid.** The closed-form prediction at a grid point is that point's own g\*, computed by the rule in item 1 on 8 training populations. The fine grid is F = j × 0.1 × F\*\_pt, j = 0…30 (F\*\_pt = g\*\_pt/q), and "audits with F = 2F\*" uses F = 2F\*\_pt (j = 20). The observed threshold is the first e on the grid at which the training search picks d = 0; it is also reported whether d stays 0 at every larger e. S2 holds at a point if |e_obs − g\*\_pt| ≤ 0.25 g\*\_pt. If g\*\_pt ≤ 0 (over-taking never pays), S2 is not testable there and the F = 2F\* arm uses twice the paper-setting F\* of that game.
9. **S3** holds at a point if harm under audits (F = 2F\*, best-responding over-takers, with the reviewer) is below harm with no reviewer and no audits; the 95% paired bootstrap interval is reported, and points where it includes 0 are flagged.
10. **Seeds.** All seeds are `stable_seed(G1_BASE or G2_BASE, role, game, ...)` (`fishery_sim/calibrated_oversight.py`), so G1 and G2 populations cannot share seeds even though the bases differ by only 20.

### Amendment 2 (14 October 2026, after a smoke run of 2–3 populations of 20 rounds; before any full run)

The smoke run gave g\* = 0 for Forest by the rule of Amendment 1, item 1: with the reviewer and no audits, over-taking lowered the cheaters' own payoff at every level. The existing S2 run shows the same at the paper setting (Forest, no audits: training payoff 64.1 at d = 0 falling to 60.2 at d = 1, `results/runs/claude_s2_compliance_deterrence_v1/adaptive_search.json`). Then F\*/2 = F\* = 2F\* = 0. Fixed now, before the full run:
- If the full run also gives F\* = 0 for a game, its Diagram 2 is run and reported at F = 0 only, and the three coinciding fines are not duplicated. P2 is then tested only at F = 0 (D3 should hold), and a D3 failure there is reported as a P2 failure.
- In G2, a grid point with g\*\_pt = 0 is not testable for S2 (as Amendment 1, item 8), and its "audits" arm uses F = 2 × (that game's G1 F\*), which is 0 if the G1 F\* is 0; that arm is then reviewer plus audits with no fine.
Nothing else from the smoke run was looked at beyond checking that every arm ran and that audits occur at rate 1/6.

### Amendment 3 (14 October 2026, **post hoc**: written after seeing the G1 results, before running G1b)

The G1 Fishery audited diagram failed P2: over-taking still paid at the paper's F\* = 2.136 t and at 2F\*. The candidate explanation, not yet tested: the paper's g\* is the gain per over-take round of a **group** of committed greedy cheaters, who share the stock depletion they cause. A **lone** over-taker passes most of that depletion on to the others, so its gain per over-take round is larger, and so is the fine that deters it.

**G1b test**, Fishery only, programmed agents, CPU:
1. **Prediction, on training populations only** (`role = "train"`, 8 populations, base `G1_BASE`; never the test populations). For each training population, g_lone = the mean over agents i of [net payoff of i when it is the only over-taker (d = 1) − net payoff of i when it complies] ÷ i's over-take rounds. Reviewer on, no audits, F = 0, the paper's greedy requesters. Then F_lone = g_lone ÷ q.
2. **Test, on the G1 test populations** (64): the Diagram-2 gain for a lone over-taker (m = 5) at F ∈ {0, ½, 1, 1.5} × F_lone.
   - **Holds** if the gain's point estimate is > 0 at ½F_lone and ≤ 0 at F_lone, and its interval excludes positive values at 1.5F_lone.
3. **Also reported (descriptive):** the gain at every m for those fines, to show how the deterring fine changes with the number of other cheaters.

The result is reported as post hoc. Nothing else in G1/G2 changes.
