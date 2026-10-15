# Results G1/G2: are Fishery, Forest and River social dilemmas at our settings, and do the programmed results survive a sweep?

**Protocol:** `protocol.md` in this folder, frozen 14 October 2026 before any G1/G2 code existed. Two amendments were appended before the full runs: Amendment 1 (written after reading the code, before anything ran) and Amendment 2 (after a 2–3 population smoke run that showed Forest's break-even fine is 0).

**Runs** (one full run each, 14 October 2026, CPU only, no LLM calls):
- G1: `results/runs/claude_g1_v1/` (95 s). Summary: `notes/claude_audit_20261005/runs/claude_g1_v1/g1_summary.json`, curves in `g1_curves.csv`.
- G2: `results/runs/claude_g2_v1/` (269 s on 8 processes; 20 grid points, 18 distinct). Summary: `notes/claude_audit_20261005/runs/claude_g2_v1/g2_summary.json`, one row per point in `g2_points.csv`.
- Figures: `notes/claude_audit_20261005/figures/fig_g1_schelling.{pdf,png}` and `fig_g2_sweeps.{pdf,png}`.
- Code: `experiments/oversight/run_g1_schelling.py`, `run_g2_sweeps.py`, `analyze_g1_g2.py`; tests in `tests/test_g1_g2.py` (16 pass; the full suite passes, 434 passed and 11 skipped).

**Status of the numbers.** The protocol's tests (D1–D3, P1, P2, S1–S3) are **verified**: they were fixed before the run and computed by the analysis script. Everything marked **post hoc** is an explanation written after seeing the results. Intervals are 95% population-bootstrap intervals (4,000 resamples), paired across k, because population c uses the same draws at every k.

**Part B (G1-LLM, the role-prompt Schelling diagram with gpt-oss-120b) was not run.** It is waiting for credit (about $3).

## Question

1. Is each game a social dilemma at the paper's settings? A dilemma means mutual restraint beats mutual greed (D1), restraint among restrained agents beats being exploited (D2), and greed tempts through fear or greed (D3). This is the check commons-game papers use, from Schelling 1973, Leibo et al. 2017 and Hughes et al. 2018.
2. With the reviewer and audits, does over-taking stop paying once the fine reaches the predicted break-even F\* (P2)?
3. Do the main programmed results still hold when one setting at a time is changed: the dilemma (S1), the stopping fine falling within 25% of the prediction (S2), and audits reducing harm (S3)?

## What was done

- **Agents.** The existing programmed policies were used unchanged:
  - Fishery: the restrained and greedy request ranges;
  - Forest: the cooperative and adversarial strategies;
  - River: C1's normal and stress dischargers.
- **Diagram 1 (no reviewer, no audits).** For k = 0…6 greedy agents, 64 populations were run for 80 rounds. R_c(m) and R_d(m) are the mean returns of a restrained or a greedy agent, where m is the number of *other* agents who are restrained.
- **Diagram 2 (reviewer and audits).** These runs used the paper's population of request policies with the half-capacity reviewer and audits at q = 1/6. For k = 0…6, agents chosen at random over-took fully (d = 1). Fines were F ∈ {0, F\*/2, F\*, 2F\*}, and returns are net of fines.
- **F\*** (Amendment 1).
  - Fishery uses the paper's value: g\* = 0.356 per over-take round, so F\* = 2.136 t.
  - The paper gives no F\* for Forest or River, so R3's rule computed it on 8 training populations before any run with a fine.
  - That rule gave Forest F\* = 0 (over-taking lowered the cheaters' own payoff at every level, G = −0.75 to −2.30 per cheater). So Forest's Diagram 2 has only F = 0 (Amendment 2).
  - River F\* = 0.853 (g\* = 0.142 discharge units per over-take round, at d = 1).
- **G2.** One factor was changed at a time from the paper values, giving 20 grid points:
  - Fishery growth r: 0.4–1.0;
  - Fishery group size N: 4–10, with K/N fixed;
  - Fishery horizon T: 20, 40, 80;
  - Forest neighbour damage: 0–0.44;
  - River synergy κ: 0.5×, 1×, 2×.
- **At each G2 point:**
  - Diagram 1 and D1–D3;
  - the predicted break-even g\*\_pt from 8 training populations;
  - the search over d ∈ {0, ¼, ½, ¾, 1} at e = j × 0.1 g\*\_pt for j = 0…30, using 8 training and 64 test populations;
  - harm with no reviewer, compared with harm under reviewer plus audits at F = 2F\*\_pt.

## What came out

### G1, Diagram 1: all three base games pass D1–D3 (P1 supported)

Source: `runs/claude_g1_v1/g1_summary.json`, `games.<game>.diagram1`.

| Game | R_c(5): all restrained | R_d(0): all greedy | R_c(0): exploited | D1 | D2 | D3 (fear / greed) | Dilemma |
|---|---:|---:|---:|---|---|---|---|
| Fishery (t) | 145.6 | 19.4 | 7.7 | +126.2 [+123.7, +128.7] | +137.9 [+135.5, +140.4] | yes / yes | **yes** |
| Forest (harvest) | 160.9 | 14.0 | 10.9 | +146.9 [+145.1, +148.7] | +150.0 [+148.2, +151.8] | yes / **no** | **yes** |
| River (discharge) | 20.9 | 16.5 | 4.2 | +4.5 [+3.9, +5.0] | +16.8 [+16.2, +17.3] | yes / yes | **yes** |

- **Fishery** is a full dilemma. A greedy agent among five restrained ones earns 409 t against 146 t for a restrained one (greed). A greedy agent among greedy ones earns 19 t against 8 t for a restrained one (fear).
- **Forest passes by fear only.** A greedy agent beats a restrained one only when at most one other agent is restrained (m = 0: +3.15 [+2.57, +3.71]; m = 1: +2.91). When most others are restrained, greed loses heavily (m = 5: −128.8 [−130.9, −126.7]).
  - The restrained agent's return jumps from 16 to 161 between m = 2 and m = 3 (Figure, panel b). With 3 or more greedy agents, harm is 99–100% of rounds and the forest fails.
  - In the terms of Leibo et al. this is a Stag Hunt, not a Prisoner's Dilemma.
- **River passes with a small D1 margin** (4.5 discharge units over 80 rounds). In River an agent's payoff is its discharge, and quality affects payoffs only through the agents' own threshold rules.

### G1, Diagram 2: P2 fails in Fishery, holds only at 2F\* in River

Source: `games.<game>.diagram2`; Figure panels d–f show R_d(m) − R_c(m), the gain from over-taking.

- **Fishery: over-taking still pays at F\* and 2F\* (P2 fails).**
  - Gain at m = 3: +135.2 [+111.6, +157.7] with no fine; +106.6 [+82.8, +129.1] at F\*; +78.0 [+54.1, +100.4] at 2F\*.
  - D3 holds at every fine, by fear and by greed. At F = 0, D3 holds as predicted.
- **River: P2 fails at F\*, holds at 2F\*.**
  - At F\*, a lone over-taker still gains +5.69 [+1.37, +10.10] (m = 5).
  - At 2F\* every point estimate is ≤ 0 (largest −2.71 [−7.20, +1.90]), so D3 fails, as predicted.
- **Forest (F = 0 only).** D3 holds, as P2 predicts at F = 0 (m = 1: +11.7 [+1.0, +23.1]).
  - But the audited Forest at F = 0 is **not** a dilemma: D1 fails (−3.68 [−4.76, −2.62]: all over-taking beats all complying) and D2's interval includes 0 (+8.2 [−4.6, +21.2]).

### G2: sweep criteria

Source: `runs/claude_g2_v1/g2_summary.json`; `g2_points.csv`.

| Factor | Values | Dilemma (S1) | Predicted g\* | Observed ÷ predicted stopping fine (S2) | Harm, none → audits at 2F\* (S3) |
|---|---|---|---|---|---|
| Fishery r | 0.4, 0.55, 0.7, 0.85, 1.0 | 5/5 | 0.33, 0.30, 0.38, 0.23, 0.37 | 1.1, 1.1, 1.1, 1.1, 0.9 | 100% → 0% (r = 1.0: 29% → 0%) |
| Fishery N | 4, 6, 8, 10 | 4/4 | 0.15, 0.38, 0.39, 0.31 | 1.0, 1.1, 1.1, 1.0 | 100% → 0% |
| Fishery T | 20, 40, 80 | 3/3 | 0.46, 0.41, 0.38 | 1.0, 1.1, 1.1 | 100% → 0% |
| Forest neighbour damage | 0, 0.11, 0.22, 0.33, 0.44 | 5/5 (fear only) | 0 at every point | not testable: over-taking never pays | 41, 45, 50, 54, 58% → 0.4–0.6% |
| River κ | 0.5×, 1×, 2× | **2/3** (2× fails D1) | 0.10, 0.18, 0.18 | **1.3**, 1.2, (1.2) | 15, 59, 99% → 0.3, 0.3, 0.5% |

- **S1: fails at one point, River κ = 2×.** D1 fails there (−3.85 [−4.68, −3.03]): with doubled synergy, six greedy dischargers earn more than six restrained ones. Every other point passes D1–D3. All Forest points pass by fear only.
- **S2: holds at 13 of 14 testable points and fails at River κ = 0.5×.** There the observed threshold is 1.3 × g\* (e = 0.134 against 0.103).
  - S2 was not applicable at the five Forest points (g\* = 0) or at River κ = 2× (not a dilemma).
  - At every testable point, once d reached 0 it stayed 0 at every larger fine.
  - At the Fishery paper point, the new seeds gave g\* = 0.378, against the paper's 0.356, with the observed threshold at 1.1 g\*.
- **S3: holds at all 20 points.** Every interval of the paired difference excludes 0. At 2F\* the best-responding over-takers chose d = 0 everywhere. Forest's "audits" arm had a zero fine (Amendment 2) and still removed nearly all harm.

## What it means

**What it shows:**
- All three games, at the paper's settings and with the paper's programmed agents, meet the standard Schelling dilemma test. The paper can claim this, with two qualifications:
  - Forest is a fear-only dilemma (Stag Hunt), driven by the forest-failure rule: a greedy agent among cooperators loses.
  - River's margin for mutual cooperation is small.
- The break-even rule for a group of committed cheaters (R3's g\*) is robust. It predicted the stopping fine within 25% at 13 of 14 testable sweep points (all 12 Fishery points within 10%, counting the paper point once per factor), across growth rates, group sizes and horizons. In River it was 20–30% late.
- Audits plus a reviewer at twice the predicted fine removed nearly all harm at every sweep point. In Fishery and River the fine deterred, so the reviewer's allowances did the work. In Forest no fine was needed, because over-taking never paid.

**What it does not show:**
- **P2 failed in Fishery** at F\* and 2F\*, and in River at F\*. The paper's F\* is the break-even for the paper's *group* of greedy requesters, each committing to one level. It is not the fine that makes a lone or random over-taker indifferent.
  - *Post hoc explanation, untested:* when few others over-take, the stock stays high and allowances are cut deeply, so taking the full request gains far more per round than g\*. This matches the paper's own statement that g\* ≤ g₁.
  - The paper should not describe F\* as the fine at which over-taking stops paying for any agent.
- The sweeps change one factor at a time. Interactions (for example high κ with fewer greedy agents) were not tested.
- Nothing here says how LLM agents fill the roles. That is Part B, pending credit.

## Design caveats

- **Schelling comparisons are across populations.** R_d(m) and R_c(m) come from populations with k and k + 1 greedy agents, and in Diagram 2 the over-takers are a random mix of request types. This follows Hughes et al. 2018, but it is not a same-agent counterfactual.
- **River payoffs are discharge.** Degrading the river costs an agent nothing directly, so River's dilemma rests on the threshold rules. Doubling κ breaks it.
- **The Forest dilemma depends on the failure rule.** That rule (half the plots below 4 for 5 rounds ends the game) has no recorded grounding (audit row H3), and it is not in the paper's harm definition.
- **Fishery harm with no reviewer is 100%** of rounds at most points, because the greedy requesters exhaust the stock. So S3 is easy there. Only r = 1.0 (29%) is a non-trivial test.
- **Small training set.** The S2 search used 8 training populations, as the protocol states. The paper's R3 used 32.
- **Grid steps.** The stopping-fine grid has steps of 0.1 g\*, so ±25% means within about two grid steps.
- **Shared paper point.** The paper point was run once and its result reused for the r, N and T factors, so it counts three times among the 20 points (18 distinct).

---

## G1b (post hoc, Amendment 3): a lone over-taker needs about three times the group's fine

**Question.** Is the P2 failure in Fishery explained by a lone over-taker gaining more per over-take round than the paper's committed cheating group? A lone over-taker passes most of the depletion it causes on to the others.

**What was done.**
- g_lone was predicted on the 8 G1 training populations only. It is the gain per over-take round of each agent when it alone over-takes in full, with the reviewer on and no audits, averaged over agents and populations.
- F_lone = g_lone ÷ q.
- The lone over-taker's gain (m = 5) was then measured on the 64 G1 test populations at 0, ½, 1 and 1.5 × F_lone.
- Code: `experiments/oversight/run_g1b_lone_breakeven.py`. Output: `runs/claude_g1b_v1/g1b_summary.json`. Runtime 18 s.

**What came out** [post hoc; verified from `g1b_summary.json`]

| | Value |
|---|---|
| g_lone (per over-take round) | **1.14 t**, against the group's g\* = 0.356 t |
| F_lone = g_lone ÷ q | **6.87 t, 3.2 × the paper's group F\*** (2.14 t) |

| Fine | Lone over-taker's gain over complying (t, 80 rounds) |
|---|---:|
| 0 | +92.2 [+62.7, +120.2] |
| ½ F_lone | +48.3 [+18.8, +76.9] |
| F_lone | +4.4 [−25.8, +33.5] |
| 1.5 F_lone | −39.5 [−70.8, −9.2] |

**Verdict against the criterion fixed in Amendment 3: narrowly fails.**
- The criterion required a point estimate ≤ 0 at F_lone. It is +4.4 t, which is 95% of the no-fine gain removed, and its interval includes 0.
- The other two parts hold: the gain is positive at ½F_lone, and clearly negative at 1.5F_lone.
- So the lone over-taker's stopping fine lies between 1.0 and 1.5 × the prediction made from training populations.

**What it means.**
- The fine that deters depends on **who is cheating and how many**. A fine sized for a group of committed cheaters (the paper's F\*) under-deters a lone cheater by a factor of about 3, because the lone cheater does not bear the depletion it causes. This is the commons externality itself.
- It fits the paper's thesis (penalties sized to *the cheater's* gain). It also means the paper's programmed-agent threshold must be described as the threshold **for that cheating group**.
- **For the LLM results:** L3/L4 used each model's *one-round* individual gain ĝ₁, so their threshold is already a per-agent quantity. This finding does not undercut them.

**What it does not show.**
- Whether intermediate group sizes interpolate smoothly. The by-m gains in the summary are descriptive.
- Anything about Forest or River.
