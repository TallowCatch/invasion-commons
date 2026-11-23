# Phase 3: hypotheses (draft 1, 23 Nov 2026)

**What this is.** The claims that follow from the overseer definition (`02_overseer_definition.md`). Each one gives:
- the part of the overseer's action it is about (look, judge, respond, share) or the agents' side;
- a precise prediction;
- the result that would refute it;
- the existing evidence;
- what is still needed, and the cheapest way to get it.

Then every experiment run so far is mapped to a hypothesis, marked keep, drop or gap. **No new experiment runs unless a hypothesis here needs it.**

Numbers quoted are from the tracker and findings log (`OVERSEER_TRACKER.md`, `claude_audit_20261005/FINDINGS_AND_IMPLICATIONS.md`). Harvest unless stated.

---

## One quantity used throughout: the sanction a violation actually meets

Each step, the overseer's look, judge and respond together determine what a violating agent can expect to lose. With random looks at k of N agents, inspection window w = 1, miss rate α and sanction F per flagged agent:

  ρ = (k / N) · (1 − α) · F.

In words: the chance of being looked at, times the chance the inspector sees the violation, times the sanction.
- *Harvest:* k = 2, N = 7, α = 0, F = 3.5 apples gives ρ = 1 apple.
- *Source:* derived from the definition (§2.1–2.3). For targeted looks, k/N is replaced by the measured share of violations that get inspected.
- ρ is a property of the overseer's action, not a theory of the agents. Whether learners respond to ρ alone is hypothesis H1.

---

## H1. Learners respond to the overseer's action only through ρ (look × judge × respond)

**Claim.** Against ordinary learners, the commons' safety S and welfare W depend on k, α and F only through ρ. There is a threshold ρ*: below it the learners strip the commons, above it they keep it.

**Prediction.**
- Two overseers with the same ρ give the same S and W, whatever their k, α and F.
- In Harvest, ρ* lies between about 0.75 and 0.9 apples.

**Refuted if** two configurations with equal ρ differ in S or W by more than the seed range.

**Evidence: supported in Harvest, with one refinement.**
- At ρ = 0.5 the orchard dies even with every agent inspected. At ρ = 1 it survives (441–477 apples). At ρ = 2 it survives (515–530) (stage 1).
- With 50% misses, a 1-apple sanction behaves like 0.5 (mistakes test).
- 15 of 16 LLM-inspector cells fall on the side of ρ* predicted from each model's measured α (F23–F25).
- The regional runs that failed had their ρ pushed below ρ* by a lower inspected share (0.65–0.75 apples) (DECENTRALISED_RESULTS).
- **Refinement (H1b):** at equal ρ, few large looks (k = 1) do worse than many small ones (k = 7) when the inspector errs (342 vs 479 apples; F20, F24). So ρ sets the outcome, but the spread of sanctions slows learning.

**Still needed.**
- The same test in a second game. Clean Up runs are in progress (24 runs, 5 done).
- A one-paragraph derivation of why learners would respond to ρ: the best response of an agent whose gain from one violation is fixed.

**Status: keep. Mostly answered.**

---

## H2. How the overseer chooses whom to look at matters only against agents who can see the looks (look policy × agent mode × capability gap)

**Claim.**
- Against ordinary learners, the look policy (random, repeat offenders first, round robin) changes how fast learners comply, but not where they end up at fixed ρ.
- Against adversarial agents who see who was inspected, predictable look policies are exploited. Random looks are not.

**Prediction.** The capability gap Δ (definition §5) is large for predictable look policies and about 0 for random ones.

**Refuted if**, with learners who see the looks, random looks lose as much as predictable ones, or predictable ones hold.

**Evidence.**
- *Ordinary learners:* repeat-offenders-first is safe sooner (by about 13M steps) but ends at the same 460–470 apples as random (finding 6). Supported.
- *Adversarial, bots only:* round robin and repeat-offenders-first are fully dodged (no catches). Random holds (caught share about 0.30) (worst-case bot preview). Supported for bots.
- *Adversarial, learners:* **missing.** The 18 worst-case runs are held, because they used separate networks, which are undertrained at 1e8 steps.

**Still needed: learners who see the looks.**
- **Proposed redesign:** run the adversarial mode with the standard *shared* network at 1e8 steps. Shared networks converge in that time.
- A shared network can still learn to dodge, because each agent's own observation includes "was I inspected last step" and "steps since I was last inspected".
- 18 runs, about 15 GPU-hours.
- This separates H2 (does seeing the looks help agents) from H6 (do separate networks matter).

**Status: keep. Central. The redesigned runs answer it.**

---

## H3. Learners exploit an inspector's patterned misses (judge × agents)

**Claim.** When the inspector's miss rate depends on the situation, α(s), learners place their violations where α(s) is high. The outcome is then worse than with a coin-flip inspector of the same average α.

**Prediction.**
- Under the patterned gpt-oss table, S and W are lower than under a coin flip at the matched average α (0.288 for overseen agents' records).
- More violations happen in high-α situations than under the coin flip.

**Refuted if** the patterned and matched coin-flip arms give the same S, W and placement of violations within seed ranges.

**Evidence.** The patterned arm is done (12 runs). The inspected share of violations missed was 0.27–0.29, and about 60% of violations were in the borderline situation (2 apples near). Learners already violated there before this inspector existed, so placement alone proves nothing. The matched coin-flip arms are running (24 of 36 runs left).

**Status: keep. Answered when job 38836358 finishes.**

---

## H4. Misses and false alarms do different things (judge)

**Claim.** Misses lower ρ, so they change what learners do. False alarms that can only fall on a legal harvest, and never on the choice to wait, do not change whether violating pays. They only harm compliant agents (H).

**Prediction.** At fixed α, raising β leaves S unchanged and raises H in proportion to β × F.

**Refuted if** β alone lowers S at fixed α and k.

**Evidence: supported.**
- Qwen 3B with reasoning never misses and wrongly flags a quarter of legal harvests. It keeps the orchard (380 apples; 480 at k = 7) and reproduces the 25%-false-alarm arm almost exactly (F23–F25).
- Wrongful sanctions take about 45% of what compliant agents collect at F = 2 apples.
- The literature reconciliation is in `llm_overseer/error_patterns/results.md`: Png 1986 and Polinsky & Shavell predict false alarms weaken compliance, unless they cannot fall on the lawful alternative (Lando 2006).

**Still needed (optional).** A map where a legal apple is always in reach, so that waiting is rarely the alternative. That would test where the reconciliation breaks.

**Status: keep. Answered for Harvest.**

---

## H5. When harm is collective, inspecting individuals cannot protect the commons (local pass, global fail)

**Claim.** The local-pass/global-fail rate is a property of the game and its rule. Where it is high, even inspecting every agent with a perfect inspector leaves the commons unsafe.

**Prediction.**
- In Harvest, LPGF is low: perfect full inspection already keeps the orchard alive in every run (k = 7, α = 0: orchard alive 100% of games).
- In Clean Up, where the violation (not cleaning) builds up slowly, LPGF is higher.

**Refuted if** a testbed with high LPGF is still fully protected by perfect full inspection, or one with low LPGF is not.

**Evidence: supported in Harvest (23 Nov; `h5_lpgf/results.md`).** 5 trained runs and 3 bot populations, 32 episodes each.
- Permanent loss (a patch losing its last apple): 0 of 580 patch extinctions came from legal eats only. Every one involved a violation. This is structural: a patch's last apples can only go legally if 4 or more agents eat them in the same step.
- Perfect full inspection (k = 7): orchard alive in 32 of 32 episodes, no patch lost.
- Temporary local damage passes the rule more often: 60–93% of newly dead cells in overseen runs come from legal eats, but 92–95% of them revive within the episode.
- The June predicate at τ = 0.5 reports 84–98% LPGF for sustainable play. That is a threshold artefact: play settles at 23–29 apples, just under 0.5 × 64.

**Still needed:** Clean Up, once its runs finish (`measure_lpgf.py` is ready).

**Status: Harvest answered (low LPGF; no case for the commons view from Harvest). Clean Up open.**

---

## H6. Learners with their own networks exploit the overseer more than parameter-sharing ones (agents' capability)

**Claim.** At the same overseer and ρ, independent learners (one network each) violate more than learners sharing one network, once both have converged.

**Prediction.** At convergence, separate networks give more violations and lower W at the same ρ.

**Refuted if**, at convergence, the two give the same outcome.

**Evidence: inconclusive.** At 1e8 steps the separate-network runs were still improving: violations falling by 4–6 and apples rising by 18–31 per 100 updates at the end, while shared runs had levelled off. The apparent 50% more violation is mostly slower learning.

**Still needed.** Separate-network runs at 3e8 steps, checked for convergence.
- **Reduced version:** central random and 2-region random only, 3 seeds = 6 runs, about 15 GPU-hours.
- The full 15 runs cost about 40.

**Why it matters for the formulation.** The agents in the definition each maximise their own return; independent networks are the faithful version. Casheekar & Tangirala 2026 show parameter sharing changes how much inspection is needed.

**Status: keep, lower priority. Reduced rerun.**

---

## H7. Splitting the overseer costs most where agents crowd unevenly, and sharing or reallocation recovers it (share; several overseers)

**Claim.** With M regional overseers on fixed equal budgets:
- the loss against one central overseer comes mainly from mismatch between budget and crowding, not from agents crossing boundaries;
- giving each region looks in proportion to how many agents are in it removes most of the loss;
- with local records only (no free hand-over), the extra catch loss rises with the locality mismatch.

**Prediction.**
- With budgets proportional to headcount, S and W are close to central.
- With local records, lost catches rank with the locality measure, as with bots (Spearman 0.80).

**Refuted if** headcount-proportional budgets still lose about as much as equal budgets.

**Evidence.**
- *Learners:* 4 strips at ρ = 1 collapse because the inspected share falls (18% vs 28%), and wasted looks are only 9–12%. Coins halves waste 28–31% of looks (DECENTRALISED_RESULTS).
- Sharing suspicion recovers most of the loss with separate networks.
- *Bots and maths:* the locality results (F21).
- Caveat: all learner runs so far hand records over for free.

**Still needed.**
- Headcount-proportional budgets: a small code change plus about 12 runs.
- Local-records overseer with learners: about 12 runs.
- Both are for the multi-overseer extension, after H1–H5.

**Status: keep, later.**

---

## H8. A fixed overseer props up behaviour rather than changing what is learned (respond × timing)

**Claim.** When the overseer is switched off after training, violation returns, whatever the look policy and ρ.

**Refuted if** some overseer configuration leaves behaviour that persists after switch-off.

**Evidence: supported in Harvest.** 18 of 18 runs revert within about 5–12M steps (finding 8).

**Status: keep. Answered.** A possible later question: whether a learned or adaptive overseer can do otherwise.

---

## H9. Violations that build up slowly need longer looks (look window w × judge)

**Claim.** If a violation is not visible within the w steps one inspection reads, the overseer must read further back (more burden) or fail.

**Prediction.** In Clean Up, raising w to match the rule's window restores protection, at a cost growing with w.

**Refuted if** short looks protect Clean Up as well as long ones.

**Evidence.** Bots: the look needed equals the rule's window (Clean Up dial, finding 15). Learners: Clean Up runs in progress.

**Status: keep. Answered when the Clean Up runs finish.**

---

## A candidate result for the theory slot (Du's template)

Du et al. pair their definition with short bounds on what locality costs. The analogue here would be a bound on what looking at k rather than N agents costs. If H1 holds, the overseer needs ρ ≥ ρ*, so

  k ≥ ρ* · N / ((1 − α) · F).

In words: the number of looks needed grows with the number of agents and the inspector's miss rate, and shrinks with the size of the sanction.

H1b says this is not the whole story when α > 0, because large sanctions with few looks slow learning. A bound that includes this spread would be the more interesting result. *To be developed only if H1 and H1b hold in a second game.*

---

## Every experiment so far, mapped

| Experiment | Hypothesis | Verdict |
|---|---|---|
| Stage 1: k, ρ and look policy (48 runs) | H1, H2 (ordinary) | keep (done) |
| Stages 2–3: switch off and on | H8 | keep (done) |
| Freezing caught agents | respond variant | done; no further work |
| Mistakes test (α, β) | H1, H1b, H4 | keep (done) |
| LLM inspector ladder (offline) | measures judge capability | keep (done) |
| LLM-inspector RL runs (40) | H1, H4 | keep (done) |
| Error-pattern runs (36) | H3 | keep (running) |
| Decentralised Harvest (69) and Coins (45) | H7 | keep (done; records-handover caveat) |
| Clean Up (24) | H1 in a second game, H9 | keep (running) |
| 14-agent Harvest (21) | H1 as N changes; H7 | keep (queued) |
| Locality, bots and trained agents | H7 | keep (done) |
| Willis commons at N = 64, 256 | H7 at scale | keep (done) |
| Worst-case bot preview | H2 (bots) | keep (done) |
| Worst-case learners, separate networks (18, held) | H2 | **redesign:** shared networks, 1e8 steps |
| Separate networks at 1e8 (15) | H6 | inconclusive; **reduced rerun at 3e8** (6 runs) |
| BazaarBench offline replay | judge capability; second testbed | keep as secondary testbed |
| Coins as a testbed | (2 players, weak dilemma) | drop further work |
| LPGF measurement | H5 | Harvest done (LPGF 0% on permanent loss); Clean Up when its runs finish |
| Headcount-proportional budgets; local records with learners | H7 | gap, later |

---

## Decisions for you

1. **H5 first (no GPU).** Measure local-pass/global-fail from trained agents we already have. *Recommend yes.*
2. **H2: redesign the held worst-case runs** to use shared networks at 1e8 steps (18 runs, about 15 GPU-hours), instead of separate networks at 3e8 (about 45). *Recommend yes.* It answers the central question cleanly.
3. **H6: the reduced separate-network rerun** (6 runs at 3e8, about 15 GPU-hours), or drop H6 for now. *Recommend the reduced rerun, after H2.*
4. **H7 extensions** (headcount budgets, local records): defer until H1–H5 are settled. *Recommend defer.*

Waiting on runs already going: H3 (error patterns) and H9 plus H1-in-a-second-game (Clean Up). No action needed.
