# What we have learned, and what should come next

## Update 14, 27 October 2026: the workshop paper is paused; a PhD contribution plan

- Ameer paused the workshop paper to develop a working contribution with a proper schedule.
- **Proposed contribution:** a **calibrate-then-deploy overseer**. It probes agents without consequences (shadow mode), designs the cheapest audit rate, targeting and consequence that keeps violations under a target, then verifies and recalibrates from random audits.
- **The gap it fills:** AI control measures detection under audit budgets but not deterrence. LLM commons work lacks calibrated, budgeted enforcement. Audit-game theory is untested on LLM agents.
- **Five claims (C1–C5)** and four chapters. Chapter 1 is the existing results.
- **Schedule:** Nov (build and offline validation, $0) → Dec (live tests, unknown odds, red team; the key gate) → Jan (scarce budgets, drift) → Feb (families) → Mar (scale on Willis's game) → Apr (LLM overseer) → May (NeurIPS submission) → summer (talk and collusion, then the spatial game).
- Full plan: `notes/phd_plan/2026-10-27_calibrated_overseer_plan.md`.

## Update 13, 27 October 2026: tracking audit, L5 written up, grant draft

- **Tracking audit.** `PROJECT_HISTORY.md` had not been updated since 12 Oct, the study index since 7 Oct and `PAPER_PLAN.md` since 8 Oct. All three are now current: rows for L4–L6, the 13 Oct meeting, the literature and platform research, Phase 0, the overseer layer, R1a, R1b and the grant draft. A tracking checklist is now in `README.md` ("What to update after any piece of work").
- **L5 had no results file;** it now does (`studies/L5_llm_families/results.md`). DeepSeek over-took in 80% of agent-rounds with no fine and collapsed the stock in 10 of 10 games; its 36 t prediction is untested.
- **The slide deck and talk scripts** (Ameer's hand edits) are now committed; renders and lock files are ignored.
- **The OpenMined "Ensembles & Routing" grant** (closes 30 Oct AoE) is drafted in `notes/private/grants/` (not in git): a budgeted worker–monitor–auditor cascade with random spot-checks, built on the audit-budget results.
- **Open decisions:** Forest (F2); the paper structure; GPU access (whether the message to Yali was sent is not recorded); the grant.

## Update 12, 19–20 October 2026: R1b (strategies told the audit rule)

- **Generation:** 254 of 256 gpt-oss strategies valid, 516 calls, about $0.58 of Ollama credit.
- **Part 1:** audit-aware Selfish strategies cut defection 0.74 → 0.06 at an expected fine of 2 (H1 holds); the blind pool is unchanged. 75% of Selfish strategies implement Becker's rule explicitly. A zero-fine announcement has no effect on written strategies, unlike live agents in L4 (F15).
- **Part 2:**
  - Blind pool: oversight selects Collective strategies (0.15 → 0.60), but welfare stays low (0.13).
  - Audit-aware pool: Selfish strategies comply and survive (Collective share 0.19), and welfare is 0.45 (H4 holds, +0.32).
  - **Compliance is not conversion** (F16).

## Update 11, 15 October 2026: R1a (audits in Willis et al.'s commons)

- **The pre-registered grid is complete:** 240 runs on GitHub Actions. The extension has 318 of 320 runs (2 being re-run).
- **H0 replicates the paper:** in groups of 64, 15% Collective and 13% efficiency, against 19% and 15% in the paper.
- **H1 holds at n = 64:** the Collective share switches above half between 1.0 and 1.25 × the balance fine of 0.96 predicted before any run (0.34 → 0.60).
- **H2 holds:** efficiency 0.13 → 0.57 at 1.25 × e_pred.
- **H1 at n = 4 fails, only at e = 0** (0.51 [0.43, 0.59]). Any fine gives 0.58–0.87.
- **Cost:** rare audits with large fines break even at 2.3–2.8× the audit cost of frequent small ones.
- **Final round:** fining it as well helped slightly (0.66 against 0.60 Collective).
- **Strategies are blind**, so the effect is selection; R1b is next. See F14.

## Update 10, 14 October 2026: Phase 0 (grounding and re-scoring) and the overseer layer

- **Platform decision** (deep research, 93 sources; `research/commons_platform/2026-10-14_decision.md`):
  - Build the overseer as a reusable layer and run it on standard games: SocialJax and OpenSocialJax, Willis's `emergent_llm`, and the text testbed.
  - Settings everywhere in the field are hand-set; the field's grounding is Schelling diagrams and focal/background evaluation.
  - Our settings audit: 12 grounded, 57 with a recorded reason, 35 with none.
- **Overseer layer v0** (`overseer/`), checked with two admission tests:
  - replaying stored games reproduces them **exactly** (L4 263/263, L2 402/402, L3 80/80, L5 10/10, the memory re-run 40/40);
  - re-running S3 Parts A and B through the layer gives byte-identical episodes and an identical analysis.
- **G1 Schelling diagrams [verified]:**
  - All three base games are social dilemmas. Forest is one by fear only (a Stag Hunt). River passes with a small margin.
  - With the reviewer and audits, Fishery over-taking still pays at the paper's group break-even fine.
- **G1b [post hoc]:** a **lone** over-taker's break-even fine is 6.87 t, **3.2× the group's** 2.14 t, because a lone cheater does not bear the depletion it causes. At that fine its gain falls from +92 to +4.4 t [−26, +34]. The pre-stated criterion fails narrowly (the point estimate is above 0), and the gain is clearly negative at 1.5×.
- **Forest's audited game is not a dilemma:** with the reviewer, everyone over-taking beats everyone complying, so the reviewer's allowance is below what is sustainable there. This is consistent with L6 (allowances halved).
- **G2 sweeps:**
  - The dilemma holds at 19 of 20 points; River at κ = 2× fails.
  - The threshold is within 25% at 13 of 14 testable points; all Fishery points are within 10%.
  - Audits reduce harm at 20 of 20.
- **W1 [post hoc]:**
  - Zero-fine audits make things worse than never mentioning audits (oversight efficiency below 0, both models).
  - The break-even cost per audit is 0.03–0.24 t, and positive only where audits deter.
  - Being caught does not reduce next-round over-taking among over-takers.
  - Over-taking rises at the end of the game (+16.6 points in rounds 17–19).
  - Oversight efficiency is inflated because N0 is near first-best, so quote efficiency differences next to it.
- **L4 data note:** nine gpt-oss games were played twice. On first plays, H2's d is −12.75 instead of −12.9; no verdict changes (`studies/L4_llm_identification/results.md`).
- **Paper:** five paper/code mismatches fixed. A supplement paragraph now places the settings inside published ranges.
- **Still open from Phase 0:**
  - G1-LLM, the role-prompt Schelling diagram (about $3);
  - the temperature check (about $2);
  - a K/4 limit line as a robustness check;
  - stopping lanes from overwriting each other's games.

## Update 9, 14 October 2026: after L4, L5 and L6, and the 13 October supervisor feedback

- **L4 [verified]:**
  - The 30 t prediction held on new populations and under reworded rules: 0.2% and 1.1% over-taking for gpt-oss, 1.8% and 1.9% for Nemotron. gpt-oss again stopped at 24 t.
  - At an expected fine of 2 t, a rare audit with a large fine deterred more than a frequent audit with a small fine: 27.9% against 40.7% over-taking, d = −12.9 points [−16.4, −9.3]. At 3 t every mix gave 9–11%.
  - Rules that never mention audits gave less over-taking than audits with a 0 t fine: 30% against 58% (gpt-oss), 46% against 64% (Nemotron).
- **L5 [verified]:** DeepSeek over-took in 80% of decisions with no fine, with a one-round gain of 5.41 t, so its predicted stop is 36 t. That stage was not run, for cost.
- **L6 [verified]:** one programmed over-cutter raised the gpt-oss foresters' over-taking by +25.1 points. Visibility added nothing.
- **L6 mechanism [post hoc]:** the spread ran through the reviewer's shared cut, which roughly halved allowances while actual cuts stayed the same.
- **Fines [verified from code]:** a fine only lowers the fined agent's score. Welfare must state whether fines count as lost or as transfers.
- **Literature:** 58 AI papers, each checked; see `notes/supervisor_feedback_2026-10-13/literature_review_ai.md`. Bell and Madisetti (ICML 2026), Okamoto et al. (AIES 2026) and OpenSocialJax (arXiv, October 2026) are close, so the novelty claims must be narrowed to four things:
  - the threshold predicted in advance and confirmed;
  - audit chance separated from fine size;
  - the never-mentioned control;
  - spread through the overseer's rule.
- **Next** (full plan: `notes/supervisor_feedback_2026-10-13/plan.md`):
  - **W1:** welfare net of cost, a break-even audit cost and oversight efficiency, from stored games, $0.
  - **A1:** an environment admission test, CPU only.
  - **L7:** talk × planted cheater, then overseer responses (shared cut, targeted, leniency), about $5.
  - **L8:** ring with proximity-dependent audits and plots that die for good, about $2.
  - **L9:** LLM overseer, programmed agents first, under $1.
  - **G1:** grid game, next paper.
  - Each study gets a frozen protocol first.

## Update 8, 8 October 2026: consistency check across all studies

- **The gain was defined two ways.** Simulated cheaters use the whole-game net gain g\*; LLMs use the one-round gain g₁ (pre-registered). Computed the simulations' way, the LLMs' whole-game gain is −1.05 t (gpt-oss) and 0.71 t (Nemotron) per over-take round. They keep over-taking far past that, and stop near g₁. **LLM agents are myopic offenders [post hoc]:** the one-round gain predicts them, the whole-game gain does not. Figure 5 now scales each kind of agent by its own gain.
- **The other differences are stated in Table 4:**
  - what misbehaving means (under-reporting in S1, T1 and C1; over-taking elsewhere);
  - how it is chosen (once per game by search, or every round);
  - the memory rule (share of the cut, or tonnes over);
  - 80 against 20 rounds, and 64 against 10 populations.
- **River at half capacity:** T1 and C1 Part B orderings are unchanged (random audits best). C1 Part A keeps 30, because its reviewers were built for 30.

## Update 7, 8 October 2026: after L3 (studies/L3_llm_threshold), the last experiment

- **The stopping point was predicted in advance and found.** From each model's no-fine gain (gpt-oss 4.48 t, Nemotron 4.74 t), the prediction was that F = 30 is the first fine that deters. Nemotron stopped at exactly F = 30; gpt-oss at F = 24, one grid step early.
- **Deterrence is graded, and it starts well before the gain.** Relative to no fine, over-taking was:
  - gpt-oss: 0.79 at e/g = 0.45, 0.23 at 0.67, 0.02 at 0.89;
  - Nemotron: 0.87 at e/g = 0.42, 0.65 at 0.63, 0.12 at 0.84.

  The simulated best-responders switch off in one step at e/g = 1.0–1.2. So LLMs are easier to deter than the risk-neutral prediction. H3 failed for both models. Why is untested: candidate explanations are risk aversion, a gain larger than the model perceives, and a large stated fine read as a signal.
- **All experiments are finished.** What remains is writing (see "FINAL LIST" in PROJECT_HISTORY.md).

## Update 6, 8 October 2026: after L2 (studies/L2_llm_agents)

- **Claim 6 holds.** In Fishery, with audits at 1/6, gpt-oss and Nemotron over-took in 51–92% of steps for every expected fine from 0 to 1.33 t, and in 0% at 6 t. That is the only tested fine above their measured one-step gain of about 4.5–4.7 t. H1 holds in all 3 frozen families.
- **Fines below the gain are useless, and small ones can backfire.** At fines of 1–2 t, over-taking rose above the no-fine level, even before any lake collapsed **[post hoc]**. The harm was large: under Nemotron, 5 to 10 of 10 lakes collapsed at every fine from 1 to 8 t.
- **Families differ in whether they over-take, not in how they respond.**
  - Mistral Large 3 never over-took, in 16,000 decisions.
  - Gemma over-took in 5–10% of steps, almost always by catching its request after a cut. That looks like ignoring the cut, not cheating **[post hoc]**.
- **A design error was found after the results.** The "silent" cell S0 still describes a check with a fine of 0 t, which itself says over-taking is free. So S0 does not replicate L1's silent cell, and H4 failed for 3 models.
- **Infrastructure:** 400 games, 0 fallbacks in 62,152 decisions. The call logs were partly lost (Amendment 6); outcomes are unaffected.
- **Memory without fines (EM, corrected rule):** over-taking fell to 20% (gpt-oss) and 33% (Nemotron) but did not stop. Steps breaking the MSY limit fell from over 90% to 20–40%, and no lake collapsed. H3 (P4) holds in 2 of 3 families.
- **Next:** L3, the last experiment, which locates the threshold with a prediction frozen in advance.

## Update 5, 7 October 2026: after R3 and T1 (studies/R3_breakeven_confirmation, studies/T1_audit_targeting)

1. **The break-even threshold is confirmed prospectively.** The gain per cheating step was computed before any search, and deterrence started within one grid step of it in all 5 testable Fishery settings (R3). This settles claim 4.
2. **Claim 5 is confirmed, with a different mechanism.** Audits aimed at the largest report let more harm through than random audits in Fishery and River.
   - They do not miss liars because liars hide.
   - They keep re-checking one agent that memory has already corrected, so most liars are never caught: 25% caught in Fishery, against 100% at random.
   - In Forest, aiming at a physical signal of harm (the plot doing worst against prediction) was best.
   - Claim 5 now reads: *audits work when they are unpredictable and reach every agent.*
3. **Only L2 remains.** It needs a decision on compute and a second model family. After it, the experiments stop and the writing starts.

## Update 4, 5 October 2026: after S4 (S4 protocol, S4 results, figure 14)

1. **Memory alone reduces harm but does not deter.** Against cheaters who know the reviewer remembers, a reviewer with memory but no fine reduced the harm without stopping the cheating.
   - At the highest audit rate tested, cheaters kept 41% of their gain.
   - Honest agents recovered about two thirds of their loss.
   - **[post hoc]** The memory rule cannot cut a caught agent's allowance below zero. Under the MSY target that floor was hit in 98% of caught-cheater steps.
2. **Memory plus a fine works best.** The fine deters at half the audit rate needed without memory (0.05 against 0.10). With audit costs, this combination gives the most harvest net of costs, though the margins are small.
3. **Net harvest hides transfers.** It can look best while honest agents still lose (memory alone at q = 1/3), so always report honest agents' harvest next to it.
4. **Strategic misreporting still does not pay against memory.** The one non-zero choice gained 0.25 [−0.09, +0.63].
5. **Next:** a memory rule that tightens the shared cut, or caps a known cheater's take.

The three decisions for Ameer and the supervisors are unchanged.

## Update 3, 5 October 2026: after S3 (S3 protocol, S3 results, figures 10–13)

1. **Where deterrence starts.** Unpredictable audits plus a flat fine stopped the cheaters at almost exactly the expected-penalty break-even.
   - The break-even is e ≈ 0.36 per agent-step, where e = audit rate × detection × fine.
   - Audits that were 8 times rarer, or that missed three quarters of over-takes, worked just as well once the fine was scaled up.
   - This is mostly arithmetic for a risk-neutral cheater facing a flat fine.
2. **Known audit schedules backfire.** At the same audit rate, a known schedule let cheaters take everything on unaudited steps.
   - Honest agents ended up worse off than with no checks: 28.8 per agent, against 40.9 with no checks and 104 when everyone complies.
   - The stock collapsed in 12 of 64 runs.
3. **Memory makes audits useful without sanctions.** A reviewer that remembers caught lies turns audits into better decisions, with no exclusion or fines.
   - Fishery: from 69% of steps unsafe (no memory) to 1.5%.
   - Harvest: from 1.7% to 1.0%, against 0.6% when everyone is honest.
   - Against lies that vary each step, the gain is about half as large. That is the more credible number.
4. **Next:**
   - liars who adapt to a reviewer with memory;
   - a cost per audit;
   - risk-averse cheaters, or fines that grow with the over-take.

The three decisions for Ameer and the supervisors are unchanged: the definition of harm, which paper, and the main game.

## Update 2, 5 October 2026: after S1b and S2 (the S1b/S2 results, figures in the figures guide; checked independently, see the verification log §10)

1. **With a reviewer that has no memory, one or two audit results per step
   barely improved its decisions.**
   - Harvest: 1.9% → 1.7% of the horizon unsafe with 2 random audits.
   - Complete information from neighbour reports reached the honest level.
   - The safety in S1's audit arms came from excluding caught cheaters,
     which also removes their extraction.
   - A reviewer that remembers caught lies has not been tested.
2. **Under the productive-level (MSY) target in Fishery, misreporting did
   little long-run damage.** The stock stayed around 62–66, and adaptive
   liars even slightly raised total harvest. Excluding fixed-level liars
   then wasted harvest.
3. **Hidden over-extraction (S2, Fishery) moved harvest from honest agents
   to cheaters** (per honest agent, 106 → 45) when unchecked.
   - Against fixed-level, group-chosen cheaters, perfect random audits of at
     least one agent per step plus any tested flat fine of 6 or more removed
     the incentive.
   - [post hoc] Expected-penalty arithmetic puts the threshold near a fine
     of 2.
   - In Harvest cheating never paid, so deterrence could not be tested
     there.
4. **Next:**
   - locate the threshold with a finer grid and lower audit rates;
   - imperfect audits;
   - cheaters who time their cheating;
   - a reviewer that remembers caught lies.

## Update, 5 October 2026: after experiments R1 and S1

Steps 1 and 3 of the plan below have now been run, each with a protocol
frozen before the run:

- R1: `studies/R1_repaired_reviewer/protocol.md` → `studies/R1_repaired_reviewer/results.md`;
- S1: `studies/S1_reporting_and_audits/protocol.md` → `studies/S1_reporting_and_audits/results.md`.

**What changed in our understanding** (wording checked by an independent
review; see the verification log §8):

1. **Finding 4 below is confirmed on fresh data (R1).**
   - With a calibrated 5% chance constraint, Harvest joint review is nearly
     exact: 0.2% of safe requests cut, 0.6% of risky ones approved.
   - The optimistic local rule approves 12.6% of risky requests.
   - The bounded local rule cuts 90% of safe requests.
   - Joint review's near-perfect score is largely by construction.
2. **Finding 3 below is tested, and refined (R1).**
   - At full inspection, an MSY-level target in Fishery gives far more
     long-run harvest than the old line: 1,393 vs 555, a difference of
     +838 [+800, +865].
   - The earlier "better information gives less harvest" pattern appears
     only under the old line *and* when unseen requests are assumed to be
     the maximum. That pessimism acted as accidental conservation.
   - With a realistic fill (each unseen agent repeats its last request),
     inspection hardly matters under either target.
3. **Limited checking is now a real trade-off, not a saturated one (R1).**
   With the realistic fill, joint review at 3 of 6 inspections cuts only
   10.7% of safe requests, but approves 23.5% of risky ones.
4. **Unverified reports break oversight (S1).**
   - Acting on under-reported requests raised the unsafe-action rate:
     - Harvest: from 0.35% to 3.0%;
     - Fishery: from 0% to 24.8%, where the stock collapsed in 61 of 64
       contexts.
   - Audits and neighbour reports brought it back to about 0, but **mainly
     by catching and excluding the cheaters**, not by improving the
     reviewer's decisions. In those arms the reviewer never needed to cut
     anything.
   - Targeted audits caught cheaters faster than random ones.
5. **Lying rarely paid (S1).**
   - It paid only in Fishery with colluding neighbours: +110 held-out
     payoff.
   - Elsewhere it lowered the liars' payoff. Under audits, that was only
     because of the fine; before fines, lying paid in several audit arms.
   - **[post hoc]** A likely reason is that the reviewer scales everyone
     equally, so the gain from a lie is shared.
   - An actor who gains *privately* from deception, and a check that
     improves decisions without excluding anyone, are the main gaps left
     between this project and scalable oversight under adversarial pressure.

**Revised next steps, replacing Steps 3–4 below:**

- **S1-ablation** (first, a few CPU minutes): audits that correct beliefs but
  never sanction, and sanctions without belief correction. This separates
  the two mechanisms.
- **S2:** make deception privately profitable. The reviewer allocates
  per-agent shares from reports. Rerun the adaptive misreporting search,
  with the sanction size as a declared factor.
- **Fishery target:** use the MSY target in Fishery for all future runs.
- **Supervisors:** agree the definition of harm with Yali and Edward. R1
  gives evidence for a productive-level target.
- **Literature:** read Becker 1968, Allingham & Sandmo 1972 and Varian 1990
  in full before citing them in the paper (see `literature_ledger.md`).

The original analysis from before R1 and S1 follows, unchanged.

---

## Part 1. What the experiments, taken together, tell us

### Finding 1. Rules that each look fine can still let the system fail, but usually for simple reasons

Every agent's request can pass a per-agent check while the shared resource
still becomes unsafe. In these experiments that happens for three reasons:

- the system was already damaged (persistence, not a new failure);
- the per-agent limit itself was unsustainable (the fixed cutoff of 2.4 per
  patch in a setting that can only sustain 2.38);
- weather or neighbour damage.

A rule that looks at the state of the resource, such as the local state
filter, removed almost all of it in the one setting tested. That evidence is
descriptive (10 populations), but it is enough to show that "local checks
fail" is not a general truth. It depends on how the local rule is designed.

### Finding 2. How information is combined matters more than who holds it

With the same target and the same possible actions, the reviewers' mistakes
follow directly from their arithmetic:

- over-counting other agents' demand causes unnecessary restriction;
- ignoring it causes missed harm;
- adding up the actual numbers is exact.

A "local" reviewer given its neighbours' numbers is identical to the
"central" one. So, **when the model is known, reports are honest and pooling
is free**, the centralised/local distinction is not the scientific variable.
The real variables are **what information is shared and how it is
combined**. Who holds the information was never varied on its own, and it
may matter once reports can be false or sharing has a cost. Edward's point
about local information and shared norms fits this.

### Finding 3. The safety target probably matters as much as the information (hypothesis, not yet tested)

In Fishery, where "safe" sits far below the resource's most productive
level:

- the reviewer with the most information produced the *lowest* long-run
  harvest (551.5 vs 1,084 for the least informed);
- none reached the sustainable maximum (about 1,400 over 80 steps).

The measured contrast is an *information* effect under one fixed target. The
explanation, that the target sits too low, fits the numbers well but was
never tested, because no run changed the target.

If it holds, the lesson is that a reviewer is only as good as the goal it
enforces. A perfectly informed reviewer with a one-step "do not cross the
line" goal will approve running the resource down to the line, because the
fixed agents keep asking for more. Step 1 below tests this directly.

### Finding 4. The Harvest results were mostly about one design constant [post hoc]

Once the weather allowance is sized for the only safety condition that
binds in these cases (the average):

- joint review in Harvest is nearly exact;
- the optimistic local reviewer misses some harm (95 of 1,368);
- the bounded local reviewer still over-restricts.

Harvest then tells the same story as Fishery. This needs prospective
confirmation, but it is a large effect, not a subtle one.

### Finding 5. The parts that would make this a scalable-oversight study have not been tested yet

The following were never tested, or never worked:

- agents who can misreport;
- reviewers who must pay to check;
- actors that adapt to the reviewer;
- a validated "stronger actor".

The actor-strength manipulation failed honestly. In the language of the
field, the experiments so far study **monitor design with honest agents and a
known model**. That is the easy corner of scalable oversight.

### Finding 6. The infrastructure is good

The confirmation is reproducible end to end. Protocols are written down,
failures are kept, and the independent units are handled correctly. This is
a solid base to build the next experiment on.

### What can currently be claimed with confidence

> In two simulated commons games where the reviewer knows the rules and
> agents report honestly, the errors of a reviewer follow from how it
> combines the requests it sees: over-counting others' demand causes
> unnecessary restriction, under-counting causes missed harm, and a
> correctly combined local calculation equals a central one. Avoiding
> one-step harm did not ensure good long-run use of the resource; in
> Fishery this is most likely because the safety line sits far below the
> resource's most productive level.

That is true and clean, but modest. On its own, I do not think it is enough
for a full conference paper. It could support a short workshop paper, as
long as the Harvest numbers are redone with a calibrated allowance (Step 1
below).

---

## Part 2. What should come next

The order matters. Each step has a clear question, a cost, and a rule for
when to stop. This follows the "scientific-experiment" and
"experiment-planner" skills you installed: decide before running, keep units
independent, and do not retry until it works.

### Step 0. Decisions to make first (no computing; for you, Yali and Edward)

1. **What counts as harm?** Choose one of:
   - **one-step:** do not cross the line next step (current);
   - **sustainability:** do not push the resource below its most
     productive level;
   - **multi-step:** do not cross the line within H steps under the
     current policies.

   This choice changes which restrictions count as "errors". I suggest
   making long-run harvest and stock co-primary with whichever immediate
   target you choose.
2. **What is the paper's contribution?** Either:
   - (a) a short, careful paper on monitor design with honest agents
     (Steps 1–2), or
   - (b) a full paper whose main result is about strategic reporting and
     costly checking (Steps 1–3).

   I recommend (b) as the goal, with (a) as a fallback if Step 3 is null.
3. **Which game is the primary test bed?** I suggest Harvest. Its safety
   line is already close to the most productive level, and it has a real
   neighbour-interaction structure. Fishery is a transparent sanity check.

### Step 1. Repair and rerun the existing design (small; about 10 minutes of CPU)

This is a repair of the measurement, not a new scientific question, but it
still needs a new written protocol because the motivation came from seeing
the data.

- **Changes:**
  - **(i) Calibrated uncertainty allowance.** Same intended risk (5%) for
    every reviewer, covering both parts of the safety rule. Check it on
    synthetic cases before the run.
  - **(ii) Labels from at least 4,000 weather draws,** so that almost
    nothing is unresolved.
  - **(iii) Unseen requests filled with a realistic estimate** (each
    agent's previous request), plus the same calibrated allowance.
  - **(iv) Decisions scored on the states each reviewer actually visits**
    (closed loop), as well as on the open-loop cases.
  - **(v) A local reviewer that receives only its neighbours' inspected
    requests,** as the honest "local information" arm.
  - **(vi) The safety target as a declared factor:** the current one-step
    line vs a sustainability line (Fishery: at least 50 left after
    harvest). This directly tests Finding 3.
- **Keep:** 64 fresh contexts per game, the same action menu and the same
  analysis plan (context bootstrap). Use new seed bases.
- **Prediction, written now so it can be checked:**
  - in Harvest at k = 6, joint and neighbour-informed local review restrict
    under 5% of safe requests and approve under 1% of risky ones;
  - optimistic local approves more than 3% of risky ones;
  - at k = 3, the realistic fill should make restriction rates differ by
    more than 10 points between methods (no longer saturated).

  If these fail, that is informative too.
- **Stop rule:** one smoke run plus one full run. No parameter changes after
  seeing results.

### Step 2. A small grid instead of two hand-picked settings (optional, cheap)

Predeclare 2 aggressiveness levels × 2 regrowth rates per game and report all
four. This answers "is the pattern general within each game?" without adding
games. Cost: about 4 × the Step 1 run.

### Step 3. The real next experiment: honest vs dishonest reports, with costly checking

**Question.** When agents *report* their intended extraction (or their patch
state), and the reviewer can only afford to *verify* a few reports, which
checking protocols keep harm low and useful activity high? How does this
change when agents choose to under-report because they know the protocol?

This connects directly to:

- **scalable oversight:** a reviewer that cannot check everything, facing
  actors who can exploit that;
- **Edward's feedback:** agents constraining each other through shared
  information;
- **the handoff's own stated next step.**

**Design sketch** (to be frozen as a protocol before any run):

- **Game.** Harvest, with the Step 0 target and the Step 1 calibrated
  allowance. Fishery as a sanity check.
- **Reports.** Each step, every agent sends a report r_i of its request.
  The true request p_i is hidden unless audited.
  - Honest agents have r_i = p_i.
  - Misreporting agents have r_i = p_i × (1 − d), with under-report level
    d ∈ {0, 0.25, 0.5}.
- **Reviewer protocols**, each with the same action menu and the same
  target:
  1. **Trust reports:** decide using the reports only.
  2. **Random audit:** verify k random agents; a caught liar is capped at 0
     next step (the penalty).
  3. **Risk-targeted audit:** verify the k agents whose reports matter most
     for the safety prediction.
  4. **Full inspection:** reference, cost = 6 checks.
  5. **Peer reports (Edward's local norm):** neighbours report each other's
     visible extraction, and the reviewer cross-checks.
- **Actors, in two layers:**
  - **(a) Known-answer layer.** Fixed honest and fixed misreporting types,
    to check the scoring works. The answers are predictable.
  - **(b) Adaptive layer.** Agents pick their under-report level d by
    search *against the protocol they face*, maximising their own payoff
    including penalties. This is the first time actors respond to the
    reviewer. Validate on held-out partner populations that the chosen d
    actually raises payoff. If it does not, report a failed manipulation.
- **Outcomes, all reported separately with no single score:**
  - harm approved;
  - safe activity restricted;
  - audits used per step and messages per step (the costs);
  - closed-loop 80-step harvest, stock and unsafe time.
- **Independent unit.** Population context; 64 per cell after a pilot of 8.
- **Hypotheses, falsifiable:**
  - **H1.** With honest agents, trusting reports equals full inspection at
    a fraction of the checking cost.
  - **H2.** With adaptive agents, trusting reports lets harm through.
    Random audits with a penalty restore safety once the audit rate exceeds
    a threshold that depends on the penalty. Risk-targeted audits reach it
    with fewer checks.
  - **H3.** Peer reporting reduces the audits needed, unless neighbours
    collude.

  Any of these can fail. A failure is a result.
- **Cost.**
  - Pilot: 8 contexts × 5 protocols × 3 audit levels × 2 actor layers,
    about 240 short episodes, seconds to minutes of CPU.
  - Full run: 64 contexts, about 2,000–4,000 episodes, comparable to the
    existing confirmation (5 CPU minutes).
- **Stop rule.**
  - If the adaptive layer does not produce misreporting that pays off
    (no actor gain), stop and report it. The reviewer comparison would
    then be uninformative.
  - No retries for significance.

### Step 4. Only after Step 3: actor capability

Make "stronger actor" mean "more search *against the reviewer*". Validate it
on held-out partners and report actor search budget separately from reviewer
checking budget.

Do not add new games or larger language models before Steps 1–3. Clean Up
should come back only if a predeclared redesign gives clean starting states
and competent policies.

---

## Part 3. Suggested message to supervisors (short version)

> The matched reviewer experiment is reproducible and its protocol is sound.
> Re-analysis shows its results are largely determined by the reviewers'
> arithmetic and, in Harvest, by the size of the weather allowance. Its
> most interesting lesson is that better-informed review gave worse
> long-run harvest in Fishery, most likely because the one-step safety line
> sits far below the resource's most productive level. That explanation
> still needs a direct test.
>
> I propose:
> (1) a cheap prospective rerun with a calibrated allowance and an agreed
> definition of harm;
> (2) a new experiment where agents can misreport and the reviewer must pay
> to verify, with actors that adapt to the checking protocol. That would make
> the scalable-oversight link real rather than motivational.
>
> Decision needed: the definition of harm, and whether the first paper waits
> for (2).
