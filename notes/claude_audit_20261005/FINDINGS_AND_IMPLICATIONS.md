# Findings and what they mean for the paper

A running list of what we have found that changes how results must be stated, or what we do next. **Add to it whenever a study or check turns something up.** Each entry gives: what was found, where the evidence is, what the paper must say, and the action, with status.

Labels:
- **[verified]**: read from data.
- **[post hoc]**: found or explained after seeing results.
- **[design]**: a property of how a game is built.

Started 14 October 2026.

---

## F1. A lone cheater needs about three times the group's deterring fine (Fishery) [verified; explanation post hoc]

- **Found.** The paper's break-even fine (F\* = 2.14 t at q = 1/6) is the fine for a *group* of four committed cheaters, who share the depletion they cause. A *lone* over-taker passes that depletion on to everyone else, so its gain per over-take round is 1.14 t against 0.356 t, and its break-even fine is 6.87 t (3.2×).
- **Test.** At that fine the lone gain falls from +92 t to +4.4 t [−26, +34]. The pre-stated criterion fails narrowly (point estimate > 0); the gain is clearly negative at 1.5×.
- **Evidence.** `studies/G1_schelling_and_sweeps/results.md` (G1 Diagram 2, G1b); `runs/claude_g1b_v1/g1b_summary.json`.
- **Paper must say.** The programmed-agent threshold is the threshold *for that cheating group*. The fine that deters depends on how many cheat. The LLM thresholds (L3/L4) use each agent's own one-round gain, so they are unaffected.
- **Action.** Add one sentence to the Results and one to the Limitations. Optionally, a figure: break-even fine against the number of cheaters. **Status: open.**

## F2. Forest under the reviewer is not a social dilemma among the agents [verified; interpretation post hoc]

- **Found** (G1 Diagram 2, F = 0):
  - With the reviewer in place, *everyone over-taking* earns each agent slightly more than *everyone complying*: 117.7 against 114.0 units over 80 rounds, D1 −3.7 [−4.8, −2.6].
  - Yet harm rises from 0.6% of rounds (all comply) to 50% (all over-take).
  - So the damage registers on the forest's harm line, but barely in the agents' own payoffs within the game.
- **Why** [design, from the code]:
  1. Each agent cuts **its own plot**. Over-cutting hurts mostly the cutter's own future, plus a 0.22 spill to each neighbour above a fixed 2.1 units. A group of greedy cheaters therefore *lose* by over-taking (S2: 64.1 → 60.2). The break-even fine is 0 (G1 Amendment 2), so Forest cannot test deterrence with programmed cheaters.
  2. The reviewer's allowance sits **below the per-plot most-productive cut**. In L6's control cells the allowance was 1.73 units, while the language-model foresters cut about 2.37, almost exactly the per-plot MSY of 2.38. The cause is the reviewer's 5% chance constraint under weather noise, combined with the extra "half the plots below 4" harm rule. So "over-taking" in Forest largely means cutting at about the most productive level, not depleting.
  3. The harm line (mean health below 10, or half the plots below 4) is stricter than anything that lowers the agents' payoffs inside 80 rounds.
- **Evidence.** `studies/G1_schelling_and_sweeps/results.md`; `runs/claude_g1_v1/g1_summary.json` (`forest.diagram2`); `studies/L6_llm_forest_spillover/results.md` (post hoc table); `research/commons_platform/sources/L06_forest_resource_model.md`.
- **What it means.**
  - Forest is better described as **protecting a resource whose damage the agents barely feel**. This is an externality on the environment, and on the future beyond the horizon, rather than a dilemma between the agents. That is still a meaningful AI-safety case (agents harming something outside their reward), but it is not the same case as Fishery.
  - Deterrence of profit-seeking cheaters cannot be tested in Forest with programmed agents.
  - L6's "over-take rate" is measured against an allowance that is below the most productive cut, so a rise in over-taking partly reflects the allowance shrinking, not the cuts growing. That is exactly L6's post hoc finding.
  - The **harm** rise in L6 (forest constraint broken in 74–86% of rounds against 31–37%) is real and comes with the planted over-cutter.
- **Paper must say.**
  - Forest's harm falls on the resource, not on the agents' own payoffs.
  - Its reviewer allowance is below the most productive cut.
  - Programmed over-taking never pays there.
  - L6 results should be reported as cuts compared with the MSY cut, and as harm, not only as over-take rates.
- **Action, choose one:**
  - (a) reframe Forest in the paper as an environmental externality;
  - (b) recalibrate Forest so that payoffs feel the harm, set the spillover kink at the MSY cut, and retune the reviewer's tolerance; then re-run the Forest studies;
  - (c) drop Forest from deterrence claims and use SocialJax Harvest for the spatial case.

  Recommendation: (a) for the current paper, (c) for the next. **Status: open, needs Ameer's decision.**

## F3. River with doubled synergy is not a dilemma [verified]

- **Found.** G2 sweep: at κ = 2×, D1 fails (−3.85): six greedy dischargers out-earn six restrained ones. At κ = 0.5× the observed stopping fine is 1.3× the prediction (outside ±25%).
- **Evidence.** `studies/G1_schelling_and_sweeps/results.md` (G2); `runs/claude_g2_v1/g2_summary.json`.
- **Paper must say.** River results hold at the paper's κ and at 0.5×. The dilemma disappears at 2×.
- **Status:** note in the supplement (open).

## F4. Announcing audits with no fine makes things worse than not mentioning audits [verified; L4 pre-registered as descriptive; W1 post hoc]

- **Found.** Over-taking: N0 30% against E0 58% (gpt-oss), 46% against 64% (Nemotron). Oversight efficiency is below 0 for both models and both wordings. Efficiency is 13–37 points worse than N0.
- **Evidence.** `studies/L4_llm_identification/results.md`; `studies/W1_welfare_cost_rescoring/results.md`.
- **Related work.** Okamoto et al. 2026 ("enforcement information paradox") and SCHEME (a monitor's presence halved sabotage) go the other way: cite and contrast.
- **Status:** in slides; to go into the paper (open).

## F5. Oversight efficiency is inflated when the no-oversight baseline is near first-best [verified]

- **Found.** N0 efficiency is 0.993 (gpt-oss) and 0.978 (Nemotron), so OE divides by a gap of 0.007–0.022. The 20-round games start at stock 70, above K/2, so the frozen efficiency exceeds 1. W1 Amendment 1 adds a stock-adjusted efficiency.
- **Paper must say.** Report efficiency differences next to OE. Use the stock-adjusted efficiency.
- **Status:** open.

## F6. Being caught does not reduce the next round's over-taking [verified, post hoc comparison]

- **Found.** Among agents who over-took, being caught changed next-round over-taking by +2.4 points [−3.6, +8.7] (gpt-oss) and +6.8 [+1.8, +12.2] (Nemotron).
  - Humans show a post-audit change (the "bomb-crater" effect; Kasper and Rablen 2022).
  - The protocol's caught-vs-not-audited contrast (+24 and +45 points) mostly measures persistence.
- **Evidence.** `studies/W1_welfare_cost_rescoring/results.md`.
- **Status:** a candidate result for the paper (open).

## F7. Over-taking rises at the end of the game [verified]

- **Found.** Rounds 17–19 against rounds 0–16: +16.6 points [+13.2, +20.2] pooled. This is biased by collapsed games dropping out of the late rounds.
- **Paper must say.** It is an end-game effect (to verify: Selten and Stoecker 1986).
- **Action.** Consider a random or unannounced horizon in future LLM studies, as Melting Pot does with random episode ends.
- **Status:** open.

## F8. Nine L4 games were played twice [verified]

- **Found.** On first plays, H2's d = −12.75 [−16.2, −9.2] instead of −12.9. E12 is 31.7% instead of 32.1%, and Q3_6 is 40.6% instead of 40.7%. No verdict changes.
- **Evidence.** `studies/L4_llm_identification/results.md` (data note); `runs/claude_l4_v1/l4_summary_firstplay.json`.
- **Paper must say.** Quote the first-play numbers.
- **Action.** Stop remote lanes from overwriting each other's games before the next remote study. **Status: open.**

## F9. Our audit budgets are 8–35 times those of AI-control studies [verified]

- **Found.** L4 used 8.8–33.8% of agent-rounds and L6 12.8%. AI control uses 0.5–2%.
- **What it means.** The scarce-oversight question is untested so far.
- **Action.** Phase 1 study C2 (an LLM monitor at 1–5%). **Status:** planned.

## F10. How grounded our settings are [verified]

- **Found.** Of 104 settings, 12 are grounded in a cited source, 57 have a recorded reason, and 35 have none. Forest is the weakest. Every platform in the field hand-sets its numbers; the field grounds them by Schelling diagrams and sweeps.
- **Evidence.** `research/commons_platform/findings/local_grounding_audit.md`.
- **Status:**
  - Mismatches fixed, and the ranges paragraph added to the supplement.
  - Schelling diagrams and sweeps done (G1/G2).
  - Still open: the K/4 robustness line, the temperature check, and G1-LLM.

## F11. The novelty landscape moved in 2026 [verified, 13–14 Oct]

- **Close papers:**
  - Bell & Madisetti (ICML 2026): voted restitution;
  - Okamoto et al. (AIES 2026): the enforcement information paradox;
  - OpenSocialJax (9 Oct 2026): LLMs in SocialJax, no overseer;
  - Large Legislative Models (2024): an LLM sets taxes for RL agents in Commons Harvest, with no audits.
- **Unclaimed:** an overseer that audits and fines LLM agents in a spatial commons, evaluated as AI control evaluates monitors.
- **Action.** Re-check `research/commons_platform/refresh_targets.md` before each submission.

## F12. In Willis et al.'s CPR, a final-round defection raises welfare [design]

- **Found.** In the last round, defecting gains +2.0 and harms nobody, and their welfare maximum assumes everyone defects then. An overseer that fines every defection lowers measured efficiency in round 20.
- **Action.** In R1a, exempt the final round (primary) and report non-exempt as a secondary result. **Status:** built into the R1a protocol.

## F13. In Willis et al.'s CPR, Selfish strategies already earn less than Collective ones in groups of 4 [verified; R1a prediction, 14 Oct]

- **Found.** In 2,000 fine-free games at the evolution's starting mix (every seat a random genotype and strategy):
  - **n = 64:** a Selfish-genotype agent earns 3.21 more per game than a Collective one and makes 3.34 more defections before the last round. The expected fine per defection that cancels this is **e_pred = 0.96**, under half the myopic one-round gain of 2.
  - **n = 4:** a Selfish agent earns 2.36 *less* per game (e_pred = −0.59, interval −0.72 to −0.44), because its extra defections drain a stock shared with only three others.
- **What it means.**
  - The paper's n = 4 result (56% Collective, 91% efficiency) is consistent with Selfish strategies being at a payoff *disadvantage* from the start. With β = 1 on normalised fitness, selection is weak, so drift keeps them near half. The paper credits group selection; the prediction points to direct self-harm in small groups as well. *Interpretation post hoc, untested.*
  - In large groups the fine that removes the Selfish advantage at the start (about 0.96 per defection, F ≈ 9.6 at q = 0.1) is about half the Becker benchmark F\* = 2/q.
- **Consequence for R1a.** The frozen fine levels at n = 4 would have been negative. Amendment 2 uses the n = 64 scale at both sizes and re-states H1 at n = 4 as "Collective > 0.5 at every e, including 0".
- **Evidence.** `runs/claude_r1a_v1/prediction.json`; `studies/R1a_willis_cpr_audits/results.md`; protocol Amendment 2.
- **Paper must say.** Any break-even fine for their CPR depends on group size, and is not 2/q. Report e_pred per n.
- **Action.** Done. R1a ran in full (see F14). At n = 4 the Selfish disadvantage alone did not lift Collective above half (0.51 [0.43, 0.59]); any fine did. **Status: closed.**

## F14. Audits and fines reverse the Selfish takeover in Willis et al.'s large groups, just above a fine predicted in advance [verified; R1a, 15 Oct]

- **Found** (n = 64, q = 0.1, 20 evolution runs per condition, blind LLM-written strategies):
  - The share of Collective strategies is 0.15 with no oversight, and 0.20 / 0.34 / **0.60** / 0.84 at 0.5 / 1.0 / 1.25 / 2.0 × the predicted balance fine (0.96 per defection, frozen before any run).
  - Welfare efficiency is 0.13 → 0.57 at 1.25× and 0.87 at 2× (fines as transfers; 0.39 and 0.74 counting fines as lost).
  - The paper's baseline replicates: 0.15 / 0.13 against 0.19 / 0.15.
  - **At the same expected fine**, rare audits with large fines break even at 2.3–2.8× the audit cost of frequent small ones (60% fewer audits).
  - With strong oversight, the surviving strategies shift from Selfish GPT/Claude to Collective Claude/Gemini/GPT.
- **Built in vs found.** With blind strategies, every effect runs through which strategies get copied, and "a big enough fine makes Collective win" is mechanical. What was found: the switch fell between 1.0 and 1.25 × a balance point predicted before any run, the size of the welfare recovery, the cost comparison, and the model shares.
- **Evidence.** `studies/R1a_willis_cpr_audits/results.md`; `runs/claude_r1a_v1/r1a_summary.json`; `figures/fig_r1a_willis_audits`.
- **Paper must say.**
  - It extends the frozen-prediction result (L3/L4) to hundreds of agents under cultural evolution, on the field's own code.
  - The strategies are blind to audits, so the effect is selection, not in-game deterrence (R1b tests that).
- **Action.**
  - R1b: regenerate strategies with the audit rule in the prompt (about $1).
  - Fix the overlapping labels in figure panels c and d.
  - All 560 runs are complete (19 Oct).
  - **Status: open (R1b next).**

## F15. When told the audit rule, LLM-written strategies obey Becker, and a zero-fine announcement does not backfire on them [verified; R1b Part 1, 19 Oct]

- **Found** (gpt-oss-120b, 127 audit-aware and 127 blind strategies, Willis et al.'s CPR):
  - Audit-aware Selfish strategies at n = 64 cut their defection from 0.74 to **0.06** at an expected fine of 2: 8% of the no-fine level [2%, 16%]. They had already halved it at 0.5.
  - The blind pool is unchanged.
  - 100% of Selfish audit-aware strategies read q or F, and 75% implement an explicit expected-value comparison; 7% use their own audit history.
  - Audit-aware Collective strategies defect less even with audits off (0.34 against 0.63).
  - No audits and audits with a zero fine give identical behaviour.
- **What it means.**
  - Deterrence works through in-game response as well as through selection (R1a).
  - **Agents that write code and agents that act live differ:** the zero-fine backfire in L4 (F4) does not appear in written strategies.
- **Evidence.** `studies/R1b_audit_aware_strategies/results.md`; `runs/claude_r1b_v1/r1b_summary.json`.
- **Paper must say.** The live-agent backfire (L4) is a property of acting LLM agents, not of the rule.
- **Action.**
  - Part 2 is done (see F16).
  - Hand-check 20 strategies.
  - **Status: open (hand check only).**

## F16. Compliance is not conversion: against strategies that know the rule, oversight works by changing behaviour, and the Selfish majority survives [verified; R1b Part 2, 20 Oct]

- **Found** (n = 64, q = 0.1, 20 runs; fine at 1.25 × the frozen balance point 0.63):
  - **Blind pool:** Collective share 0.15 → 0.60, but welfare only 0.13: selection works, behaviour barely changes.
  - **Audit-aware pool:** Collective share stays at 0.19, defections fall 63%, and welfare is 0.45, so H4 holds at +0.32 [+0.29, +0.34]. At 2×: 0.64 against 0.42.
  - **Without audits,** the audit-aware pool falls back to 0.03 efficiency.
- **What it means.**
  - An overseer of agents that can respond produces compliant Selfish agents, not cooperative ones.
  - The share of "good types" is the wrong success measure; behaviour and welfare are the right ones (this supports W1's oversight-efficiency framing).
  - Compliance under audits does not show what the agents do without them.
- **Paper must say.** This is the mechanism result for the flagship paper: routes to deterrence and what metric to use. The "compliance under monitoring is not alignment" angle connects to the AI-control literature.
- **Evidence.** `studies/R1b_audit_aware_strategies/results.md`; `runs/claude_r1b_v1/r1b_part2.csv`.
- **Status:** open (hand check of 20 strategies; a test of removing oversight after evolution would make the point directly).

## F17. With learning agents in Harvest, the expected penalty sets a sharp threshold, one check per step is enough, and proxy targeting is worse than random [verified; phase 2 stage 1, 16 Nov]

- **Found** (SocialJax Harvest Open, 7 IPPO agents with individual rewards, 1e8 steps, 3 seeds; rule m = 3; perfect inspector; agents cannot see checks):
  - **No overseer:** the orchard dies in every env (68 apples per episode; 45% of eats break the rule).
  - **Random checks** (expected penalty per over-harvest = k/7 × penalty): an expected 0.5 apples changes nothing at any k (even k = 7). At 1 apple every run saves the orchard (441–477 apples). At 2 the agents eat 515–530, more than 7 restrained bots (500).
  - **Cheapest working overseer:** k = 1 with a 14-apple penalty (515 apples with 1,000 checks per episode, against 530 with 7,000).
  - **A 50-step freeze at k = 2 fails** (77 apples).
  - **History targeting** ends level with random but protects the orchard much sooner.
  - **Location targeting fails:** 150 apples, catching 5% of over-harvests against 28% for random. It catches only 4% even before learning, because it checks agents on bare ground, while an over-harvester stands beside the last apples.
- **What it means.**
  - The Fishery result "the expected fine decides" (L4, F14) carries over to learning agents in a spatial game, with a threshold near the scripted-bot break-even (about 7 apples).
  - The budget question becomes how large a penalty one check can carry. That depends on false alarms, which are not yet tested.
  - Targeting is only as good as its proxy (as in Fishery T1).
- **Paper must say.** This is the baseline result for the ICML overseer paper: a budgeted overseer works against learners once the expected penalty is above the threshold (about 1 apple per over-harvest, the apple it gains). Targeting needs a good signal, and its gains here are speed, not end state, because parameter-shared agents have no persistent cheater.
- **Evidence.** `notes/socialjax/phase2_ippo/results.md`, `summary.json`, `fig_checks_by_penalty.png`, `fig_learning_curves.png`.
- **Status:** open. Two k = 4 cells have 2 of 3 seeds (reruns queued). Late switch-on and switch-off (stages 2–3) are pending, and so is a finer sweep between 0.5 and 1 apple expected. (Wording corrected 16 Nov: k × P is 7 times the expected penalty.)

## F18. In Harvest, removing the overseer brings over-harvesting straight back, and fixing agents that already over-harvest needs frequent checks or targeting [lead; stages 2–3, 21 Nov]

- **Found** (interim; 9 of 18 switch-off runs, 7 of 21 switch-on runs, seed 1 only):
  - **Switch-off:** every arm falls from 408–482 apples to 68 with the orchard dead, within about 5–12M steps of continued learning.
  - **Switch-on** at an average penalty of 1 apple, after 50M steps: random k = 1/2/4/7 give 120/172/315/403 apples; history k = 2 gives 352; freeze gives 76.
- **What it means.**
  - With learning agents in a spatial game, compliance is not conversion (as F16 found for LLM strategies). The overseer has to stay.
  - The average penalty (F17) decides whether deterrence holds in the long run. But how quickly an overseer repairs existing over-harvesting depends on how often it checks and on targeting, so budget and targeting matter more for repair than for prevention.
- **Evidence.** `notes/socialjax/phase2_ippo/results.md`; `notes/OVERSEER_TRACKER.md`.
- **Status:** verified (22 Nov, all 39 runs). Switch-off 18/18 return to 68 apples. Switch-on at 1 apple: k = 1/2/4/7 give 149/197/274/325, history k = 2 gives 348 [346–352].

---

## F19. Splitting a budgeted overseer by region costs budget and invites migration; what overseers share decides which [lead; scripted bots, 22 Nov]

- **Found** (scripted bots, SocialJax Harvest, Coins and Clean Up; regional overseers at the same total budget as one central overseer; 10 seeds; protocols written first):
  - **Waste.** Regions with fewer agents than checks leave checks unused: 22–29% with 4 strips in Harvest, 26% in 2-player Coins. Random checks lose about that share of catches. In Coins this is enough to make theft pay again at the penalty where it stops paying centrally (thief 86 vs 81 coins; honest 83).
  - **Local memory.** Migrator bots, which cheat only where the local overseer has not checked them, over-harvest 37 times a game under 2 regional random overseers, against 7 under one (k = 2).
  - **Sharing.** Sharing suspicion (who was caught) recovers most of the targeting lost to splitting: a fixed cheater is caught 53% → 68% of the time, against 81% centrally. Sharing coverage (who was just checked) helps a little more against fixed cheaters but invites migration (12 → 27 over-harvests).
  - **Detectability (Clean Up).** A check proves free-riding only if it reads the rule's whole 100-step window: 0% caught below 100 steps, 32% at 100, 100% at 150+. Presuming guilt on short looks wrongly freezes honest cleaners and collapses the commons (908 → 196 apples).
- **What it means.**
  - Decentralisation is not free at a fixed budget.
  - The communication rule is a design choice with an adversary in mind.
  - Clean Up gives the general formulation's detectability property (C4) a measurable dial.
  - Whether learning agents discover migration is the GPU question.
- **Evidence.** `notes/socialjax/DECENTRALISED_SUMMARY.md`; `notes/socialjax/{harvest,coins,cleanup}/results.md`.
- **Status:** open (bots only; 159 IPPO runs listed for CREATE).
- **Update (23 Nov):** with IPPO learners the budget-placement cost holds and is larger (F23). The migration gain did not appear, and this design can't detect it.

## F20. An imperfect overseer: misses scale the deterrent as predicted, and few-big-check overseers fall behind many-small-check ones [verified; mistakes test, 22 Nov]

- **Found** (Harvest, IPPO, 1e8 steps, 3 seeds per cell; protocol written before the runs):
  - **Misses 50%:** an average penalty of 1 apple behaves like 0.5 (orchard dead, 68 apples); 2 apples behaves like 1 (k = 7: 479 against 477 for the perfect inspector at 1 apple).
  - **False alarms on 25% of honest harvests:** the orchard survives at both c = 1 (effective 0.75) and c = 2. So the learners' threshold lies between 0.5 and 0.75 apples.
  - **Honest cost:** wrongful fines are about 0.25 apples per legal apple at both k = 1 and k = 7, as C2 predicts.
  - **Few big checks fall behind:** at the same average penalty, k = 1 does worse than k = 7 in every mistakes arm, with seed ranges not overlapping (342 vs 479; 382 vs 482; 470 vs 509).
- **What it means.**
  - C1 holds with misses.
  - "One check is nearly as good as checking everyone" (F17) holds only for a perfect inspector. With mistakes, uneven losses (rare, large, sometimes wrongful fines) slow learning, so budget spread matters for a realistic overseer.
  - This is the first "less capable overseer" result for the bounded evaluation.
- **Evidence.** `notes/socialjax/phase2_ippo/results.md` (mistakes section), `mistakes_protocol.md`.
- **Revised (23 Nov, F24):** the 25%-false-alarm arm's over-harvests still cost the full penalty, so it does not place the threshold at 0.5–0.75. On the fine an over-harvest actually costs, the threshold is about 0.75–0.9 (higher at k = 1).

---

## F21. Locality: what makes splitting the overseer cost catches is agents moving between regions and how unevenly they crowd, not where the harm lands [lead; bots and fixed strategies, 22 Nov]

- **Found** (protocol written before any run; 13 of 21 predictions held, 4 in part, 3 failed, 1 untested):
  - **The existing regional overseer hands records over for free.** An agent's last-w-steps record travels with it, so findings 12–16 and the 159 decentralised GPU runs test only where checks can go. A new `LocalRecordOverseer` (records="local") tests evidence lost at boundaries.
  - **Measured locality:** crossing rate χ is 0.006–0.07 per agent-step and harm containment ζ is 0.9–1.0 in Harvest; Coins ζ = 0.67; Clean Up ζ = 0.50 with 2 strips (0.26 with 4, 1.00 for a river/orchard split).
  - **The P_0 formula is exact only if agents never come back** (error ≤ 0.002); with returns it is off by up to 0.73, because agents are caught late rather than never. A Markov version fits within 0.003.
  - **D1:** ξ_loc ranks the catches lost to local records (Spearman 0.80, 65 points from 7 maps) but only weakly ranks the total gap (0.42). Most of the gap is placement: fixed per-region budgets while agents crowd unevenly (up to 0.20 of violations at w = 1).
  - **Scale (Willis et al.'s commons, N = 64 and 256):** fixed groups lose nothing (|Δ| ≤ 0.01) despite ξ_loc = 1 − 1/M. An overseer counting only its own group's welfare deters its own group at every M ≥ 4 (derived, not simulated).
- **What it means.**
  - Split ξ_loc: χ plus budget density govern catching; ζ governs the overseers' objectives.
  - Replace P_0 in §5.2 of the formulation with the Markov form; report time-to-catch alongside catch share.
  - Budgets that follow where agents are are the obvious next test.
- **Next.** Trained-agent χ and ζ: done 23 Nov (F25). A learning-agent version of the gap is still to do.
- **Evidence.** `notes/socialjax/locality/results.md`, `protocol.md`, `tables.md`; code `overseer/socialjax/{local_records,locality,rooms,corridor}.py`, `experiments/scale/`.

---

## F22. LLM inspectors: a step at 7B, and what the inspector is shown matters as much as its size [lead; 600 fixed items, 22 Nov]

- **Found** (protocol written before any run; Qwen2.5 0.5B–7B local, gpt-oss 20B/120B cloud):
  - At w = 1, inspectors are near perfect or useless. Qwen ≤ 3B gives the same answer every time; Qwen 7B and both gpt-oss models are perfect on text summaries.
  - On raw maps, where the inspector must count apples itself, plain Qwen 7B misses every violation; only gpt-oss protects (120B: α 0.03). Brief reasoning rescues 3B on summaries and brings 7B on raw maps to the threshold, at 35–90× the output tokens.
  - Longer records hurt even strong models (α + β_eat up to 0.27–0.31 at w = 50). A check costs about 330 tokens on a summary and 900 on raw maps with gpt-oss.
  - BazaarBench: all models agree poorly with GPT-5's labels (α + β 0.70–0.83), largely over whether a privately written plan counts as a violation.
- **What it means.**
  - "Overseer capability" is not one dial: model size, what the record shows, and reasoning budget each move the inspector across C1's protect/fail line.
  - Bounded evaluation needs the inspector's α, β measured on the rendering it will actually see.
  - BazaarBench labels need a public rubric before they can serve as ground truth.
- **Next.** 40 RL runs with LLM inspectors: done 23 Nov (F24).
- **Evidence.** `notes/llm_overseer/SUMMARY.md`, `results/tables.md`, `protocol.md`; code `overseer/llm/`, `experiments/llm_overseer/`.

---

## F23. With learning agents, splitting a budgeted overseer by region fails when it pushes the fine a violation actually costs below the threshold [verified for placement; IPPO, 23 Nov]

- **Found** (IPPO, 1e8 steps, 3 seeds per cell; Harvest 67 of 69 runs, Coins 45 of 45; regional overseers with the same total budget as one central overseer; records handed over between regions for free, so this tests check placement only):
  - **Harvest, 4 strips, average penalty 1 apple:** the orchard dies at k = 2 and k = 4. Random checks get 76–86 apples, against 458–476 centrally; sharing plus repeat offenders first gets 175–224, alive in 2–67% of games. Over-harvests are caught less often (18% vs 28% at k = 2; 38% vs 57% at k = 4), so each costs 0.65–0.75 apples instead of 1.
  - **Coins, 2 halves, at break-even (1 coin per theft):** 26–29 thefts a game against 7–9 centrally. 28–31% of checks go unused and the catch share falls by the same third, so a theft costs 0.68 coins.
  - **Cheap cases:** 2 Harvest strips lose 1–4 points of catch share and up to about 60 apples at c = 1. At c = 2 every split arm works.
  - **The bot predictions, against learners:** waste is smaller in Harvest than for bots (0–13% vs 6–29%) but the catch loss with 4 strips is larger than the waste (crowding). The bots' 5× migration gain did not appear (1.1–1.3×), though the design cannot detect deliberate migration. Local targeting did not beat central at k = 4 (439 vs 484 apples).
- **What it means.**
  - F19's "decentralisation is not free" holds with learners, through one mechanism: fixed per-region budgets lower the realised deterrent. Set the penalty from the achieved catch rate, or let budgets follow agents.
  - Migration and evidence lost at boundaries are still bot-only results.
- **Evidence.** `notes/socialjax/DECENTRALISED_RESULTS.md`; `notes/socialjax/DECENTRALISED_SUMMARY.md`.
- **Status:** verified for check placement. Open for evidence loss (local-records overseer, untrained) and for deliberate migration (separate-nets and worst-case lists).

## F24. LLM inspectors in Harvest RL: C1's protect/fail line mostly holds, but misses and false alarms don't trade one for one [verified; 40 runs, 23 Nov]

- **Found** (IPPO, 1e8 steps, random checks making each LLM's measured w = 1 mistakes as coin flips; 3 seeds, 1 for the Qwen 7B direct control; predictions in `notes/llm_overseer/protocol.md` §6):
  - C1's classes held in 15 of 16 cells: "fails" cells end at 68–69 apples with the orchard dead; "protects" cells keep it, except one.
  - **The exception:** gpt-oss-20B on raw maps (α 0.10, β 0.08, C1 e 0.82) at k = 1, c = 1: 210 apples, orchard alive in 63% of games. Qwen 3B + reasoning on summaries (α 0, β 0.23, C1 e 0.77) keeps the orchard at 380, and at k = 7 reaches 480 (perfect: 477) against gpt-oss's 361.
  - Qwen 3B + reasoning reproduces the 25%-false-alarm arm (380 vs 382; 480 vs 482). Qwen 7B + reasoning (α 0.47) reproduces the 50%-miss arm (dead at c = 1; 389 and 494 at c = 2).
  - k = 1 collected clearly fewer apples than k = 7 in 4 of 5 surviving pairs.
  - Wrongful fines per legal apple = β·c (built in). At β 0.23 and c = 2 they take about 45% of what agents collect, while the orchard survives.
- **What it means.**
  - Deterrence follows the fine an over-harvest actually costs, (1 − α)·c. False alarms add a cost to honest agents (C2) but barely weaken deterrence. C1's symmetric (1 − α − β) is wrong in the β direction.
  - This revises F20's "the learners' threshold lies between 0.5 and 0.75". The 25%-false-alarm arm still cost over-harvests the full 1 apple. On the realised scale, the threshold is about 0.75–0.9 at k = 2–7 (F23 and the gpt-oss k = 7 cell), and between 0.9 and 1.0 at k = 1.
  - For choosing an LLM inspector: minimise misses first. A model that over-flags is safer for the commons and costlier for honest agents.
- **Evidence.** `notes/llm_overseer/rl_results.md`; baselines in `notes/socialjax/phase2_ippo/results.md`.
- **Status:** verified for fixed-rate mistakes. Correlated mistakes (hard records missed more often) are untested.
- **Update (23 Nov, F26):** on learners' records gpt-oss-20B misses 0.215–0.288 of violations, not 0.10, and its misses are strongly situation-dependent; the gpt-oss cells here used a too-good α. The β reading (false alarms barely weaken deterrence) is consistent with law-and-economics once one notes that β falls only on legal harvests, never on waiting (Lando 2006; Lando & Mungan 2018; `notes/llm_overseer/error_patterns/results.md` §7).

## F25. Trained agents cross region boundaries about half as often as bots, and their harm is as local [lead; 11 stage 1 runs, 23 Nov]

- **Found** (trained IPPO agents from stage 1 played 32 episodes under their own central overseer; 1 seed per overseer setting, 3 with no overseer):
  - ζ matches the bots within 0.06 on every layout (1.000 for 2 strips).
  - χ is about half the bots' on strips (0.011–0.015 vs 0.028; 0.027–0.044 vs 0.073). Under a working overseer, band crossings are 2.5× those of agents that have stripped the orchard (0.034 vs 0.013).
  - ξ_loc at w = 50 on 2 strips is 0.27–0.47, against 0.70 for bots. Violators stay in their region longer than (1 − χ)^{w−1} says, as with bots.
- **What it means.** F21 holds for learners: Harvest's harm is local and the non-locality comes from movement. For learners, evidence lost at boundaries should be smaller than the bot estimates, so placement (F23) is the main cost. χ depends on the overseer, so a gap formula should use χ measured under it.
- **Evidence.** `notes/socialjax/locality/results.md` §8.
- **Status:** lead (1 seed per setting; central overseers only).

---

## F26. A real LLM inspector's misses depend on the situation, enough to matter for deterrence; its false alarms barely do [verified offline; RL untested, 23 Nov]

- **Found** (protocol written before any run, `notes/llm_overseer/error_patterns/protocol.md`; 13,159 eats from 5 trained stage 1 policies; 2,463 judged by gpt-oss-20B on raw maps, stratified by apples around the eat; 179 by Qwen 7B + reasoning):
  - gpt-oss-20B's miss rate runs from 0.00 to 0.48 by situation: it rises with the count (0.03 / 0.17 / 0.38 at 0 / 1 / 2 apples around) and with apples in the cells that don't count (z = +9.4), the over-counting mistake. Held-out log-likelihood gain over a constant rate: +0.128 nats per item [+0.103, +0.153].
  - Its false alarms (25 in 1,263 honest eats) sit mostly on the borderline legal eat (0.06 at exactly 3 apples, about 0.01 elsewhere); no model beats a constant out of sample.
  - Bank averages differ from the ladder's bot-record numbers: α 0.215 (0.288 for overseen agents, who over-harvest at the borderline), β 0.039; Qwen 7B + reasoning α 0.57, β 0.10 (ladder 0.47 / 0.04), within the ladder's wide intervals.
  - An agent keeping its over-harvests to the worst-checked cell would pay 0.52c per over-harvest instead of the coin flip's 0.79c, across F24's threshold at c = 1.
- **What it means.**
  - The coin-flip model of an LLM inspector is wrong for misses and adequate for false alarms. Inspectors should be described by α(s), not one α, and measured on the records the agents being overseen produce.
  - The error model is local (a 5 × 5 patch), so a regional overseer can apply it without global information.
  - F24 vs law and economics (Png 1986; Polinsky & Shavell 2000): not a conflict. β lowers deterrence only when the alternative to a violation is a legal act that can be wrongly flagged; in Harvest it is mostly waiting.
- **Next.** The 36-run job list (patterned table vs coin flips at the bank and overseen-agent averages; k ∈ {1, 7}, c ∈ {1, 2}, 3 seeds): `experiments/llm_overseer/runs_error_patterns.tsv`. The most informative cell is k = 7, c = 1.
- **Evidence.** `notes/llm_overseer/error_patterns/results.md`, `fit_tables.md`; code `overseer/socialjax/inspector_table.py`, `experiments/llm_overseer/{error_bank,judge_bank,fit_error_model,error_pattern_runs}.py`; tests `tests/test_inspector_table.py`.
- **Status:** verified offline for gpt-oss-20B (Qwen: averages only). Deterrence effect untested.

---

## F27. In Harvest, every permanent loss of the commons involves a rule violation, so inspecting individuals is enough there [verified offline; 5 trained runs + 3 bot populations, 23 Nov]

- **Found** (H5; protocol written before measuring; 32 episodes per policy):
  - All 580 permanent losses (a patch losing its last apple) involved a rule violation. No overseen policy lost a patch to legal eats.
  - Perfect full inspection (k = 7) lost none.
  - Legal-only loss is structurally near impossible: it needs 4+ agents eating 4+ close apples in one step, which never happened in 256 episodes.
  - Temporary damage often passes the rule in overseen runs (60–93% of newly dead cells), but 92–95% of those cells regrow within the episode.
  - Side finding: random k = 1 and history k = 2 still lost a patch in 3 and 1 of 32 episodes through unchecked violations (a budget failure).
- **What it means.**
  - Local-pass/global-fail is negligible in Harvest. An overseer that only inspects individuals is the right first formulation there.
  - The commons view (the planned next step) must be justified in a game where harm builds up from legal-looking behaviour; Clean Up is the test.
  - The June τ = 0.5 LPGF reading (84–98%) is a threshold artefact (sustainable play settles near 23–29 apples), so LPGF must be counted as harm creation, not time spent below a line.
- **Next.** The same measures on Clean Up params when those runs finish (script ready).
- **Evidence.** `notes/formulation/h5_lpgf/results.md`, `protocol.md`; code `experiments/socialjax/measure_lpgf.py`.

---

## F28. Where an inspector errs matters more than how often: an LLM inspector whose misses fall on mild violations protects better than a random-error one with the same average miss rate [verified; 35 of 36 runs, 3 seeds, 23 Nov]

- **Found** (H3; gpt-oss-20B patterned inspector vs coin flips at the bank average α = 0.215 and the overseen-agent average α = 0.288; Harvest, IPPO, 1e8 steps):
  - **At a 1-apple average sanction, the result is the opposite of H3's prediction.**
    - Patterned inspector: orchard alive in 92–100% of games, 255–362 apples (k = 1 and k = 7).
    - Both coin flips: orchard dead in every run, 69–75 apples.
  - **Learners did move their violations into the inspector's blind spot.**
    - Under the patterned inspector, 63–66% of violations are borderline (2 apples left nearby; the inspector misses up to 48%).
    - Under the coin flips, most violations take the last apples (0 left; 14–16 per episode against 8–10).
  - But the blind spot sits on the **least harmful** violations: 2 apples left still regrow. The inspector almost never misses the harmful ones (0 apples left: miss rate 0.03).
  - At a 2-apple average sanction every arm survives, with the same placement of violations (80–88% borderline).
- **What it means.**
  - H1 needs refining. Learners respond to the sanction a violation meets *in each situation*, ρ(s) = (k/N)(1 − α(s))F, and place violations where ρ(s) is low. The commons survives if the low-ρ situations are the low-harm ones.
  - Inspector quality should be judged by its miss rate on *harmful* violations, not its average.
  - A uniform error model can call a usable inspector useless: the earlier coin-flip runs would have written off this one.
  - H3 is refuted in its outcome (patterned was better, not worse) and supported in its mechanism (learners moved violations to where misses are likely).
- **Caveats.**
  - One inspector, one game, 3 seeds.
  - The pattern happened to align blind spots with mild violations. An inspector whose blind spot sits on harmful violations should do worse than a coin flip. That is untested, and a clean next test.
- **Evidence.** `results/create_pull/E_*`; design `notes/llm_overseer/error_patterns/`.

## F29. Agents who can see when they are looked at dodge every predictable look policy; random looks cannot be dodged [verified; 24 runs, 3 seeds, 24 Nov]

- **Found** (H2; Harvest, shared-network IPPO, 1e8 steps, k = 2 of 7, 1-apple average sanction; each agent's observation shows whether it was checked last step and how long since its last check):
  - Repeat offenders first, round robin and two overseers sharing a repeat-offender list: 69–72 apples, orchard dead, 1–4% of violations caught. No overseer gives 68.
  - Random looks, one or two overseers: 431–457 apples, alive, 27–28% caught, the same as when the checks are hidden.
  - With checks hidden, every look policy ends at 439–487 apples, alive.
  - 30% random mixed into repeat offenders first: 265 apples [170, 417]; one seed of three kept the orchard.
- **What it means.**
  - The capability gap Δ (ordinary minus visible) is about 400 apples for predictable looks and under 10 for random ones. The look action must be random if agents can observe it.
  - Learners find the pattern from their own observation channels, so this needs no assumption that agents know the schedule. That is the difference from Stackelberg security games, which assume the attacker observes the defender's mixed strategy (Pita et al. 2008).
  - Earlier "repeat offenders first is safer sooner" (finding 6) holds only when agents cannot see the checks.
- **Caveats.** One game, one sanction level, k = 2. The reveal channels are generous (exact last-check information).
- **Evidence.** `results/create_pull/WS_*`, `OS_*`; `experiments/socialjax/analyze_formulation_runs.py`.

## F30. Learners follow an inspector's blind spot wherever it is: put it on harmful violations and the commons pays [verified; 12 runs, 3 seeds, 24 Nov]

- **Found** (H3b; gpt-oss-20B's table with its miss rates for 0 and 2 apples left swapped, rescaled to the same bank-average miss rate 0.215):
  - c = 2: 485 and 491 apples (k = 1, 7) against the coin flip's 515 and 535; seed ranges do not overlap. Violations that take the last apples: 5.7 and 7.7 per episode against 0.8 and 0.4. Orchard alive in every arm.
  - c = 1: inverted dies like the coin flip (69–72 apples); the original table survives (283–351).
- **What it means.** With F28, the refined H1 holds in both directions: learners place violations where the situation-specific sanction ρ(s) is low. An inspector's value is set by its miss rate on harmful violations.
- **Caveats.** A constructed inspector, not a measured one; one game.
- **Evidence.** `results/create_pull/E_gptoss20b_inverted_*`.

## F31. Separate networks end where shared networks do once trained long enough [verified for random looks; 6 runs, 3 seeds, 24 Nov]

- **Found** (H6; separate networks at 3e8 steps): central random 447 apples, 39.6 violations (shared at 1e8: 460, 38.6); two overseers random 438, 45.7 (shared: 439, 41.1). Curves levelled off (last-10% change −2 apples).
- **What it means.** The extra violation seen at 1e8 (F23 era, `SN_*`) was slow learning. Shared networks are an acceptable, 3× cheaper stand-in here.
- **Caveats.** Ordinary mode and random looks only.
- **Evidence.** `results/create_pull/SL_*`.

---

*Index entry:* this file is referenced from `what_we_learned_and_next.md` and from Ameer's Claude memory.
