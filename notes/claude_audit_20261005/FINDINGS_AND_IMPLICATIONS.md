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
- **Next.** Trained-agent χ and ζ (CREATE script prepared, not submitted); a learning-agent version of the gap.
- **Evidence.** `notes/socialjax/locality/results.md`, `protocol.md`, `tables.md`; code `overseer/socialjax/{local_records,locality,rooms,corridor}.py`, `experiments/scale/`.

---

*Index entry:* this file is referenced from `what_we_learned_and_next.md` and from Ameer's Claude memory.
