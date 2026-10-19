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

---

*Index entry:* this file is referenced from `what_we_learned_and_next.md` and from Ameer's Claude memory.
