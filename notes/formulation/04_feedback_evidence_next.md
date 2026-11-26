# Phase 4 plan: all feedback, the evidence we have, and what is left (26 Nov 2026)

**Goal (Ameer, 26 Nov).** A formulation of a *partially capable overseer* in scalable-oversight problems for multi-agent systems. It must be:
- grounded in the literature and in our results;
- oriented around all of Yali's feedback;
- built on what already exists (AI Control, AI-Control Games, MARL planners, security games, Du's networked MARL), so the contribution is a worthwhile extension, not a reinvention.

This note:
1. maps every piece of supervisor feedback to what we have and what is missing;
2. grades the evidence behind each claim;
3. sets out the optimised-overseer question;
4. proposes the remaining work with costs, followed by a devil's-advocate pass.

Nothing here has been run. Proposed hypotheses (H10–H12) need Ameer's approval before any run.

---

## 1. Why the November questions slipped

The formulation phase (23 Nov) narrowed on purpose to one thing: the overseer's action (look, judge, respond, share), tested in SocialJax. That answered several of Yali's questions without saying so, and left others untouched. The table below makes the mapping explicit. In future, every phase document should carry this table (see [[supervisor-feedback-integration]]).

## 2. Every piece of feedback, against what we have

| # | Feedback (date) | What we have now | Gap | Action |
|---|---|---|---|---|
| Y1 | Overseer power by size and intelligence: can a smaller or less capable overseer keep agents in check? (2 Nov) | The judge's capability enters only through its miss and false-alarm rates on the situations that matter (F28, F30). 40 RL runs with 7 LLM judges: the predicted side of the threshold held in 15 of 16 cells (F24). The judge ladder: Qwen 0.5–7B fail on raw maps; gpt-oss 20B/120B protect (`llm_overseer/SUMMARY.md`) | No **capability curve**: outcome against overseer size, with a prediction. Situation-specific miss rates exist for one model only. The RL "smarts" dial (overseer network size) was never built, because the overseer is not learned | **WP3** (§5) |
| Y1b | How do we bound the evaluation? (2 Nov) | No overseer as the floor and a full overseer as the ceiling, with "share of gap recovered" η (finding 4). Worst case = agents who see the checks; capability gap Δ (F29) | The worst case so far is one adversary type (visible checks). No optimised overseer, so the overseer side of the bound is a fixed protocol | **WP2** |
| Y2 | BazaarBench, with a token cost on the overseer (1 vs 10 units) (2 Nov) | Tokens per check measured for each judge (gpt-oss 20B: about 982). Looks priced as 1 + 0.01 × steps read (H9). BazaarBench offline replay: findings 17–19 | No **cost frontier** (outcome against oversight spend). No live overseer in BazaarBench | **WP4** (frontier from existing data); a live BazaarBench overseer is planned, not promised |
| Y3 | A detailed plan of agents and experiments; random vs targeted (2 Nov) | Answered: H2 / F29. Predictable looks are fully dodged by agents who see them; random holds | Only one mix level between random and targeted, so no best mix | **WP2** |
| Y4 | Does the formulation hold in general, or only in SocialJax? Model it on DMPO and Ma et al. 2024: one problem class with theory, many testbeds (16 Nov) | Three conditions now say **when budgeted inspection of individuals can work**: (i) the sanction a harmful violation meets, ρ(s_harm), above a threshold (F24, F28, F30, F34); (ii) the look reads at least the rule's window, w ≥ W (F33); (iii) low local-pass/global-fail (F27, F32). Each was measured in Harvest and/or Clean Up | Not yet written as one general problem class. Condition (i) is tested in one game only (Harvest, at two sizes). BazaarBench has no Phase 3 evidence | **WP5** |
| O1 | Cost of monitoring: is auditing worth it? (13 Oct) | η and checks per game (finding 4). Window cost (H9) | No frontier | **WP4** |
| O2 | Agents communicating: does cheating spread, and how should the overseer react? Ground it in Willis et al. (13 Oct) | Overseer-to-overseer sharing only (H7, F23) | **Agent-to-agent communication untouched** | Deferred, stated as a limitation (§6). Needs its own protocol |
| O3 | Individual utility vs social welfare; how to choose an environment (13 Oct) | Objective vector (S, W, H, B) in `02_overseer_definition.md` §6; testbed criteria §8; LPGF as a testbed property (F27, F32) | Fine | Keep |
| O4 | A metric that shows the overseer works (13 Oct) | η, Δ, caught share, realised miss rate | Fine | Keep |
| O5 | Make the overseer an LLM (13 Oct) | An LLM as the **judge** (F24–F26, F28, F30) | An LLM never chooses **whom to look at** | Part of WP2, as an optional LLM leader (Gasztowtt-style), only if WP2's search works |

## 3. How strong the evidence is

Below, every main comparison at the seed level. Apples eaten per episode over the last 10% of training, 3 seeds per arm. The exact two-sided permutation test is used (`p`). **With 3 seeds per arm, the smallest p any comparison can reach is 0.10.** So every "solid" result sits at that floor: the seed ranges are apart, but no test can say more.

| Claim | Arm A | Arm B | Difference | Ranges apart | p |
|---|---|---|---|---|---|
| H2: repeat offenders first, hidden vs visible checks | 470, 465, 487 | 71, 71, 72 | +403 | yes | 0.10 |
| H2: random, hidden vs visible | 441, 466, 473 | 450, 468, 454 | +2 | no | 0.90 |
| H2: 30% random mix, hidden vs visible | 488, 491, 481 | 417, 170, 207 | +222 | yes | 0.10 |
| H3: gpt-oss table vs coin flip (k = 1, c = 1) | 300, 269, 279 | 74, 75, 73 | +209 | yes | 0.10 |
| H3b: coin flip vs inverted (k = 1, c = 2) | 502, 533, 509 | 497, 479, 479 | +30 | yes, by 5 | 0.10 |
| H3b: coin flip vs inverted (k = 7, c = 2) | 533, 540, 531 | 506, 481, 485 | +44 | yes | 0.10 |
| H1: threshold, c = 0.5 vs 1 (k = 2) | 68, 68, 68 | 441, 466, 473 | −392 | yes | 0.10 |
| **H1: k = 1 vs k = 7 at the same ρ (c = 1)** | 453, 410, 457 | 484, 479, 468 | **−37** | **yes** | 0.10 |
| H6: shared vs separate networks (3e8) | 441, 466, 473 | 448, 433, 459 | +13 | no | 0.40 |
| H9: w = 150 vs 50 (Clean Up) | 75, 227, 89 | 0, 0, 36 | +118 | yes | 0.10 |
| H1 in Clean Up: c = 2 vs 1 | 109, 142, 313 | 75, 227, 89 | +58 | no | 0.50 |

**Three things follow.**
1. **H1 in its strict form is false.** At the same ρ, k = 1 gives fewer apples than k = 7, even with a perfect inspector. What ρ decides is **survival**, on which side of the threshold the orchard lands. Above the threshold, how the sanction is spread (few big looks vs many small ones) still changes welfare: H1b, now seen without inspector errors too. The formulation must say this: *ρ sets the regime, the spread sets the level.*
2. **The large effects are safe** (H2, H3, the H1 threshold): differences of 200–400 apples with no overlap. **The moderate ones are not yet:** H3b's 30–44 apples, H1b's 37, and anything in Clean Up.
3. **Cheapest upgrade: 2 more seeds** on the key arms (5 per arm, smallest possible p ≈ 0.008). See WP1.

**Already grounded and documented:** each claim has an F-entry (`FINDINGS_AND_IMPLICATIONS.md` F27–F34), a hypothesis written before its runs (`03_hypotheses.md`), raw runs (`results/create_pull/`) and an analysis script (`experiments/socialjax/analyze_formulation_runs.py`, `measure_lpgf.py`).

## 4. Should the overseer be optimised? Yes: compare fixed protocols with the best protocol in each capability class

**Why it is needed.**
- The formulation calls the overseer a *player*. A player's protocol is only meaningful against the best it could do with the same capability. A fixed protocol that fails might fail through bad design, not lack of capability.
- The capability gap Δ is defined for a given overseer policy. The natural object is the gap for the **best** policy of each capability class (k, w, judge, F): that is what "a partially capable overseer can or cannot keep agents in check" means.

**How the literature does it.** We build on this rather than invent:
- **Griffin, Shlegeris & Abate 2024 (AI-Control Games).** They find optimal protocols by reducing a two-objective partially observable game (safety and usefulness) to zero-sum games, and trace the Pareto frontier. Their protocols are parameterised (audit thresholds). → We do the same over a parameterised look policy.
- **Gasztowtt et al. 2024 (Stackelberg-Markov game, LLM legislator).** The leader commits to parameters each episode; the followers learn in the game those parameters define; the leader is judged by what they learn. → This is the solution concept we use: protocol search in an outer loop, with learners retrained for each candidate protocol.
- **Zheng et al. 2022 (AI Economist).** The planner itself is learned with RL, in two levels. → The heavier option, kept for later.
- **Wang et al. 2019 (green security games).** A double oracle against best-responding attackers. → For the adversarial mode, if outer search is not enough.
- **Greenblatt et al. 2024 (AI Control).** Protocols are evaluated against the best red-team attack found. → Already our worst-case convention.

**Proposed design (WP2).** Fix the capability class used throughout: k = 2 of 7, c = 1, perfect judge. Optimise the look policy within that class.
- **The protocol family:** the mix ε, the probability that a look is random rather than repeat-offenders-first.
  - We already have ε = 0, 0.3 and 1 in both agent modes (12 + 6 runs).
  - Add ε ∈ {0.5, 0.7, 0.9} in both modes: 18 runs.
- **What we already see** (a hint, not a result): with hidden checks the 30% mix is the best arm (487 vs random 460); with visible checks random is best.

**H10 (proposed).** The best look policy depends on what the agents can observe:
- with hidden checks, some targeting beats pure random;
- with visible checks, the best ε is at or near 1;
- the optimised overseer's capability gap is smaller than that of any fixed predictable policy.

*Refuted if* one ε is best in both modes, or random is best in both.

**Later, only if H10 holds:** an LLM leader that proposes ε (and k per region) between training rounds, as in Gasztowtt. This would also answer O5.

**What WP2 does not do.** It does not co-train a learning overseer with the agents. That is unstable and gives a different solution concept (co-learning, not leader-follower). Stated as future work.

## 5. Where to go: the remaining work

Each package maps to feedback and to a hypothesis. Costs assume about 40 minutes per 1e8-step IPPO run on a large GPU, and 6–8 GPUs at a time under our quota.

| WP | What | Feedback | Hypothesis | Runs | GPU-hours | Ollama $ |
|---|---|---|---|---|---|---|
| **WP1** | 2 more seeds (4, 5) on the key arms: H1 (k = 1/7, c = 0.5/1), H2 (6 arms), H3/H3b (12 arms), Clean Up H9 (4 arms) | evidence | existing | about 56 | about 40 | 0 |
| **WP2** | Optimised overseer: ε search, both agent modes | Y1b, Y3, O5 | H10 | 18 | about 12 | 0 |
| **WP3** | Capability curve. Situation-specific miss and false-alarm tables for open judges of increasing size (Qwen 2.5 3B / 7B / 14B / 32B / 72B, served on CREATE with vLLM, on the existing 2,463-item bank). Predict each judge's outcome from ρ(s_harm) *before* the RL runs, then check with RL at c = 1 and 2 | Y1 | H11 | about 30 RL + judging jobs | about 25 | 0 (open models) |
| **WP4** | Cost frontier: outcome against oversight spend (checks × tokens per check × steps read), from existing runs plus WP3's token counts | Y2, O1 | none (descriptive) | 0 | 0 | 0 |
| **WP5** | The general formulation (draft 2): one budgeted-overseer problem class; game-specific parts = rule, harm predicate, gain; three sufficiency conditions (§2, Y4). Instances: Harvest (all hold), Clean Up (condition iii fails), BazaarBench (planned) | Y4 | H12 | 0 to start | 0 | 0 |

**H11 (proposed).** A judge keeps the commons if and only if its predicted ρ(s_harm) = (k/N)(1 − α_harm)F is above the threshold measured in stage 1 (about 0.75–0.9). Overseer size matters only through α_harm, so a small judge with a low miss rate on harmful situations protects as well as a large one.

*Refuted if* two judges with the same α_harm land on different sides, or the prediction misclassifies more than 1 of 10 cells.

This is the sequential version of Engels et al. 2025 (scaling laws for scalable oversight), who list single-step oversight as their limitation.

**H12 (proposed).** The three conditions predict where budgeted individual inspection works. A **second game where all three hold** must show the H1 threshold too.

*Candidates for that game:*
- (a) Clean Up with a rule that also protects the commons. That is the commons view (Decision 1). It needs a rule change and about 12 runs.
- (b) Another SocialJax commons game.
- (c) BazaarBench with a live overseer. The most aligned with Yali's group, the most expensive, and it needs Ziyan Wang and Shuqing Shi.

**Order:** WP1 + WP2 together (overnight, about 74 runs) → WP3 (needs vLLM set up on CREATE first; effort unknown) → WP4 (desk work, any time) → WP5 written alongside, the second-game test last.

## 6. Devil's advocate

**Steel-man.** Every piece now answers a named piece of feedback. The optimised overseer and the capability curve turn "we tried some protocols" into "this is the best a partially capable overseer can do, and this is how its power scales". Both extend AI-Control Games and Engels et al. into sequential multi-agent commons, which those papers name as missing.

**The three biggest risks.**
1. **The ε search is too narrow to count as "optimised".** One parameter is a weak optimiser.
   - *Probability:* medium; a reviewer could say so.
   - *Fix:* call it what it is, the best protocol in a stated family (Griffin et al. also optimise over a parameterised family). Add k-per-region as a second parameter only if ε shows structure.
2. **WP3 depends on infrastructure we haven't tested** (vLLM on CREATE; Qwen 32B/72B memory).
   - *Probability:* medium.
   - *Detectability:* high, in a one-job smoke test.
   - *Fix:* a smoke test first. Fall back to the four local Qwen sizes plus gpt-oss 20B/120B on Ollama: about 2.4M tokens per table, about $0.77 each, $8.2 left, $5 reserve. That allows at most 3 cloud tables.
3. **Generality stays thin.** Two SocialJax games, one of which fails a condition, plus a planned BazaarBench. Yali's own bar (DMPO, Ma et al.) is many testbeds with theory.
   - *Probability:* high.
   - *Fix:* make the three conditions the theory, falsifiable and stated per testbed. Choose the second-game test (H12) early. Be explicit that BazaarBench is the next instance, not a current one.

**Best alternative.** Skip WP2 and WP3. Write the formulation now from current evidence, plus WP1's seeds only. This is right if the deadline (ICML, about 22 Jan) is the binding constraint. Wrong if the paper's claim is "partially capable overseer", which needs WP3's capability curve to say anything about capability.

**How hard to undo.** WP1–WP4 are runs and analysis, easy to stop. Choosing the second game for H12 is costly to change later.

**What we forgot.**
- Agent communication (O2) is out of scope for now. That must be said in the paper's limits.
- The overseer's own cost of computing the protocol (the WP2 outer loop) is not counted in the cost frontier.
- Clean Up's training instability could be a hyperparameter issue. Nobody has checked the SocialJax paper's Clean Up IPPO curves against ours.

**Verdict.**
1. Approve WP1 + WP2 (about 74 runs, about 52 GPU-hours, no Ollama).
2. Start the vLLM smoke test for WP3.
3. Write WP5 draft 2 in parallel.
4. Decide H12's second game within the week.

## 7. Decisions for Ameer

1. Approve **H10** (optimised overseer) and **WP1 + WP2**?
2. Approve **H11** and the WP3 plan (open judges on CREATE; cloud fallback capped at 3 tables)?
3. **H12's second game:** (a) Clean Up with a commons-protecting rule, (b) another SocialJax game, or (c) BazaarBench live (needs Yali's group)?
4. Agent communication (O2): confirm it is deferred and stated as a limitation?
