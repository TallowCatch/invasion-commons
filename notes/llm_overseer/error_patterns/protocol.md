# Patterned LLM inspector errors: protocol and predictions (written 23 Nov 2026, before any rollout or judge run)

**The question.** The 40 RL runs with LLM inspectors (`../rl_results.md`) treat each LLM as a coin flip: a check misses a violation with a fixed chance α and flags an honest harvest with a fixed chance β, whatever the situation. A real LLM's mistakes depend on the record it reads. **Does a state-dependent error pattern, measured from a real LLM, change deterrence compared with a coin flip at the same average α and β?** This protocol covers the offline measurement (situation bank, judging, error model), the trainer option that plays the measured pattern, and the RL job list. The RL runs themselves are not run here.

**What was already seen when this was written.** The ladder's w = 1 results (`../SUMMARY.md`): gpt-oss-20B on raw maps α 0.10, β_eat 0.08 (30 violations, 26 honest eats); Qwen 7B + reasoning on raw maps α 0.47, β_eat 0.04. Their token costs per raw w = 1 check (from the ladder's judged replies, not their verdicts): gpt-oss-20B about 510 in + 440 out tokens and 3.6 s; Qwen 7B + reasoning about 490 in + 270 out and 32 s locally. No per-situation breakdown of any judge's mistakes was looked at before writing the predictions.

## 1. Situation bank

- **Agents.** Trained IPPO parameters (shared network) from stage 1: `A_none_seed1` (no overseer: strips the orchard), `B_random_k1_c1.0_seed1`, `B_random_k2_c1.0_seed1`, `B_random_k7_c2.0_seed1`, `D_history_k2_c1.0_seed1` (all keep the orchard). Each is played for full 1,000-step episodes with the policy sampling its actions, under no overseer (their behaviour does not depend on the overseer when nothing is frozen and nothing is revealed).
- **Items.** At w = 1 the trainer applies a miss only to a checked violation and a false alarm only to a checked honest harvest. Both are apple-eating steps, so **every eat is one item**, and steps without an eat cannot produce an inspector mistake. Every eat in the rollouts goes into the bank.
- **Episodes.** A_none eats about 68 apples a game, mostly as violations; the overseen agents eat 440–530 with about 7–11% violations. Planned: 16 episodes of A_none and 3 of each overseen policy, so about 1,100 + 5,800 = 7,000 eats, with violations from both kinds of agent. Exact counts are reported.
- **Ground truth** is the wrapper's: violation if the apple had fewer than 3 apples among the 12 cells within two squares, on the map before the move (`overseer/socialjax/harvest_overseer.py`).
- **Features**, all on the map before the move, in the 5 × 5 patch the raw rendering shows around the apple taken:
  - `near`: apples in the 12 counted cells (0–12). It fixes the label (violation ⇔ near ≤ 2);
  - `outer`: apples in the other 12 cells of the patch (corners and their neighbours), which do not count and may be miscounted in;
  - `players`: players in the patch other than the centre (the eater itself is usually one of them);
  - `walls`: wall or off-map cells in the patch;
  - `source`: which policy produced the item (a diagnostic, not a model feature, since the trainer cannot know it).
- Each item is rendered with the existing raw renderer (`overseer/llm/harvest_records.render_raw`, w = 1) and saved with its features.

**Amendment 1 (23 Nov, after a 5-episode smoke rollout and a 3-item cloud smoke test, before any judging).**
- The smoke rollout showed violations at near = 0 and 1 are scarce (about 30 per episode from A_none, a few from each overseen policy), so the episode counts were **doubled: 32 episodes of A_none and 6 of each overseen policy**.
- The bank (`experiments/llm_overseer/data/error_bank.jsonl.gz`): **13,159 eats, 1,717 violations** (near 0 / 1 / 2: 497 / 500 / 720). Honest eats: near 3: 7,232; 4: 2,080; 5: 1,227; 6: 520; 7: 220; 8: 45; 9: 35; 10: 83; none at 11–12. Learners eat at exactly the threshold most of the time (55% of all eats are at near = 3).
- With the quotas of section 2 the subset is therefore **2,463 items** (1,200 violations, 1,263 honest eats), about 2.3M cloud tokens at the smoke test's 980 tokens a check, so the whole subset fits under the cap.
- The 3-item cloud smoke test (2,952 tokens) is in the token log and its verdicts are kept (the run resumes from them).

## 2. Judging

- **Inspectors** (the two borderline ones in `../SUMMARY.md`):
  - **gpt-oss-20B on raw maps**, Ollama cloud, direct prompt, `think: low`, up to 2,048 output tokens (as in the ladder);
  - **Qwen2.5-7B-Instruct + reasoning on raw maps**, local, reason prompt, `--num-ctx 4096`, up to 768 output tokens.
- **Judging subset, stratified by `near`.** Within each value of `near`, items are drawn at random from the bank, so the other features keep their natural mix given `near`. Planned for gpt-oss-20B: up to 400 items per near value for 0, 1, 2 (violations), and 300 / 250 / 200 / 200 for near 3 / 4 / 5 / 6, and 150 per value for 7–12 (or all there are). The subset is shuffled once (seed 0) and judged in that order, so if a run stops early, what was judged is still a random part of the subset.
- **Qwen 7B + reasoning** is too slow for all of it (about 32 s an item on this M1). It judges the first 600 items of the same shuffled order (paired with gpt-oss on those items). If time allows, it continues.
- **Cloud cap: 3,000,000 tokens in total** for this study (in + out, gpt-oss thinking included; about $1 at about $0.32 per million). Every call is logged to `notes/llm_overseer/error_patterns/cloud_tokens.tsv`; the run stops at the cap. At about 950 tokens a check this allows about 3,100 items.
- **No verdict** counts as no flag (as in the ladder).
- **Determinism.** Temperature 0 and a fixed seed: the same record gets the same verdict. The error model below turns this into a probability per feature cell; that is a coarser pattern than the LLM's real one (section 8).

**Amendment 2 (23 Nov, during judging, before any analysis of verdicts).** The first parallel start of both judges ran the disk out of space within 2 minutes (Qwen 7B swapping on the 8 GB M1; 12 gpt-oss and 1 Qwen verdicts were written, all intact). Up to 4 in-flight cloud calls at the crash were not logged (at most about 4,000 tokens). gpt-oss was restarted alone; Qwen 7B was restarted under a guard that stops it if free disk falls below 1.2 GB, with `--num-ctx 2048` (the setting of the ladder's own reason-mode runs, `../protocol.md` Amendment 2; the longest prompt plus 768 output tokens fits in it).

**Amendment 3 (23 Nov, during judging).** Ameer capped the local Qwen judging at 200 items to free the laptop, with Qwen as the secondary inspector: fit its pattern if 200 are enough, otherwise report its averages only. The disk guard stopped it at 179. After both fits, one rule was added (so it was written after seeing the Qwen fit): a trainer table is written only if M3 beats M0 on held-out items for α or β (the lower end of the interval above 0). gpt-oss passes; Qwen does not, so the job list has gpt-oss only.

## 3. Error model

- **α(s)** = P(no flag | violation, features s); **β(s)** = P(flag | honest eat, features s).
- **Models compared**, fitted separately for violations and honest eats:
  - M0: constant (the coin flip);
  - M1: one rate per `near` value (violations: 0, 1, 2; honest eats: 3, 4, 5, 6, 7–8, 9–12, merged further where a cell has under 20 items);
  - M2: logistic regression on `near` (as categories), `outer`, `players`, `walls`;
  - M3: the binned table actually used in the trainer: `near` × (`outer` in two or three bins), chosen from M2's coefficients.
- **Test.** Held-out log-likelihood per item, 5-fold cross-validation (folds fixed by seed), M1–M3 against M0, with a paired bootstrap 95% interval on the difference (2,000 resamples). Also Brier score and a calibration table of predicted vs observed rate.
- **How far α and β vary:** the range of the fitted table's entries over cells with at least 30 items, with 95% Wilson intervals, and the spread weighted by how often each cell occurs.
- **Matched average.** The coin flip to compare with uses α̅ and β̅ averaged over the **pooled bank at its natural frequencies** (every eat counts once; A_none supplies most violations). As a sensitivity, the same average over the overseen policies only. If the two differ by more than 0.05, both coin flips go in the job list.

## 4. Trainer option

- `--inspector-table <json>` (off by default). At w = 1 in Harvest, a check's miss chance is α(s) and its false-alarm chance β(s), looked up from the table for the checked agent's features this step (computed in JAX from the map before the move and the agent's cell). Everything else is unchanged: the same random numbers are drawn, compared with a per-agent rate instead of one number.
- Two new per-episode measures in every run (the existing ones are unchanged): checks that fell on a violation and on an honest eat, so the realised α = 1 − detected / checked violations and realised β = false alarms / checked honest eats; and in Harvest, violations and eats by `near` value.
- Tests (`tests/test_inspector_table.py`): the default path is unchanged; a table with constant entries reproduces `--miss-rate a --false-alarm b` exactly; the JAX features match the renderer's patch; the lookup returns the right cell.

## 5. RL job list (written, not submitted)

Per inspector: patterned table vs coin flip at the matched average, k ∈ {1, 7}, c ∈ {1, 2} (penalty 7c/k per catch), random checks, 3 seeds, 1e8 steps from scratch, as in `../rl_results.md`: 2 inspectors × 2 arms × 4 cells × 3 seeds = 48 runs (plus sensitivity coin flips if section 3 calls for them).

## 6. Predictions (written before any rollout or judging)

**E1. Misses cluster on the borderline violation.** α is higher at near = 2 than at near = 0:
- gpt-oss-20B: α(0) ≤ 0.05 and α(2) ≥ 0.15;
- Qwen 7B + reasoning: α(2) − α(0) ≥ 0.2.
The reason: an apple with no apples around it is obviously isolated; one with 2 needs an exact count.

**E2. False alarms cluster on the borderline legal eat.** β(3) is at least twice the average β over near ≥ 6, for both inspectors.

**E3. Apples in the cells that don't count push the judge towards "legal".** Given `near`, more `outer` apples raise α and lower β (the over-counting mistake predicted in `../protocol.md` P2). In M2, the coefficient of `outer` on P(flag) is negative for both labels, for both inspectors. No prediction for `players` or `walls`.

**E4. State dependence beats the coin flip on held-out items.** For both inspectors, M1 or M2 beats M0 in held-out log-likelihood with the bootstrap interval above 0, and the table's α ranges over at least 0.15 across well-populated cells.

**E5. The matched averages move.** On learners' records the averages differ from the ladder's bot-record numbers (α 0.10 / β 0.08; α 0.47 / β 0.04) by more than their ladder intervals for at least one of the four numbers, because the learners' mix of near values is not the bots'.

**E6. What we expect in RL (to be tested by the job list).** If E1 holds, learners under the patterned inspector shift their over-harvests to near = 2 apples, where checks miss most:
- the share of violations at near = 2 is at least 10 points higher than under the matched coin flip;
- the realised fine per over-harvest, (1 − realised α)·c, is lower than the coin flip's;
- apples and orchard survival are worse where the coin flip sits near the threshold (gpt-oss-20B at k = 1, c = 1; Qwen 7B + reasoning at c = 2, k = 1).
- A counter-effect: a near-2 over-harvest leaves more apples around it than a near-0 one, so it does less harm to regrowth. The orchard may suffer less than the realised α suggests.
- If E2 holds, honest agents can steer away from the borderline legal apples, so wrongful fines per legal apple are lower than β̅·c.

**What would show that patterned errors don't matter:**
- offline: α and β vary by less than 0.10 across cells with at least 30 items, or no model beats M0 on held-out items (interval includes 0). Then the coin-flip runs stand;
- in RL: patterned and matched coin flip runs fall within each other's seed ranges on apples, orchard survival and fines per over-harvest, in all four cells for both inspectors.

## 7. Related work behind the design choices

References checked by web search on 23 Nov 2026 (title, authors, year). ★ = in Ameer's reading matrix.

- **Deterrence as chance × fine.** Becker (1968), *Crime and Punishment: An Economic Approach*, JPE 76(2):169–217: the expected fine is the chance of conviction times the fine. Our realised fine per over-harvest, (1 − α)·(k/7)·P, is that quantity, and a state-dependent α makes it differ across apples.
- **Enforcement errors and deterrence.** Png (1986), *Optimal Subsidies and Damages in the Presence of Judicial Error*, Int. Rev. Law & Econ. 6(1):101–105, and Polinsky & Shavell (2000), *The Economic Theory of Public Enforcement of Law*, JEL 38(1):45–76, give the standard view: wrongful acquittal makes violating pay more and wrongful conviction makes complying pay less, so both shrink the gap and weaken deterrence. (Png's own naming of the two error types is the reverse of today's, so we name them by meaning.) Kaplow & Shavell (1994), *Accuracy in the Determination of Liability*, J. Law & Econ. 37(1):1–15, ground the value of accuracy in deterrence. Lando (2006), *Does Wrongful Conviction Lower Deterrence?*, J. Legal Studies 35(2):327–337, shows the conclusion depends on whether the error risk changes with the act chosen; Garoupa & Rizzolli (2012), *Wrongful Convictions Do Lower Deterrence*, JITE 168(2):224–231, reply that in general it does. Lando & Mungan (2018), *The Effect of Type-1 Error on Deterrence*, IRLE 53:1–8: the effect of wrongful conviction scales with how often the lawful choice is exposed to judgement. Kaplow (2011), *On the Optimal Burden of Proof*, JPE 119(6):1104–1140: errors also chill benign activity.
- **Our F24 result against this theory.** F24 found false alarms barely weakened deterrence. That is not a conflict once one asks what β falls on. In our trainer a false alarm can only fall on a **legal harvest**, never on "not harvesting". So, per check:
  - over-harvest vs. wait: the expected fine gap is (1 − α)·P, with no β;
  - over-harvest vs. a legal harvest: the gap is (1 − α − β)·P, as in Png.
  The law-and-economics prediction (β weakens deterrence) applies only where the agent's real alternative to an illegal apple is a legal one. Lando & Mungan's exposure point says the same: β matters in proportion to how often the lawful choice is judged, and "wait" is never judged here. F24's reading — in Harvest the alternative is often to wait — fits. What β does do in our runs is what Kaplow calls chilling: it taxes legal eating (C2). This study tests one more step: if β is patterned, the chilling falls on particular legal apples, and agents can steer around it.
- **Inspection games.** Avenhaus, von Stengel & Zamir (2002), *Inspection Games*, Handbook of Game Theory Vol. 3, ch. 51, pp. 1947–1987, treat inspections with errors of the first kind (false alarm) and second kind (non-detection) as statistical tests; Avenhaus & Canty (1996), *Compliance Quantified*, CUP. Their errors are usually fixed rates of a test; ours are a function of the situation, which is what a learner can respond to.
- **Instance-dependent errors.** The coin flip is "class-conditional noise": the error rate depends only on the true label. Frénay & Verleysen (2014), *Classification in the Presence of Label Noise: A Survey*, IEEE TNNLS 25(5):845–869, separate this from noise that also depends on the features; Chen et al. (2021), *Beyond Class-Conditional Assumption*, AAAI 35(13):11442–11450, argue real noise is rarely class-conditional. Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS D&B, document LLM-judge biases (position, verbosity) that depend on the input. This is why α(s), β(s) is fitted on features, and why held-out likelihood against a constant rate is the test.
- **Calibration of the error model.** Guo et al. (2017), *On Calibration of Modern Neural Networks*, ICML: we report calibration of the fitted rates alongside likelihood.
- **Learners exploiting a state-dependent evaluator.** Everitt et al. (2017), *Reinforcement Learning with a Corrupted Reward Channel*, IJCAI: when reward corruption depends on the state, RL agents seek out the corrupted states. Skalse et al. (2022), *Defining and Characterizing Reward Hacking* (printed at NeurIPS as "…Reward Gaming"): almost any imperfect proxy is exploitable in principle. Gao, Schulman & Hilton (2023), *Scaling Laws for Reward Model Overoptimization*, ICML: optimising hard against an imperfect evaluator eventually lowers the true objective. Hardt et al. (2016), *Strategic Classification*, ITCS: agents shift their features to game a known classifier. Our agents cannot read the table, but they can learn it from fines, which is the Everitt et al. setting with a fine-sized signal. E6 is the prediction this literature motivates.
- **LLMs as overseers.** Bowman et al. (2022), *Measuring Progress on Scalable Oversight for Large Language Models*, arXiv:2211.03540; Engels et al. (2025)★, *Scaling Laws For Scalable Oversight*, NeurIPS (the ladder's closest prior work); Greenblatt et al. (2024)★, *AI Control: Improving Safety Despite Intentional Subversion*, ICML (trusted monitoring with a fixed audit budget, the monitor's false-positive rate set by that budget). Both control and scalable-oversight work score monitors by average rates; this study asks whether the average is enough once the monitored agents learn.
- **Monitoring and audits in RL and agents.** Parisi et al. (2024)★, *Monitored Markov Decision Processes*, AAMAS (the reward is seen only when a monitor shows it; our inspector is a noisy monitor). Wang et al. (2019)★, *Deep RL for Green Security Games with Real-Time Information*, AAAI (randomised patrols against adaptive attackers). Gans & Holden (2026)★, *When Does Randomized Oversight Align AI Agents That Can Conceal?*, NBER WP 35878 (audits without errors; stronger audits make undeterred violations better hidden). Makins et al. (2026)★, *Multi-Agent AI Control: Distributed Attacks Hamper Per-Instance Monitors*, arXiv:2607.07368 (per-agent monitors have blind spots coordinated agents exploit). Ye & Steinhardt (2026)★, *Norm Enforcement for AI Agents*, arXiv:2607.09766 (agents weaponise false reports). Zhu, Zuo & Sun (2026)★, *Blindfold*, arXiv:2610.37055 (an agent that sees the audit coin biases the audit; here the coin stays hidden).
- **Locality (Yali Du's group).** Du et al. (2022), *Scalable Model-based Policy Optimization for Decentralized Networked Systems* (DMPO), IROS, arXiv:2207.06559, and Ma, Li, Du, Dong & Yang (2024), *Efficient and Scalable Reinforcement Learning for Large-Scale Network Control*, Nature Machine Intelligence 6:1006–1020: an agent's dynamics depend only on its κ-hop neighbourhood, and truncating to that neighbourhood costs an error that decays exponentially with κ. The link here is narrow: our error features are all **local** (the 5 × 5 patch), so the inspector's error model is itself a κ-local function of the state, with κ = 2 cells. That is what lets the table run per agent inside a decentralised overseer. Beyond that, these papers do not speak to inspector error; we don't claim more.
- **BazaarBench★** (Wang et al. 2026, arXiv:2610.06748): its LLM-judge labels (GPT-5, high reasoning) were checked by humans on 200 cases (94% raw agreement), with some systematic disagreements. It is a reminder that a judge's errors are patterned even when its average agreement is high; this study measures the pattern directly on a task with exact ground truth.

## 8. Limitations stated up front

- **The table is coarser than the LLM.** At temperature 0 the LLM's verdict is a fixed function of the full record; the table keeps only what four local features capture. Learners in the patterned runs can exploit only that part. Anything the LLM gets wrong for reasons the features miss (for example, an exact arrangement of apples) becomes a coin flip within its cell.
- **Learners can only learn the pattern from fines.** With k = 1 of 7 checked, an agent sees a fine on about 1 in 7 over-harvests, so learning a near-dependent α needs many episodes; 1e8 steps may or may not be enough.
- **The bank is from policies trained against perfect inspectors or none.** Learners trained against the patterned inspector may visit situations the bank under-samples (section 3 reports the cells with few items; the table merges thin cells).
- One prompt per inspector, one seed, 4-bit local weights for Qwen; gpt-oss via Ollama's cloud.
