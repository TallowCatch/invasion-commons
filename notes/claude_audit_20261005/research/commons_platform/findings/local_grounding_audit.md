# Local grounding audit: every setting of the current paper and the L-series

14 October 2026. Sub-agent 6 of the commons-platform deep research (`../plan.md`). Read-only audit of the repository on branch `claude/r1-s1-oversight-experiments`; no code was changed and nothing was run except short prints of configuration values. Evidence, with exact code lines and quotes, is in `../sources/L01`–`L12`.

## How to read the table

- **Verdict.** **G**: the value or form is tied to a cited source in the notes or paper. **C**: chosen for convenience, by a pilot, by a calibration rule or by the study design, and the reason is written down. **U**: no reason recorded (sometimes the notes say outright that the value is a choice, without saying why).
- G is generous. Most G rows rest on a *concept* from a source read at secondary or abstract depth (for example, logistic growth and MSY from Schaefer 1954 via Wikipedia, `literature_ledger.md:42`), not on a value taken from data.
- **After results** marks settings chosen or changed after seeing outcomes of the same or an earlier study ("pilot", "post hoc", "after", "amended"). Most are dated and disclosed in the protocols.
- Paths are relative to the repository root; `S:` = `notes/claude_audit_20261005/studies/`; `RR:` = `notes/research_review/`.

## The table

### Fishery: resource model (L01)

| ID | Setting | Value | Set at | Recorded justification | Verdict | After results |
|---|---|---|---|---|---|---|
| F1 | Growth law | logistic, x' = y + r y(1 − y/K), y = x − catch | `fishery_sim/env.py:183-185` | "The sustainability target is the logistic **MSY level**: stock K/2 left after harvest (Schaefer 1954, secondary source)" (`S:R1_repaired_reviewer/protocol.md:38-39`); paper cites May et al. 1978 (`body.tex:39`) | G | |
| F2 | Regrowth r | 0.7 (grids 0.5/0.7/0.9 in R2; ×1.0/×0.85 in R5) | `experiments/oversight/claude_oversight_common.py:35` | "These are declared pilot choices, not externally calibrated values." (`RR:MATCHED_OVERSIGHT_PILOT_PROTOCOL.md:57-59`) | U | 0.7 is the "piloted level" (R2 protocol:28) |
| F3 | Capacity K | 100 t | `claude_oversight_common.py:34` | same quote | U | |
| F4 | Initial stock | 70 t | `claude_oversight_common.py:34` | same quote (70 ≈ the MSY pre-harvest level 67.5; not recorded as a reason) | U | |
| F5 | Max catch per agent | 6 t (6 × 6 = 36 t ≈ 2 × MSY of 17.5 t) | `claude_oversight_common.py:35`; `fishery_sim/llm_actor.py:25` | same quote | U | |
| F6 | Observation noise, weather | none; deterministic | `claude_oversight_common.py:35`; `fishery_sim/fishery_oversight.py:25-26` | "No weather variants or repeated weather seeds are invented for this deterministic game." (`RR:MATCHED_OVERSIGHT_PILOT_PROTOCOL.md:60-61`) | C | |
| F7 | Collapse rule | stock < 10 for 5 rounds in a row | `fishery_sim/config.py:16-17` | "danger threshold 10 and the existing collapse patience. These are declared pilot choices" (same file) | U | |
| F8 | Over-demand | catches scaled down proportionally | `env.py:174-177` | none | U | |
| F9 | r ≤ 1 | enforced | `fishery_oversight.py:27-28` | "Monotone accounting bounds require regen_rate <= 1" | C | |
| F10 | Number of agents N | 6 | `claude_oversight_common.py:34` | "Use six agents." (no reason; `RR:MIXED_POLICY_REPAIR_PROTOCOL.md:18`) | U | |
| F11 | Horizon, programmed | 80 rounds | `claude_oversight_common.py:33` | listed under "Kept from the original design" (`S:R1_repaired_reviewer/protocol.md:52-60`) | U | |
| F12 | Horizon, LLM | 20 rounds | `run_l2_llm_agents.py:36` | "**Horizon: 20 steps**, not 80, to keep the pilot cheap." (`S:L1_llm_actor_pilot/protocol.md:37`) | C | |

### Programmed agents, shared across games (L02)

| ID | Setting | Value | Set at | Recorded justification | Verdict | After results |
|---|---|---|---|---|---|---|
| A1 | Policy form | request "low" share below own stock threshold, "high" above | `claude_oversight_common.py:45-46` | "These are design choices, not measured actor-capability levels." (`RR:HELDOUT_POLICY_PILOT_PROTOCOL.md:25`) | U | |
| A2 | Thresholds | U(25, 65) t | `claude_oversight_common.py:41` | same quote | U | |
| A3 | Honest request ranges | low U(5%, 20%), high U(20%, 40%) of 6 t | `claude_oversight_common.py:37` | same quote | U | |
| A4 | Greedy request ranges | low U(45%, 65%), high U(75%, 95%) | `claude_oversight_common.py:37` | same quote | U | |
| A5 | Greedy count | 4 of 6 in Fishery, 2 of 6 in Forest (grid 2, 4) | `claude_oversight_common.py:21, 33` | "chosen because they produced both safe and risky requests" (`notes/claude_audit_20261005/background/design_issues.md:201-202`) | C | pilot |
| A6 | Under-report factor (all games) | report = 0.5 × request | `run_r2_robustness_reviewer_model.py:55`; `run_t1_audit_targeting.py:33`; `run_c1_compositional_harm.py:34`; `run_r4_river_grid.py:27` | "as in C1, T1 and R2's fixed liars" (`S:R4_river_grid/protocol.md:24`); 0.5 is the largest of S1's fixed levels {0, 0.25, 0.5} | U | |
| A7 | Over-take family | take a + d(p − a), d ∈ {0, ¼, ½, ¾, 1}, one d per game | `run_s2_compliance_deterrence.py:34`; `run_s4_adaptive_memory_cost.py:37` | "a lower bound on an adaptive adversary" (`body.tex:131`) | C | |
| A8 | Search training set | 8 populations (S1–S4); 32 (R2, R3, S5) | `run_s2_compliance_deterrence.py:44`; `run_r2_robustness_reviewer_model.py:66`; `run_s5_stronger_attackers.py:54` | "Use 32 instead of 8" after S3 Part C over-fitted (`background/caveats_assessment.md:136-137, 180`) | C | after S3; paper says 32 for all |
| A9 | Reaction to a catch | continue or stop | `run_s4_adaptive_memory_cost.py:38` | S4 protocol | C | |

### Reviewer, all games (L03)

| ID | Setting | Value | Set at | Recorded justification | Verdict | After results |
|---|---|---|---|---|---|---|
| R1 | Shared-cut menu λ | {1, ¾, ½, ¼, 0} | `fishery_sim/calibrated_oversight.py:23` | "The finite menu is shared; its resolution is a limitation, not a separately tuned advantage." (`RR:MATCHED_OVERSIGHT_PILOT_PROTOCOL.md:37-38`) | C | |
| R2 | Choice rule | largest λ whose prediction meets the target, else 0 | `calibrated_oversight.py:112-118` | same | C | |
| R3 | Reviewer types | joint; local bounded; local optimistic | `calibrated_oversight.py:101-110` | design contrast (`supplementary.tex:65-66`) | C | |
| R4 | Fishery target | stock left after fishing ≥ K/2 = 50 | `calibrated_oversight.py:96-97` | Schaefer 1954 (secondary), May 1978 (abstract) (`R1 protocol:38-39`; `body.tex:39`) | G | introduced to test a post hoc explanation (R1 protocol:120); default "because of R1" (S1b protocol:110) |
| R5 | One-step line | next stock ≥ 10 | `calibrated_oversight.py:94-95` | "no standard literature name" (`literature_ledger.md:70`) | U | |
| R6 | Chance-constraint tolerance (Forest, River) | 5% | `calibrated_oversight.py:52`; `fishery_sim/two_reagent.py:42` | "The reviewer's 5% rule is a **chance constraint** (Charnes & Cooper 1959)." (`R1 protocol:37`; concept only, no source for 5%) | U | |
| R7 | Monte Carlo draws | 400 (reviewer); 2,000 / 4,000 (labels) | `calibrated_oversight.py:52, 72` | R1 protocol:45-47 | C | |
| R8 | Reviewer's model | true model; R2 misspecifications ×0.5/×2 noise, ×0.75/×1.25 r and K, learned, Allee A = 20 | `run_r2_robustness_reviewer_model.py` Part B | R2 protocol Part B (sizes not sourced) | C | |

### Audits, fines, memory, programmed studies (L04)

| ID | Setting | Value | Set at | Recorded justification | Verdict | After results |
|---|---|---|---|---|---|---|
| P1 | Main audit rate q | 1/6 per agent-round (one of six) | `run_r2_robustness_reviewer_model.py:57`; `run_s3_threshold_timing_memory.py:34` | "Our k of 6 per step is a much larger share (17–100%) ... Results at k/6 are not comparable to the 2% regime." (`literature_ledger.md:84`) | C | |
| P2 | Audit-rate grids | S3 {1/6, 0.10, 0.05, 0.02}; S4 {0.01 ... 1/3}; S5 {0.02 ... 1/6} | `run_s3_...:34`; `run_s4_...:36`; `run_s5_...:38` | S3, S4, S5 protocols | C | |
| P3 | Detection rate s | 1, 0.5, 0.25 | `run_s3_...:35` | S3 protocol:57 | C | |
| P4 | Audits in the memory grid (Fig. 2a) | 2 per round in Fishery and Forest (rand2, 1/3 per agent); 1 per round in River | `run_r2_...:52` (R5 reuses); `run_r4_river_grid.py:79-80` | "Two random audits per step" (R2 protocol:43-44), no reason | U | paper says 1/6 per agent (`supplementary.tex:22`) |
| P5 | S1 fine | 12 per catch | `run_s1_reporting_audit.py:33` | "two steps of one agent's maximum harvest" (S1 Amendment 1, after the smoke run) | C | after smoke run |
| P6 | S1 exclusion | 5 rounds | `run_s1_reporting_audit.py:32` | none | U | |
| P7 | S2 fines | 0, 6, 12, 24 | `run_s2_compliance_deterrence.py:33` | none found | U | |
| P8 | S3 expected-fine grid | 0–1 (11 values) | `run_s3_...:33` | S2's threshold was "near a fine of 2, below the grid" (S3 protocol:25-27) | C | after S2 |
| P9 | S4, S5 fine | 6 | `run_s4_...:34`; `run_s5_...:35` | "from S3, deterrence starts at e = q × 6 ≈ 0.36" (S4 protocol:98-99) | C | after S3 |
| P10 | R2/R3 expected-fine grid | e = j × 0.2 g*, j = 0..10 | `run_r3_breakeven.py`; R3 protocol | R3 protocol; R2's gain was redefined "*after* seeing the data" (R3 protocol:18-20) | C | R2 post hoc |
| P11 | Memory rules | rescale reports by mean caught lie; targeted allowance; tighter shared cut | `two_reagent.py:239-247`; `run_s4_...:44-48` | concept: "records of past offences give enforcement leverage" (Harrington 1988, `body.tex:24`); rules are the project's own | C | |

### Harm lines and metrics (L05)

| ID | Setting | Value | Set at | Recorded justification | Verdict | After results |
|---|---|---|---|---|---|---|
| H1 | Fishery harm | stock after fishing < 50 of 100 | `run_l2_llm_agents.py:326-327`; `calibrated_oversight.py:96-97` | "**Harm** is the resource below half capacity, where it regrows fastest." (`PAPER_PLAN.md:58`); logistic B_MSY = K/2 | G | decided 8 Oct, labelled post hoc (`runs/social_metrics/README.md:1`) |
| H2 | Forest harm level | mean plot health < 10 of 20 | `fishery_sim/harvest.py:23` | same; the line predates the MSY reading ("the safety line sits close to the most productive level", `background/september_reviewer_experiment.md:41-42`) | G | rationalised post hoc |
| H3 | Forest extra failure | at least half the plots < 4; collapse after 5 rounds | `harvest.py:28-30`; `calibrated_oversight.py:35-37` | none; omitted from `body.tex:41` | U | |
| H4 | River harm | quality < 50 of 100 | `run_r4_river_grid.py:28` | "our modelling analogue of MSY, not a regulatory standard" (`literature_ledger.md:123`) | G | post hoc for C1/T1; prospective in R4 |
| H5 | River pre-registered line | quality < 30 | `two_reagent.py:39` | "A step is unsafe if Q' < 30." (C1 protocol:41; no reason; TMDL kept as background only) | U | |
| H6 | Social metrics | efficiency = harvest ÷ (T·rK/4); equality = 1 − Gini; survival | `experiments/oversight/social_metrics.py` | "copying GovSim and Perolat et al. 2017" (`PAPER_PLAN.md:59`; full text checked 8 Oct) | G | post hoc |

### Forest: resource model and programmed agents (L06, L07)

| ID | Setting | Value | Set at | Recorded justification | Verdict | After results |
|---|---|---|---|---|---|---|
| W1 | Growth law | logistic per plot + weather + neighbour loss | `harvest.py:644-651` | logistic as F1 | G | |
| W2 | Spillover form | each neighbour loses 0.22 × (cut − 2.1) | `harvest.py:644-648` | none | U | |
| W3 | Base regrowth | 0.56 (`hard_h1` tier) | `fishery_sim/harvest_benchmarks.py:23` | "`medium_h1`: current default tuned setting - `hard_h1`: lower regeneration, higher weather noise, stronger neighbor externality" (`notes/cycle_logs/harvest_commons_next_run_plan.md:38-40`) | U | hand-tuned tier |
| W4 | Regrowth multiplier | ×0.85 (grid ×1.0) → r = 0.476 | `claude_oversight_common.py:25` | "slow renewal ... chosen because they produced both safe and risky requests" (`design_issues.md:201-202`) | C | pilot |
| W5 | Plot capacity K | 20 | `harvest.py:17` | none | U | |
| W6 | Initial plot health | 12.5 | `harvest_benchmarks.py:22` | none (≈ MSY pre-harvest level 12.38; not recorded) | U | |
| W7 | Weather | N(0, 0.42²) per plot per round | `harvest_benchmarks.py:24` | "The weather is **environmental stochasticity** (May et al. 1978)" (`R1 protocol:36`; concept only) | U | |
| W8 | Externality | 0.22 | `harvest_benchmarks.py:186` | "the scenario labels were parameter packages, not validated irrigation or forestry field models" (`PROJECT_HISTORY.md:70`); code lists Nagendra & Ostrom 2012 for the institutional label only | U | |
| W9 | "Sustainable" cut | 0.35 × 6 = 2.1 | `harvest.py:21` | none; per-plot MSY under the game's own r is 2.38 | U | |
| W10 | Max cut | 6 | `harvest.py:24` | none | U | |
| W11 | Ring | 6 plots, 2 neighbours each | `harvest.py:410-411` | "Harm travels only to neighbours on a ring of six plots" (`S:L6_llm_forest_spillover/protocol.md:11`) | C | |
| W12 | Communication, side payments | off | `claude_oversight_common.py:24` | "Communication, side payments and legacy governors are disabled." (`RR:MATCHED_OVERSIGHT_PILOT_PROTOCOL.md:34`) | C | |
| W13 | Forest strategy ranges | cooperative and adversarial seeds (three-band thresholds, cut fractions) | `fishery_sim/harvest_evolution.py:534-588` | none | U | |

### Forest LLM study L6 (L07)

| ID | Setting | Value | Set at | Recorded justification | Verdict | After results |
|---|---|---|---|---|---|---|
| K1 | Roles | 5 LLM foresters, 1 programmed agent at position 0 | `run_l6_llm_forest.py:33-35` | ring distance 1, 2, 3 from position 0 (L6 protocol:31-34) | C | |
| K2 | Programmed behaviour | requests 0.35; control cuts its allowance; defector cuts 6 every round | `run_l6_llm_forest.py:33, 214` | "In both conditions its request is identical, so the reviewer sees the same thing." (L6 protocol:30) | C | |
| K3 | Horizon | 20 | `run_l6_llm_forest.py:32` | "**20 rounds**, as in the Fishery LLM studies." (L6 protocol:22) | C | |
| K4 | Audits | q = 1/6, nested draws; position 0 never checked | `run_l6_llm_forest.py:32, 224-226` | "It is a fixed manipulation, not a participant." (L6 protocol:47) | C | |
| K5 | Fine | 0 (default 12; gate: < 15% → 0, > 85% → 24) | `run_l6_llm_forest.py:38, 300-302`; `S:L6_llm_forest_spillover/f_choice.json` | pilot over-take rate 4% → F = 0 (L6 Amendment 1) | C | pilot |
| K6 | Visibility | hidden vs neighbours' last cut and allowance | `run_l6_llm_forest.py:102-109` | L6 protocol:40-42 | C | |
| K7 | Sample | 10 populations, gpt-oss-120b only | L6 Amendment 1 | budget | C | |

### River (L08)

| ID | Setting | Value | Set at | Recorded justification | Verdict | After results |
|---|---|---|---|---|---|---|
| V1 | Recovery law | logistic in quality, K = 100 | `two_reagent.py:76-77, 99-100` | logistic as F1 | G | |
| V2 | Damage form | D = κXY + λ(X + Y); 3 X and 3 Y dischargers | `two_reagent.py:12, 35-36, 80-87` | "Harm comes from a **combination** of actions, the case Hu & Wang (2026) call "compositional harm"" (C1 protocol:20-21); Cedergreen 2014 for "synergy" (ledger, 7 Oct) | C | |
| V3 | κ, r, σ, ranges, thresholds | κ = 32, r = 1.0, σ = 3; `wide_low` ranges; thresholds U(35, 75) | `two_reagent.py:50-54, 280-281` | calibration rule: no-reviewer risky share in [30%, 60%], target 45% (C1 Amendment 1) | C | calibration, after an unsaved "coarse look" |
| V4 | λ/κ | 0.05 (λ = 1.6) | `two_reagent.py:57` | "ratio fixed before the search" (no reason) | U | |
| V5 | λ′ (additive control) | 22.92, matching rule | `two_reagent.py:131-135` | C1 Amendment 1 | C | |
| V6 | Start, capacity, per-agent max | Q₀ = 80, Q_max = 100, p_max = 1 | `two_reagent.py:38-41` | none | U | |
| V7 | Stress agents | 1 X + 1 Y (2g); 2 X + 2 Y (4g) | `two_reagent.py:254`; `run_r4_river_grid.py:51-62` | C1 protocol:31-32; R4 protocol:20 | C | |
| V8 | Audits | one per round; random or largest-effect targeted | `run_c1_compositional_harm.py:123`; `two_reagent.py:235-236` | C1 protocol:57-60 | C | |
| V9 | R4 grid and target | regrowth ×1.0/×0.85; reviewer aims at quality ≥ 50 | `run_r4_river_grid.py:25, 28` | "the reviewer then aims at the same line the harm is measured against" (R4 protocol:18) | C | |

### LLM Fishery studies L1–L5 (L09, L10)

| ID | Setting | Value | Set at | Recorded justification | Verdict | After results |
|---|---|---|---|---|---|---|
| M1 | LLM vs programmed | 4 LLM (the stress slots) + 2 honest rule-followers who take their allowance | `run_l2_llm_agents.py:253, 289` | "**The 4 stress agents are LLM actors.**" (`S:L1_llm_actor_pilot/protocol.md:38`) | C | inherits pilot mix4 |
| M2 | Number of families | 3 frozen + Mistral addition; DeepSeek screen in L5 | `run_l2_llm_agents.py:38-39` | "Recent studies that make cross-model claims use **3–4 families**." (L2 protocol:23-24, with a table of six studies) | G | |
| M3 | Which models | gpt-oss-120b, Nemotron 3 Super, Gemma 4 31B, Mistral Large 3; DeepSeek V4.1 Flash | `run_l2_llm_agents.py:38-39`; `run_l4_llm_identification.py:34-36` | "the three that Ollama Cloud's free plan serves on this account" (L2 protocol:26); Mistral Large 3 "based on cost and speed only" (L2 Amendment 4) | C | |
| M4 | Temperature | 0.7 (provider default where not accepted) | `fishery_sim/llm_actor.py:38`; `experiments/oversight/llm_clients.py:20-24, 46` | "**Sampling:** temperature 0.7, with a seed per call" (L1 protocol:81), no reason | U | |
| M5 | Seeds | fixed per call from population, cell, agent, round, phase | `run_l2_llm_agents.py:115, 271-272` | "Ollama Cloud may not honour seeds" (L1 protocol:82-84) | C | |
| M6 | Decision format | JSON; request, then catch after seeing the allowance | `run_l2_llm_agents.py:270-300` | "one structured decision per agent per step (Guzman Piedrahita et al. 2025, arXiv:2506.23276)" (L1 protocol:14-15) | G | |
| M7 | Units and catch range | tonnes; catch 0–6 whatever the request | `llm_actor.py:165-175` | L1 Amendments 2 and 3 (over-taking "impossible by construction" when capped) | C | pilot |
| M8 | History and visibility | own last 3 rounds; never other agents' actions | `llm_actor.py:156` | "the last 3 steps only, fixed in length" (L1 protocol:92), no reason | U | |
| M9 | Permission wording | explicit main wording; silent and paraphrase controls | `run_l2_llm_agents.py:148-190` | "permission to misreport as an explicit factor (Bhandary et al. 2026, arXiv:2606.28456)" (L1 protocol:18-19) | G | explicit made main after L1 showed 0% without it |
| M10 | Comprehension check | 3 questions, option order shuffled | `run_l2_llm_agents.py:193-224` | "neutral rules plus a comprehension check (Fontana et al., arXiv:2406.13605)" (L1 protocol:16) | G | |
| M11 | Pilot gate thresholds | ≥ 95% valid first answers; mean comprehension ≥ 2 of 3 | `run_l2_llm_agents.py:408` | L2 protocol:118-121, no source | U | |
| M12 | Re-prompt and fallback | one re-prompt; fallback to last request / allowance | `run_l2_llm_agents.py:228-246, 274-279, 295-298` | L1 protocol:89-91 | C | |
| M13 | Over-take margin | 0.06 t (1% of the 6 t maximum) | `run_l2_llm_agents.py:60`; `run_l6_llm_forest.py:32` | "Known and accepted" (L2 Amendment 3) | C | |
| M14 | Memory (EM) | allowance − mean recorded excess in tonnes, ≥ 0; no fine | `run_l2_llm_agents.py:59, 283-288, 316` | "This is what the EM prompt already says" (L2 Amendment 5) | C | after a rule fault |
| M15 | Rules content | K = 100, fastest growth at 50, collapse below 10, max 6, q and F stated | `llm_actor.py:101-128` | `supplementary.tex:212-220` | C | |
| M16 | Audit rate, L2/L3 | 1/6, independent per agent and round | `run_l2_llm_agents.py:35, 302` | "Audit rate q = 1/6, independent per agent and step." (L2 protocol:56), inherited | C | |
| M17 | Audit rates, L4 | 1/12, 1/6, 1/3 at equal e = 2, 3 | `run_l4_llm_identification.py:29, 75-77` | expected fine (Becker 1968), severity (Allingham & Sandmo 1972), certainty (Nagin 2013) (L4 protocol:16-18); values chosen to match e | C | e = 2, 3 chosen from the L3 curve |
| M18 | L2 fine grid | 0, 1, 2, 4, 8, 36 t | `run_l2_llm_agents.py:41-44` | proposed in L1 results; 36 so that "the expected fine is at least as large as the largest possible one-round gain" (L1 Amendment 1) | C | after L1 |
| M19 | L3 fine grid | 12, 18, 24, 30 t | `run_l2_llm_agents.py:46-47` | "no tested expected fine lies between 1.33 t ... and 6 t" (L3 protocol:7) | C | after L2 (frozen before L3) |
| M20 | L4 grid and rule | 6–48 t in steps of 6; F* = smallest F with F/6 > ĝ₁ | `run_l4_llm_identification.py:28, 49-55` | L4 protocol:47-50 | C | |
| M21 | Fine form | flat per catch; caught agent keeps its catch | `run_l2_llm_agents.py:305-306` | `supplementary.tex:210`; Stigler 1970 cited only post hoc | C | |
| M22 | Stopping criterion | ≤ 5% over-take rounds at that fine and every larger one | `experiments/oversight/analyze_l3.py`; L3 protocol:44-46 | "an operational criterion rather than an indifference point" (`body.tex:111`) | C | |
| M23 | No-check control N0 | "No one checks the actual catches." | `run_l4_llm_identification.py:45` | answers "the "silent" control still mentioned a check with a 0 t fine" (L4 protocol:10, 36) | C | |
| M24 | L5 qualifying rule | E0 over-take ≥ 20% and CI lower bound > 5% | L5 protocol | L5 protocol | C | |

### Samples and analysis (L11)

| ID | Setting | Value | Set at | Recorded justification | Verdict | After results |
|---|---|---|---|---|---|---|
| N1 | Programmed test populations | 64 per cell | `run_r2_...:66`; `run_s1_...:40`; `run_c1_...:41`; `run_r4_...:36` | inherited from 23 Sep; no power argument found | U | |
| N2 | LLM populations | 10 per cell (L2, L3, L4 threshold, L5, L6) | `run_l2_llm_agents.py:37`; `run_l4_...:38` | cost (L2 protocol:69-77); power computed afterwards (`body.tex:77`) | C | |
| N3 | LLM populations, L4 matched cells | 20 | `run_l4_...:38` | "With n = 20, the half-width of the 95% interval is about 0.04–0.08" (L4 protocol:66) | C | |
| N4 | Bootstrap and seeds | 4,000 paired resamples over populations; fixed seed bases per study | each runner's `SEEDS`; protocols | protocols | C | |
| N5 | Games per population | 1 sampled game | L2 results:156 | stated as a limitation (`body.tex:131`) | C | |

## Counts

| Verdict | Rows | Share |
|---|---:|---:|
| G | 12 | 12% |
| C | 57 | 55% |
| U | 35 | 34% |
| **Total** | **104** | |

(Counted by script over the rows above.)

The 12 G rows are mostly *forms*, not values:
- logistic regrowth in the three games (F1, W1, V1);
- the MSY / half-capacity line, as reviewer target and harm line (R4, H1, H2, H4);
- the social metrics (H6);
- the LLM decision format and comprehension check (M6, M10);
- the number of LLM families and the permission factor (M2, M9).

**No numerical value of any resource or agent parameter is tied to a cited source.** 24 rows are flagged as chosen or changed after seeing outcomes (last column); nearly all of these are dated in the protocols.

By group, the resource and programmed-agent parameters are the weakest: 24 of their 43 rows (F, A, W, V) are U. Forest is the worst: 9 of 13 rows are U. The reviewer, audit and LLM-protocol rows are mostly C, with written reasons.

## Mismatches between paper text and code (fix before submission)

1. `supplementary.tex:22` says audits are random with probability 1/6 per agent and round (T1 and C1: one per round). The Figure 2a memory grid (R2, R5) audits **two** agents per round in Fishery and Forest (1/3 per agent); River (R4) audits one.
2. `body.tex:67` says the search uses 32 training populations. S1–S4 used 8; only R2, R3 and S5 used 32. The pilot deterrence result in `body.tex:96` comes from S3 (8).
3. `body.tex:41` defines Forest harm as mean plot health below 10 as a 5% chance constraint; the code also counts "at least half the plots below 4".
4. `body.tex:45` calls 2.1 the "sustainable" cut; under the game's own regrowth the per-plot MSY is 2.38, and 2.1 does not move when r changes.
5. The L6 Forest LLM run used F = 0, so its "checks" carry no consequence; any Forest LLM claim must say so.
6. The brief's "tolerance 0.06" is the LLM over-take margin (0.06 t). The reviewer's chance-constraint tolerance is 0.05.

## Settings most exposed to a "not grounded" criticism, and what to tie each to

General knowledge, marked **to verify**: none of the anchors below were checked in this audit. Other sub-agents (GovSim, Melting Pot, SocialJax, human CPR experiments) may already have the exact values.

1. **Fishery r = 0.7, K = 100, start 70, max catch 6, N = 6 (F2–F5, F10).** The notes call these "declared pilot choices, not externally calibrated values".
   - *Anchors (to verify):* the Schaefer surplus-production model and published intrinsic growth rates. FishBase's resilience classes put r of about 0.6–1.5 in "high" and 0.2–1 in "medium", so 0.7 sits in a realistic band. GovSim uses a 100-ton lake, 5 agents and a stock that doubles each month up to 100; human CPR lab games (Ostrom, Gardner & Walker 1994) use groups of 8.
   - *Cheap fixes:* state the dimensionless facts the design needs. Start 70 ≈ the MSY pre-harvest level 67.5. Unconstrained demand of 36 t is about 2× MSY, so a dilemma exists. Then show a Schelling diagram, in the style of Leibo et al. 2017 SSDs, to confirm the game is a social dilemma at these values.
2. **The programmed policies (A1–A4, W13): thresholds U(25, 65), ranges 5–20/20–40 and 45–65/75–95, Forest strategy seeds.** These are "design choices, not measured actor-capability levels".
   - *Anchors (to verify):* behavioural types from public-goods and CPR experiments, for example Fischbacher, Gächter & Fehr 2001 (about half conditional cooperators, about a third free riders), could justify "2 or 4 greedy of 6".
   - Melting Pot's practice is to treat scripted or trained "background populations" as test opponents, not as calibrated humans. Reframing the programmed agents that way (focal vs background) uses an accepted convention.
3. **Forest parameters (W2, W3, W5–W10): externality 0.22, "sustainable" 2.1, base r 0.56, weather 0.42, K = 20, start 12.5.** These are hand-tuned tiers; the Nagendra & Ostrom citation in the code is only a label. This is the most exposed game.
   - *Anchors (to verify):* neighbour-dependent regrowth is used in Janssen et al. 2010 (*Science*, spatial foraging) and in Perolat et al. 2017 / Melting Pot `commons_harvest`, where regrowth probability depends on the number of nearby apples.
   - *Cheap fixes:* set the externality kink at the per-plot MSY (r K/4) instead of a fixed 2.1, and report a sweep of the externality (the R2 grid did not vary it).
4. **LLM temperature 0.7 (M4).**
   - *Anchors (to verify):* check what GovSim, SanctSim/Guzman Piedrahita et al. and Willis et al. used. Many LLM-agent game papers use 0 or the provider default.
   - A small temperature check (0 and 1.0 at E0 and E30) would close this.
5. **Chance-constraint tolerance 5% (R6) and the Forest harm "as a 5% chance".**
   - *Anchor (to verify):* the ICES precautionary approach, which requires the probability that spawning-stock biomass falls below B_lim to be at most 5% per year. That is a direct fisheries analogue of a 5% one-step chance constraint.
6. **Harm line at K/2, adopted post hoc (H1, H2, H4).**
   - B_MSY is normally a *target* reference point. *Limit* points are lower: the US "overfished" threshold (MSST) is often ½ B_MSY, and ICES B_lim is well below B_MSY (to verify). The ledger already notes that UNFSA states the MSY limit for fishing mortality, not stock.
   - *Fix:* call K/2 a target, and report a lower limit line (for example K/4) as a robustness check.
7. **Audit rate 1/6 and fine sizes (P1, P5–P9, M16–M20).** Becker gives the logic, not the numbers.
   - *Anchors (to verify):* tax-evasion lab experiments (Alm and co-authors) commonly use audit probabilities of about 0.05–0.3 and penalties of 1.5–3× the evaded amount. AI control uses audit budgets of about 2% (Greenblatt et al. 2024; already in the ledger).
   - L4 already varies q. Expressing F as a multiple of the one-round gain, as the paper does, is the most defensible framing.
8. **LLM horizon 20 and history of 3 (F12, M8).** Twenty rounds was chosen for cost.
   - *Anchors (to verify):* GovSim runs 12 "months". Known end-games produce end-game defection (Selten & Stoecker 1986).
   - *Fix:* report over-taking by round, to show whether the last rounds drive the threshold.
   - The ban on seeing other agents' actions is unexplained; GovSim agents see others' catches (to verify).
9. **Under-report factor 0.5 (A6).** Inherited, never varied except R2's noisy liars (mean one half).
   - *Anchors (to verify):* tax-evasion experiments let the subject choose the declared share. S1's own search over d is the defensible version, so cite it.
10. **River λ/κ = 0.05, Q₀ = 80, line 30, p_max = 1 (V4, V6, H5).** The calibration rule (target about 45% risky steps) is documented and is closer to field practice than anything in Fishery or Forest.
    - *Anchors (to verify):* SSD substrates are generally tuned by hand until a dilemma appears. Cedergreen 2014's definition of synergy (a mixture more than twice as potent as concentration addition predicts) could give κ a measurable meaning.
11. **64 test populations (N1).** No precision argument is recorded. A one-line power statement, like the one L4 gives for 20, would close this.

## What is already well defended

- Every study since R1 has a protocol frozen before its code or outcomes, with dated amendments that state what was seen. This answers most "chosen after results" worries, because the choice is visible.
- The forms of the dynamics (logistic regrowth, MSY at K/2) and the outcome metrics (GovSim, Perolat 2017) follow the fisheries and commons-game conventions.
- The LLM interface follows named methodological sources: one structured decision per step, a comprehension check, permission as a factor, and 3–4 families.
- Robustness grids exist: R2 (6 Fishery and 4 Forest settings, 8 of them never piloted), and R4/R5, which put all three games on one 2×2 grid.
