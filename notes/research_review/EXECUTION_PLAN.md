# Execution plan: validate the mechanism before expanding the benchmark

## Goal and boundary

Produce a defensible first PhD benchmark contribution that measures the relationship between local compliance and global resource risk, isolates relevant oversight mechanisms, and can be reproduced outside this working directory.

The existing project is the starting point. No environment rebuild, live LLM rollout, new model purchase, or full new grid is justified immediately. The earlier "full overseer ablation next" recommendation needs one prerequisite: fix the measurement and comparison design so that a larger run will answer an identifiable question.

## Completed during this review

- Read the latest paper, supporting notes, source implementation, and current evidence; checked relevant primary literature.
- Recomputed descriptive architecture means, equal-gap contrasts, inactive-overseer invariance, threshold invariance, all four local/global categories, selected-case onset/persistence, and prompt-anchor similarity.
- Added `experiments/audit_research_evidence.py`, its tests, and hashed evidence outputs.
- Created a factor-separated figure, exported as PDF/SVG/300-dpi PNG, without pooling settings into a scalar gap.
- Ran the existing paper-input and threshold-completeness checks into a new audit directory rather than overwriting previous reports.
- Ran 43 relevant tests successfully, including six new audit tests. No expensive experiment was started.

## What can be recovered, and what needs simulation

| Check | Existing evidence sufficient? | Tool/code status | Cost and next action |
| --- | --- | --- | --- |
| Separate actor and overseer effects descriptively | Yes, 360 Stage A rows. | New `audit_research_evidence.py`; factor figure completed. | Seconds. Do not interpret this as causal identification. |
| None/local invariance and repeated threshold trajectories | Yes. | New audit completed; `audit_threshold_sweep_completeness.py` also exists. | Seconds. Use active-condition masks and deduplicate inactive settings for inference. |
| Four local/global categories | Yes, for the existing averaged estimand. | New audit completed. | Seconds. These are averages of episode fractions; pooled step probabilities need original numerators/denominators. |
| Paired architecture effects with uncertainty | Yes for total architecture-plus-selection outcomes. | New script supplies paired differences; existing `analyze_harvest_oversight_stageA.py` is a starting point. Run-cluster uncertainty and inactive-factor handling need tightening. | Seconds to a few minutes. Five independent evolutionary runs per cell limit precision. |
| Held-out regime and generation trajectories | Yes, generation-history CSV has the needed columns. | `analyze_stagea_stress_regimes.py` exists but its "nominal" label and aggregation need correction. | Seconds to minutes. Collapse generations within each run before estimating run-level uncertainty. |
| Selected-case onset versus persistence | Yes, for the supplied trace. | New audit completed. | Seconds. All 29 highlighted LPGF steps are persistent unsafe states. |
| Onset/conditional error over all episodes | Not from run means alone. Some primitives may be in per-agent archives; pre-state failed fraction and episode boundaries must be verified. | Need streaming trace inventory/reconstruction and provenance checks; existing case extractor is a starting point. | Minutes for a streaming archive audit; otherwise a small frozen-policy replay. Do not assume a summary CSV encodes transitions. |
| Total return, survival and normalized burden | Some quantities are logged at episode/agent level but dropped by top-level summaries. | Extend summarization; inspect archived episode fields first. | Minutes if recoverable. Never reconstruct mean episode totals by multiplying two unrelated means. |
| Threshold relabelling | Completed for the current matrix. | Existing replay/merge/analyze scripts; new invariance check. | No new full sweep. Future step primitives permit offline recalculation. |
| Held-out generator calibration | Existing entrant tables are not a common-parent/common-opponent experiment. | New calibration harness needed around existing generators and `evaluate_harvest_population`. | Small CPU experiment after protocol is frozen. |
| Communication versus targeting versus enforcement | No. Existing packages change several things together. | New frozen-population intervention harness needed. Existing population evaluator can be reused. | Small paired CPU experiment first; no evolutionary grid initially. |
| Individual overseer-limit effects across settings | Only a reduced one-slice check exists. | Runner/analyzer/auditor exist; scenario grouping and cost/capacity semantics need correction. | Frozen-policy screen first. Full evolution is a later gated run. |
| Direct anchor sampler versus model-generated policies | Anchor similarity is recoverable now; behavioural counterfactual is not. | Anchor function and governance-map evaluator exist. Add source manifest/control-bank export and paired comparisons. | No model inference needed. CPU evaluation only. |
| Clean-clone reproduction | Current file-presence checks are insufficient. | Existing manifest/check scripts help; release packaging still needed. | Developer work plus a small rerun, no GPU. |

## Phase 1: lock definitions and repair the analysis

Do this before collecting more substantive results.

1. Freeze the audited input files and hashes. Keep the current manuscript as an archival draft until its claims are revised deliberately.
2. Make an implementation-to-formulation table: user observation/action/payoff; governor observation/state/signal/intervention; researcher-controlled parameters; logged diagnostic. State the timing of cap announcements and enforcement.
3. Remove the scalar gap from primary inference. Use actor budget, recall/dropout, delay, capacity count, and cost as separate fields. For absent mechanisms use an inactive marker rather than an implied overseer ability.
4. Correct ablation grouping to retain scenario, independent run identity, and relevant factors. Correct aggregate-test versus held-out-regime labels. Define final-generation and across-generation analyses separately.
5. Preserve historical cost behaviour but name it accurately. Decide and test capacity endpoints before any revised run; version the protocol if behaviour changes. Explicitly distinguish planned targets, executed targets, and realised clipping.
6. Report four diagnostic outcomes, approval coverage, duration and return units. Make rank ordering explicit or move winner counts to the appendix.

**Tests required:** two scenarios never pool accidentally; duplicated inactive settings do not tighten intervals; generation rows are not independent samples; zero capacity really means the declared behaviour; deterministic repeated inputs replay identically with the same spatial ordering; no-intervention conditions are invariant to unused governor parameters.

**Stop/go condition:** one page of equations and timing can be traced to code, and every plotted axis corresponds to an active and specified treatment. More data should not precede this.

## Phase 2: the smallest informative mechanism experiment

**Question:** Does a local request check miss ongoing risk, miss the onset of failure from a safe state, or both? Which part of the intervention package changes that behaviour?

Freeze identical policies and initial/environment seeds across mechanisms. First use existing no-governor policy snapshots, with original agent order preserved, as a clearly exploratory screen. These snapshots retain their source-run identity; sampling several populations from one history does not create new independent evolutionary replicates.

Use both stress settings and low/high generation sources. A screen with five source-run populations per source, eight environment seeds, and nine mechanism variants is 1,440 evaluation episodes:

    2 settings x 2 strategy sources x 5 populations x 8 seeds x 9 mechanisms

The nine variants are:

- No messages and no intervention.
- Messages/reciprocity only.
- Uniform cap enforcement, messages off.
- Uniform cap enforcement, messages on.
- Neighbourhood cap enforcement, messages off.
- Neighbourhood cap enforcement, messages on.
- An actual local request filter enforcing the specified local predicate.
- A state-aware local filter using the permitted local observations, including neighbour mean, and declared local dynamics, with no privileged full-system state. Specify and validate its local safety objective before comparison.
- A cap announcement with no enforced clipping, using otherwise matched policies and information.

The four cap-enforcement variants form a matched communication-by-targeting comparison. Keep credits fixed or explicitly disabled within that factorial comparison; do not allow "messages on" to silently introduce a second transfer treatment. The local-filter and signal-only cases answer different questions and should not be interpreted as equal-authority architectures without qualification. The state-aware filter prevents the comparison from relying exclusively on a weak fixed cutoff that ignores resource state.

For all episodes, log requested and realised actions, pre/post resource states, pre/post global safety, approval, intended/executed targets, costs, episode duration, source policies, and RNG identifiers. Archive initial states and the complete configuration. A deterministic trace-order check should precede aggregate analysis.

Evaluate approved failure onset, persistent mismatch, conditional unsafe occupancy given approval, approval coverage, resource health, total return, and intervention burden. Include a no-weather/no-spillover control on a preselected subset to distinguish stochastic shocks and interaction effects. Do not search thresholds until a desired positive result appears.

Add a model-aware one-step joint-safety controller as a separately labelled privileged reference if feasible. It can use known transition dynamics to screen or uniformly rescale joint extraction, with an explicit safety objective and stochastic assumption. It is not a formal shield or guaranteed-safe baseline unless those guarantees are actually proved. It should not be described as a reproduction of an existing shielding algorithm without matching its assumptions.

**Decision gate:** If the local filter still produces approved safe-to-unsafe transitions under controlled conditions, investigate why the predicate is insufficient. If it prevents onset but fails to indicate recovery, write a narrower diagnostic paper. If weather or previous violations explain the effect, report that and abandon the stronger all-approved-joint-actions claim. A negative mechanism result can improve the benchmark; it must not be repaired by post-hoc threshold selection.

This screen is exploratory. A confirmatory protocol should use fresh independent population-generation seeds, prespecified outcomes, and a sample-size/precision target chosen from the screen. Do not turn five archived source runs into a claim of broad statistical validation.

## Phase 3: independent actor-generation validation

**Question:** Does more generation effort produce higher held-out objective performance under common conditions?

Use common parent strategies and opponent populations, fixed downstream evaluation conditions, and held-out seeds never used to select candidates. Generate nested candidate sets so comparisons can distinguish additional search from arbitrary candidate changes. Cross candidate count and selection horizon rather than increasing both together.

An initial design is two stress settings, twelve independently generated parent/opponent populations, candidate counts 1/6/12, and internal horizons 30/60. Evaluate each selected policy on sixteen shared held-out environment seeds. This gives 2,304 selected-policy test episodes plus short candidate-selection evaluations. The current search class clamps candidate count to at least two; implement the one-candidate control explicitly rather than silently requesting K=1 from that class.

Report held-out entrant return with the same collapse penalty used for selection, raw total payoff, other-agent welfare, resource outcomes, and selection generalization. Pair comparisons within parent/opponent context. Rotate the entrant's ring position or hold it fixed and state the restriction; candidate placement and neighbours must not vary accidentally.

**Decision gate:** A reliable performance increase supports a within-task generator-quality ordering. It does not establish that the actor is stronger than the governor on a shared scale. If the performance ordering is weak or reverses on held-out seeds, retain the descriptive budget labels. Do not substitute increased damage for improved task capability after observing results.

## Phase 4: corrected overseer-limit ablation

Proceed after Phases 1-3 define the outcomes and matched mechanism comparison. Separate information available to the governor from intervention execution; the existing recall parameter is not an observation-quality manipulation.

Start on frozen actors, isolate delay/dropout/actual capacity at a fixed cost, and evaluate cost separately as a price. Extend only the interactions supported by a clear mechanism, such as delay by capacity. If delay dominates a slice, investigate action timing and cap-response behaviour before declaring a general finding about oversight.

The earlier full evolutionary proposal contains 420 run jobs. Of these, 180 bundled strong/limited/weak jobs potentially overlap Stage A; 240 isolated-limit jobs are new. Reuse requires exact matching of code, configuration, seed schedule, population order, aggregation, and input provenance. Do not assume identical labels imply reusable runs.

Use remote CPU shards for the later evolution stage, with an explicit manifest, bounded job duration, resumable outputs, unique keys, and a completeness merge. A workflow succeeding is not evidence of scientific validity. The full command in the old ablation plan should not be launched unchanged because its summary currently pools scenarios.

## Phase 5: optional policy-source control and release

Before adding another language model, export the existing numerical anchors as a no-LLM policy bank. Match attitude, nonce, population composition, and evaluation seeds to the current banks. Compare behavioural outputs, not only exact parameter equality. This can determine whether model generation adds useful variation beyond an already effective designed sampler.

Keep this secondary to the first paper's mechanism question. If the direct sampler reproduces the main pattern, present the LLM component as a compatible strategy-source demonstration rather than a separate autonomy/capability result. Live-agent memory, tools, and training would be a distinct project.

Prepare a release with an updated entry-point README, pinned dependencies, data/config/code checksums, compact public result tables, a trace example, archive download instructions, and one small command sequence rebuilding a figure from a clean checkout. Keep raw archives outside git when appropriate, but make them identifiable and retrievable. Include a data-generation manifest, not only an artifact filename.

## Runtime and compute assumptions

A timing-only check during this review evaluated sixteen frozen episodes in about 0.37 seconds on the current Python environment, approximately 0.023 seconds per episode. This is a rough calibration, not a throughput guarantee. Episode length, search work, trace logging, filesystem load, and hardware differ. No scientific result was inferred from that timing sample.

| Work | Planning estimate | Basis / caveat |
| --- | --- | --- |
| Existing-data audit and separated-factor figure | Seconds to a few minutes. | Small run/threshold CSVs; completed locally. |
| Streaming larger archives / reconstructing episode summaries | Approximately 5-30 minutes to inspect and process initially. | Unmeasured I/O estimate; schema/provenance work may require additional development time. |
| 1,440-episode frozen mechanism screen | Approximately 2-10 CPU minutes including logging. | Raw timing extrapolates to about 33 seconds; buffer allows setup/trace overhead. Controller lookahead is additional. |
| 2,304-episode generator calibration plus short searches | Approximately 5-20 CPU minutes initially. | Raw episode cost is small; pairing, candidate search, and validation add overhead. Measure the actual harness first. |
| Direct-sampler LLM control, e.g. 1,024 evaluation episodes | Approximately 2-10 CPU minutes. | No inference or model download. Population configuration must match the comparison being made. |
| 240 new full evolutionary ablation jobs | Approximately 8-24 CPU hours as an initial budget envelope, not a measured forecast. | About 576,000 train/held-out episode evaluations before candidate-search and logging overhead. Raw frozen-episode extrapolation alone is about 3.7 CPU hours. |
| All 420 proposed evolutionary jobs | Approximately 16-40 CPU hours as an initial envelope. | About 1,008,000 evaluations plus candidate search/logging. Reuse valid existing jobs where possible. |

For any full evolutionary run, time one representative job with the final logging and search settings before estimating quota or elapsed time. Remote wall time depends on actual concurrency, queueing, and the slowest shard. No paid GPU or API is needed for the recommended work. Existing GitHub Actions tooling is usable infrastructure, but account allowance/storage availability must be checked at dispatch rather than assumed here.

## Scripts to keep, fix, and add

| Status | Script / component | Action |
| --- | --- | --- |
| Added in this pass | `experiments/audit_research_evidence.py` | Rebuild audit CSVs, hashes, and factor-separated figure from existing evidence. |
| Added in this pass | `tests/test_research_evidence_audit.py` | Test joint outcomes, onset/persistence, invariance, and paired differences. |
| Keep | `audit_threshold_sweep_completeness.py`, replay shards/merge, `check_paper_inputs.py` | Preserve recovered evidence; strengthen release checks rather than rerun the grid. |
| Fix before use | `run_overseer_limit_ablation.py`, `analyze_overseer_limit_ablation.py` | Retain scenario and active factors through analysis; test cost/capacity semantics. |
| Fix before use | `analyze_stagea_stress_regimes.py` | Correct aggregate-test label and run-level uncertainty/estimand handling. |
| Verify before use | `extract_harvest_oversight_case.py` | Match original agent order and source episode provenance. |
| Extend | `analyze_harvest_oversight_stageA.py`, `plot_scalable_oversight_paper_v5.py` | Separate factors; report paired effects, units, coverage, and cluster-aware uncertainty. |
| Write | Frozen-policy mechanism harness and trace analyzer. | Named protocol, matched mechanisms, transition diagnostics and replay checks. |
| Write | Common-opponent generator calibration harness. | Independent selection/evaluation, crossed search factors, correct K=1 control. |
| Write/extend | Direct-anchor bank export and paired source comparator. | Reuse existing anchor function and `run_harvest_llm_governance_map.py`; no model calls. |

New harness descriptions are implementation tasks, not currently available commands.

## Exact first command

From the repository root, the first safe command is:

```bash
python -m experiments.audit_research_evidence
```

It has already been run in this pass. It produces `notes/research_review/evidence/` and does not start simulations or model inference. Verification:

```bash
python -m pytest -q tests/test_research_evidence_audit.py tests/test_harvest_institutional_commons.py tests/test_harvest_invasion.py tests/test_study_extensions.py
```

The next implementation task should be Phase 1 fixes plus the frozen-policy trace/provenance harness. There is deliberately no full-sweep launch command recommended until those checks pass.

## When this becomes a defensible first-paper package

The paper should have one explicit question, an implementation-faithful game/mechanism formulation, independent generator-quality evidence if capability remains in the headline, a matched mechanism comparison, diagnostic coverage/onset/persistence analysis, clearly scoped uncertainty, and a clean reproducibility path. It should compare its contribution directly with existing resource-use societies and multi-agent safety methods.

If approved onset is absent, the paper can still study the limits of local compliance as a risk/recovery indicator, but must not claim to demonstrate that safe-looking joint actions caused collapse. If model anchors explain the policy-bank result, report the model interface as supporting infrastructure. If hybrid is not best after matching mechanisms, keep that result. None of these outcomes requires abandoning the project; each determines the claim the evidence can support.
