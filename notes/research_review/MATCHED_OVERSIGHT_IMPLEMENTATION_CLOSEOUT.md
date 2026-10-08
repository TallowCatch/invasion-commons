# Matched Oversight: Implementation and Pilot Closeout

22 September 2026. Development evidence, not a replacement publication result.

## What Was Completed

The immediate question is now precise: **what does access to the joint action
add when local and joint monitors must protect the same resource condition and
have the same power to change actions?** This repairs a confound in comparing
a local request cutoff with a global resource-health rule.

| Requested work | Status and boundary |
| --- | --- |
| Reusable scientific workflow | Four skills installed or authored and validated; local MLflow tracking works. |
| Matched Harvest comparison | Implemented, tested and piloted with paired policies, states and weather. Local reports are pooled, not independent local governance. |
| Direct oversight decisions | Identical frozen proposals judged by every method; risky approvals, safe rejections, uncertain labels and checking effort recorded. |
| Fishery transfer | Native single-stock transitions verified; pilot completed, but harmful-proposal coverage failed. |
| Cleanup transfer | Native SocialJax adapter, evidence boundaries, rejection authority and replay verified. Mechanics smoke passed. Monitors and validated observation-limited policies are still missing. |
| Further game families | Deferred. No RICE-N, new LLM model, training or large sweep. |

The prior paper and historical results were not rewritten to imply these new
comparisons were already validated. Existing unrelated working-tree edits were
left intact; nothing was staged or committed.

## What the Pilot Actually Shows

Completed **512 Harvest episodes and 16 Fishery episodes**, using four newly
sampled policy contexts. Eight weather seeds are nested within each Harvest
context. The simulation/reference-label invocation took **21.32 seconds** and
produced 5.66 MB of compressed episode blocks, excluding analysis and tracking
copies. There was no policy training or strategy search in this run.

Harvest monitors share the global safety predicate and uniform request-reduction
menu. The local nominal model omits neighbouring damage; the conservative local
model bounds unknown neighbours. Joint models use actual joint requests.
Weather-aware variants use a margin calculated from the simulator's known noise
distribution. The results therefore concern information and model assumptions,
not a validated scalar actor-overseer capability gap.

| Harvest method | Unsafe steps: base | Unsafe steps: slow regrowth |
| --- | ---: | ---: |
| No intervention | 76.42% | 96.88% |
| Local reports, nominal model | 15.12% | 13.95% |
| Joint requests, nominal model | 6.41% | 8.71% |
| Local reports, weather margin | 0.00% | 0.00% |
| Conservative local reports, bounds and weather | 0.00% | 0.00% |
| Joint requests, weather margin | 0.00% | 0.00% |

This selected table illustrates the mechanism checks; the machine-generated
report includes all eight methods, returns, resource health and checking costs.
Zero observed unsafe steps is not a zero-risk guarantee. Occupancy is measured
over observed steps; early-terminated and full-horizon episodes have different
denominators. Collapse, episode length and onset counts must accompany this
measure in any publication analysis.

On the **same initially safe base-Harvest snapshots**, each method judged 272
proposals: 21 reference-risky, 243 reference-safe and eight unresolved. Nominal
local reports accepted 10 of the 21 risky proposals; nominal joint checks
accepted seven. With a weather margin, joint checks accepted none and rejected
19 of 243 safe proposals, versus 136 safe rejections for conservative local
reports with bounds and weather. This supports investigating the cost of
worst-case neighbour bounds.

However, local reports with just the weather margin also accepted none of these
risky proposals and rejected only 11 safe proposals. They omit neighbour damage,
so this is empirical performance on the sampled cases, not a general bound.
**The pilot does not establish that joint checking is always necessary or best.**
These proposal counts contain dependent states and scaled variants; they are
not 272 independent replications. Four policy contexts support exploratory
comparisons only, with wide context-level uncertainty.

The initially safe slow-regrowth snapshots contained **no risky proposals**.
The 96 Fishery proposals per method were also all safe. Their risky-approval
rates are unavailable, not zero. Fishery's engineering transfer works; its
evaluation corpus is not yet sufficient to test detection of harmful proposals.

## Cleanup Mechanics Check

The separate 300-step, seed-17 probe uses the pinned original SocialJax game.
The no-action control never recovered resource productivity. A privileged
script with four cleaners and three harvesters reached the declared target at
step 19, remained inside it throughout the final 100 steps, and harvested 240
apples during that tail while continuing to clean.

The probe took **10.04 wall seconds / 11.13 CPU seconds**, including compilation.
It passes the declared mechanics gate, but the script sees the full simulator
state and is not an oversight monitor. Cleanup still needs observation-limited
productive policies, damaging proposals, and matched monitor implementations.
Suppressing actions cannot force missing maintenance; that is the specific
additional oversight problem this game can test.

## Verification and Artifacts

- Repository suite: **98 passed, 10 skipped** in 15.62 seconds. The skips are
  optional native Cleanup tests in the ordinary project environment.
- Isolated Cleanup suite: **27 passed**, including those native tests, in 6.69
  seconds. One upstream JAX dtype FutureWarning remains; the runtime is pinned.
- Completed-run resume now preserves raw records and timing, verifies checksums,
  and refuses corrupted evidence. All **528 original episode hashes** reverified;
  no completed pilot was rerun. Original source snapshots remain beside results.
- Main runner: `experiments/run_matched_oversight.py`; offline analysis:
  `experiments/analyze_matched_oversight.py`.
- Completed pilot: `results/runs/matched_oversight_v1_pilot/`. Read
  `analysis/experiment.md`, `analysis/outcomes.csv`, `analysis/decision_quality.csv`
  and `analysis/paired_contrasts.csv`; figures have PDF, SVG and PNG exports.
- Cleanup: `results/runs/cleanup_oversight_smoke_20260922/report.json`, native
  snapshots, and local source copies. Full contract: `CLEANUP_ADAPTER.md`.
- Local MLflow: `results/scientific_tracking/mlflow.db`, experiment
  `commons-matched-oversight`, run `1836ba58328a44baae481fc1155f834d`.
- Skills/provenance: `SCIENTIFIC_SKILLS_INSTALL.md`. Experiment design:
  `MATCHED_OVERSIGHT_PILOT_PROTOCOL.md`; literature rationale remains in
  `literature_20260921/`.

Artifacts currently live in ignored local result directories. They are not a
public benchmark release. A publication release needs an explicit artifact
manifest/archive, dependency installation check and reproducible entrypoints.

## Decision: Stop This Pilot, Repair Coverage Next

1. **Build and freeze a balanced decision-case suite before a larger run.**
   Include naturally sampled cases and separately labelled boundary challenges,
   with initially safe and recovery states, safe and risky proposals, and known
   counts. For Fishery, test aggregate demand near the stock-safety boundary.
   Validate against native transitions; never tune until one monitor wins.
2. **Retain strong local baselines.** The useful comparison is actual joint
   information versus defensible local bounds at the same safety target and
   authority. Do not rely only on the optimistic local model. Report safety,
   useful activity and total checking/communication effort together.
3. **Finish Cleanup's policy and information contracts before benchmarking it.**
   Distinguish keeping a productive system safe from restoring an already
   polluted one. Validate policies without hidden-state access and specify what
   evidence a shared monitor may receive. No comparative Cleanup claim yet.
4. **Then test constrained oversight.** Independently vary evidence access or
   checking resources against fixed proposal populations. The current candidate
   budget only limits repair search. Actor search resources and overseer
   resources must remain separate, with measured outcomes rather than rank
   subtraction used as a capability measure.

For the supervisor: "I repaired the comparison so the monitors share a safety
target and intervention options. The pilot now records which proposals they
accept or reject, not just resource outcomes. It shows that uncertainty and
conservative assumptions matter, while strong local checks remain competitive.
Fishery needs better harmful-proposal coverage; Cleanup's native integration
works, but its policies and monitors are not yet ready for comparative claims."

## Follow-Up, 23 September 2026

The fixed decision-case coverage repair has now been completed. It supplied
safe and risky constructed cases in Fishery and both Harvest regimes, while
preserving the above pilot as historical evidence. See
`DECISION_CASE_COVERAGE_CLOSEOUT.md` for the new counts, strong-local comparison,
artifact paths and next research gate. The statement above about Fishery
coverage refers only to the original naturally sampled pilot proposals.
