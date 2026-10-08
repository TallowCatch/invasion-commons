# Completed checks and the next research question

13 September 2026. This supersedes the experiment recommendations in the preceding audit. Original results and the previous manuscript are preserved.

## The direction

**Study when an individual resource-use check misses system failure, and what information a monitor needs to prevent it without unnecessarily restricting useful activity.**

Keep Harvest as the main game. Its local patches, shared effects, and regrowth make this question testable. Fishery remains the earlier single-stock study, useful for isolating quotas and sanctions. Their names describe distinct experimental structures. Neither provides evidence about real fisheries, agriculture, or deployed AI infrastructure.

This can develop into a benchmark paper with a defensible measurement protocol and informative controls. Publication readiness is not established by these pilots. The current results do not justify a general capability-gap scale, a universal failure of local oversight, or a universally best architecture.

## What was fixed

- Kept strategy generation and oversight limits separate. Their assigned ranks are not comparable measurements.
- Stopped counting unused overseer variants and repeated generations as independent runs in the repaired summaries.
- Kept stress settings separate in the overseer-ablation summarizer. The held-out average is now labelled as an average, not a nominal test condition.
- Added separate records for requested and allowed actions, safe-to-unsafe transitions, persistent unsafe states, and actions that fail a local check while the global state is safe.
- Added local filters and separated announcements, communication, and enforced caps using the same policies and seeds.
- Corrected policy restoration: agent positions are reconstructed from selection history, not the final fitness ranking. All 120 restored positions match the archived per-agent records.
- Preserved historical capacity rounding by default; new checks explicitly allow zero targets at zero capacity.
- Replaced the paper's headline gap plot with matched-policy and held-out search plots. Kept the original plots and draft in the archive.

Ten paired legacy episodes matched the old simulator on six numerical outcomes. The complete test suite passed: 61 tests. Software checks support implementation consistency, not scientific validity by themselves.

## What actually ran

All jobs ran as short, resumable background terminal processes on CPU. No new model calls, GPU rental, paid APIs, remote workflows, or large evolutionary sweeps were used.

| Check | Purpose | Evaluation episodes | Recorded run time |
| --- | --- | ---: | ---: |
| Frozen-policy base screen | Separate mechanisms and test meaningful local controls | 3,520 | 101 s |
| Held-out search test | Check whether extra search helps beyond its selection episodes | 2,304 | 47 s |
| Saved policies vs numerical prompt templates | Check whether the broad result requires a model | 1,536 | 25 s |
| Existing slow-regrowth follow-up | Test the same mechanisms where prior logs showed more failures | 3,520 | 103 s |

Total: 10,880 evaluation episodes, plus 576 short candidate-selection evaluations included in the search job's time. These are this machine's measured run times, excluding source-archive scanning, analysis, tests, and manuscript work. Raw output occupies about 65 MB. All four jobs completed.

The follow-up was chosen after seeing sparse failures in the base screen. It uses an existing held-out regime without changing the safety thresholds. It is exploratory, not independent confirmation.

## The results that change the story

**1. A fixed cutoff can miss genuine onset, but the quality of the local rule matters.** Under high stress and slower regrowth, the mean fraction of unsafe steps is:

| Mechanism | Unsafe steps |
| --- | ---: |
| No intervention | 19.70% |
| Fixed local harvest cutoff | 5.33% |
| Local rule using its patch's predicted regrowth | 0.125% |
| Joint-action, zero-weather reference | 8.30% |

Each mean covers ten fixed populations and eight weather seeds per population. Under the base regrowth setting, the fixed filter has zero unsafe steps. Both settings must remain in the paper.

In the slow-regrowth condition, 16 of 80 fixed-filter episodes enter an unsafe state although all allowed actions have met the cutoff throughout. With weather removed, one fixed population still produces this failure; eight seed labels repeat that same deterministic case. Removing both weather and spillovers eliminates those onsets. This is stronger evidence than the old 29-step example, which only showed a resource that was already unsafe.

The local state-based rule has eight predicted-local-pass onset steps across six episodes with weather and spillovers, and none in the removal controls. This is limited evidence about a specific one-step predictor. Local oversight cannot fairly be represented only by a fixed extraction cutoff.

In this same high-stress/slow-regrowth check, the local state rule improves both mean health (11.12 to 11.79) and total population return (980.48 to 1,028.95) relative to no intervention. Uniform caps protect more resource (health 14.74) but yield lower return (908.18). These are descriptive means, not universal trade-off claims. The logged prevented-harvest quantity measures removed nominal requests; some requested harvest might already have been unavailable.

**2. More information did not automatically make the reference safer.** The joint reference protects mean health and failed-patch fraction, while the local state filter tries to keep every predicted local patch above 10. Both ignore future weather. The joint reference can operate close to its global boundary and then become unsafe after a disturbance. These different objectives prevent an information-only comparison. Calling the reference an oracle would be incorrect.

**3. The structured policies already respond to announced caps.** Announcement alone produces no unsafe steps in these screens. Adding enforcement lowers return in the base screen by 11.45 to 39.07 per episode across the four setting/generation cells. That matters because the old interpretation attributed protection too readily to enforcement. It does not establish that arbitrary agents will obey announcements, or that norms have been learned.

**4. Search has some held-out benefit, but not a reliable three-level ordering.** Six candidates outperform one in moderate stress by 8.98 score points at the shorter search horizon (nominal 95% interval 2.51 to 15.44). In high stress, all paired score intervals include zero. Twelve-candidate improvements are uncertain. These intervals use twelve parent contexts, not thousands of supposedly independent episode replicates, and are exploratory without multiple-comparison correction.

**5. The model-policy result needs a narrower interpretation.** At fully exploitative composition, both saved-model policies and direct numerical-template policies collapse in every no-intervention test cell; neither collapses under uniform caps. Numerical outcomes differ, but the broad pattern does not require language-model generation. The current result supports an interface for evaluating saved model-produced policies, not unique model discovery or broad live-agent conclusions. Adding a third model would not resolve that issue.

## Why the next experiment should change

Do not expand the bundled-overseer grid yet. The more useful next comparison is:

**Give local and system-level monitors the same safety target and intervention authority. Vary which current actions they can see, and whether they allow for uncertain regrowth.**

1. Write the common target first: the same global unsafe predicate, or the same stricter per-patch constraint for both. Report how the latter relates to global safety. Do not switch targets between monitors.
2. Define a common intervention objective, such as minimizing prevented extraction while satisfying that target. Give both mechanisms the same clipping authority initially. Capacity limits can follow after the information comparison works.
3. Compare local observations with joint-request access. For the local monitor, specify the missing-neighbour assumption explicitly, then test a conservative bound as a control. Do not silently treat missing actions as zero.
4. Compare zero-weather predictions with a stated uncertainty allowance. Separate states where even zero extraction cannot meet the requirement from failures of an otherwise feasible intervention. The current joint reference shows why this is necessary.
5. Use fresh populations, retain both ordinary and slow-regrowth settings, and declare the paired contrasts and run-level uncertainty before inspecting results. Start small; select a larger sample using the pilot variation and the effect size worth detecting.

This next design needs a short protocol before execution. It is a change in scientific comparison, not another unexplained parameter sweep. Expected runtime must be measured after implementing the common decision rule; the present fast filters do not justify a time estimate for an unknown optimizer.

## Literature and the claim boundary

| Source | Why it is relevant | What it does not establish for us |
| --- | --- | --- |
| [Leibo et al., 2017](https://arxiv.org/abs/1702.03037) | Sequential social dilemmas concern policies and repeated interaction. | Originality of the commons setting or our numeric parameters. |
| [Elsayed-Aly et al., 2021](https://arxiv.org/abs/2101.11196) | Multi-agent shielding makes joint-action safety and intervention a real comparison family. | A formal safety guarantee for our filters. |
| [Engels et al., 2025](https://arxiv.org/abs/2504.18530) | Oversight capability can be measured through defined task performance. | Validity of subtracting our preset ranks. |
| [Piatti et al., 2024, GovSim](https://arxiv.org/abs/2404.16698) | Language-model societies, resource use, and communication already have direct precedents. | Novelty from adding model-generated harvest policies alone. |

The paper also retains Melting Pot, SocialJax, Ostrom, and strategy-generation references. The benchmark's cutoff values, new filters, and experiment schedule are our stated design choices, not literature-backed constants. The next study should compare precisely against shielding and decentralized safety work before any novelty claim is strengthened.

## Files and reproduction

Current paper: `paper/paper_v5_scalable_oversight_commons/main.tex` and `main.pdf`. Previous draft: `archive/pre_validation_20260913/` inside that folder. Main new output files are under `notes/research_review/completed_checks/`; the slow-regrowth results have their own subfolder. Raw resumable blocks and manifests are under `results/runs/validation_v1/`.

New experiment entry points:

```bash
python -m experiments.validate_harvest_mechanisms prepare --output-dir results/runs/validation_fresh/sources
python -m experiments.validate_harvest_mechanisms mechanisms --sources results/runs/validation_fresh/sources --output-dir results/runs/validation_fresh/mechanisms
python -m experiments.validate_harvest_mechanisms calibration --output-dir results/runs/validation_fresh/calibration
python -m experiments.validate_harvest_mechanisms mechanisms --evaluation-regime slow_regen --sources results/runs/validation_fresh/sources --output-dir results/runs/validation_fresh/slow_regrowth
python -m experiments.check_harvest_policy_sources --output-dir results/runs/validation_fresh/policy_sources --summary-dir results/runs/validation_fresh/summary
python -m experiments.analyze_harvest_validation --root results/runs/validation_fresh --output results/runs/validation_fresh/summary
```

These commands are documented, not automatically scheduled. CPU dependencies are recorded in `requirements-validation.txt`. Preparing original populations requires the Stage A strategy and agent histories. The source control requires the two saved v3 banks. These files are locally available but a complete externally accessible release is still required. Calibration does not require those archived data inputs.

To rebuild the completed matched/search analysis without rerunning simulations:

```bash
python -m experiments.analyze_harvest_validation
python -m pytest -q tests
```

Manifests reject resumes after relevant code changes. The calibration and base-mechanism runners at launch are archived with their completed outputs: later edits added a numerical filter buffer and the regime CLI option. Do not overwrite those manifests to force a resume. Use a fresh directory for current-code reproduction. Analysis of completed blocks uses recorded configurations and does not rerun policies.

Main implementation changes: `fishery_sim/harvest.py`; the extractor; the overseer and stress summarizers; `validate_harvest_mechanisms.py`; `analyze_harvest_validation.py`; `check_harvest_policy_sources.py`; and `tests/test_harvest_validation.py`. Documentation and paper assets were updated separately. Unrelated dirty files were not staged or committed.

## What to tell Yali

> I have checked whether the result depends on how I defined a local rule. A fixed harvest limit can miss system failure, but a rule that uses the local resource state prevents most of it. I also found that some protection comes from agents responding to announced limits before enforcement. I am now separating those effects and dropping the single capability-gap scale. The next question is what information a monitor needs when agents affect one another, comparing monitors with the same safety target and intervention limits. Harvest remains the controlled test environment, with the earlier Fishery and saved-model experiments as supporting studies.
