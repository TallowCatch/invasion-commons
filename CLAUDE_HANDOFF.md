# Project handoff for Claude

Read this file first, then inspect the linked source files before editing or
making scientific claims. This is a research project, not a request to make the
paper sound impressive. Preserve uncommitted user files. Do not launch a large
experiment without an explicit question, baseline, resource estimate, and
small pilot.

## Repository and current state

- GitHub: https://github.com/TallowCatch/invasion-commons
- Working branch: `codex/harvest-oversight-gap-stagea`
- Draft PR: https://github.com/TallowCatch/invasion-commons/pull/1
- Current paper: `paper/paper_v5_scalable_oversight_commons/main.tex` and its
  compiled `main.pdf` on this branch. The `main` branch is older; use the
  working branch for the current research.
- Main evidence/claim audit: `notes/research_review/PAPER_CLOSEOUT_20260924.md`.
- Full chronological explanation: `PROJECT_HISTORY.md`.
- Source-data and reproduction instructions:
  `paper/paper_v5_scalable_oversight_commons/data/README.md`.
- The PR is draft because the paper is supervisor-ready, not submission-ready.
  It includes a long research history, not one isolated feature. Do not
  interpret the older drafts as current findings.

## Research question in plain language

Several agents repeatedly ask to use a shared, renewable resource. A reviewer
may inspect requests and reduce them. What information must the reviewer see
or receive to reject harmful *combined* requests without unnecessarily
restricting safe work? How does the answer change when checking is limited?

This is relevant to scalable oversight, but the strongest completed result
isolates information and checking effort, **not** a validated gap between a
strong AI actor and a weak AI overseer. The old subtraction of actor and
overseer preset ranks is not a measured common capability scale.

## How the project reached this question

1. Fishery began as a simple repeated game with one renewable stock. It made
   aggregate extraction, collapse, quotas, sanctions, monitoring, and strategy
   replacement measurable.
2. Harvest added local resource patches, neighbour effects, stochastic renewal,
   communication/credit, and local versus system-level interventions. These are
   game structures, not literal claims about agriculture or real-world policy.
3. Early experiments compared governance packages under mutation, strategy
   search, institutional frictions, and some PPO checks. These packages often
   changed several factors together, so their winner maps are exploratory.
4. Stage A made local-action and global-resource safety explicit. Its 72
   condition rows and 18 ranking cells found hybrid first in 13 cells, local
   first in 4, and global signal first in 1. A full threshold *relabeling* grid
   tested nearby safety definitions on saved trajectories; it did not create
   9,000 new independent runs.
5. Two small local open-weight LLMs generated **offline structured Harvest
   strategy artifacts**. The saved Qwen2.5 3B and Llama 3.2 3B banks showed that
   the interface can accept model-produced policies. Numeric-template controls
   reproduced broad patterns; this is not a live LLM-agent result.
6. Reviewer comparisons then held the one-step safety target and intervention
   menu fixed while changing how many proposed actions a reviewer inspects.
   A conservative local bound, an optimistic local rule, and a joint reviewer
   expose missed harmful requests versus unnecessary restriction. A later
   coupled-local calculation, using the same inspected contributions, matched
   joint decisions on saved proposals. That is an important counterexample to
   any claim that local review is inherently inferior.
7. A third Clean Up game was attempted but did not pass a predeclared policy
   admission gate. It has no accepted paper result. Different scenario names
   inside Harvest are not independent game families.

## Main confirmed evidence and caveats

The fresh matched-reviewer comparison has 64 independent population contexts
per game and 1,920 episodes total. The independent unit is the context, not
each timestep or stochastic continuation.

| Game, full inspection | Resolved-safe original requests | Resolved-risky requests | Joint reviewer | Coarse bounded local | Optimistic local |
| --- | ---: | ---: | --- | --- | --- |
| Fishery | 694 | 63 | 0 safe restricted, 0 risky approved | 100 safe restricted, 0 risky approved | 0 safe restricted, 63 risky approved |
| Harvest | 3,687 | 1,368 | 2,521 safe restricted, 0 risky approved | 3,687 safe restricted, 0 risky approved | 2,407 safe restricted, 0 risky approved |

Harvest also has 923 unresolved safety labels; they must not be silently
counted as safe. The reference labels are one-step labels, with Fishery's
calculable transition and Harvest's sampled uncertain renewal. The reviewers
share an intervention menu but not necessarily the same *total* computation
or communication cost; the manuscript distinguishes those costs.

At full inspection, Fishery joint review has **lower** mean 80-step return
(551.5 vs 784.7) and lower mean stock (16.88 vs 25.58) than bounded local
review, even though it interferes less with immediately safe requests. This
demonstrates why immediate decision quality and long-run resource outcomes
must be reported separately. These numbers are specific to selected settings.

The post-hoc coupled-local replay has 11,249 saved proposals and 33,747 paired
budget decisions, with zero disagreements against joint review. It reuses the
same cohort, so it is **not** independent confirmation. The paper should not
claim that centralized control is necessary, that hybrid always wins, that
local-pass/global-fail alone is novel, or that the benchmark is already a
general five-game suite.

## Supervisor feedback and intellectual direction

Yali asked for a direct research question, a stakeholder/game formulation,
clear decisions and safety outcomes, literature positioning, and a credible
link to scalable oversight. Edward questioned external centralized control
and emphasized that agents may constrain each other through shared norms and
local information. The current paper therefore compares **information and
review protocols**, not a real-world central authority. The longer-term
question is how strategic agents provide truthful reports, at what checking
and communication cost, and whether guidance can become stable behavior.

The next scientifically useful experiment is a preregistered report-sharing
or auditing contrast under the same target and action menu, with truthful and
manipulated reports and explicit communication/inspection costs. Separately,
if the title or claim is to emphasize rising actor capability, first verify on
held-out conditions that more strategy-generation resources produce stronger
behavior under a declared measure. Keep actor search, reviewer checks, delay,
and information **separate**, not collapsed into one unvalidated score. Do not
add games or LLMs merely to increase a count.

## Reproduce and inspect

From a clean checkout of the working branch:

```sh
python -m pip install -r requirements-validation.txt
python -m pytest -q tests
python -m experiments.plot_reviewer_decisions \
  --input-dir paper/paper_v5_scalable_oversight_commons/data \
  --output-dir paper/paper_v5_scalable_oversight_commons/figures
```

The complete confirmation cohort is committed as
`paper/paper_v5_scalable_oversight_commons/data/sources/budgeted_reviewer_confirmation_v1.tar.gz`.
The data README gives extraction and replay commands. Check
`data/sources/manifest.json` and `data/provenance.json` before quoting results.
Many `results/` outputs are intentionally ignored; do not assume they exist
in a clean clone. The paper PDF is a convenient read, but `main.tex` is the
source of record.

## GitHub Actions issue, October 2026

The three historical sweeps were wired to branch pushes. A documentation push
accidentally queued the large Harvest matrix and threshold replay. Their
unintended runs were cancelled. The threshold aggregation from that push
failed because its saved source run `27685651404` had **zero surviving GitHub
artifacts**, so `gh run download` returned `no valid artifacts found to
download`. This is artifact expiry, not a new scientific failure. The full
threshold summary was already recovered and committed in
`paper/paper_v5_scalable_oversight_commons/data/sources/supplementary/harvest_oversight_gap_threshold_replay_full_grid_recovered.csv`.
The workflows now trigger expensive runs only through an explicit request
file (for replay/aggregate) or a manual dispatch; normal PRs use the small
`Validation` test workflow. A historical red run remains in Actions history
and should not be rerun against expired artifacts.

## Suggested first task for Claude

Read the current paper, closeout audit, project history, and curated data
README. Summarize the *supported* claim, the main open scientific decision,
and any contradictions you find between paper prose and committed evidence.
Do not rewrite the paper or start experiments until that audit is complete.
