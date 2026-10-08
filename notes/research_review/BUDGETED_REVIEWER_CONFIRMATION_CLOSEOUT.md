# Fresh-Seed Reviewer Confirmation: Closeout

23 September 2026. The [frozen local protocol](BUDGETED_REVIEWER_CONFIRMATION_PROTOCOL.md)
chose **immediate decision quality at a fixed one-step safety target** as the
primary question. Long-run resource and return outcomes are secondary. This
choice follows the original oversight question: whether limited information
causes a monitor to allow harmful joint requests or restrict safe ones. It
does not turn search budget or information access into a single capability
score.

## Execution and coverage

The engineering smoke passed (14 episodes). The one planned fresh-seed run
completed **1,920 episodes**, **11,249 original frozen proposals**, and
**101,241 matched reviewer decisions** in 288.4 wall seconds using one local
CPU worker and 19.5 MB of saved files. Manifest, source snapshots, block
checksums, pairing and completion verification passed. The ordinary test suite
has 226 passed and 11 skipped; optional native Cleanup tests use a separate
runtime. There was no paid/cloud compute, LLM inference or training. Results
are in `results/runs/budgeted_reviewer_confirmation_v1/`.

Both selected settings met the predeclared class-coverage floor of 20
independent contexts per resolved class:

| Setting | Resolved-safe original requests | Resolved-risky | Unresolved | Contexts with safe / risky |
| --- | ---: | ---: | ---: | ---: |
| Fishery `mix4`, deterministic stock | 694 | 63 | 0 | 64 / 63 |
| Harvest `mix2`, slow renewal | 3,687 | 1,368 | 923 | 64 / 62 |

The settings were chosen **after** the earlier development pilot for
two-class coverage. Fresh population seeds make these new observations, but
they do not validate other mixtures, renewal settings, or game families.
Harvest's 923 unresolved reference labels are excluded from the safe/risky
denominators, not treated as safe. The Harvest reference is a 128-draw
one-step Monte Carlo decision, not ground-truth long-run safety.

## Primary decision trade-off

The table counts original **joint** proposals from initially safe states.
"Safe restricted" means the original request was scaled down, possibly only
partly; it does not mean all useful activity was stopped. All monitors use
the same global target and action menu, and the same inspected subset at a
given budget.

| Setting, 6 inspected | Joint review | Bounded local | Optimistic local |
| --- | ---: | ---: | ---: |
| Fishery: risky approved / 63 | 0 | 0 | 63 |
| Fishery: safe restricted / 694 | 0 | 100 | 0 |
| Harvest: risky approved / 1,368 | 0 | 0 | 0 |
| Harvest: safe restricted / 3,687 | 2,521 | 3,687 | 2,407 |

For joint minus bounded local at budget 6, the safe-restriction difference
was **-14.4 percentage points** in Fishery (95% context-bootstrap interval
-18.6 to -10.9) and **-31.6 points** in Harvest (-41.7 to -21.1). Neither
approved a resolved-risky original proposal in these settings. Joint minus
optimistic local was **-100 points** on Fishery risky approval: the optimistic
rule approved all 63; both left all 694 safe requests unchanged. In Harvest,
the optimistic local rule restricted **3.1 points fewer** safe proposals than
joint (joint minus optimistic: +2.3 to +4.1 points), while both approved
0/1,368 resolved-risky proposals. Hence the tested Harvest setting supports
an information-versus-conservatism comparison, **not** a unique advantage of
joint review over every strong local rule.

At zero and three inspections, all three Harvest modes restricted nearly
every resolved-safe original request; their upper bound on uninspected
requests is deliberately conservative. Fishery joint review reduced safe
restrictions from 229/694 at zero inspections to 107/694 at three and 0/694
at six, with no resolved-risky approvals. These budget curves are supportive
diagnostics, not separate independent confirmations.

The predeclared binary restriction metric conceals how much activity remains.
A **post-hoc**, case-weighted diagnostic from the saved decisions found the
mean fraction of resolved-safe original requests retained at budget 6:
Fishery joint 1.000, bounded local 0.955, optimistic local 1.000; Harvest
joint 0.774, bounded local 0.493, optimistic local 0.782. No safe Fishery or
budget-6 Harvest original request was completely stopped in this diagnostic;
the differences were partial scaling. This diagnostic was computed after
seeing the primary results and has no uncertainty interval. It is saved
separately in `posthoc/resolved_safe_retention.csv` and must not be presented
as a predeclared endpoint.

## Long-run boundary

The fixed-horizon unsafe fraction was zero for joint and bounded local review
at all three budgets in both selected settings. At budget 6, Fishery joint
review had mean 80-step return **551.5** versus **784.7** for bounded local,
and lower mean stock (**16.88** versus **25.58**). Allowing more immediately
safe activity can leave less stock for later harvest. Optimistic local review
had **0.861** fixed-horizon unsafe fraction in Fishery, versus **0.864** with
no oversight; in Harvest all three monitored modes had zero observed unsafe
steps, versus **0.421** without oversight. Harvest joint and optimistic local
at budget 6 had similar mean return (**653.7** versus **653.2**). These are
different closed-loop trajectories; one-step labels do not establish their
causal long-run ranking. The absorbing-failure convention is used only for
the fixed-horizon unsafe rate, and terminal events are separately logged.

## What this establishes, and what it does not

The completed comparison shows that request information can change the
monitor's immediate safety/activity trade-off under **matched target and
authority**. It also shows a boundary: a less conservative local rule matches
or slightly exceeds joint review's safe-activity retention in Harvest without
an observed harmful approval there. The Fishery optimistic rule, in contrast,
fails to account for aggregate extraction. The case mix and dynamics decide
which difference is exposed. These are conditional results for fixed policy
generators, known transition forms and two chosen settings. A zero empirical
harmful-approval difference yields a degenerate bootstrap interval when both
methods make identical decisions; it is **not proof of a zero true risk**.

This run does not validate a general actor--overseer capability gap, a learned
weaker overseer, live LLM agents, or a five-game benchmark. The bounded local
Harvest rule pools patch reports into a global decision, so it is not purely
decentralized governance. Fishery and Harvest use different resource units;
their rates are shown side by side, not pooled. This comparison is a rigorous
oversight-information study but does not alone complete the broader
"scalable oversight under rising actor capability" paper story.
The monitors know the simulator's transition rules, making full-information
joint review a diagnostic reference rather than a deployable learned monitor.
The modes share inspection and candidate limits, but total component checks
and communication differ and are logged rather than assumed equal.

## Next paper gate

Stop this experiment as predeclared. The working manuscript now includes the
matched-information result as a **mechanism study**, while preserving the
earlier actor-generation benchmark as background. The most defensible primary
claim of this new study is about immediate oversight decisions, not universal
architecture superiority or a general capability gap. A final paper framing
still needs supervisor input: retain this within a broader benchmark paper or
make a narrower oversight-information paper. For a stronger scalable-oversight
claim, the next *separate* test would hold this reviewer protocol fixed while
validating distinct actor strategy-generation resources on new policy
contexts. It must measure the resulting proposal distribution and reviewer
trade-off directly, not assign a scalar rank. Supervisor feedback on that
scope should precede another sweep.

Key artifacts: `analysis/primary_decision_quality.csv`,
`analysis/primary_paired_contrasts.csv`, `analysis/outcomes.csv`,
`analysis/paired_contrasts.csv`, `posthoc/resolved_safe_retention.csv`,
`analysis/experiment.md`, and `completion.json` in the run directory.
