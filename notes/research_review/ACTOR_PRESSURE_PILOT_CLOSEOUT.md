# Actor-search pressure pilot: completed, not confirmed

23 September 2026. The [protocol](ACTOR_PRESSURE_PILOT_PROTOCOL.md) was saved
before the four-context pilot. One amendment after an engineering smoke changed
the background from five cooperative partners to three cooperative plus two
aggressive partners, because the 12-step smoke produced no risky reference
cases. The original smoke is retained separately. No outcome threshold,
reviewer rule, or search budget was changed after seeing pilot outcomes.

## Why this was run

The recent [matched-reviewer confirmation](BUDGETED_REVIEWER_CONFIRMATION_CLOSEOUT.md)
showed an information-versus-safe-activity trade-off under fixed actors. It
did not test whether review becomes harder when actors get more search. The
older Stage A rank did not validate that claim either. This pilot changes
only the actor's number of candidate strategies and keeps reviewer rules,
inspection budgets, safety target, and allowed request-scaling actions fixed.

The intended paper contribution, if later validated, is a measurement
protocol for the **safety/useful-activity trade-off when interacting actors
grow more strategic under limited review**. That is narrower than inventing
commons games or multi-agent monitoring. [Leibo et al.](https://arxiv.org/abs/1702.03037),
[Melting Pot](https://proceedings.mlr.press/v139/leibo21a.html), and
[SocialJax](https://arxiv.org/abs/2503.14576) establish the game lineage;
[GovSim](https://arxiv.org/abs/2404.16698) already tests model agents in a
renewable commons; [Kenton et al.](https://arxiv.org/abs/2407.04622) and
[a scalable-oversight benchmark](https://arxiv.org/abs/2504.03731) motivate
careful reviewer baselines. [Multi-Agent AI Control](https://arxiv.org/abs/2607.07368)
also studies distributed attacks on per-agent monitors. The remaining
distinctive question here is **sequential resource consequences with the same
review target and intervention choices**, not a first-of-kind oversight claim.

## What ran

- Four independently generated candidate/partner contexts. Within each,
  candidate sets were nested at 1, 8 and 32 candidates, with two 30-step
  training evaluations each. The selected actor occupied one of six slots.
- Two new test weather seeds and a new mixed partner population per context;
  80-step Harvest episodes with no review and three fixed reviewers at
  inspection budgets 0, 3 and 6.
- 256 short selection evaluations, 240 test episode blocks, 407 initially
  safe frozen requests and 3,663 matched reviewer decisions. The full run
  took 14.5 seconds of runner wall time and saved about 3.0 MB before figures.
  No paid/cloud compute, LLM inference or model training was used.
- Of the 407 initially safe frozen requests, the 128-draw one-step reference
  resolved 131 as safe and 196 as risky; 80 were unresolved and excluded
  from conditional error rates. A further 1,388 recorded steps began with
  the system already unsafe and were excluded from this immediate-decision
  analysis. That high occupancy is itself an important difficulty problem.

Run: `results/runs/actor_pressure_pilot_mix2_v1/`. The directory contains
source snapshots, a protocol/hash manifest, selected candidates, individual
episode blocks, frozen proposals, decisions, context-level CSVs, checksums,
and vector/raster figures. The result directory is ignored by Git; it is a
local research artifact, not a reproducible public release.

From the repository root, the commands were:

```bash
pytest -q tests/test_actor_pressure_pilot.py
python -m experiments.run_actor_pressure_pilot \
  --output-dir results/runs/actor_pressure_pilot_mix2_v1 \
  --profile pilot --max-seconds 600 --max-bytes 150000000
python -m experiments.plot_actor_pressure_pilot \
  --summary-csv results/runs/actor_pressure_pilot_mix2_v1/analysis/context_summary.csv \
  --output-dir results/runs/actor_pressure_pilot_mix2_v1/analysis/figures
pytest -q tests
```

## Actor-strength manipulation check

The training score rises by construction with nested search, so it is not
enough to call the selected actors stronger. More search selected a different
policy at 8 versus 1 in only 1/4 contexts; 32 versus 1 changed the selection
in 4/4 contexts. The held-out payoff changed, but the response was uneven.

| Search candidates | Mean training score | Mean held-out entrant payoff | Mean no-review unsafe fraction | Resolved risky / safe / unresolved requests |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 58.25 | 139.86 | 0.811 | 55 / 46 / 26 |
| 8 | 65.74 | 177.93 | 0.803 | 60 / 46 / 26 |
| 32 | 66.66 | 181.46 | 0.780 | 81 / 39 / 28 |

These are descriptive means over only four contexts. The mean payoff rise
from 1 to 8 is largely driven by one context where the 1-candidate policy
performed poorly. At 32 versus 1, held-out payoff increased in 3/4 contexts
and fell slightly in one. More search did **not** produce a monotone increase
in unsafe occupancy. Different numbers of risky proposals also reflect
different trajectory lengths and states, not just a changed propensity at
the same state. The pilot therefore validates that search can change selected
behavior, but **does not validate a simple stronger-actor = more system harm
ladder**.

## Fixed reviewer decisions

At six inspected requests, the descriptive pooled frozen-request counts were:

| Actor candidates | Reviewer | Resolved risky approved | Resolved safe requests restricted |
| ---: | --- | ---: | ---: |
| 1 | Joint | 0 / 55 | 37 / 46 |
| 1 | Local bounded | 0 / 55 | 46 / 46 |
| 1 | Local optimistic | 0 / 55 | 33 / 46 |
| 8 | Joint | 0 / 60 | 37 / 46 |
| 8 | Local bounded | 0 / 60 | 46 / 46 |
| 8 | Local optimistic | 0 / 60 | 33 / 46 |
| 32 | Joint | 0 / 81 | 31 / 39 |
| 32 | Local bounded | 0 / 81 | 39 / 39 |
| 32 | Local optimistic | 0 / 81 | 26 / 39 |

These counts are **not independent trials**: many requests come from the
same four contexts and repeated game steps. No population interval or
significance test is warranted. At inspection budgets 0 and 3, all reviewers
also approved zero resolved-risky originals, while they restricted almost
all resolved-safe originals. The local optimistic rule spared more safe
requests at full inspection in this Harvest setting. "Restricted" includes
partial scaling, not complete stoppage of work. Zero observed risky approvals
is not proof that the true rate is zero.

At actor budget 32, no review averaged 0.780 fixed-horizon unsafe fraction,
mean patch health 9.71 and welfare 718.1. All three full-inspection reviewer
arms had zero observed unsafe steps; joint review averaged patch health 11.02
and welfare 687.7, bounded local 11.75 and 646.0, optimistic local 10.98 and
688.9. This is a four-context, one-setting comparison of different closed-loop
trajectories, not evidence of a universally optimal reviewer or welfare policy.

## Figures and their role

The figures are in `results/runs/actor_pressure_pilot_mix2_v1/analysis/figures/`,
each as PDF, SVG and 300-dpi PNG. Their `figure_manifest.json` records the
summary CSV and plotting-script hashes.

1. `actor_pressure_training_heldout`: paired context lines test whether
   search's guaranteed training advantage survives new partners and weather.
2. `actor_pressure_decision_tradeoff`: separate panels for harmful approval
   and safe restriction by inspection budget; context points show the four
   independent units, and dashed no-review lines provide the baseline.
3. `actor_pressure_resources_welfare`: long-run unsafe fraction, patch health,
   welfare and reviewer component checks at full inspection. It keeps
   ecological, activity and computational costs separate.

No figure carries a confidence band because four contexts do not support a
meaningful population interval here. The exact counts remain in
`analysis/context_summary.csv`, rather than being hidden in plot styling.

The pilot contract did not set a numeric minimum practically meaningful
effect or a power-based sample size. It therefore cannot be promoted to a
confirmatory effect test after looking at these results. Any later study must
set its effect and precision targets prospectively.

## Decision for the paper

**Stop this pilot.** Engineering is valid, but the proposed interaction
between actor search and reviewer errors is not demonstrated. No-review
failure is often saturated, and the fixed reviewers make no resolved-risky
approval errors in this sample. Adding more repetitions of this same
configuration would improve precision around a weakly discriminating task,
not by itself create the missing mechanism.

The current first-paper core should remain the matched Fishery/Harvest
information study. The actor-search study is a transparent pilot and a
design boundary, not a new headline result. The next *separate*, prospective
experiment should first construct non-saturated, informative actor requests
under fixed review: for example, compare reward-seeking search with a
reviewer-aware search objective on training contexts, then test both on new
partners without retuning reviewer thresholds. Its admission gate is not a
favorable result; it is verified behavioral difference plus adequate safe and
risky cases across independent contexts. Only then is a larger, uncertainty-
bearing actor-by-inspection experiment justified. An additional game or LLM
model is not the remedy for this pilot's missing discrimination.

## Minimum remaining paper program

1. **First-paper core, already run:** keep the fresh-seed matched Fishery and
   Harvest reviewer comparison. The primary figure should show the paired
   harmful-approval versus safe-restriction trade-off at 0/3/6 inspections;
   an exact-count table and a long-run resource/welfare panel supply context.
   State the game-specific limits rather than pooling their returns.
2. **Capability-pressure gate, still open:** predeclare a new actor-search
   study that includes a reward-seeking baseline and, only if scientifically
   justified, a reviewer-aware actor. Select on training partners, evaluate
   on new partners, and keep reviewer target/actions/budgets fixed. Require
   a non-saturated case distribution and report actor held-out return,
   proposal risk, reviewer errors and useful activity separately. Use a
   development set to calibrate case coverage, then fresh independent
   contexts for confirmation. This pilot cannot substitute for that test.
3. **Game transfer, conditional:** do not call Clean Up a third result until
   its policy-admission gate yields both useful safe activity and harmful
   requests. Add a further game only if it tests a new failure mechanism,
   such as delayed maintenance, not to make a five-row benchmark table.
4. **Model-generated strategies, later:** rerun the same fixed-reviewer
   protocol with validated structured LLM policies and a numeric-template
   control. Do not treat model count or live inference as a contribution by
   itself.
5. **Release gate:** make the input manifests, ignored result data or a
   reproducible result-generation route, checks, figures, and exact commands
   available from a clean checkout. The current local pilot bundle alone
   does not meet that requirement.

Verification: `pytest -q tests` gave **232 passed, 11 skipped**. The two
skipped groups include optional native Cleanup tests; two external-library
deprecation warnings are unrelated to this pilot. No manuscript claims were
changed on the basis of the pilot.
