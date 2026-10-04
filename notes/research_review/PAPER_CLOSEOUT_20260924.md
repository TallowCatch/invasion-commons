# Paper closeout audit: oversight information in sequential commons

24 September 2026. This is a decision record for the current manuscript, not
a claim that it is submission-ready. The working paper is
`paper/paper_v5_scalable_oversight_commons/main.tex`.

## One question and one honest answer

**Question:** When agents propose joint use of a renewable resource, what must
reviewers observe or share to reject harmful proposals while retaining safe
activity?

The current answer is conditional. In two selected six-agent games, a joint
one-step calculation restricts fewer safe original requests than a *coarse*
local bound. However, a local calculation that uses the same inspected
requests and known neighbouring effects exactly reproduces the joint
decisions on the saved cohort. The design therefore tests the value and cost
of information, not the superiority of a centralized institution. A strong
local state rule also defeats the claim that every local check fails.

This is a **mechanism study / pilot benchmark protocol** in two game
families. It is not yet a broad benchmark suite and does not validate a
general actor--overseer capability scale. The original governance-package
study and saved language-model policies support the development history,
not those stronger claims.

## Evidence inventory

| Evidence | Status | Main implication |
| --- | --- | --- |
| Original Harvest package matrix | Exploratory; packages change several factors together | Motivates matched comparisons; do not rank institutions from its winner cells. |
| Fixed-cutoff and local-state controls | Frozen-policy exploratory checks | A fixed local rule misses some slow-regrowth failures; the state-aware local rule is a strong counterexample. |
| Fresh limited-inspection comparison | 64 independent population contexts in each of two pilot-selected settings; 1,920 episodes | Joint review restricts fewer safe proposals than the coarse bounded rule; optimistic local performance depends on the game. |
| Coupled-local replay | Post-hoc, same saved cohort: 11,249 proposals and 33,747 paired budget decisions | Zero decision disagreements with joint review; no new independent observations. |
| Cleanup admission | Failed | All tested native episodes began outside the safe/productive set; no productive-vs-free-rider comparison was established. |
| Saved Qwen/Llama policies | Small offline artifact test | Structured policy interface works; numerical prompt anchors alone reproduce the broad collapse/protection pattern. |

The replay protocol is `COUPLED_LOCAL_REPLAY_PROTOCOL_20260924.md` and its
local-file result is `results/runs/coupled_local_replay_20260924/summary.json`.
The fresh comparison remains the main independent evidence; its prior local
protocol and interpretation are `BUDGETED_REVIEWER_CONFIRMATION_PROTOCOL.md`
and `BUDGETED_REVIEWER_CONFIRMATION_CLOSEOUT.md`.

## What the relevant papers actually show

These are figure and evaluation precedents, **not** templates to copy.

| Paper | Figure/measurement practice | Consequence for this paper |
| --- | --- | --- |
| [SocialJax](https://arxiv.org/html/2503.14576) | Environment gallery, training-return panels, Schelling diagrams, throughput and game-specific cooperation metrics. | Use a game/interface diagram and a game-coverage table. Do **not** add training curves: our main experiment does not train reviewers. A Schelling diagram would be relevant only after validating cooperative/defective policy classes. |
| [Safe MARL via Shielding](https://arxiv.org/html/2101.11196) | Action-correction schematic, safety specification, minimum-interference criterion, joint/factored shielding comparisons. | Draw the proposal -> limited view -> decision -> executed action path. Measure harmful approvals and unnecessary intervention, with common authority. |
| [Contract-Based Compositional Shielding](https://arxiv.org/html/2606.14130) | Tiny counterexample to explain why naive local masks discard safe actions; reward curves with run-level intervals across six environments. | A small worked two-agent extraction example is more informative than a decorative flowchart. Compare to coordinated local rules, not only the naive bound. We make no deterministic guarantee. |
| [Scalable Oversight Benchmark](https://arxiv.org/html/2504.03731) | Explicit protocol metric, protocol comparisons, separate capability analysis. | A paper needs a clear estimand. Our inspection budget is a resource, not a scalar intelligence gap; plot decision quality against it. |
| [GovSim](https://arxiv.org/html/2404.16698) | Survival/gain/overuse metrics, time-series resource examples, newcomer and communication ablations. | Keep one explanatory trajectory and report resource/return alongside decision metrics. A single trace is illustrative, not an effect estimate. |
| [Multi-Agent AI Control](https://arxiv.org/abs/2607.07368) and [local-monitor compositional harm](https://arxiv.org/abs/2607.11751) | Distributed harm under limited per-instance views, with measured attack and monitor changes. | Local-pass/global-fail is not novel by itself. The distinctive test must be renewable-resource dynamics plus the information/safe-activity trade-off. |

The literature reading register in `literature_20260921/READING_LOG.md` records
full-text coverage for the principal sources. This is a focused search, not an
exhaustive priority claim. A 2026 paper should not be called peer reviewed
solely because it is on arXiv.

## Figure architecture for the finished paper

1. **Problem and interface:** a compact vector diagram of proposed requests,
   public resource state, an inspection mask, reviewer calculation, common
   scale menu, and next resource. The main paper now uses TikZ for this.
2. **Primary empirical result:** two game panels of safe requests scaled down
   against 0/3/6 inspected requests, with 95% context-cluster bootstrap
   intervals, denominator counts, and risky-approval counts in caption/table.
   Source and vector exports: `experiments/plot_reviewer_decisions.py` and
   `paper/paper_v5_scalable_oversight_commons/figures/fig08_reviewer_decisions.pdf`.
3. **Long-run consequence:** Table 3 in the manuscript now reports paired
   differences in resource health and total return, with context-level
   bootstrap intervals. Fishery's immediate safe-activity gain reverses for
   mean stock and return against bounded local review. The reproducible
   context differences and summary are in `paper/.../data/analysis/`.
4. **Worked case:** the main paper now gives Fishery context 0, step 14:
   six visible requests total 14.70, leaving predicted stock 10.77 above
   the cutoff; the coarse local bound substitutes 20.65 and halves the
   request. It is a mechanism illustration, not a separate effect estimate.
   The old 29-step local-pass/global-fail trace shows persistence, not onset.
5. **Appendix:** Stage A package matrix, threshold relabelling sensitivity,
   search calibration, LLM policy-artifact controls, and cost-accounting
   details. A winner map alone is not the primary result.

Do not plot the old rank-subtraction "capability gap" as if it were measured
intelligence. Do not use multi-game training curves without training agents.
Use vector PDF/SVG, 7--9 pt effective type at final column width, a
colorblind-safe palette plus line/marker redundancy, explicit axis units,
and captions that identify samples and uncertainty.

## Game-family decision

Adding more games is scientifically useful **only if** the same reviewer
question can be asked with a meaningful safe/risky reference and working
policies in each game. Distinct names are not independent dynamics.

| Candidate | New mechanism | Present status | Admission decision |
| --- | --- | --- | --- |
| Fishery | Additive extraction from one stock | Working | Keep as transparent aggregate-demand case. |
| Harvest | Local patches, neighbour spillovers, stochastic renewal | Working | Keep as coupled-state case. |
| SocialJax Clean Up | Maintenance public good and delayed apple regrowth | Adapter exists but policy admission failed | Do not include results yet; redesign starts and observation-only policies under a separate predeclared gate. |
| GovSim pasture/pollution | Different narrative surface in a shared resource simulator | Local source inspection shows they inherit the same common concurrent resource dynamics as fishing | Do not count them as independent game families without demonstrating different transitions and safety targets. |
| RICE-N / climate-style game | Delayed global stock and negotiation | Literature candidate only; no adapter or admitted policies | Candidate for a later paper, not an immediate extra row. |

For a third game, require: (a) meaningful safe starts and unsafe onsets; (b)
competent fixed policies with a productive/damaging contrast; (c) the same
*evaluation questions* and matched target, information, and intervention
authority **within that game**; (d) enough resolved safe and risky contexts
to estimate both errors; (e) game-native metrics alongside the common rubric.
Clean Up would need a keep-or-stay decision, not a copied harvest-fraction
scaler: blocking an apple-collection or cleaning action can itself damage
future productive capacity. A reviewer cannot force agents to clean unless
that new authority is explicitly modeled. Its productive and free-rider
policies must first harvest and separate economically on safe starts; they
currently do not. This is a medium engineering project with high scientific
admission risk, not a free benchmark row. Failed admission is reported, not
tuned repeatedly until it passes. No current third family meets this gate.

## Remaining work, in order

1. **Scientific design:** decide whether the first submission is the narrow
   two-game information study or a larger benchmark release. The current data
   support the former. The main unresolved mechanism is how truthful
   information is obtained from strategic agents, at what total cost.
2. **One focused next experiment:** predeclare a limited-communication or
   report-auditing contrast, with the same safety target and action menu.
   Account for truthful versus manipulated reports, number of inspected
   requests, total messages, and safe activity retained. Run a small admission
   pilot before fresh confirmation. Do not present the post-hoc coupled replay
   as that confirmation.
3. **Paper edits:** lead with the matched reviewer study; shorten Stage A to
   motivation, move the small offline LLM test to appendix, and distinguish
   diagnosis of one-step proposals from closed-loop protection. Add the
   explicit result provenance. The paired long-run table and worked case
   have been added. The figure should not outrun the evidence.
4. **Release:** pin dependencies; make a small public smoke and result
   manifest; ensure a clean checkout can reproduce main tables/figures without
   ignored local files; resolve permissions/licences before redistributing
   external game code or data. Compile and visually check the final PDF.

## Stop rule

Do not launch five games, a broad LLM sweep, or paid compute to fill a figure.
First establish a distinct admitted third game or a genuinely informative
communication/auditing effect. A null or boundary result remains publishable
if the question, comparator and uncertainty are sound. Until then, the safe
claim is **a controlled two-game investigation of oversight information**, not
"scalable oversight solved" or a general benchmark of AI capabilities.

## Submission shape

The nearest mature community is multi-agent systems, where [AAMAS 2026
proceedings](https://www.ifaamas.org/Proceedings/aamas2026/forms/contents.htm)
include technical studies of agent interaction, evaluation, and safety.
An AAMAS-style full paper would need a prospectively specified report-sharing
or auditing test, stronger coverage beyond two pilot-selected settings, and
an accessible reproducer. The present manuscript is a supervisor-ready
working paper, not a justified full-conference submission yet. A focused
workshop version could report the two-game information result and its
counterexample honestly, if the venue accepts preliminary mechanism studies.
[FAccT's 2026 call](https://facctconference.org/2026/cfp.html) explicitly
places purely hypothetical work without deep social engagement outside its
scope, so this simulator-only study should not be pitched there as a policy
paper. Submission dates and formats must be rechecked against the eventual
year's official call rather than assumed from these examples.
