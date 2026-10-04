# Research Direction After the Literature Audit

21 September 2026. Research and planning only. Read alongside
[the source/read-depth record](READING_LOG.md) and
[the proposed experimental protocol](OVERSIGHT_PROTOCOL_DRAFT.md).
This updates the scientific scope in the 17 September execution plan; it does
not replace completed results or authorize its experimental commands.

## Decision in Plain Language

Keep the project about **oversight in sequential commons**. The next study
should ask:

> How much information and checking effort does an overseer need to identify
> harmful joint resource-use decisions, and how much useful activity is lost
> when that information is unavailable?

The important extension is an actual, budgeted judgment task. A monitor must
judge a proposed action before seeing its outcome, with a declared view of the
system and a measurable checking budget. This makes the connection to scalable
oversight more direct than comparing governance packages whose limits are
assigned low/medium/high labels.

**Do not commission five independent families immediately.** Start with the
matched comparison in repository Harvest, transfer it to Fishery, and add an
established Cleanup implementation. These are three games covering shared and
spatial extraction plus maintenance. A delayed-pollution game is the strongest
subsequent addition. Five games can be a useful release goal, but the literature
does not make five a requirement, and five independent families have not yet
been justified or sourced.

Working title for the intended extension:
**Scalable Oversight in Sequential Commons: Evaluating Limited Information and
Checking Budgets.** Retain the current manuscript's more cautious title until
the budgeted supervision experiment is complete.

## 1. What the Existing Work Establishes

The current draft and the completed-checks report support a real, bounded
contribution: a working resource-game pipeline with reproducible population
experiments, action/outcome logging, and controls that expose misleading
interpretations of local/global safety disagreement.

| Finding in the repository | Scientific interpretation | Remaining restriction |
| --- | --- | --- |
| A fixed local cutoff permits clean-prefix failure onset in 16/80 high-stress, slow-regrowth episodes. | A particular individual rule is inadequate under these dynamics; this is more informative than a trace of persistent damage. | Ten populations and repeated weather trials are not 80 independent policy discoveries. The slow-regrowth follow-up is exploratory. |
| Unsafe occupancy falls from 5.33% with the fixed cutoff to 0.125% with the state-aware local filter. | Better local reasoning is a strong alternative to more centralized observation. | This refutes a broad claim that local oversight necessarily fails. |
| The joint reference has 8.30% unsafe occupancy in the same screen. | Prediction assumptions and safety targets matter. | It protects mean health/failure share; the local filter protects every predicted patch. The comparison does not isolate information. |
| More search has a setting-dependent held-out benefit, not a reliable three-level ordering. | Search is a controlled computational treatment whose benefits must be measured. | No validated actor-minus-overseer capability scale follows. |
| Announcements already change the structured policies. | Guidance and enforcement need separate controls. | Compliance is coded into those policies; no learned cultural norm or persistent internalization is demonstrated. |
| Direct numerical prompt templates reproduce broad model-bank collapse/protection patterns. | The saved-policy interface works across policy sources. | Current results do not show that LLM generation is essential or supplies uniquely difficult policies. |

Source: [completed checks](../RESULTS_AND_DIRECTION.md), current manuscript, and
`experiments/validate_harvest_mechanisms.py`, especially `local_state` and
`joint_reference`. The historical threshold matrix checks relabelling
sensitivity; its 9,000 rows are not 9,000 independent system experiments.

The current study has scientific content. It is still an exploratory
single-main-game study, and several controls have weakened the original
headline. Those corrections should remain visible rather than be hidden by
adding environments.

## 2. What the Closest Papers Change

These are method comparisons from complete readings, not abstract-only claims.

| Closest work | Established contribution and important scope | Consequence for this project |
| --- | --- | --- |
| [GovSim, Piatti et al., NeurIPS 2024](https://arxiv.org/abs/2404.16698), Sections 2-5 and appendices | LLM societies use renewable resources, communicate, receive reasoning interventions, and face greedy newcomers. The three narrative scenarios have mathematically equivalent resource dynamics. | Neither LLM commons nor adversarial newcomers are new. Borrow its task-grounded capability questions, not three scenario names as evidence of breadth. |
| [SocialJax, Guo et al., ICLR 2026 version](https://arxiv.org/abs/2503.14576), environment descriptions and dilemma analysis | A fast social-dilemma suite with policy-level cooperation/defection analysis and game-specific behavioural metrics. Its nine environments include three related Harvest variants. | Reuse an established game. Validate that our chosen policy populations exhibit the intended dilemma. Do not call every cooperation game a renewable commons. |
| [Elsayed-Aly et al., AAMAS 2021](https://arxiv.org/abs/2101.11196), centralized/factored shielding | Joint-action correction and coordinated local shields already address multi-agent safety. Guarantees rely on a suitable safety abstraction and stated assumptions. | Include a meaningful model-aware safety reference. Do not claim to discover compositional safety or describe all prior shields as requiring a fully known concrete simulator. |
| [Adalat et al., EUMAS 2026 accepted version](https://arxiv.org/abs/2606.14130), contract construction and assumptions | Local temporal obligations can be jointly certified and executed without central runtime control. Safe optimality depends on representability; offline certification/library costs matter. | Decentralization is not synonymous with weak safety. Compare coordination/evidence costs and conservatism; our empirical noisy-game controls do not inherit a formal guarantee. |
| [Engels et al., NeurIPS 2025](https://arxiv.org/abs/2504.18530), Section 2, Section 5, Appendix F | Fits task-specific actor/guard performance from pairings. Explicitly warns that different oversight capabilities need not transfer as one scale. Notes limitations in modelling dependencies between oversight steps. | Separate budgets from measured ability. Persistent resource state and repeated judgments are a defensible extension; assigned-rank subtraction is not. |
| [Sudhir et al., BiAlign 2025 workshop](https://arxiv.org/abs/2504.03731) | Evaluates whether a protocol rewards truth over deception, rather than merely whether its judge is often correct. Experiments have narrow task/model coverage. | Ask whether oversight accepts harmful proposals selectively, not only whether average resource health improves. Include approval/rejection quality and pressure from held-out policies. |
| [Alqithami, MCA 2026](https://www.mdpi.com/2297-8747/31/3/69), Sections 5-6, 9, 12; author revision read | An institutional wrapper already combines monitoring, delayed sanctions and review in Harvest/Cleanup. The pilot monitors a narrow beam-use rule; review is probabilistic. Explicit review queues/capacity and richer norms remain open. | Adding monitoring, delay and cost is insufficient novelty. A measured review task for joint consequences, with limited access and effort, addresses a more precise missing piece. |
| [Hu and Wang, July 2026 preprint](https://arxiv.org/abs/2607.11751), main analysis and Appendix D | Directly studies local monitors missing composed harmful outputs. Its own stronger fragment-aware local rule succeeds on a controlled code setting where weaker local statistics fail. | The wording "local checks pass while global harm occurs" is already directly occupied. Strong local baselines and sequential non-code consequences can distinguish our study; priority is not established. |
| [RICE-N, Zhang et al., 2022 report](https://arxiv.org/abs/2208.07004), Sections 3-7 and Appendix K | Couples production, emissions, carbon reservoirs, temperature, capital and negotiations. Distinguishes binding action masks from voluntary agreements and warns about simulation-to-world claims. | A genuine delayed-effect extension exists. It is substantially more involved than renaming Harvest and must preserve the distinction between observation, guidance and imposed control. |

Two other close lines need a full methods/code pass before final novelty claims:
[Pretorius et al.](https://arxiv.org/abs/2010.07777) already study information
structures in networked common-pool control, and
[Incentivising Monitoring](https://ojs.aaai.org/index.php/AAAI/article/view/10610)
is relevant if monitors themselves have incentives. Their presence is a reason
to narrow the claim, not to leave them out of related work. HDO was inaccessible
in full through the available fetching routes and is not used to justify a
theorem or experimental choice.

## 3. The Specific Gap Worth Testing

The synthesis suggests a useful intersection, not a proven first-in-literature
claim:

**Budgeted verification of joint decisions in evolving resource systems,
with selfish users, uncertain transitions, and explicit costs of unnecessary
restriction.**

This follows three concrete openings: sequential dependence identified in
Scaling Laws; actual capacity-limited review and richer monitored rules left
open by IML; and the need to separate missing evidence from weak local rules
in shielding/compositional-monitor work. Our specific combination and protocol
would be a new design to validate, not something those citations already prove.

The benchmark should let another researcher replace the monitor while holding
the game, proposals, safety target, evidence budget and intervention authority
fixed. The useful output is a curve of missed harmful decisions versus checking
cost and lost useful activity. A universal hybrid winner is unnecessary.

The study earns a scalable-oversight interpretation if it measures supervision
performance under resource asymmetry on a defined task. Showing only that a
cap stops extraction earns a governance/controller comparison. Showing only
more agents or longer episodes earns a system-scaling result. A broad claim
about weaker AI supervising stronger AI needs task-specific performance
evidence beyond any of these labels.

## 4. Five Families: What Would Actually Count?

There is no canonical requirement that a benchmark use five independent
families. Here "different family" should mean a different causal problem, not
different art, nouns, seeds or regeneration constants. Require a written
account of the new state dynamics, incentive conflict, failure mechanism and
monitoring difficulty. Related games remain related even if implemented by
different authors; environment count is not statistical independence.

| Candidate mechanism | Available game/source | What it adds | Decision |
| --- | --- | --- | --- |
| Renewable extraction | Existing Fishery; GovSim | Aggregate withdrawals can outpace renewal; a simple accounting reference is possible. | Keep Fishery. GovSim would be a future language-based replication, not three new families. |
| Spatial extraction with spillovers | Existing Harvest; SocialJax Harvest | Nearby actions change local resources and regeneration; evidence has a spatial boundary. | Keep our Harvest as the development game. Treat it as a distinct structure within extraction, not automatically a wholly independent family. |
| Maintenance of a productive commons | SocialJax Cleanup | Production depends on costly cleaning; low extraction alone does not supply maintenance. | Highest-value first external adapter. Must validate competent cleaning and harvesting policies. |
| Accumulating delayed harm | RICE-N | Present production affects emissions and later temperature/capital; one-step checks may miss later consequences. | Strong optional fourth game after the common protocol works. Use the original implementation or explicitly label a simplified derivative. |
| Multi-resource ecological composition | SocialJax Mushrooms | Resource types have different regrowth/reward consequences; preserving total stock may hide composition changes. | Possible fifth task, conditional on a non-arbitrary viability target and a distinct failure analysis. Not accepted into the core merely to reach five. |

A five-**independent-family** alternative could replace the related extraction
variants with congestion and investment/repair games. That is a larger,
different project. The congestion literature was discovered, but persistent
queue dynamics and a compatible risk label have not been verified here.
Investment/repair can also overlap with Cleanup unless its dynamics genuinely
add something. Neither is a ready-to-run recommendation.

**Recommended scope:** commit to Harvest + Fishery + Cleanup first. Keep RICE-N
as the next breadth test. Add Mushrooms only if its admission checks pass. If
all five are eventually included, describe five tasks spanning several resource
mechanisms, with their relationships visible. Do not advertise five independent
families. SocialJax Harvest Open/Closed remain useful boundary/implementation
checks, but are weaker additions for mechanism diversity than Cleanup or RICE-N.

## 5. A Consistent Rubric Does Not Require Identical Actions

Every admitted game must provide: user observations and actions; a persistent
shared state; a private-gain/collective-outcome conflict; a justified unsafe-state
definition; a monitor evidence interface; declared intervention options; useful
baseline policies; and held-out evaluation contexts.

The common questions are: did the monitor approve a risky proposal, did it
reject a safe one, what did checking cost, what happened after intervention,
and how much useful activity remained? Raw resource units and actions stay
game-specific. Harvest's cutoff of 10 and its policy fields do not transfer to
temperature, cleaning or queues.

Admission requires both productive and damaging policies. A game where all
actors do nothing, all policies immediately collapse, or a monitor wins simply
by freezing the simulation is not yet a useful oversight benchmark. Following
the social-dilemma lineage, test returns under cooperative/defecting population
mixtures; do not infer incentive conflict just from a familiar game name.

## 6. Feedback Carried Forward

| Feedback | Concrete response in the next study |
| --- | --- |
| Yali: what problem does the paper solve? | Judge future shared-resource harm under limited evidence; distinguish judgment quality from enforcement effectiveness. |
| Yali: model all stakeholders | General-sum resource users; explicit monitor query/verdict decisions; separate executor; experimenter chooses limits. A fixed monitor is not a learned strategic opponent. |
| Yali: ground the method | Closest-work comparison above, common protocol below, no invented source for numerical thresholds. |
| Group: several axes cannot simply become one capability gap | Separate actor candidate budget, search horizon, information access and reviewer work; calibrate each on held-out tasks. No replacement weighted sum. |
| Group: why do no-overseer lines vary with overseer strength? | Do not repeat inactive-overseer settings or put those conditions on a reviewer-strength axis. |
| Group: local failure with global safety? | Report the full decision/outcome table, including unnecessary restriction, approval coverage and uncertain reference labels. |
| Ed, as reported: decentralized guidance and norms matter | Shared evidence can be passed between peers. Keep information pooling separate from authority. Enforced masks remain enforced masks; normative compliance and persistence are future tests. |
| User: every experiment needs a reason | The first experiment fixes an unequal comparison; Cleanup tests maintenance; RICE-N tests delayed harm. No automatic third LLM or five-game training sweep. |

## 7. Claims to Keep, Test or Drop

- **Keep:** this is a controlled sequential-commons evaluation with real
  diagnostic and reproducibility work already completed.
- **Keep narrowly:** a particular fixed individual rule misses failure in
  specified settings; improved local reasoning prevents much of it.
- **Test next:** the benefit of shared evidence at matched safety targets and
  checking budgets, including whether that benefit survives another mechanism.
- **Drop:** a validated universal scalar capability gap, universal inadequacy of
  local oversight, universal hybrid superiority, or the first discovery of
  compositional harm.
- **Keep secondary:** offline model-generated policy evaluation. Improve it
  only if unanchored generation or iterative search adds useful behavioural
  coverage beyond an equally budgeted numerical baseline.
- **Do not infer:** institutional legitimacy, cultural norm learning, direct
  deployment safety, or policy prescriptions from this simulator.

## Suggested Message to Yali

> I reviewed the closest oversight and commons papers alongside our controls.
> The sharper question is how much information and checking effort a monitor
> needs to catch harmful joint decisions without unnecessarily restricting
> useful activity. A state-aware local check already performs well in our
> current game, so I want to compare it fairly with shared-information checks,
> using the same safety target. I would then test the same protocol in a
> single-stock game and Cleanup, where agents must maintain the resource.
> Actor search and reviewer resources would be measured separately. Does this
> give the right scope for a first benchmark paper, before adding delayed-harm
> games or further LLM experiments?
