# Supervisor brief: oversight information in sequential commons

24 September 2026. Working question: when several agents propose to use a
renewable resource, what information must a reviewer inspect or share to
reject harmful joint proposals without unnecessarily scaling down safe ones?

Fishery is a transparent shared-stock case: total extraction matters. Harvest
adds local patches, neighbour damage and stochastic renewal. The overseer is a
fixed decision rule, not a trained strategic player. All reviewers in the
matched test use the same one-step global safety target, the same public
resource state, and the same uniform action-scaling options. The tested factor
is how current requests enter their prediction. We evaluate immediate
harmful approvals and safe restrictions, plus resource health and return over
the subsequent trajectory.

The fresh comparison used 64 independent policy contexts per game, with
1,920 complete episodes. Joint review restricted fewer safe original
requests than a coarse conservative local calculation in both selected
settings, with no additional resolved-risky approvals observed. An optimistic
local calculation was unsafe in Fishery but performed similarly to joint
review in Harvest. Better immediate discrimination did not guarantee better
long-run return. The Harvest reference leaves 923 initially-safe proposals
unresolved; those are not counted as safe.

The paired long-run analysis makes the trade-off concrete. Against the
conservative local rule at full inspection, Fishery joint review has a mean
total-return difference of -233.2 (95% context-bootstrap interval -259.8 to
-207.6) and a mean-stock difference of -8.70 (-9.76 to -7.69), even though it
restricts fewer originally safe requests. A worked safe-start Fishery request
in the paper shows exactly where the coarse local bound wastes information.

We then checked a serious alternative explanation. The coarse local rule did
not use all inspected neighbouring information. A post-hoc coupled-local
calculation that does use it matched joint review on all 33,747 paired
budget decisions from the saved cohort. This is an algebraic/implementation
boundary, not fresh confirmation. It means the current evidence does not
support an intrinsic advantage for centralized oversight. The more useful
question is what information needs to be exchanged, at what communication or
verification cost, especially when agents can misreport.

This is presently a **two-game mechanism study and benchmark protocol**, not
a broad benchmark suite or a demonstrated actor--overseer intelligence gap.
The earlier architecture matrix motivated the matched experiment, but its
packages bundle several changes. The small offline model-generated policy
test establishes interface compatibility; prompt anchors also reproduce its
broad result. A third game, SocialJax Clean Up, has an adapter but did not
pass the policy/safe-start admission gate, so it is not counted as evidence.

The next methodological choice is whether to make the first paper this
focused information study or to run a new, prospectively specified test of
limited or manipulable information. I recommend the latter as the one
substantive addition: compare truthful local reports, report auditing under
a fixed inspection budget, and joint review using the same safety target and
intervention menu. A small policy/safety admission pilot must precede fresh
confirmation. Adding more game names or LLMs now would not answer this
question.

Feedback requested: Is the information-cost question the right first-paper
scope? What model of local reports and verification would be scientifically
credible? Should the resulting first submission be presented as a two-game
oversight mechanism study, while broader game coverage remains a separate
benchmark-release objective?

Evidence and caveats: `BUDGETED_REVIEWER_CONFIRMATION_CLOSEOUT.md`,
`COUPLED_LOCAL_REPLAY_PROTOCOL_20260924.md`, and
`PAPER_CLOSEOUT_20260924.md` in this directory.
