# Cleanup Parser-Repair Follow-up

23 September 2026. This is a **new bounded engineering attempt**, proposed
after `CLEANUP_POLICY_ADMISSION_CLOSEOUT.md` documented a first-step parser
exception and a tested observation-only repair. It does not turn the failed
first attempt into a completed result. The preceding report and code remain
available. No policy outcomes have yet been observed.

The sole question is whether the repaired fixed controller can run the
original admission matrix and meet its predeclared criteria. Use the same
`productive` and `free_rider` variants, reset seeds 17 and 43, 180 steps,
native per-agent observations, private serializable memory, paired transition
seed stream, native productive-capacity target, metrics and admission checks
specified in `CLEANUP_POLICY_ADMISSION_PROTOCOL.md`. No change to thresholds,
agent roles, behavior, environment configuration or score definition. The
repair resolves ambiguous ego-row decoding by explicit heading memory and a
fixed initial tie-break. It is not an outcome-driven policy search.

Before the attempt, rerun the native policy/adapter tests in the isolated
pinned SocialJax environment. Then run one CPU process, at most four episodes
and 720 native transitions, 120 wall and CPU seconds and 5 MB report. Save a
new report path. Any exception, timeout, missing metric or failed admission
criterion is reported as `not admitted`; do not adjust the policy and retry
within this contract. The test remains a two-context development admission
gate, not evidence for an oversight architecture or population generalization.
