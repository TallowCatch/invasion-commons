# Cleanup Policy Admission: Engineering Stop

23 September 2026. This is a separate policy-admission gate, **not** a local
versus joint monitor comparison or a paper result. The prospective contract is
`CLEANUP_POLICY_ADMISSION_PROTOCOL.md`; it was written before any new policy
outcomes. The pinned SocialJax checkout and isolated CPU environment were valid.

## Attempt and result

The one authorized local smoke attempt used
`SOCIALJAX_SOURCE=/tmp/commons-cleanup-socialjax`
and `/tmp/commons-cleanup-env/bin/python -m experiments.admit_cleanup_policies
--output /tmp/commons-cleanup-policy-admission.json`. It exited 1 in about 4.4
wall seconds with `ValueError: native observation does not identify a unique
ego row` in the first version of `fishery_sim/cleanup_policies.py`. No completed
episode report or admission JSON was produced. The exact number of native steps
before the exception was not recorded. Therefore **no policy competence,
productive/free-rider contrast, or admission outcome is measurable from this
attempt**. Cleanup is **not admitted** for an oversight comparison.

This was an observation-decoding engineering failure, not evidence for or
against ecological competence. A second visible agent can occupy the other
candidate ego row. After this failure, the parser was changed to use explicit
heading memory, with a fixed initial tie-break, when both rows are marked; a
synthetic regression test was added. This is a post-attempt code change and
has **not** been evaluated in a policy rollout. There was no threshold, seed,
role, or outcome-driven tuning, and the frozen one-attempt stop rule was honored.

## Verification and limits

Before the attempt, the new admission and existing Cleanup tests passed
(35 passed). After the parser repair, they passed again (36 passed, one
pre-existing JAX int32-to-int16 scatter FutureWarning). The native
noninterference test changed hidden pollution labels and other agents'
observations while holding agent 0's observation and memory fixed; its action
and next memory were identical. These tests validate the interface boundary
and selected mechanics, not whole-episode productivity. No cloud compute,
training, package install, existing source modification, or paper edit occurred.

Decision: **stop**. Any new rollout of the repaired parser requires a new
prospective, bounded attempt and must not be described as a continuation of
the frozen one-attempt result. The current heuristic also carries an unresolved
heading-estimation limitation: its initial absolute heading is a fixed guess,
not a value supplied by the native observation. It might fail even if the
parser no longer throws. A later study must measure that honestly, alongside
native harvest, dirt removal, and productive-capacity persistence.
