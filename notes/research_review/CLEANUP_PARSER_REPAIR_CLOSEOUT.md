# Cleanup Parser-Repair Attempt: Not Admitted

23 September 2026. The separate follow-up contract is
`CLEANUP_PARSER_REPAIR_PROTOCOL.md`. The first failed parser attempt remains
documented in `CLEANUP_POLICY_ADMISSION_CLOSEOUT.md`; it was not overwritten.
The native pinned SocialJax adapter/policy tests passed **36 tests** before
this follow-up, with one upstream JAX dtype FutureWarning. The ordinary repo
suite has 223 passed and 11 optional-native skips.

The one permitted CPU follow-up completed four episodes and 720 transitions
in 3.80 wall / 6.30 CPU seconds. It produced
`results/runs/cleanup_policy_admission_v2.json`. The report's embedded
`protocol` field names the original admission criteria; the distinct
follow-up contract above records why this second attempt was authorized.
Source SHA-256: `cleanup_policies.py`
`7dea2816a19de7821e6ead2297e6ed7f16cefc0ba4fb9f3608337725d2b434f0`;
`admit_cleanup_policies.py`
`26ab96d448d76b91f503ca4e0cbceab994955bbbaaa7075fad1e2f1125dd5128`.
Report SHA-256:
`8e6639735b0c17a855d58a31fef554f083023564eba08a3204371ca1d5519cad`.

**Admission failed for both reset seeds.** Every episode began outside the
productive-capacity safe set, and neither policy reached it or harvested any
apples over 180 steps. The productive variant proposed 106 and 85 clean
actions and removed 13 and 6 preexisting dirt cells, respectively, for seeds
17 and 43. The free-rider variant proposed no cleaning and removed none, but
also harvested nothing. Productive maintenance therefore had a measured local
effect without restoring ecological capacity or establishing the required
productive-versus-free-rider contrast. No safe-to-unsafe onset occurred because
all trajectories started unsafe; that is not a successful safety outcome.

Stop the admission sequence. Do not include Cleanup in cross-game oversight
figures or claim that the current observation-limited policy is competent. The
next Cleanup question would need a *new* design for reachable productive
starts or sufficient observation-limited cleaning competence, plus an
economically active damaging policy. Any changed reset, threshold, controller
or horizon must be declared before outcomes; repeated attempts until the gate
passes would be selection on the result. Fishery/Harvest reviewer work can
continue independently.
