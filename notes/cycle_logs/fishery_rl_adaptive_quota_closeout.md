# Fishery RL Adaptive Quota Closeout

## Decision

- Promote `adaptive_quota` as the main Fishery RL headline in `paper_v2`.
- Keep the rest of the paper structure unchanged:
  - Fishery Commons remains the first study about central governance signals under invasion pressure.
  - Harvest Commons remains the second study about governance architecture.
- Present the RL result as a learned-policy validation of the Fishery ranking, not as a replacement for the invasion study.

## Evidence

- Direct `none` vs `adaptive_quota` confirmatory run on `medium_v1`, `5` PPO seeds:
  - collapse: `0.8141 -> 0.0`
  - mean stock: `97.5574 -> 186.0458`
  - mean welfare: `4770.1950 -> 9902.3497`
- Governance follow-up under the same PPO protocol:
  - `adaptive_quota`: collapse `0.0`, stock `186.0458`, welfare `9902.3497`
  - `monitoring_sanctions`: collapse `0.0016`, stock `126.5182`, welfare `6800.7801`
  - `temporary_closure`: collapse `0.0`, stock `124.0505`, welfare `4719.5819`

## Paper Implication

- The Fishery claim is now stronger than ``adaptive quotas win in hand-written threshold runs.''
- The stronger claim is:
  - the invasion study identifies `adaptive_quota` as the best overall top-down signal,
  - and a learned-policy PPO validation in the same medium tier recovers the same winner.
- This keeps the main causal story intact while reducing the risk that the result is specific to one policy family.

## Artifacts

- [fishery_rl_medium_v1_adaptive_quota_confirmatory_note.md](/Users/ameerfiras/invasion-commons/invasion-commons/results/runs/rl_fishery/curated/fishery_rl_medium_v1_adaptive_quota_confirmatory_note.md)
- [fishery_rl_medium_v1_governance_followup_note.md](/Users/ameerfiras/invasion-commons/invasion-commons/results/runs/rl_fishery/curated/fishery_rl_medium_v1_governance_followup_note.md)
- [main.pdf](/Users/ameerfiras/invasion-commons/invasion-commons/paper/paper_v2/main.pdf)
