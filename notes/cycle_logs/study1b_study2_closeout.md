# study1b and study2 implementation note

## Study 1b
The fishery substrate is now extended for mechanism analysis and broader robustness tests. The invasion pipeline logs requested harvest, realized harvest, audit rate, quota clipping, repeat-offender rate, closure activation, and stock recovery lag. Governance can now be compared across five top-down conditions: none, monitoring, monitoring plus sanctions, adaptive quota, and temporary closure.

The invasion loop also supports partner-mix presets and additional non-LLM injectors. This makes it possible to test whether the governance ranking is stable across ecological regimes, initial population composition, and adversarial pressure.

Key scripts:
- `python -m experiments.run_governance_ablation --condition-set study1b --partner-mix balanced`
- `python -m experiments.run_study1b --injector-mode mutation`
- `python -m experiments.summarize_study1b --runs-csv ... --output ...`

## Study 2
A second commons substrate now exists as a minimal harvest commons environment. Agents can communicate with neighbors, offer side-payments, and act under an optional government cap. This creates three governance families to compare: top-down only, bottom-up only, and hybrid.

Key script:
- `python -m experiments.run_harvest_study`

## Why this matters
The fishery is still the controlled identification environment, but the broader goal is governance for decentralized self-interested systems under uncertainty and strategic turnover. The harvest commons study is the first step away from fish-specific claims toward more general institution design for AI societies.
