# Harvest Oversight Gap Stage A Closeout

## Purpose

Stage A tests the current scalable-oversight framing in Harvest Commons. The question is how local oversight, a global oversight signal, and hybrid oversight behave when actor capability rises relative to overseer capability.

Actor capability is varied through the strength of the strategy entrant process. Low actor capability uses mutation entrants. Medium and high actor capability use search-generated entrants with larger search budgets and longer internal evaluation. Overseer capability is varied through detection recall, enforcement delay, targeting capacity, and budget cost.

## Run Status

The Stage A run completed successfully and produced the expected summary tables, figures, and concrete episode trace.

The main table contains 72 condition rows. The ranking table contains 18 decision cells, corresponding to two stress settings, three actor capability levels, and three overseer capability levels.

## Main Result

Hybrid oversight is the most frequent top-ranked architecture, but it does not win every cell.

Hybrid oversight ranks first in 13 of 18 decision cells. Local oversight ranks first in four cells. The global signal ranks first in one cell.

The high-coupling commons setting is clean: hybrid oversight ranks first in all 9 cells. The moderate-coupling commons setting is mixed: hybrid oversight ranks first in four cells, local oversight in four cells, and the global signal in one cell.

This means Stage A should be written as an architecture-comparison result under capability pressure, not as a claim that one oversight architecture always dominates.

## Safety Metrics

The capability-gap dashboard separates two safety outcomes.

Global unsafe rate measures how often the overall Harvest state becomes unsafe. Local-pass/global-fail rate measures how often local action checks pass while the aggregate state becomes unsafe.

No oversight and local oversight leave higher global unsafe rates than architectures with a global signal. Hybrid oversight keeps global unsafe rate low across most capability gaps. The global signal shows a rising local-pass/global-fail rate as actor capability exceeds overseer capability.

This supports the scalable-oversight framing because it shows that local checks can miss system-level failure.

## Concrete Failure Case

The case extractor found an episode trace under local oversight in the high-coupling setting with a limited overseer. The trace contains 29 local-pass/global-fail steps.

This case is useful for explanation. It shows an episode in which local action checks pass at multiple steps while mean patch health falls below the global safety threshold. The paper should use this as the concrete example so the result is not only presented as a matrix.

## Files

Main summary:

`results/runs/showcase/curated/harvest_oversight_gap_stageA_summary.md`

Main result table:

`results/runs/showcase/curated/harvest_oversight_gap_stageA_table.csv`

Architecture ranking:

`results/runs/showcase/curated/harvest_oversight_gap_stageA_ranking.csv`

Capability-gap dashboard:

`results/runs/showcase/curated/harvest_oversight_gap_stageA_oversight_gap.png`

Winner map:

`results/runs/showcase/curated/harvest_oversight_gap_stageA_oversight_winners.png`

Concrete case trace:

`results/runs/showcase/curated/harvest_oversight_gap_stageA_oversight_case_trace.png`

Concrete case summary:

`results/runs/showcase/curated/harvest_oversight_gap_stageA_oversight_case_summary.md`

## Interpretation For The Next Manuscript

The clean paper claim is:

This work evaluates oversight architectures in a shared-resource multi-agent benchmark as actor capability rises relative to overseer capability.

The result should be described carefully. Hybrid oversight is strongest overall and especially strong in high-coupling commons. In moderate-coupling commons, the ranking depends on actor capability and overseer capability. Architectures with a global signal reduce global unsafe states, while the local-pass/global-fail metric shows why local action checks alone can miss aggregate failure.

## Next Step

The next bridge is LLM-generated structured strategies. The project should not move directly to live LLM agents. The first LLM module should generate valid Harvest strategy specifications, sample them into the same oversight matrix, and test whether Stage A patterns carry over when strategies are produced by language models.
