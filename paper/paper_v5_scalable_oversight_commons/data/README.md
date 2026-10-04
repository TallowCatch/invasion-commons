# Curated paper data

The `analysis/` tables and `provenance.json` are generated from the completed
`budgeted_reviewer_confirmation_v1` run by:

```sh
python -m experiments.export_reviewer_paper_data \
  --input-dir results/runs/budgeted_reviewer_confirmation_v1 \
  --output-dir paper/paper_v5_scalable_oversight_commons/data
```

They contain aggregate counts by independent population context, primary
decision counts, closed-loop summary means, and episode-level outcomes. The
export also computes paired differences by population context at six
inspections, using 4,000 fixed-seed bootstrap resamples for the summary
intervals. The tables are sufficient to regenerate the main decision figure
and the long-run paired table without the ignored raw run directory:

```sh
python -m experiments.plot_reviewer_decisions \
  --input-dir paper/paper_v5_scalable_oversight_commons/data \
  --output-dir paper/paper_v5_scalable_oversight_commons/figures
```

The `sources/` directory is the bounded public source-data bundle for this
draft. It contains the complete fresh reviewer confirmation run as one
compressed archive, selected historical Stage A and LLM tables, and a SHA-256
manifest. Unpack the confirmation cohort into `results/runs/` to rerun the
paper analysis and the post-hoc coupled-local replay. Its 1,920 blocks include
step-level traces; the small CSVs above remain convenient for figures.

```sh
mkdir -p results/runs
tar -xzf paper/paper_v5_scalable_oversight_commons/data/sources/budgeted_reviewer_confirmation_v1.tar.gz -C results/runs
python -m experiments.replay_coupled_local \
  --input-dir results/runs/budgeted_reviewer_confirmation_v1 \
  --output-dir results/runs/coupled_local_replay_reproduced
```

This is **not** an archive of every exploratory run in the repository's
history. The historical tables support the background discussion, while the
full archived cohort supports auditing the current matched-reviewer result.
Rerunning agent generation from scratch also requires the documented
environment and model versions; saved outcomes alone do not prove a fresh
independent replication.
