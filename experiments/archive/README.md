# Archived experiment scripts

These scripts belong to earlier studies. They are not part of the current
paper (`paper/paper_v5_scalable_oversight_commons`) or of the current oversight
experiments (R1, S1, S1b, S2). They were moved here on 5 October 2026 (`harvest_2026q2/` in a second step the same day). They
were not edited, apart from changing their own `experiments.*` module paths.

| Folder | What it holds |
| --- | --- |
| `fishery_2026q1/` | The Fishery study from February–March 2026: invasion runs, the governance ablation, Study 1b, the paper v1/v2 summaries and figures, and Fishery PPO. |
| `harvest_2026q2/` | Earlier Harvest work: Stage A invasion matrix and threshold-replay shards, the LLM strategy bridge, Harvest RL, the actor-pressure pilot, the decision suite, and Clean Up. Some tests and GitHub workflows still run these, with updated paths. |
| `oneoff/` | GIF makers, the showcase report, talk figures, the results organiser, and the LLM setup check. |

Run them from the repository root as before, with the longer module path.
For example:

```bash
python -m experiments.archive.fishery_2026q1.run_single
```

Config paths such as `experiments/configs/base.yaml` are relative to the
repository root, so they are unchanged.

Notes written before the move (for example `notes/cycle_logs/`) still use
the old paths, such as `experiments.run_invasion`. The full old → new table is
in `experiments/README.md`.
