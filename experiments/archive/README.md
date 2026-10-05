# Archived experiment scripts

These scripts belong to earlier studies. They are not part of the current
paper (`paper/paper_v5_scalable_oversight_commons`) or of the current oversight
experiments (R1, S1, S1b, S2). They were moved here on 5 October 2026. They
were not edited, apart from changing their own `experiments.*` module paths.

| Folder | What it holds |
| --- | --- |
| `fishery_2026q1/` | The Fishery study from February–March 2026: invasion runs, the governance ablation, Study 1b, the paper v1/v2 summaries and figures, and Fishery PPO. |
| `oneoff/` | GIF makers, the showcase report, talk figures, the results organiser, and the LLM setup check. |

Run them from the repository root as before, with the longer module path.
For example:

```bash
python -m experiments.archive.fishery_2026q1.run_single
```

Config paths such as `experiments/configs/base.yaml` are relative to the
repository root, so they are unchanged.

Notes written before the move (for example `notes/cycle_logs/`) still use
the old paths, such as `experiments.run_invasion`. Read
`experiments.<name>` there as `experiments.archive.<folder>.<name>`.
