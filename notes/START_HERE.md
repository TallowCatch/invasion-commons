# Start here (budgeted overseer work, from Nov 2026)

**The project in three lines.** An overseer that can only check a few agents at a time, studied in sequential social dilemmas, which are games where everyone gains by holding back but each is tempted to take more. Four questions:
1. How few checks are enough?
2. Whom should they target?
3. What happens when there are several overseers, each seeing only its own region (decentralised, in the sense of Du et al. 2022 and Ma et al. 2024)?
4. How capable must an LLM overseer be?

## Which document for which job

| I want to… | Read |
|---|---|
| see what we've found, with numbers | `OVERSEER_TRACKER.md` |
| see the problem written as formulas (one overseer) | `phd_plan/2026-11-21_general_formulation.md` |
| see the decentralised version (several overseers) | `phd_plan/2026-11-23_decentralised_overseer_formulation.md` |
| know what's new against existing work | `phd_plan/2026-11-22_whats_new.md` |
| see the critique and the open gaps | `phd_plan/2026-11-22_devils_advocate_formulation.md` |
| see the plan and the choice of testbeds | `phd_plan/2026-11-21_way_forward.md`, `phd_plan/2026-11-23_testbed_candidates.md` |
| run the Harvest learning-agent experiments | `socialjax/README.md`, `socialjax/phase2_ippo/` (each study has protocol.md written before the runs, then results.md) |
| run the decentralised experiments | `socialjax/DECENTRALISED_README.md`, `socialjax/DECENTRALISED_SUMMARY.md` |
| see the BazaarBench (LLM marketplace) replay | `bazaarbench/offline_replay/SUMMARY.md` |
| see the history of everything | `../PROJECT_HISTORY.md`, `claude_audit_20261005/FINDINGS_AND_IMPLICATIONS.md` |

## From clone to first run

1. `bash scripts/socialjax_local_setup.sh`. This clones SocialJax (commit 18e2171) into `external/SocialJax` and makes `.venv-socialjax` with CPU JAX.
2. `.venv-socialjax/bin/python -m pytest -q tests/test_socialjax_overseer.py tests/test_decentralised_overseer.py`. Expect all to pass.
3. Run a bot preview on the CPU, for example `python -m experiments.socialjax.run_decentralised_bots`. See its README for the arguments.
4. For GPU training on CREATE, follow `socialjax/README.md` (the CREATE section) and `DECENTRALISED_README.md`. The job script is `scripts/create/ippo_overseer_array.sbatch`; it reads line `SLURM_ARRAY_TASK_ID` of a run list.

## Gotchas

**On CREATE:**
- **Login needs MFA** approved for your current address at portal.arc.kcl.ac.uk/mfa. Without it, ssh is reset before login.
- **Nodes arc-hpc-comp035 and arc-hpc-comp040 have broken GPUs.** Jobs there silently ran on CPU or crashed, so always pass `--exclude=arc-hpc-comp035,arc-hpc-comp040`.
- **Use the `gpu` partition.** `interruptible_gpu` gets preempted, and a run loses everything because there's no mid-run checkpoint.
- **Short time limits start sooner.** Measured times for a 1e8-step run:

  | Run | Time |
  |---|---|
  | central Harvest | about 20–40 min |
  | Harvest with sharing | about 70 min |
  | Coins | about 18 min |
  | Clean Up | about 55 min |

- **The 14-agent Harvest needs `--num-envs 128`,** or it runs out of GPU memory.

**In SocialJax:** most games default to a shared reward (`shared_rewards=True`), which removes the dilemma. Our wrappers set it to False, and any new game must too.

**Units:**
- One apple is worth 7 reward in SocialJax, so a penalty of P apples removes 7P reward.
- The **average penalty** per over-harvest is (k/N) × P. It is **not** k × P, which is 7 times larger in Harvest; an early version of the notes had this wrong.

**Code changes to know about:**
- **False alarms** (since 21 Nov) fall only on honest harvests, not on any checked agent.
- **One region in the decentralised overseer** reproduces the central overseer exactly (tested).

**Results:**
- Agents share one network unless a run says otherwise, so "repeat offenders" are not individuals.
- BazaarBench results are offline replays: the traders never react to the overseer.
