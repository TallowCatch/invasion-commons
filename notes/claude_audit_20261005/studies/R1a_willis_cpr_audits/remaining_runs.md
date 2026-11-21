# R1a: grid runs still to do (all of them), with ready commands

**Update:** the grid is now running on GitHub Actions (workflow `r1a-willis-cpr.yml`, 56 jobs of one condition × 10 runs, results on branch `r1a-results`), including the conditions dropped by Amendment 3. Those remain an extension outside H1–H3. The commands below are for the **parked runner v2** (branch `r1a-runner-v2`; protocol Amendment 5), not for the runner the grid is using. To re-run missing runs now, re-dispatch the workflow, which runs only what is missing.

Written 14 October 2026, 15:45 UTC. Protocol: `protocol.md` (Amendments 1–4). **No grid run has been done yet.** The prediction, the pilot and the timing probes are done; R = 20 is fixed (Amendment 3); the grid runs on GitHub Actions lanes (Amendment 4).

## Set-up on a lane

```bash
git submodule update --init external/emergent_llm      # Willis et al., pinned at be78ff9
python3 -m pip install numpy pandas                     # nothing else is needed (matplotlib optional)
export PYTHONPATH=.:external/emergent_llm/src
R1A="python3 -m experiments.oversight.run_r1a_willis_audits"
```

The lane also needs `notes/claude_audit_20261005/runs/claude_r1a_v1/prediction.json` (committed), or `--prediction PATH`.

## The command

```bash
$R1A grid [selection] --shard I --of N --out DIR [--processes P] [--max-minutes M]
$R1A grid [selection] --list-jobs           # prints the job list with indices, then exits
```

- **Selection** (every lane must pass the same selection flags, or none):
  - `--runs 0-19` (default; R = 20; also `0-4,10`);
  - `--n 64 4`, `--q 0.1` (baselines have q = 0), `--e-mult 0 0.5 1.0 1.25 2.0` (multiples of e_pred(64) = 0.962; 0 = baseline), `--exempt yes|no|any`;
  - `--conditions NAME ...`;
  - `--include-dropped`: adds the conditions dropped by Amendment 3, listed *after* the analysed grid, so the analysed jobs keep their indices.
- **Sharding:** a lane runs the jobs whose index in the fixed job list satisfies `index % N == I`. The list is condition-major in the protocol's priority order, then run index, so each lane gets a mix of fast and slow jobs.
- **Output:** one file per job, `DIR/<condition>/run_XXX.json.gz`, written by atomic rename. A job whose file exists is skipped, never overwritten, so lanes can share one results branch. Logs go to `DIR/logs/shard_I_of_N.jsonl`. `DIR/grid_conditions.json` has the same content on every lane.
- **Time limit:** `--max-minutes M` starts no new job after M minutes, lets running jobs finish, and exits 0. Leave at least 30 minutes between M and the CI job's limit, since one n = 4 fined job takes about 19 minutes on an M1 core and probably longer on a hosted runner.
- **Completion:** `DIR/DONE_SHARD` (and `DIR/DONE_shard_I_of_N`, which does not collide between lanes) is written only when every job of the shard exists. Re-running the same shard command finishes whatever is missing.
- **Exit code:** 0, or 1 if any job raised an exception (logged with its traceback text).

Default output directory: `results/runs/claude_r1a_v1/runs`, which the analysis reads (`analyze_r1a --runs-dir DIR` for another).

## Cost per run (from the pilot and the probes, Amendment 3)

Single-process CPU on this machine (Apple M1), 2,000 generations:

| Run type | Seconds per run |
|---|---|
| n = 64, no oversight | ≈ 540 (pilot: 0.271 s per generation) |
| n = 64, audited | ≈ 560 |
| n = 4, no oversight | ≈ 670 |
| n = 4, audited | ≈ 1,130 (one audit draw per agent-round: 9,728 rounds per generation) |

The analysed grid totals about 181,600 CPU-seconds (50 CPU-hours) here. A hosted runner core is probably 1.2–1.5× slower, so about 60–75 CPU-hours. Example: 8 lanes × 4 processes ≈ 2.0–2.4 h each, inside the 6 h job limit.

## The analysed grid: 240 jobs (indices without `--include-dropped`)

| Job indices | Condition | n | q | e ÷ e_pred(64) | e | F | Final round exempt | Arm | s/run |
|---|---|---|---|---|---|---|---|---|---|
| 0–19 | `n64_base` | 64 | 0 | 0 | 0.000 | 0.000 | yes | baseline (H0) | 540 |
| 20–39 | `n4_base` | 4 | 0 | 0 | 0.000 | 0.000 | yes | baseline (H0) | 670 |
| 40–59 | `n64_q0.1_e1.25` | 64 | 0.1 | 1.25 | 1.202 | 12.025 | yes | primary (H1, H2) | 560 |
| 60–79 | `n64_q0.1_e0.5` | 64 | 0.1 | 0.5 | 0.481 | 4.810 | yes | primary | 560 |
| 80–99 | `n64_q0.1_e1.0` | 64 | 0.1 | 1 | 0.962 | 9.620 | yes | primary | 560 |
| 100–119 | `n64_q0.1_e2.0` | 64 | 0.1 | 2 | 1.924 | 19.239 | yes | primary | 560 |
| 120–139 | `n4_q0.1_e1.25` | 4 | 0.1 | 1.25 | 1.202 | 12.025 | yes | primary | 1,130 |
| 140–159 | `n4_q0.1_e0.5` | 4 | 0.1 | 0.5 | 0.481 | 4.810 | yes | primary | 1,130 |
| 160–179 | `n4_q0.1_e1.0` | 4 | 0.1 | 1 | 0.962 | 9.620 | yes | primary | 1,130 |
| 180–199 | `n4_q0.1_e2.0` | 4 | 0.1 | 2 | 1.924 | 19.239 | yes | primary | 1,130 |
| 200–219 | `n64_q0.1_e1.25_noexempt` | 64 | 0.1 | 1.25 | 1.202 | 12.025 | no | secondary | 560 |
| 220–239 | `n64_q0.1_e2.0_noexempt` | 64 | 0.1 | 2 | 1.924 | 19.239 | no | secondary | 560 |

Each condition runs 0–19; job index = 20 × (row number − 1) + run. `--list-jobs` prints all 240 lines.

### Ready commands

```bash
# 8 lanes: run this in each CI job of a matrix I = 0..7 (4 processes; no new job after 5 h)
$R1A grid --shard $I --of 8 --processes 4 --max-minutes 300 --out results/runs/claude_r1a_v1/runs
# one condition only, e.g. H2's cell with 25 runs
$R1A grid --n 64 --q 0.1 --e-mult 1.25 --exempt yes --runs 0-24 --out results/runs/claude_r1a_v1/runs
# the baselines only (H0)
$R1A grid --e-mult 0 --out results/runs/claude_r1a_v1/runs
```

Then, once every `DONE_shard_I_of_8` exists:

```bash
PYTHONPATH=.:external/emergent_llm/src python3 -m experiments.oversight.analyze_r1a
```

## Optional, outside the pre-registered grid: the 16 conditions dropped by Amendment 3

With `--include-dropped` the job list has 560 jobs; indices 0–239 are the analysed grid above and 240–559 are the dropped conditions (q = 0.1 at e = 0.75 and 1.5 × e_pred; all twelve q = 0.25 conditions). Run them only as a labelled extension, for example:

```bash
$R1A grid --include-dropped --conditions n64_q0.25_e1.25 n64_q0.25_e0.5 n64_q0.25_e1.0 n64_q0.25_e2.0 \
    --shard $I --of 4 --processes 4 --max-minutes 300
```

Cost at R = 20: n = 64, 8 conditions × 20 × 560 s; n = 4, 8 × 20 × 1,130 s; about 270,000 CPU-seconds (75 CPU-hours) here.
