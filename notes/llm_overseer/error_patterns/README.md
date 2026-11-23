# Patterned LLM inspector errors: how to rerun, and the CREATE job list

Protocol and predictions: `protocol.md`. Results: `results.md`. Generated fit tables: `fit_tables.md`, `fit.json`.

## 1. Offline pipeline (laptop)

```bash
# situation bank: trained stage 1 agents, every eat is one item (about 1 minute on the M1)
PYTHONPATH=. .venv-socialjax/bin/python -m experiments.llm_overseer.error_bank --params-root results/create_pull
# judge the stratified subset (resumes; cloud tokens logged to cloud_tokens.tsv, capped)
PYTHONPATH=. .venv/bin/python -m experiments.llm_overseer.judge_bank --model gpt-oss-20b --think low \
    --max-tokens 2048 --workers 4 --token-cap 2980000 --token-log notes/llm_overseer/error_patterns/cloud_tokens.tsv
PYTHONPATH=. .venv/bin/python -m experiments.llm_overseer.judge_bank --model qwen2.5-7b --mode reason \
    --num-ctx 2048 --limit 600
# fit alpha(s), beta(s); writes tables/<tag>.json (the trainer's --inspector-table), fit.json, fit_tables.md
PYTHONPATH=. .venv-socialjax/bin/python -m experiments.llm_overseer.fit_error_model
# job list
PYTHONPATH=. python3 -m experiments.llm_overseer.error_pattern_runs > experiments/llm_overseer/runs_error_patterns.tsv
```

Files: `experiments/llm_overseer/data/error_bank.jsonl.gz` (13,159 items, force-added), `judge/*.jsonl` (one line per verdict, with the reply text), `judge/subset.tsv` (the judging order), `cloud_tokens.tsv`.

## 2. The trainer option

`--inspector-table notes/llm_overseer/error_patterns/tables/<tag>.json` (Harvest, `--window 1`, with `--miss-rate` and `--false-alarm` left at 0). A check's miss chance is α(s) and its false-alarm chance β(s) for the checked agent's situation: apples within two squares of the apple it took (`near`), and, if the table has it, apples in the other cells of the 5 × 5 patch (`outer`), on the map before the move. Code: `overseer/socialjax/inspector_table.py`, `overseer/socialjax/decentralised.py`. Tests: `tests/test_inspector_table.py` (default path identical to c0f39d9; a constant table equals the coin flip exactly; features equal the LLM's patch; lookup and per-situation application).

New measures in every run's `metrics.npz` / `summary.json` (existing ones unchanged):
- `checks_on_violations`, `checks_on_honest_eats`: realised α = 1 − `detected` / `checks_on_violations`; realised β = `wrongly_stopped` / `checks_on_honest_eats`;
- `violations_by_near` (3: near 0, 1, 2) and `eats_by_near` (13): where agents over-harvest and eat. These test E6 directly (do learners move their over-harvests to where the inspector misses?).

## 3. CREATE job list (not submitted)

`experiments/llm_overseer/runs_error_patterns.tsv`: per inspector (gpt-oss-20B on raw maps; Qwen 7B + reasoning on raw maps), the patterned table vs a coin flip at the matched average α, β, at k ∈ {1, 7}, c ∈ {1, 2}, seeds 1–3, random checks, 1e8 steps from scratch. Names `E_<inspector>_<arm>_k<k>_c<c>_seed<s>`, arm `table` or `coin` (and `coinov`, the overseen-policies average, where it differs from the bank average by more than 0.05). See `results.md` §5 for the count and the matched numbers.

The runs need this branch's trainer (`--inspector-table`) and the table files, so the array script must run from a checkout of `claude/llm-error-patterns` (the script's `REPO` variable).

On CREATE (`WORK=/scratch/users/$USER/socialjax_work`, as in the array script):

```bash
# once: a checkout of this branch beside the default one (the table files are in it)
cd $WORK && git clone -b claude/llm-error-patterns https://github.com/TallowCatch/invasion-commons.git invasion-commons-errpat
cp $WORK/invasion-commons-errpat/experiments/llm_overseer/runs_error_patterns.tsv ~/ippo/
N=$(wc -l < ~/ippo/runs_error_patterns.tsv)
sbatch --array=1-$N%8 --exclude=arc-hpc-comp035,arc-hpc-comp040 \
  --export=ALL,RUNS=$HOME/ippo/runs_error_patterns.tsv,REPO=$WORK/invasion-commons-errpat ~/ippo/ippo_overseer_array.sbatch
```

Outputs go to `/scratch/users/$USER/ippo_overseer/E_*/`. Each run is 1e8 steps (18–42 min on one GPU, as the L_ runs). To analyse: mean of the last 10 updates per seed, as in `../rl_results.md`, plus realised α, β, and `violations_by_near` against the coin flip.
