# LLM inspector ladder: how to run

Plan and predictions: `protocol.md` (written before any judging; run changes in its Amendments). Results: `results/tables.md` (generated) and the judged replies in `results/judge/`.

All commands run from the repository root.

## 1. Datasets (already built; rebuild only to check)

**Harvest** (committed: `experiments/llm_overseer/data/harvest_judge.jsonl.gz`, 3,000 records). Needs the CPU SocialJax venv:

```bash
bash scripts/socialjax_local_setup.sh        # once: external/SocialJax and .venv-socialjax
PYTHONPATH=. .venv-socialjax/bin/python -m experiments.llm_overseer.make_harvest_dataset
```

About 25 seconds on an M1. Fixed seed, so a rebuild gives the same file (check with `git diff --stat`).

**BazaarBench** (not committed: CC-BY-4.0 texts are kept out of git; only the index is committed):

```bash
bash experiments/bazaarbench/download.sh     # once: about 258 MB into data/bazaarbench/raw (git-ignored)
python3 -m experiments.llm_overseer.make_bazaar_dataset
# -> data/bazaarbench/llm_judge/bazaar_judge.jsonl.gz (836 examples); experiments/llm_overseer/data/bazaar_index.tsv
```

## 2. Judging with Ollama (the main path)

Needs an Ollama daemon (`ollama serve`, default http://localhost:11434) and only the Python standard library plus numpy.

**Local tier** (Qwen2.5-Instruct, Ollama's default 4-bit weights):

```bash
for m in qwen2.5:0.5b-instruct qwen2.5:1.5b-instruct qwen2.5:3b-instruct qwen2.5:7b-instruct; do ollama pull $m; done
for m in qwen2.5-0.5b qwen2.5-1.5b qwen2.5-3b qwen2.5-7b; do
  PYTHONPATH=. python3 -m experiments.llm_overseer.run_judge --model $m
done
```

Each run judges the same 600 items (`overseer/llm/subset.py`) and writes `results/llm_overseer/judge/<tag>_direct.jsonl` (git-ignored; copies of the finished runs are in `notes/llm_overseer/results/judge/`). Runs resume where they stopped. Disk: the four models take about 7.5 GB; check `df -h` before pulling the 7B.

**Cloud tier** (gpt-oss through the local daemon, which must be signed in to Ollama; no key is read by this code). Cap: at most 2 cloud models, 600 items each, about 1.5M tokens in total. The token log is shared, so the cap holds across both models:

```bash
for m in gpt-oss-20b gpt-oss-120b; do
  PYTHONPATH=. python3 -m experiments.llm_overseer.run_judge --model $m --think low --max-tokens 2048 \
      --token-cap 1500000 --token-log results/llm_overseer/cloud_tokens.tsv
done
```

A 402/429 or a usage/credit message stops the run at once; 5 errors in a row stop it too.

**Other options:** `--mode reason` (brief reasoning, then "VERDICT: YES/NO"; default 768 output tokens), `--limit 50` (smoke test spread over every dataset and rendering), `--datasets harvest`, `--backend transformers --model Qwen/Qwen2.5-0.5B-Instruct` (CPU, no Ollama; needs `pip install torch transformers`).

## 3. Analysis

```bash
PYTHONPATH=. python3 -m experiments.llm_overseer.analyze --judge-dir results/llm_overseer/judge --out notes/llm_overseer/results
```

Writes `results/tables.md` (α, β, β_eat, confusion, tokens per check, C1 classes) and `results/summary.json`.

## 4. Harvest RL with each model's α and β (CREATE)

The generator reads the measured w = 1 numbers (α, and β on honest harvests) and writes the `phase2_runs.py` format:

```bash
PYTHONPATH=. python3 -m experiments.llm_overseer.harvest_runs --summary notes/llm_overseer/results/summary.json > runs_llm.tsv
# smaller: merge inspectors whose (α, β) round alike to 0.05, and give 1 seed to cells C1 predicts fail
PYTHONPATH=. python3 -m experiments.llm_overseer.harvest_runs --dedupe 0.05 --one-seed-if-fails > runs_llm_small.tsv
```

**The chosen list is committed:** `experiments/llm_overseer/runs_llm.tsv` (40 runs). It covers the four in-between or control inspectors: gpt-oss-20B on raw maps (α 0.10, β_eat 0.08); Qwen 3B with reasoning on the summary (0.00, 0.23); Qwen 7B with reasoning on raw maps (0.47, 0.04), which C1 says fails at c = 1 and protects at c = 2; and Qwen 7B direct on raw maps (1.00, 0.04) as a no-deterrence control with 1 seed. Inspectors with α ≈ β ≈ 0 repeat the existing perfect-inspector runs and are left out.

The trainer is unchanged (`--miss-rate`, `--false-alarm` exist since the mistakes test), so the existing array script runs these as they are. On CREATE (from the parent session; N = 40):

```bash
scp experiments/llm_overseer/runs_llm.tsv create:~/ippo/
sbatch --array=1-N%8 --exclude=arc-hpc-comp035,arc-hpc-comp040 \
    --export=ALL,RUNS=$HOME/ippo/runs_llm.tsv ~/ippo/ippo_overseer_array.sbatch
```

Outputs go to `/scratch/users/$USER/ippo_overseer/L_<model>_<rendering>_k<k>_c<c>_seed<s>/`. Each run is 1e8 steps (18–38 min on one GPU).

## 5. Optional: vLLM on CREATE (untested)

`scripts/create/llm_judge.sbatch` has the setup (a separate venv at `/scratch/users/$USER/vllm_venv`, weights in `/scratch/users/$USER/hf_cache`) and the submit commands for Qwen2.5-Instruct 0.5B–32B in bf16. It was not run: the plan changed to Ollama on 22 Nov.

## 6. Tests

```bash
PYTHONPATH=. .venv-socialjax/bin/python -m pytest -q tests/test_llm_overseer.py    # 20 tests; 1 needs JAX + SocialJax
```

They check, among other things, that counting the 12 cells in every raw map and reading every summary give back the wrapper's label exactly, and that the token cap stops a run.
