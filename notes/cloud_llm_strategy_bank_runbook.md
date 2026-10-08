# Cloud Runbook: Low-Cost LLM Strategy Banks

This runbook is for the Harvest LLM bridge. The purpose is narrow: rent a GPU briefly, generate structured Harvest strategy banks with stronger open-weight models, download the CSV outputs, and stop the machine. The cloud run is not part of the scientific contribution. It is only a way to avoid crashing the laptop and to test stronger model-generated strategies.

## Why cloud now

The local MacBook Air can run small models, but it is tight on RAM and disk. That is enough for smoke tests with small open models, but it is not a good setup for repeated strategy generation with larger models. A short GPU rental lets us run stronger open-weight models such as Qwen 14B, Llama 8B, or Gemma 12B while keeping the experiment reproducible and cheap.

The research claim should remain focused on governance and oversight. The cloud model only supplies candidate strategies. Those strategies are still validated, deduplicated, stored as CSV rows, and evaluated in the same Harvest benchmark.

## Cheapest practical option

Use a short-lived GPU pod with at least 24 GB VRAM.

Recommended first target:

`RunPod Community Cloud, RTX A5000 or L4`

Reason:

These are cheap enough for a short run and have enough VRAM for 8B to 14B quantized models through Ollama. RunPod lists Community Cloud prices such as RTX A5000 around `$0.27/hr`, L4 around `$0.39/hr`, A40 around `$0.44/hr`, RTX 3090 around `$0.46/hr`, and RTX 4090 around `$0.69/hr` on its pricing page. Storage can add small extra cost, so the pod should be stopped or deleted after downloading the CSVs.

Absolute cheapest alternative:

`Vast.ai interruptible instance`

Reason:

Vast.ai uses live market pricing and offers interruptible instances that are advertised as more than 50% cheaper than on-demand. This is cheaper but more annoying. It is acceptable for this task because the strategy-bank builder writes partial CSVs frequently, so interruption is not catastrophic.

## Model order

Run these in this order:

1. `qwen2.5:14b`
2. `llama3.1:8b`
3. `gemma3:12b`, only if the first two work cleanly

Why this order:

Qwen is strong at structured JSON-style output, which matters because the strategy format is strict. Llama gives a second model family. Gemma should move to cloud because it was unstable on the laptop.

For a first cloud pilot, use `32` cooperative and `32` exploitative strategies per model. If the outputs are valid, diverse, and behaviorally separated, scale the best two models to `64 + 64`.

## Cloud setup

Start a GPU pod with Ubuntu and SSH access. Then run:

```bash
sudo apt-get update
sudo apt-get install -y git curl python3 python3-venv python3-pip
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama:

```bash
ollama serve
```

In a second shell on the same pod:

```bash
git clone https://github.com/TallowCatch/invasion-commons.git
cd invasion-commons
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scipy pytest requests
```

Pull the first model:

```bash
ollama pull qwen2.5:14b
```

Generate the bank:

```bash
python -m experiments.build_harvest_strategy_bank \
  --providers ollama \
  --models qwen2.5:14b \
  --attitudes cooperative,exploitative \
  --target-per-pair 32 \
  --max-attempts-per-pair 256 \
  --reference-tier medium_h1 \
  --seed 211 \
  --output-prefix results/runs/showcase/curated/harvest_llm_bridge_cloud_qwen14b_stageB32 \
  --temperature 0.30 \
  --timeout-s 120 \
  --max-output-tokens 650 \
  --progress-every 8 \
  --partial-save-every 1
```

Repeat for Llama:

```bash
ollama pull llama3.1:8b
python -m experiments.build_harvest_strategy_bank \
  --providers ollama \
  --models llama3.1:8b \
  --attitudes cooperative,exploitative \
  --target-per-pair 32 \
  --max-attempts-per-pair 256 \
  --reference-tier medium_h1 \
  --seed 223 \
  --output-prefix results/runs/showcase/curated/harvest_llm_bridge_cloud_llama31_8b_stageB32 \
  --temperature 0.30 \
  --timeout-s 120 \
  --max-output-tokens 650 \
  --progress-every 8 \
  --partial-save-every 1
```

Only try Gemma after those succeed:

```bash
ollama pull gemma3:12b
python -m experiments.build_harvest_strategy_bank \
  --providers ollama \
  --models gemma3:12b \
  --attitudes cooperative,exploitative \
  --target-per-pair 32 \
  --max-attempts-per-pair 320 \
  --reference-tier medium_h1 \
  --seed 229 \
  --output-prefix results/runs/showcase/curated/harvest_llm_bridge_cloud_gemma3_12b_stageB32 \
  --temperature 0.35 \
  --timeout-s 180 \
  --max-output-tokens 650 \
  --progress-every 8 \
  --partial-save-every 1
```

## Files to download

Download only the generated CSV files:

```text
results/runs/showcase/curated/harvest_llm_bridge_cloud_qwen14b_stageB32_bank.csv
results/runs/showcase/curated/harvest_llm_bridge_cloud_qwen14b_stageB32_summary.csv
results/runs/showcase/curated/harvest_llm_bridge_cloud_llama31_8b_stageB32_bank.csv
results/runs/showcase/curated/harvest_llm_bridge_cloud_llama31_8b_stageB32_summary.csv
```

If Gemma is run:

```text
results/runs/showcase/curated/harvest_llm_bridge_cloud_gemma3_12b_stageB32_bank.csv
results/runs/showcase/curated/harvest_llm_bridge_cloud_gemma3_12b_stageB32_summary.csv
```

Stop or delete the cloud pod immediately after downloading the files.

## Local evaluation after download

After the bank CSVs are copied into the local repo, run:

```bash
python -m experiments.run_harvest_llm_governance_map \
  --bank-csv results/runs/showcase/curated/harvest_llm_bridge_cloud_qwen14b_stageB32_bank.csv \
  --scenarios community_irrigation,forest_co_management \
  --conditions none,bottom_up_only,top_down_only,hybrid \
  --governance-friction-regime ideal \
  --exploitative-shares 0.0,0.5,1.0 \
  --n-populations 40 \
  --population-size 6 \
  --evaluation-seeds 8 \
  --seed 239 \
  --output-prefix results/runs/showcase/curated/harvest_llm_bridge_cloud_qwen14b_stageB32_map
```

Run the same command for each downloaded bank by changing `--bank-csv`, `--seed`, and `--output-prefix`.

## Decision gate

Keep a model for paper evidence only if all of these hold:

1. It reaches at least `32 + 32` accepted unique strategies.
2. The duplicate rate is not extreme.
3. Cooperative and exploitative banks produce behaviorally different outcomes under no oversight.
4. The governance map separates local, global signal, and hybrid oversight.

If a model produces valid JSON but no behavioral separation, report it as a diagnostic model, not as main evidence.

## Paper framing

Write this as:

The LLM bridge uses open-weight models to generate structured strategy populations. The generated strategies are validated and evaluated inside the same Harvest oversight benchmark. This tests whether the Stage A governance findings survive when strategies are model-generated rather than hand-designed or search-generated.

Do not write this as:

Live LLM agents were deployed in the environment.

Do not make the cloud provider part of the methodology. The method is structured model-generated strategy evaluation. The cloud machine is only compute.
