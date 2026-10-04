#!/usr/bin/env bash
set -euo pipefail

# Generate Harvest LLM strategy banks on a temporary cloud GPU pod.
# This script assumes Ollama is installed and reachable on localhost:11434.

MODELS="${MODELS:-qwen2.5:14b llama3.1:8b}"
TARGET_PER_PAIR="${TARGET_PER_PAIR:-32}"
MAX_ATTEMPTS_PER_PAIR="${MAX_ATTEMPTS_PER_PAIR:-256}"
REFERENCE_TIER="${REFERENCE_TIER:-medium_h1}"
TEMPERATURE="${TEMPERATURE:-0.30}"
TIMEOUT_S="${TIMEOUT_S:-180}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-650}"
RUN_ID="${RUN_ID:-stageB32}"
OUTPUT_DIR="${OUTPUT_DIR:-results/runs/showcase/curated}"
BASE_SEED="${BASE_SEED:-211}"

mkdir -p "${OUTPUT_DIR}"

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama is not responding on localhost:11434."
  echo "Start it in another shell with: ollama serve"
  exit 1
fi

model_label() {
  printf "%s" "$1" | tr ':./' '___' | tr -cd '[:alnum:]_'
}

i=0
for model in ${MODELS}; do
  label="$(model_label "${model}")"
  seed="$((BASE_SEED + i * 12))"
  prefix="${OUTPUT_DIR}/harvest_llm_bridge_cloud_${label}_${RUN_ID}"
  echo "Pulling model: ${model}"
  ollama pull "${model}"
  echo "Generating strategy bank: ${model}"
  python -m experiments.build_harvest_strategy_bank \
    --providers ollama \
    --models "${model}" \
    --attitudes cooperative,exploitative \
    --target-per-pair "${TARGET_PER_PAIR}" \
    --max-attempts-per-pair "${MAX_ATTEMPTS_PER_PAIR}" \
    --reference-tier "${REFERENCE_TIER}" \
    --seed "${seed}" \
    --output-prefix "${prefix}" \
    --temperature "${TEMPERATURE}" \
    --timeout-s "${TIMEOUT_S}" \
    --max-output-tokens "${MAX_OUTPUT_TOKENS}" \
    --progress-every 8 \
    --partial-save-every 1
  i="$((i + 1))"
done

archive="${OUTPUT_DIR}/harvest_llm_bridge_cloud_${RUN_ID}_outputs.tgz"
tar -czf "${archive}" "${OUTPUT_DIR}"/harvest_llm_bridge_cloud_*_"${RUN_ID}"_*.csv
echo "Saved cloud output archive: ${archive}"
