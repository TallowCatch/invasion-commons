#!/usr/bin/env bash
set -euo pipefail

# Evaluate one or more generated Harvest LLM strategy banks through the local
# governance-map pipeline.

BANK_GLOB="${BANK_GLOB:-results/runs/showcase/curated/harvest_llm_bridge_cloud_*_stageB32_bank.csv}"
SCENARIOS="${SCENARIOS:-community_irrigation,forest_co_management}"
CONDITIONS="${CONDITIONS:-none,bottom_up_only,top_down_only,hybrid}"
EXPLOITATIVE_SHARES="${EXPLOITATIVE_SHARES:-0.0,0.5,1.0}"
N_POPULATIONS="${N_POPULATIONS:-40}"
EVALUATION_SEEDS="${EVALUATION_SEEDS:-8}"
POPULATION_SIZE="${POPULATION_SIZE:-6}"
BASE_SEED="${BASE_SEED:-239}"

shopt -s nullglob
banks=( ${BANK_GLOB} )
if [ "${#banks[@]}" -eq 0 ]; then
  echo "No bank CSVs matched: ${BANK_GLOB}"
  exit 1
fi

i=0
for bank_csv in "${banks[@]}"; do
  stem="${bank_csv%_bank.csv}"
  seed="$((BASE_SEED + i * 12))"
  output_prefix="${stem}_map"
  echo "Evaluating bank: ${bank_csv}"
  python -m experiments.archive.harvest_2026q2.run_harvest_llm_governance_map \
    --bank-csv "${bank_csv}" \
    --scenarios "${SCENARIOS}" \
    --conditions "${CONDITIONS}" \
    --governance-friction-regime ideal \
    --exploitative-shares "${EXPLOITATIVE_SHARES}" \
    --n-populations "${N_POPULATIONS}" \
    --population-size "${POPULATION_SIZE}" \
    --evaluation-seeds "${EVALUATION_SEEDS}" \
    --seed "${seed}" \
    --output-prefix "${output_prefix}"
  i="$((i + 1))"
done
