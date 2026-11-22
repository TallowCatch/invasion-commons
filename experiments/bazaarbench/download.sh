#!/bin/bash
# Download the two BazaarBench cells used by the offline overseer replay.
#
# Source: HuggingFace dataset BazaarBench/bazaarbench-analysis-v2-gpt5-judge (CC-BY-4.0).
# Cells: base market GPT-5.5, tested model gpt-oss-120b, ordinary (L1) and exploit (L3)
# instructions. About 270 MB in total. The judgments/shards.ndjson files repeat the
# per-bundle decisions already in judgments/records.ndjson, so they are skipped.
#
# Usage: bash experiments/bazaarbench/download.sh   (from the repo root)
# Output goes to data/bazaarbench/raw/<cell>/, which is git-ignored.
set -euo pipefail
BASE=https://huggingface.co/datasets/BazaarBench/bazaarbench-analysis-v2-gpt5-judge/resolve/main
OUT=data/bazaarbench/raw
CELLS="${CELLS:-gpt55_L1_gptoss120b gpt55_L3_gptoss120b}"
mkdir -p "$OUT"
for f in README.md ANALYSIS.md RELEASE.json registry.json; do
  [ -s "$OUT/$f" ] || curl -sL --retry 3 -m 600 "$BASE/$f" -o "$OUT/$f"
done
for c in $CELLS; do
  mkdir -p "$OUT/$c"
  for f in "judgments/$c/records.ndjson" "judgments/$c/ledger.json" \
           "bundles/$c/bundles.ndjson" "bundles/$c/ledger.json" \
           "plans/$c/plan.json" "status/$c.json"; do
    dest="$OUT/$c/$(echo "$f" | cut -d/ -f1)_$(basename "$f")"
    [ -s "$dest" ] || curl -sL --retry 3 -m 1800 "$BASE/$f" -o "$dest"
    echo "$(wc -c < "$dest") $dest"
  done
done
du -sh "$OUT"
