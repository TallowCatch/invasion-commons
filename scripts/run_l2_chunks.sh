#!/usr/bin/env bash
# Run experiment L2 in chunks on the Ollama Cloud free plan, one model after another.
# Protocol: notes/claude_audit_20261005/studies/L2_llm_agents/protocol.md
#
# Needs: the Ollama app running and signed in (`ollama signin`), and the repo's .venv.
# Usage (from the repository root):   bash scripts/run_l2_chunks.sh [WAIT_MINUTES]
# Stop any time with Ctrl-C; rerunning resumes at the next unfinished episode.
# Exit codes of the runner: 0 = model done, 3 = usage limit (wait and retry), 4 = token cap reached, other = error.
set -u
cd "$(dirname "$0")/.."
WAIT_MIN="${1:-45}"
OUT="results/runs/claude_l2_v1"
MODELS=("gpt-oss:120b-cloud" "gemma4:31b-cloud" "nemotron-3-super:cloud")
source .venv/bin/activate
mkdir -p "$OUT"

for m in "${MODELS[@]}"; do
  slug="${m//[:.]/_}"
  while true; do
    if [ -f "$OUT/$slug/DONE" ]; then echo "[$(date '+%F %T')] $m already complete"; break; fi
    echo "[$(date '+%F %T')] running $m"
    caffeinate -i env PYTHONPATH=. python -m experiments.oversight.run_l2_llm_agents full --model "$m" --out "$OUT/$slug" \
      2>&1 | tee -a "$OUT/$slug.log"
    code=${PIPESTATUS[0]}
    if [ "$code" -eq 0 ]; then echo "[$(date '+%F %T')] $m complete"; break; fi
    if [ "$code" -eq 3 ]; then
      echo "[$(date '+%F %T')] usage limit reached for $m; waiting $WAIT_MIN minutes before resuming"
      caffeinate -i sleep $((WAIT_MIN * 60))
      continue
    fi
    echo "[$(date '+%F %T')] $m stopped with exit code $code; not retrying automatically. See $OUT/$slug.log"
    exit "$code"
  done
done
echo "[$(date '+%F %T')] L2 finished for all models"
