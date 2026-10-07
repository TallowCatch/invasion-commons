#!/usr/bin/env bash
# One GitHub Actions job of experiment L2 (protocol: notes/claude_audit_20261005/studies/L2_llm_agents/protocol.md, Amendment 1).
# Usage: bash scripts/l2_ci_step.sh check|pilot|run STORE_DIR MAX_MINUTES
# STORE_DIR holds the l2-results branch: claude_l2_v1/<model>/..., claude_l2_pilot_v2/<model>/..., gates.json
set -u
MODE="$1"; STORE="$2"; MAXMIN="${3:-330}"
next() { echo "$1" > "$STORE/NEXT"; }  # tells the workflow what to do next: done | wait | now
rm -f "$STORE/NEXT"  # a job that crashes leaves no NEXT, so it does not start another job
MODELS=("gpt-oss:120b-cloud" "gemma4:31b-cloud" "nemotron-3-super:cloud")
start=$(date +%s)
run() { PYTHONPATH=. python -m experiments.oversight.run_l2_llm_agents "$@"; }
if [ "$MODE" = "check" ]; then run check; exit $?; fi
if [ "$MODE" = "pilot" ]; then
  for m in "gemma4:31b-cloud" "nemotron-3-super:cloud"; do
    slug="${m//[:.]/_}"
    run pilot --model "$m" --out "$STORE/claude_l2_pilot_v2/$slug"; code=$?
    [ "$code" -eq 3 ] && { echo "usage limit during pilot; rerun later"; exit 0; }
  done
  exit 0
fi
# Finish any pilot gate that has not completed yet (resumable; stops at the usage limit like the full run).
for m in "gemma4:31b-cloud" "nemotron-3-super:cloud"; do
  slug="${m//[:.]/_}"
  if [ ! -f "$STORE/claude_l2_pilot_v2/$slug/pilot_gate.json" ]; then
    run pilot --model "$m" --out "$STORE/claude_l2_pilot_v2/$slug"; code=$?
    [ "$code" -eq 3 ] && { echo "usage limit during pilot; next job resumes"; next wait; exit 0; }
  fi
done
# Gate record (protocol rule: >= 95% valid first-try answers and mean comprehension >= 2 of 3).
# gpt-oss passed its pilot locally on 2026-10-07 (notes/claude_audit_20261005/runs/claude_l2_pilot_v1/);
# gemma4 and nemotron are judged from their rerun pilots in $STORE/claude_l2_pilot_v2 (Amendment 1).
python - "$STORE" <<'PY'
import json, os, sys
store = sys.argv[1]
path = os.path.join(store, "gates.json")
gates = json.load(open(path)) if os.path.exists(path) else {}
gates.setdefault("gpt-oss:120b-cloud", True)  # local pilot: valid 1.00, comprehension 3.0
for m in ("gemma4:31b-cloud", "nemotron-3-super:cloud"):
    f = os.path.join(store, "claude_l2_pilot_v2", m.replace(":", "_").replace(".", "_"), "pilot_gate.json")
    if m not in gates and os.path.exists(f):
        gates[m] = bool(json.load(open(f))["passed"])
json.dump(gates, open(path, "w"), indent=1)
print("gates:", gates)
PY
# full run: models whose gate passed, one after another, until the usage limit or the time limit
for m in "${MODELS[@]}"; do
  slug="${m//[:.]/_}"
  python - "$STORE/gates.json" "$m" <<'PY' || { echo "skip $m: pilot gate not passed (gates.json)"; continue; }
import json, sys
g = json.load(open(sys.argv[1])) if __import__("os").path.exists(sys.argv[1]) else {}
sys.exit(0 if g.get(sys.argv[2]) is True else 1)
PY
  [ -f "$STORE/claude_l2_v1/$slug/DONE" ] && { echo "$m complete"; continue; }
  left=$(( MAXMIN - ($(date +%s) - start) / 60 ))
  [ "$left" -le 10 ] && { echo "time limit reached"; next now; exit 0; }
  run full --model "$m" --out "$STORE/claude_l2_v1/$slug" --max-minutes "$left"; code=$?
  case $code in
    0) echo "$m complete"; continue ;;
    3) echo "usage limit reached; next job resumes after a wait"; next wait; exit 0 ;;
    5|6) echo "stopped (code $code); next job resumes"; next now; exit 0 ;;
    *) echo "$m failed with code $code"; next done; exit "$code" ;;
  esac
done
echo "all gated models complete"
next done
