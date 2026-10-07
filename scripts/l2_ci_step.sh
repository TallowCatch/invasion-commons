#!/usr/bin/env bash
# One GitHub Actions job of experiment L2 (protocol: notes/claude_audit_20261005/studies/L2_llm_agents/protocol.md, Amendments 1-2).
# Usage: bash scripts/l2_ci_step.sh check|run STORE_DIR MAX_MINUTES LANE
# STORE_DIR holds the l2-results branch: claude_l2_v1/<model>/..., claude_l2_pilot_v2/<model>/..., gates.json
# Amendment 2 (Ollama Pro, 3 models at a time): three lanes run in parallel, each working through its own list of
# units in order. A unit is a model (all 10 contexts) or MODEL@LO-HI (only those contexts; files get the suffix _ctxLO-HI).
# No unit appears in two lanes, so parallel jobs never write the same file. Rebalancing = editing the lists below
# between jobs (never while a lane is running, or two lanes could run the same game).
# Amendment 3: the memory cell EM is held back from this run until its interface fault is fixed.
set -u
MODE="$1"; STORE="$2"; MAXMIN="${3:-330}"; LANE="${4:-A}"
LANE_A=("gpt-oss:120b-cloud" "nemotron-3-super:cloud@5-7")
LANE_B=("gemma4:31b-cloud" "mistral-large-3:675b-cloud" "nemotron-3-super:cloud@8-9")
LANE_C=("nemotron-3-super:cloud@0-4")
ADDED_PILOTS=("mistral-large-3:675b-cloud")  # pilot gate run before the full run (Amendments 2, 4)
next() { echo "$1" > NEXT; }  # tells the workflow what to do next: done | wait | now (outside the store, never committed)
rm -f NEXT "$STORE/NEXT"  # (store/NEXT was used before Amendment 2) a job that crashes leaves no NEXT, so it does not start another job
run() { PYTHONPATH=. python -m experiments.oversight.run_l2_llm_agents "$@"; }
if [ "$MODE" = "check" ]; then run check; exit $?; fi
# Save progress to the l2-results branch every 20 minutes while the job runs, so it is visible early.
if [ -d "$STORE/.git" ]; then
  ( while sleep 1200; do bash scripts/l2_save.sh "$STORE" "L2 progress (lane $LANE, in job)"; done ) &
  SAVER=$!
  trap 'kill $SAVER 2>/dev/null' EXIT
fi
case "$LANE" in A) UNITS=("${LANE_A[@]}") ;; B) UNITS=("${LANE_B[@]}") ;; C) UNITS=("${LANE_C[@]}") ;;
  *) echo "unknown lane $LANE"; exit 2 ;; esac
start=$(date +%s)
left() { echo $(( MAXMIN - ($(date +%s) - start) / 60 )); }
# Gate record (protocol rule: >= 95% valid first-try answers and mean comprehension >= 2 of 3); written only if it changes.
update_gates() {
python - "$STORE" <<'PY'
import json, os, sys
store = sys.argv[1]
path = os.path.join(store, "gates.json")
gates = json.load(open(path)) if os.path.exists(path) else {}
old = dict(gates)
gates.setdefault("gpt-oss:120b-cloud", True)  # local pilot 2026-10-07: valid 1.00, comprehension 3.0
for m in ("gemma4:31b-cloud", "nemotron-3-super:cloud", "mistral-large-3:675b-cloud"):
    f = os.path.join(store, "claude_l2_pilot_v2", m.replace(":", "_").replace(".", "_"), "pilot_gate.json")
    if m not in gates and os.path.exists(f):
        gates[m] = bool(json.load(open(f))["passed"])
if gates != old:
    json.dump(gates, open(path, "w"), indent=1)
print("gates:", gates)
PY
}
gate() {  # 0 = passed, 1 = failed, 2 = no gate yet
python - "$STORE/gates.json" "$1" <<'PY'
import json, os, sys
g = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}
sys.exit(2 if sys.argv[2] not in g else (0 if g[sys.argv[2]] is True else 1))
PY
}
update_gates
for u in "${UNITS[@]}"; do
  m="${u%@*}"; ctx=""; sfx=""
  [ "$u" != "$m" ] && { ctx="${u#*@}"; sfx="_ctx$ctx"; }
  slug="${m//[:.]/_}"
  [ -f "$STORE/claude_l2_v1/$slug/DONE$sfx" ] && { echo "$u complete"; continue; }
  gate "$m"; g=$?
  if [ "$g" -eq 2 ] && [[ " ${ADDED_PILOTS[*]} " == *" $m "* ]]; then
    echo "pilot gate for $m"
    run pilot --model "$m" --out "$STORE/claude_l2_pilot_v2/$slug"; code=$?
    [ "$code" -eq 3 ] && { echo "usage limit during pilot; next job resumes"; next wait; exit 0; }
    update_gates; gate "$m"; g=$?
  fi
  [ "$g" -ne 0 ] && { echo "skip $u: pilot gate not passed (gates.json)"; continue; }
  # no new game starts in the last 35 minutes, so a slow game (nemotron ~23 min) ends before the 350-minute job timeout
  [ "$(left)" -le 40 ] && { echo "time limit reached"; next now; exit 0; }
  args=(full --model "$m" --out "$STORE/claude_l2_v1/$slug" --max-minutes "$(( $(left) - 35 ))" --skip-cells EM)
  [ -n "$ctx" ] && args+=(--contexts "$ctx")
  run "${args[@]}"; code=$?
  case $code in
    0) echo "$u complete"; continue ;;
    3) echo "usage limit reached; next job resumes after a wait"; next wait; exit 0 ;;
    5|6) echo "stopped (code $code); next job resumes"; next now; exit 0 ;;
    *) echo "$u failed with code $code"; next done; exit "$code" ;;
  esac
done
echo "lane $LANE complete"
next done
