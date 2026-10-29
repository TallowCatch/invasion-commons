#!/usr/bin/env bash
# One lane of L4/L5 on this Mac (protocols: studies/L4_llm_identification, studies/L5_llm_families).
# Usage: bash scripts/l4_lane.sh LANE_NAME "STUDY STAGE MODEL [LO-HI]" ["STUDY STAGE MODEL [LO-HI]" ...]
# Each unit runs to completion before the next. Usage limit (exit 3): wait 15 min and resume. Network (5): wait 2 min.
# Budget cap (4) or any other failure: the lane stops and says why. The runner prints progress only, never outcomes.
set -u
LANE="$1"; shift
LOG="results/runs/l4_lanes/lane_${LANE}.log"; mkdir -p results/runs/l4_lanes
for unit in "$@"; do
  read -r study stage model ctx <<< "$unit"
  args=(stage"$stage" --study "$study" --model "$model"); [ -n "${ctx:-}" ] && args+=(--contexts "$ctx")
  for attempt in $(seq 1 200); do
    echo "$(date -u +%FT%TZ) lane $LANE: $unit attempt $attempt" >> "$LOG"
    PYTHONPATH=. python3 -m experiments.oversight.run_l4_llm_identification "${args[@]}" >> "$LOG" 2>&1
    code=$?
    case $code in
      0) echo "$(date -u +%FT%TZ) lane $LANE: $unit DONE" >> "$LOG"; break ;;
      3) sleep 900 ;;
      5|6) sleep 120 ;;
      *) echo "$(date -u +%FT%TZ) lane $LANE: $unit STOPPED (exit $code)" >> "$LOG"; break ;;
    esac
  done
done
echo "$(date -u +%FT%TZ) lane $LANE: all units finished" >> "$LOG"
