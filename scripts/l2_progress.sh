#!/usr/bin/env bash
# Show L2 progress from the l2-results branch (no checkout needed). Usage: bash scripts/l2_progress.sh
set -u
cd "$(dirname "$0")/.."
git fetch -q origin l2-results 2>/dev/null || { echo "No results yet (branch l2-results does not exist)."; exit 0; }
R=origin/l2-results
echo "Last update: $(git log -1 --format='%cd (%s)' --date=local $R)"
echo
echo "Pilot gates:"
git show "$R:gates.json" 2>/dev/null || echo "  gates.json not written yet"
for f in $(git ls-tree -r --name-only $R | grep 'claude_l2_pilot_v2/.*/pilot_gate.json'); do
  echo "  $f: $(git show "$R:$f" | tr -d '\n ' )"
done
echo
echo "Full run (100 games per model: 10 cells x 10 contexts; the memory cell EM is held back, Amendment 3):"
for m in gpt-oss_120b-cloud gemma4_31b-cloud nemotron-3-super_cloud mistral-large-3_675b-cloud; do
  n=$(git ls-tree -r --name-only $R | grep "^claude_l2_v1/$m/episodes/" | grep -vc "/EM__")
  st=$(for f in $(git ls-tree -r --name-only $R | grep "^claude_l2_v1/$m/STATUS"); do
    git show "$R:$f" | python3 -c 'import sys,json
part=sys.argv[1].split("STATUS")[-1].lstrip("_") or "all contexts"
try:
  d=json.load(sys.stdin); print("%s: %s, %.1fM tokens;" % (part, d["status"], d["tokens"] / 1e6))
except Exception: pass' "$f"; done | tr '\n' ' ')
  calls=$(git ls-tree -r --name-only $R | grep -c "^claude_l2_v1/$m/calls")
  [ -z "$st" ] && { [ "$n" -gt 0 ] || [ "$calls" -gt 0 ] && st="in progress" || st="not started"; }
  printf "  %-24s %3d / 100 games   %s\n" "$m" "$n" "$st"
done
echo
echo "Recent GitHub Actions runs:"
gh run list --workflow l2-llm-agents.yml --limit 8 2>/dev/null || echo "  (install/sign in to gh to see runs)"
