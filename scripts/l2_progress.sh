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
echo "Full run (110 games per model):"
for m in gpt-oss_120b-cloud gemma4_31b-cloud nemotron-3-super_cloud; do
  n=$(git ls-tree -r --name-only $R | grep -c "^claude_l2_v1/$m/episodes/")
  st=$(git show "$R:claude_l2_v1/$m/STATUS" 2>/dev/null | python3 -c 'import sys,json
try:
  d=json.load(sys.stdin); print(f"{d[\"status\"]}, {d[\"tokens\"]/1e6:.1f}M tokens")
except Exception: print("not started")')
  printf "  %-24s %3d / 110 games   %s\n" "$m" "$n" "$st"
done
echo
echo "Recent GitHub Actions runs:"
gh run list --workflow l2-llm-agents.yml --limit 5 2>/dev/null || echo "  (install/sign in to gh to see runs)"
