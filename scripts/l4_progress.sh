#!/usr/bin/env bash
# Progress of L4, L5 and L6 on GitHub Actions: games saved on the l2-results branch (every 20 minutes) and the jobs running.
# Shows counts only, never outcomes.  Usage: bash scripts/l4_progress.sh
set -u
git fetch -q origin l2-results
tree() { git ls-tree -r --name-only origin/l2-results -- "$1" 2>/dev/null | grep -c '/episodes/.*\.json$'; }
printf "%-34s %s\n" "study / model" "games saved (planned)"
for row in "claude_l4_v1/gpt-oss_120b-cloud 190" "claude_l4_v1/nemotron-3-super_cloud 190" \
           "claude_l5_v1/deepseek-v4_1-flash_cloud 30" \
           "claude_l6_pilot_v1/gpt-oss_120b-cloud 2" "claude_l6_v1/gpt-oss_120b-cloud 40" \
           "claude_l6_pilot_v1/nemotron-3-super_cloud 2" "claude_l6_v1/nemotron-3-super_cloud 40"; do
  set -- $row; printf "%-34s %s (%s)\n" "$1" "$(tree "$1")" "$2"
done
for f in claude_l4_v1/prediction.json claude_l5_v1/prediction.json claude_l6_v1/f_choice.json; do
  git cat-file -e "origin/l2-results:$f" 2>/dev/null && echo "frozen: $f" || echo "not yet: $f"
done
gh run list --workflow l2-llm-agents.yml --limit 4 2>/dev/null
