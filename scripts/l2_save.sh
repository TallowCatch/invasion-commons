#!/usr/bin/env bash
# Commit and push the L2 store (the l2-results branch). Parallel lanes write different files, so a push that loses the
# race is rebased onto the other lanes' commits and retried. Usage: bash scripts/l2_save.sh STORE_DIR MESSAGE
set -u
STORE="$1"; MSG="$2"
cd "$STORE" || exit 0
git config user.name "l2-runner"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
git commit -q -m "$MSG $(date -u +%Y-%m-%dT%H:%MZ)" || { echo "nothing new"; exit 0; }
for i in 1 2 3 4 5; do
  git push -q origin HEAD:l2-results && exit 0
  git pull -q --rebase origin l2-results || { git rebase --abort; echo "rebase failed"; }
  sleep $((RANDOM % 20 + 5))
done
echo "push failed after 5 tries; the next save retries"
exit 0
