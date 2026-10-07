#!/usr/bin/env bash
# Commit and push the L2 store (the l2-results branch). Parallel lanes write different files, so a push that loses the
# race is rebased onto the other lanes' commits and retried. Files still being written by the runner are stashed
# during the rebase (autostash) and committed by the next save. Usage: bash scripts/l2_save.sh STORE_DIR MESSAGE
set -u
STORE="$1"; MSG="$2"
cd "$STORE" || exit 0
git config user.name "l2-runner"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
for i in 1 2 3 4 5 6; do
  git add -A
  git commit -q -m "$MSG $(date -u +%Y-%m-%dT%H:%MZ)" >/dev/null || true
  git push -q origin HEAD:l2-results 2>/dev/null && { echo "saved"; exit 0; }
  git pull -q --rebase --autostash origin l2-results || { git rebase --abort 2>/dev/null; echo "rebase failed"; }
  sleep $((RANDOM % 15 + 3))
done
echo "push failed after 6 tries; the next save retries"
exit 0
