#!/usr/bin/env bash
# Commit and push the L2 store (the l2-results branch) without ever touching the files the runner is writing.
# The runner keeps its log files open, so git must not rewrite them: a rebase in the store replaced the files on disk and
# the runner's later lines went to a deleted file (found 2026-10-08; Amendment 6). Instead, the files this job changed
# (from `git status`, which only reads) are copied into a separate clone, STORE.sync, and committed and pushed from there.
# Parallel lanes write different files, so a lost push race is rebased in the sync clone and retried.
# Usage: bash scripts/l2_save.sh STORE_DIR MESSAGE
set -u
STORE="$1"; MSG="$2"
[ -d "$STORE/.git" ] || exit 0
SYNC="${STORE%/}.sync"
if [ ! -d "$SYNC/.git" ]; then
  git clone -q --branch l2-results --single-branch "$(git -C "$STORE" remote get-url origin)" "$SYNC" || exit 0
fi
git -C "$SYNC" config user.name "l2-runner"
git -C "$SYNC" config user.email "41898282+github-actions[bot]@users.noreply.github.com"
for i in 1 2 3 4 5 6; do
  git -C "$SYNC" pull -q --rebase origin l2-results 2>/dev/null || git -C "$SYNC" rebase --abort 2>/dev/null
  # copy every file this job created or changed (never *.tmp, which is a game being written)
  git -C "$STORE" status --porcelain=v1 -z --untracked-files=all | tr '\0' '\n' | sed -n 's/^.. //p' | grep -v '\.tmp$' |
    while IFS= read -r f; do
      [ -f "$STORE/$f" ] || continue
      mkdir -p "$SYNC/$(dirname "$f")" && cp -p "$STORE/$f" "$SYNC/$f"
    done
  git -C "$SYNC" add -A
  git -C "$SYNC" commit -q -m "$MSG $(date -u +%Y-%m-%dT%H:%MZ)" >/dev/null || true
  git -C "$SYNC" push -q origin HEAD:l2-results 2>/dev/null && { echo "saved"; exit 0; }
  sleep $((RANDOM % 15 + 3))
done
echo "push failed after 6 tries; the next save retries"
exit 0
