#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${1:-https://github.com/FarihaMehzabin/Domain-Adaptation}"
MESSAGE="${2:-backup: $(date -u +%Y-%m-%dT%H:%M:%SZ)}"

cd /workspace

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init -b main
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

if ! git config user.name >/dev/null; then
  git config user.name "Codex Backup"
fi
if ! git config user.email >/dev/null; then
  git config user.email "codex-backup@local"
fi

if git lfs version >/dev/null 2>&1; then
  git lfs install --local >/dev/null
fi

git add -A
if git diff --cached --quiet; then
  echo "No changes to commit."
  exit 0
fi

git commit -m "$MESSAGE"
git push -u origin main
