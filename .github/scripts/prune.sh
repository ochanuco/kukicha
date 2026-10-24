#!/usr/bin/env bash
# prune.sh <keep>
# 公開済みの r<run>-<sha7> の Release を新しい順に <keep> 件だけ残し、残りをタグごと消す。
set -euo pipefail

keep=$1
gh release list --limit 100 --exclude-drafts --json tagName,createdAt \
  --jq "[.[] | select(.tagName | test(\"^r[0-9]+-[0-9a-f]{7}\$\"))] | sort_by(.createdAt) | reverse | .[$keep:][] | .tagName" |
while read -r tag; do
  gh release delete "$tag" --yes --cleanup-tag
  echo "deleted release $tag"
done
