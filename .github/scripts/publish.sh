#!/usr/bin/env bash
# kukicha-win-x64.zip を Release $VERSION として $SHA に対して公開する。
# $SHA の公開済み Release があればそれを使う。
set -euo pipefail

existing=$(gh release list --limit 50 --exclude-drafts --json tagName \
  --jq ".[].tagName | select(test(\"^r[0-9]+-${SHA:0:7}\$\"))" | head -1)
if [[ -n "$existing" && "$(gh release view "$existing" --json targetCommitish --jq .targetCommitish)" == "$SHA" ]]; then
  version=$existing
  echo "reusing release $version"
else
  version=$VERSION
  if [[ "$(gh release view "$version" --json isDraft --jq .isDraft 2>/dev/null || true)" == true ]]; then
    gh release delete "$version" --yes
  fi
  # immutable release は、asset を付けた下書きを公開して初めて固定される
  gh release create "$version" kukicha-win-x64.zip --draft --target "$SHA" --title "$version" --notes "kukicha @ $SHA"
  gh release edit "$version" --draft=false
  echo "published release $version"
fi
