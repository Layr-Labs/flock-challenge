#!/usr/bin/env sh
# yukon-trace-config-v14
# Thin transport only: the CLI owns identity, checkpoints, source handling, retry, and upload.
agent="${1:-unknown}"
kind="${2:-capture}"
detail="${3:-}"
repo="$(CDPATH= cd -- "$(dirname -- "$0")/../.." 2>/dev/null && pwd)" || repo="$PWD"
payload="$(cat 2>/dev/null || true)"
if [ "$kind" = "session" ]; then
  cd "$repo" 2>/dev/null || exit 1
  printf '%s' "$payload" | yukon trace session "$agent"
  exit $?
fi
( cd "$repo" 2>/dev/null && printf '%s' "$payload" | yukon trace hook "$agent" >/dev/null 2>&1 & ) >/dev/null 2>&1 || true
# Codex consumes command-hook stdout. UserPromptSubmit must remain empty because plain text is
# injected into the prompt; the other capture hooks accept this explicit continue response.
[ "$agent" = "codex" ] && [ "$detail" != "UserPromptSubmit" ] && printf '{"continue":true}\n'
exit 0
