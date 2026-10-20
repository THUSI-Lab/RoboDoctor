#!/usr/bin/env bash
# Plan and supervise pi05 with the independent Codex agent.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime_env.sh"
exec "$ROBODOJO_PYTHON" -m robodoctor.launch --evaluation-method pi05_plus_gpt \
  --max-decisions 0 --run-timeout 14400 "$@"
