#!/usr/bin/env bash
# One command owns the simulator, controller and an isolated Codex CLI process.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime_env.sh"
exec "$ROBODOJO_PYTHON" -m robodoctor.launch "$@"
