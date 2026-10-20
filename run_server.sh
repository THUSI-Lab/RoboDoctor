#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime_env.sh"
set +o pipefail
source "$ROBODOCTOR_ROOT/hybrid_rollout/robodojo/robodojo_server/runtime.sh"
set -o pipefail
exec "$ROBODOJO_PYTHON" -m robodoctor.server "$@"
