#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime_env.sh"
export PRISMWAM_MODEL="${PRISMWAM_MODEL:-prismwam5b}"
exec "$ROBODOJO_PYTHON" -m robodoctor.pipeline "$@"
