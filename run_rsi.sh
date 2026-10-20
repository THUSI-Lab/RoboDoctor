#!/usr/bin/env bash
# Iterative self-improving policy doctor. Experience is optional.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime_env.sh"
export PRISMWAM_MODEL="${PRISMWAM_MODEL:-prismwam5b}"
exec "$ROBODOJO_PYTHON" -m robodoctor.launch --evaluation-method rsi_policy_doctor \
  --max-decisions 0 --run-timeout 14400 "$@"
