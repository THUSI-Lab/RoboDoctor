#!/usr/bin/env bash
# Install only NVRTC into this repository; do not change the shared Python env.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime_env.sh"
exec "$ROBODOJO_PYTHON" -m pip install --disable-pip-version-check --no-deps --no-cache-dir \
    --target "$ROBODOCTOR_ROOT/runtime/curobo_nvrtc" --upgrade \
    'nvidia-cuda-nvrtc-cu12==12.9.86'
