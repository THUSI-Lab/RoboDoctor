#!/usr/bin/env bash
# Shared direct/hybrid/RSI runtime with explicit simulator and policy roots.
ROBODOCTOR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROBODOCTOR_ROOT/runtime.local.sh" ]]; then
  source "$ROBODOCTOR_ROOT/runtime.local.sh"
fi
if [[ -z "${ROBODOJO_SOURCE:-}" || -z "${ROBODOJO_PYTHON:-}" ]]; then
  echo 'Set ROBODOJO_SOURCE and ROBODOJO_PYTHON in runtime.local.sh; see README.md.' >&2
  return 1 2>/dev/null || exit 1
fi
export ROBODOJO_SOURCE ROBODOJO_PYTHON
export CODEX_CONFIG="${CODEX_CONFIG:-$ROBODOCTOR_ROOT/codex.toml}"
export PYTHONPATH="$ROBODOCTOR_ROOT:$ROBODOCTOR_ROOT/third_party/prismwam/src:$ROBODOJO_SOURCE:$ROBODOJO_SOURCE/XPolicyLab:$ROBODOJO_SOURCE/third_party/curobo${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${OPENPI_SOURCE:-}" ]]; then
  export PYTHONPATH="$OPENPI_SOURCE/packages/openpi-client/src:$PYTHONPATH"
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TASK_ROOT="${TASK_ROOT:-$ROBODOCTOR_ROOT/runtime}"
if [[ -n "${SIM_EXTRA_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$SIM_EXTRA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
