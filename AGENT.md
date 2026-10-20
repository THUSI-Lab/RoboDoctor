# RoboDoctor contributor instructions

## Scope and layout

This is a standalone public source distribution, without Git metadata. The
project is RoboDoctor and the Python import package is `robodoctor`.
`robodoctor/` contains launchers, direct/hybrid/RSI controllers, prompts, geometry
and the PrismWAM adapter. `hybrid_rollout/` contains only the runtime dependency
closure. `third_party/prismwam/` contains vendored inference dependencies.
`scripts/` contains model downloads and environment checks; `tests/` contains
regression tests. Keep one README.md containing all user-facing setup and usage.
Preserve required third-party notices in THIRD_PARTY_NOTICES.md and the
PrismWAM licensing/provenance files. There is no root project LICENSE.
Use only the public PrismWAM implementation and PRISMWAM_* runtime settings.

## Configuration and credentials

Use explicit environment variables and runtime.local.sh for machine paths.
Never hardcode personal paths, private endpoints, API keys, account identities,
authentication state, experiment artifacts or model weights into source.
Commit only credential-free templates. Do not add compatibility packages under
retired project names. Model download manifests pin revisions, sizes and SHA256.

## Behavior to preserve

Maintain direct, Pi05 hybrid, PrismWAM RSI, and iterative pipeline entry points.
Policy weights remain fixed. Ground robot actions in fresh RGB-D/FK evidence;
RSI interventions remain limited to 1–5 physical control steps. Preserve
observation history, experience persistence, policy-prefix semantics, process
cleanup and the distinction between invalid evaluations and zero task scores.
Keep supported runtime scripts executable and package all required prompt/config
assets. New dependencies must work from this directory alone.
Keep direct and RSI prompts separate. After the three gate episodes, only the
RSI branch runs Harness improvement from iter2 experience. All six default tail
episodes use the same validated run-local RSI prompt pair; direct skips editing
and retains its original prompts. Editing failures must stop before the tail.

## Validation

Install the project and test dependencies in the explicitly configured simulator
environment: `python -m pip install -e ".[test]"`. Load runtime.local.sh through
`source ./runtime_env.sh`, then run `"$ROBODOJO_PYTHON" -m pytest -q`.
Run `bash -n` on every shell script, build and inspect the wheel, and check
`run_pipeline.sh --dry-run`. Model download plans can be checked without weights:
`python scripts/download_prismwam.py --output /tmp/robodoctor-models --dry-run`.
The full FK/IK tests need RoboDojo robot assets through ROBODOJO_SOURCE.
GPU rendering, policy inference and model API calls require separately installed
assets, a configured GPU environment and credentials. Report which checks ran;
do not claim a complete GPU episode passed based only on unit tests or a dry run.
Keep generated caches/builds/results out of the public distribution.
