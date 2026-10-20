#!/usr/bin/env python3
"""Check explicit simulator/policy configuration without starting an episode."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "prismwam"), default="direct")
    args = parser.parse_args()
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    def env_path(name, *, directory=False):
        value = os.environ.get(name)
        if not value:
            errors.append(f"Set {name} in runtime.local.sh")
            return None
        path = Path(value).expanduser()
        require(path.is_absolute(), f"{name} must be an absolute path: {path}")
        require(path.is_dir() if directory else path.is_file(), f"{name} missing: {path}")
        return path

    def probe(python, code, label):
        if python is None or not python.is_file():
            return
        try:
            result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=60)
            require(result.returncode == 0, f"{label}: {result.stderr.strip() or result.stdout.strip()}")
            if result.returncode == 0:
                print(f"OK {label}: {result.stdout.strip()}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{label}: {exc}")

    simulator = env_path("ROBODOJO_SOURCE", directory=True)
    sim_python = env_path("ROBODOJO_PYTHON")
    if simulator:
        for name in ("Robots", "Object", "Material", "Eval_Layout"):
            require((simulator / "Assets" / name).is_dir(), f"Missing RoboDojo Assets/{name}; run init_assets.sh")
        for name in ("XPolicyLab", "third_party/IsaacLab", "third_party/curobo"):
            require((simulator / name).is_dir(), f"Missing RoboDojo submodule: {name}")
        layout = simulator / "Assets/Eval_Layout/RoboDojo/arx_x5/0/build_tower_0.json"
        require(layout.is_file(), f"Missing smoke-test layout: {layout}")
        if layout.is_file():
            with layout.open("rb") as stream:
                require(not stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"),
                        "Assets are Git LFS pointers; finish git lfs pull")
    codex = shutil.which(os.environ.get("CODEX_BIN") or "codex")
    require(codex is not None, "Codex CLI not found; install it or set CODEX_BIN")
    config = Path(os.environ.get("CODEX_CONFIG") or ROOT / "codex.toml")
    require(config.is_file(), f"Codex provider config missing: {config}")
    if config.is_file():
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            provider_config = tomllib.loads(config.read_text())
            provider = provider_config.get("model_providers", {}).get(provider_config.get("model_provider"), {})
            key = provider.get("env_key") or "OPENAI_API_KEY"
            require(bool(os.environ.get(key) or os.environ.get("ROLLOUT_OPENAI_API_KEY_FILE")),
                    f"Set {key} or ROLLOUT_OPENAI_API_KEY_FILE for the selected Codex provider")
        except (ValueError, ImportError) as exc:
            errors.append(f"Cannot read Codex config: {exc}")
    probe(sim_python, "import importlib.metadata as m, importlib.util as u, torch; "
          "assert m.version('isaacsim').startswith('5.1.'); "
          "assert all(u.find_spec(x) for x in ['isaaclab', 'curobo', 'cv2', 'robodoctor']); "
          "assert torch.cuda.is_available(); print('Isaac Sim 5.1 / CUDA / Python imports')", "simulator Python")

    if args.mode == "prismwam":
        policy_python = env_path("PRISMWAM_PYTHON")
        checkpoint = env_path("PRISMWAM_CHECKPOINT")
        policy_config = env_path("PRISMWAM_CONFIG")
        assets = env_path("PRISMWAM_ASSETS_ROOT", directory=True)
        depth = env_path("PRISMWAM_DEPTH_VAE")
        manifest = json.loads((ROOT / "robodoctor/prismwam/assets.json").read_text())
        if checkpoint and checkpoint.is_file():
            require(checkpoint.stat().st_size == manifest["checkpoint"]["files"][0]["size"],
                    "RoboDojo checkpoint size mismatch; finish download_prismwam.py")
        if assets:
            for name in ("rgb_vae", "wan22", "tokenizer"):
                group = manifest[name]
                for record in group["files"]:
                    folder = assets if name == "rgb_vae" else assets / group["repo_id"]
                    path = folder / record["path"]
                    require(path.is_file() and path.stat().st_size == record["size"], f"Missing/incomplete model asset: {path}")
        if depth and depth.is_file():
            require(depth.stat().st_size == manifest["depth_vae"]["files"][0]["size"],
                    "RoboDojo depth VAE size mismatch; finish download_prismwam.py")
        else:
            errors.append("Matching PrismWAM depth VAE (including latent stats) is required; "
                          "run scripts/download_prismwam.py and set PRISMWAM_DEPTH_VAE")
        probe(policy_python, "import torch, prismwam, robodoctor.prismwam_policy as p; "
              "assert torch.cuda.is_available(); print(p.__file__)", "repository policy imports")
        if policy_config and policy_config.is_file() and depth and depth.is_file() and assets:
            probe(policy_python,
                  "import os; from pathlib import Path; from omegaconf import OmegaConf; "
                  "from robodoctor.prismwam_policy import _action_stats_path, _load_policy_config; "
                  "p=Path(os.environ['PRISMWAM_CONFIG']); c=_load_policy_config(p); "
                  "print(_action_stats_path(c.data.train,p.parent))",
                  "policy config / action normalization")
    if errors:
        for error in errors:
            print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(1)
    print("Environment preflight passed. Use the guide's smoke test to verify rendering and inference.")


if __name__ == "__main__":
    main()
