"""Released PrismWAM checkpoints on the Hugging Face Hub.

All files live in one model repository, one folder per release::

    THUSI-Lab/PrismWAM
    ├── robotwin/           model folder: config.yaml, policy.pt, VAEs, statistics
    ├── robodojo/
    ├── libero/
    └── init/               training initializers: depth/RGB VAEs, action-expert init

A folder downloaded to ``<local_dir>/<name>`` keeps the same relative layout, so
``weights/init/...`` is exactly where the training recipes look.
"""

from __future__ import annotations

import os
from pathlib import Path

HF_REPO_ID = "THUSI-Lab/PrismWAM"
HF_URL = f"https://huggingface.co/{HF_REPO_ID}"
POLICIES = ("robotwin", "robodojo", "libero")
TEXT_ENCODER_FILES = (
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)


def download(
    name: str,
    local_dir: str | Path = "weights",
    *,
    repo_id: str = HF_REPO_ID,
    revision: str | None = None,
) -> Path:
    """Download one release folder, e.g. ``"robotwin"`` or ``"init"``."""
    from huggingface_hub import snapshot_download

    name = name.strip("/")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=[f"{name}/**"],
        local_dir=str(local_dir),
    )
    path = Path(local_dir) / name
    if not path.is_dir():
        raise FileNotFoundError(f"{repo_id} has no folder named {name!r}.")
    return path


def download_text_encoder(model_id: str = "Wan-AI/Wan2.2-TI2V-5B") -> Path:
    """Fetch the T5 encoder and tokenizer that encode instructions, if missing.

    They go to ``<PRISMWAM_WEIGHTS_ROOT or weights>/<model_id>/`` (about 11 GB),
    where the Wan loader reads them.
    """
    from huggingface_hub import snapshot_download

    from prismwam.models.wan22.helpers.io import PRETRAINED_ROOT_ENV

    root = Path(os.environ.get(PRETRAINED_ROOT_ENV, "weights")) / model_id
    missing = [name for name in TEXT_ENCODER_FILES if not (root / name).is_file()]
    if missing:
        snapshot_download(repo_id=model_id, allow_patterns=missing, local_dir=str(root))
    return root


def is_bundle(path: str | Path) -> bool:
    path = Path(path)
    return (path / "config.yaml").is_file() and (path / "policy.pt").is_file()


def resolve_bundle(name_or_path: str | Path, local_dir: str | Path = "weights") -> Path:
    """Return a local model folder, downloading a released model by name if needed."""
    path = Path(name_or_path).expanduser()
    if is_bundle(path):
        return path
    if str(name_or_path) in POLICIES:
        local = Path(local_dir) / str(name_or_path)
        return local if is_bundle(local) else download(str(name_or_path), local_dir)
    raise FileNotFoundError(
        f"{name_or_path!r} is neither a PrismWAM model folder nor one of {POLICIES}."
    )


def bundle_files(bundle: str | Path) -> tuple[Path, Path]:
    """The (config, weights) files of a model folder."""
    bundle = Path(bundle).resolve()
    return bundle / "config.yaml", bundle / "policy.pt"


def load_pretrained(
    name_or_path: str | Path,
    *,
    device: str = "cuda:0",
    dtype=None,
    local_dir: str | Path = "weights",
):
    """Load a released model by name (``"robotwin"``) or from a local model folder.

    >>> from prismwam import load_pretrained
    >>> policy = load_pretrained("robotwin")          # downloads to weights/robotwin
    >>> out = policy.predict(observation, seed=0)    # out["action"]: [32, 14]
    """
    import torch

    from prismwam.deployment.policy import load_policy

    config, checkpoint = bundle_files(resolve_bundle(name_or_path, local_dir))
    return load_policy(config, checkpoint, device=device, dtype=dtype or torch.bfloat16)
