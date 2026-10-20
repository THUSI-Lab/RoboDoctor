#!/usr/bin/env python3
"""Download pinned public PrismWAM inference assets into an explicit directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


MANIFEST = Path(__file__).resolve().parents[1] / "robodoctor/prismwam/assets.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, record: dict) -> None:
    if path.stat().st_size != record["size"]:
        raise ValueError(f"Incomplete download: {path}")
    if record.get("sha256") and sha256(path) != record["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")


def download_depth(url: str, destination: Path, expected_sha256: str | None) -> None:
    if not url.startswith("https://"):
        raise ValueError("The depth VAE download URL must use HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(".pt.partial")
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        if expected_sha256 and sha256(temporary) != expected_sha256:
            raise ValueError(f"Depth VAE SHA256 mismatch; partial file retained: {temporary}")
        temporary.replace(destination)
    if expected_sha256 and sha256(destination) != expected_sha256:
        raise ValueError(f"Depth VAE SHA256 mismatch: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Explicit model download directory")
    parser.add_argument("--dry-run", action="store_true", help="Print pinned URLs/destinations without downloading")
    parser.add_argument("--depth-vae-url", help="Optional HTTPS mirror for the matching released depth VAE")
    parser.add_argument("--depth-vae-sha256", help="Depth VAE digest; defaults to the pinned release digest")
    args = parser.parse_args()
    if args.depth_vae_sha256 and not args.depth_vae_url:
        parser.error("--depth-vae-sha256 requires --depth-vae-url")
    manifest = json.loads(MANIFEST.read_text())
    output = args.output.expanduser().resolve()
    if not args.dry_run:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            parser.error("Install huggingface-hub in this Python environment first")
        output.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoint", "depth_vae", "rgb_vae", "inference_config", "wan22", "tokenizer"):
        group = manifest[name]
        folder = output if name in ("checkpoint", "depth_vae", "rgb_vae", "inference_config") else output / group["repo_id"]
        for record in group["files"]:
            if name == "depth_vae" and args.depth_vae_url:
                destination = folder / record["path"]
                print(f'{args.depth_vae_url}\n  -> {destination}', flush=True)
                if not args.dry_run:
                    download_depth(args.depth_vae_url, destination,
                                   args.depth_vae_sha256 or record["sha256"])
                continue
            url = f'https://huggingface.co/{group["repo_id"]}/resolve/{group["revision"]}/{record["path"]}'
            print(f'{url}\n  -> {folder / record["path"]}', flush=True)
            if not args.dry_run:
                path = Path(hf_hub_download(repo_id=group["repo_id"], revision=group["revision"],
                                          filename=record["path"], local_dir=folder))
                verify(path, record)


if __name__ == "__main__":
    main()
