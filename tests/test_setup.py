import json
import os
from pathlib import Path
import subprocess

import pytest
from omegaconf import OmegaConf

from robodoctor.prismwam_policy import (
    _action_stats_path, _checkpoint_config_path, _remap_local_assets, _load_policy_config, checkpoint_for_variant,
)


def test_checkpoint_requires_explicit_input(monkeypatch):
    monkeypatch.delenv('PRISMWAM_CHECKPOINT', raising=False)
    with pytest.raises(ValueError, match='PRISMWAM_CHECKPOINT'):
        checkpoint_for_variant('prismwam5b')
    with pytest.raises(ValueError, match='PRISMWAM_CHECKPOINT'):
        checkpoint_for_variant('5b')


def test_explicit_config_is_authoritative(tmp_path, monkeypatch):
    missing = tmp_path / 'missing.yaml'
    (tmp_path / 'config.yaml').write_text('model: {}')
    monkeypatch.setenv('PRISMWAM_CONFIG', str(missing))
    with pytest.raises(FileNotFoundError, match='PRISMWAM_CONFIG'):
        _checkpoint_config_path(tmp_path / 'robodojo/policy.pt')


def test_shallow_checkpoint_without_config_reports_actionable_error(monkeypatch):
    monkeypatch.delenv('PRISMWAM_CONFIG', raising=False)
    with pytest.raises(FileNotFoundError, match='Set PRISMWAM_CONFIG'):
        _checkpoint_config_path(Path('/nonexistent.pt'))


def test_preview_config_resolves_only_explicit_frozen_assets(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    depth = tmp_path / 'depth.pt'
    rgb = tmp_path / 'robodojo/rgb_vae.safetensors'
    depth.touch()
    rgb.parent.mkdir(parents=True)
    rgb.touch()
    monkeypatch.setenv('PRISMWAM_DEPTH_VAE', str(depth))
    monkeypatch.setenv('PRISMWAM_ASSETS_ROOT', str(tmp_path))
    config_path = root / 'robodoctor/prismwam/config.yaml'
    config = OmegaConf.load(config_path)
    _remap_local_assets(config.model, config_path.parent)
    assert config.model.depth_vae_checkpoint_path == str(depth)
    assert config.model.rgb_vae_checkpoint_path == str(rgb)
    assert _action_stats_path(config.data.train, config_path.parent).is_file()
    assert config.model.checkpoint_init_only
    assert config.model.depth_dit_checkpoint_path is None
    assert config.model.action_dit_pretrained_path is None
    assert config.model.skip_rgb_dit_load_from_pretrain
    assert config.model.skip_action_dit_load_from_pretrain


def test_missing_frozen_asset_is_rejected_without_path_guessing(tmp_path):
    config = {'depth_vae_checkpoint_path': 'missing.pt'}
    with pytest.raises(FileNotFoundError, match='depth_vae_checkpoint_path'):
        _remap_local_assets(config, tmp_path)


def test_public_deployment_statistics_override_training_cache(tmp_path):
    for name in ('depth.pt', 'rgb.pt', 'action_stats.pt'):
        (tmp_path/name).touch()
    path = tmp_path/'config.yaml'
    path.write_text('model:\n  _target_: prismwam.models.factory\n'
                    '  depth_vae_checkpoint_path: depth.pt\n  rgb_vae_checkpoint_path: rgb.pt\n'
                    'data:\n  train:\n    action_stats_cache_path: missing_training_cache.pt\n'
                    'deployment:\n  statistics_path: action_stats.pt\n')
    cfg = _load_policy_config(path)
    assert _action_stats_path(cfg.data.train, tmp_path) == tmp_path/'action_stats.pt'


def test_runtime_env_has_no_machine_fallback(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / 'runtime_env.sh'
    script.write_text((root / 'runtime_env.sh').read_text())
    environment = dict(os.environ)
    for key in ('ROBODOJO_SOURCE', 'ROBODOJO_PYTHON'):
        environment.pop(key, None)
    result = subprocess.run(['bash', '-c', 'source "$1"', 'test', str(script)],
                            env=environment, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Set ROBODOJO_SOURCE and ROBODOJO_PYTHON' in result.stderr
    (tmp_path / 'runtime.local.sh').write_text(
        'export ROBODOJO_SOURCE="/explicit/simulator"\n'
        'export ROBODOJO_PYTHON="/explicit/python"\n')
    result = subprocess.run(['bash', '-c', 'source "$1"; env', 'test', str(script)],
                            env=environment, capture_output=True, text=True, check=True)
    assert 'ROBODOJO_SOURCE=/explicit/simulator' in result.stdout


def test_download_plan_uses_public_pinned_checkpoint_and_depth_vae(tmp_path, capsys):
    from scripts.download_prismwam import MANIFEST, main
    import sys
    from unittest.mock import patch
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['checkpoint']['repo_id'] == 'THUSI-Lab/PrismWAM'
    assert len(manifest['checkpoint']['revision']) == 40
    assert manifest['depth_vae']['repo_id'] == manifest['checkpoint']['repo_id']
    assert manifest['depth_vae']['revision'] == manifest['checkpoint']['revision']
    depth = manifest['depth_vae']['files'][0]
    assert depth['path'] == 'robodojo/depth_vae.pt'
    assert len(depth['sha256']) == 64
    destination = tmp_path / 'not-created'
    with patch.object(sys, 'argv', ['download', '--output', str(destination), '--dry-run']):
        main()
    plan = capsys.readouterr().out
    assert f"/resolve/{manifest['depth_vae']['revision']}/{depth['path']}" in plan
    assert str(destination / depth['path']) in plan
    assert not destination.exists()


@pytest.mark.parametrize('corrupt_depth', [False, True])
def test_download_fetches_and_checks_depth_vae_by_default(tmp_path, monkeypatch, corrupt_depth):
    import hashlib
    import sys
    from types import SimpleNamespace
    from scripts import download_prismwam as download

    manifest = json.loads(download.MANIFEST.read_text())
    for group in manifest.values():
        for record in group['files']:
            record.update(size=4, sha256=hashlib.sha256(b'good').hexdigest())
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(download, 'MANIFEST', manifest_path)
    calls = []

    def fetch(*, repo_id, revision, filename, local_dir):
        calls.append((repo_id, revision, filename))
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'evil' if corrupt_depth and filename == 'robodojo/depth_vae.pt' else b'good')
        return str(path)

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=fetch))
    output = tmp_path / 'models'
    monkeypatch.setattr(sys, 'argv', ['download', '--output', str(output)])
    if corrupt_depth:
        with pytest.raises(ValueError, match='SHA256 mismatch.*robodojo/depth_vae.pt'):
            download.main()
        assert len(calls) == 2  # Stop before downloading other assets.
    else:
        download.main()
        expected = [(g['repo_id'], g['revision'], r['path'])
                    for name in ('checkpoint', 'depth_vae', 'rgb_vae', 'inference_config', 'wan22', 'tokenizer')
                    for g in [manifest[name]] for r in g['files']]
        assert calls == expected
        assert (output / 'robodojo/depth_vae.pt').read_bytes() == b'good'


def test_download_integrity_rejects_corruption(tmp_path):
    import hashlib
    from scripts.download_prismwam import verify
    asset = tmp_path / 'asset.pt'
    expected = {'size': 4, 'sha256': hashlib.sha256(b'good').hexdigest()}
    asset.write_bytes(b'good')
    verify(asset, expected)
    asset.write_bytes(b'evil')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        verify(asset, expected)
    asset.write_bytes(b'partial')
    with pytest.raises(ValueError, match='Incomplete download'):
        verify(asset, expected)


def test_auxiliary_deployment_path_requires_absolute_explicit_value(monkeypatch, tmp_path):
    from hybrid_rollout.robodojo.io import configured_path
    monkeypatch.delenv('ROLLOUT_SHARED_ROOT', raising=False)
    with pytest.raises(ValueError, match='Set ROLLOUT_SHARED_ROOT explicitly'):
        configured_path('ROLLOUT_SHARED_ROOT')
    monkeypatch.setenv('ROLLOUT_SHARED_ROOT', 'relative/path')
    with pytest.raises(ValueError, match='absolute path'):
        configured_path('ROLLOUT_SHARED_ROOT')
    monkeypatch.setenv('ROLLOUT_SHARED_ROOT', str(tmp_path))
    assert configured_path('ROLLOUT_SHARED_ROOT') == tmp_path
