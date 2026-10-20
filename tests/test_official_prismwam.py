"""Official policy interface checks without loading weights or contacting an API."""
from pathlib import Path
from types import SimpleNamespace
import inspect
import json
import hashlib

import numpy as np
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def test_official_assets_pin_matching_config_and_statistics():
    manifest = json.loads((ROOT / 'robodoctor/prismwam/assets.json').read_text())
    groups = [manifest[key] for key in ('checkpoint', 'depth_vae', 'rgb_vae', 'inference_config')]
    assert {group['repo_id'] for group in groups} == {'THUSI-Lab/PrismWAM'}
    assert len({group['revision'] for group in groups}) == 1
    stats = next(record for record in groups[-1]['files'] if record['path'].endswith('action_stats.pt'))
    assert hashlib.sha256((ROOT / 'robodoctor/prismwam/action_stats.pt').read_bytes()).hexdigest() == stats['sha256']
    config = OmegaConf.load(ROOT / 'robodoctor/prismwam/config.yaml')
    assert config.model._target_ == 'prismwam.models.wan22.rgb_memory.create_prismwam_rgb_depth_action'
    assert config.deployment.action_horizon == 32
    assert config.deployment.replan_steps == 16
    assert config.model.rgb_memory.sparse_recent_action_offset == 32
    assert manifest['tokenizer']['repo_id'] == config.model.tokenizer_model_id == manifest['wan22']['repo_id']
    assert manifest['tokenizer']['revision'] == manifest['wan22']['revision']
    assert all(len(record['sha256']) == 64 for group in manifest.values() for record in group['files'])


@pytest.fixture
def official_adapter(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('git', reason='Install the PrismWAM dependencies for model interface tests')
    from prismwam.models.wan22.rgb_memory import RGBMemoryPrismWAM
    from prismwam.models.wan22.sparse_memory import RGBMemoryConfig
    from prismwam.deployment import policy as deployment
    from prismwam.hub import TEXT_ENCODER_FILES
    from robodoctor.prismwam_policy import PrismWAMPolicy

    config = OmegaConf.load(ROOT / 'robodoctor/prismwam/config.yaml')
    checkpoint = tmp_path / 'policy.pt'
    checkpoint.touch()
    for name in ('depth_vae.pt', 'rgb_vae.safetensors'):
        (tmp_path / name).touch()
    config.model.depth_vae_checkpoint_path = 'depth_vae.pt'
    config.model.rgb_vae_checkpoint_path = 'rgb_vae.safetensors'
    config.data.train.pop('action_stats_cache_path')
    config.deployment.statistics_path = str(ROOT / 'robodoctor/prismwam/action_stats.pt')
    config.data.train.video_size = [32, 32]
    OmegaConf.save(config, tmp_path / 'config.yaml')
    monkeypatch.setenv('PRISMWAM_CONFIG', str(tmp_path / 'config.yaml'))
    monkeypatch.setenv('PRISMWAM_ASSETS_ROOT', str(tmp_path))
    for name in TEXT_ENCODER_FILES:
        path = tmp_path / config.model.model_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    signature = inspect.signature(RGBMemoryPrismWAM.infer_action)
    calls = []

    class Model:
        action_expert = SimpleNamespace(action_dim=14)

        def eval(self):
            return self

        rgb_history_num_frames = 3
        rgb_history_frame_stride = 4
        depth_history_num_frames = 1
        device = 'cpu'
        torch_dtype = torch.float32
        rgb_memory_config = RGBMemoryConfig(**OmegaConf.to_container(config.model.rgb_memory))

        def reset_rgb_memory(self):
            pass

        def infer_action(self, **kwargs):
            signature.bind(self, **kwargs)  # Enforce the actual official model signature.
            calls.append(kwargs)
            return {'action': torch.zeros(32, 14)}

    Model.infer_action.__signature__ = signature

    real_signature = inspect.signature(deployment.load_policy)

    def load(config, checkpoint, **kwargs):
        real_signature.bind(config, checkpoint, **kwargs)
        prepared = deployment.prepare_model_config(config)
        assert prepared['checkpoint_init_only']
        assert prepared['load_text_encoder']
        assert not prepared['redirect_common_files']
        assert prepared['tokenizer_model_id'] == 'Wan-AI/Wan2.2-TI2V-5B'
        assert config['deployment']['num_inference_steps'] == 10
        assert kwargs['evaluation']['inference_shifts'] == dict(depth=5., action=5.)
        statistics = deployment.ActionStatistics.load(kwargs['statistics_path'], 14)
        return deployment.PrismWAMPolicy(Model(), config, statistics=statistics)

    monkeypatch.setattr(deployment, 'load_policy', load)
    policy = PrismWAMPolicy(checkpoint, device='cpu')
    observation = {'instruction': 'Build a tower', 'states': np.zeros(14, np.float32)}
    for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
        observation[camera] = np.zeros((8, 8, 3), np.uint8)
        observation[camera + '_depth'] = np.ones((8, 8), np.float32)
    return SimpleNamespace(policy=policy, observation=observation, calls=calls, assets=tmp_path)


def test_official_portable_policy_observe_and_infer(official_adapter):
    policy, observation = official_adapter.policy, official_adapter.observation
    for _ in range(6):
        policy.observe(observation)
    before = policy.history.step
    actions, record = policy.infer(observation, record_observation=False)
    assert policy.metadata['execution_horizon'] == 16
    assert policy.metadata['rgb_memory']
    assert actions.shape == (32, 14) and np.isfinite(actions).all()
    assert record['inference_index'] == 0
    assert policy.history.step == before == 5
    call = official_adapter.calls[0]
    assert call['rgb_history_images'].shape[2] == 3
    assert 'negative_prompt' not in call and 'text_cfg_scale' not in call
    assert call['action_horizon'] == 32
    policy.reset()
    assert policy.history.step == -1


def test_sparse_history_preserves_native_steps_and_short_interventions(official_adapter):
    policy = official_adapter.policy
    for step in range(40):
        observation = dict(official_adapter.observation)
        for camera in policy.CAMERAS:
            observation[camera] = np.full((8, 8, 3), step, np.uint8)
        policy.observe(observation)
    assert policy.history.indices() == [0, 7, 39]
    assert len(policy.history.frames) == 33
    history = policy.history.tensor().float().numpy()
    np.testing.assert_allclose(history[0, :, :, 0, 0],
                               np.tile(np.array([0, 7, 39]) * (2 / 255.) - 1, (3, 1)), atol=1e-6)
    # A short RSI correction is three observed actions, not a synthetic 16-step pad.
    for _ in range(3):
        policy.observe(observation)
    before = policy.history.step
    policy.infer(observation, record_observation=False)
    assert policy.history.step == before == 42
    assert policy.history.indices() == [0, 10, 42]


def test_camera_layout_depth_units_and_action_normalization(official_adapter):
    policy = official_adapter.policy
    obs = dict(official_adapter.observation)
    for camera, value in zip(policy.CAMERAS, [0, 127, 255]):
        obs[camera] = np.full((8, 8, 3), value, np.uint8)
    obs['cam_high_depth'] = np.full((8, 8), 0.75, np.float32)
    obs['cam_left_wrist_depth'] = np.full((8, 8), 0., np.float32)
    obs['cam_right_wrist_depth'] = np.full((8, 8), np.nan, np.float32)
    actions, _ = policy.infer(obs)
    call = official_adapter.calls[-1]
    rgb, depth = call['rgb_input_image'], call['input_image']
    assert tuple(rgb.shape) == (1, 3, 32, 32)
    assert tuple(depth.shape) == (1, 1, 32, 32)
    assert float(rgb[0, 0, 0, 0]) == -1
    assert float(rgb[0, 0, -1, -1]) == 1
    assert float(depth[0, 0, 0, 0]) == 0
    assert float(depth[0, 0, -1, 0]) == -1
    assert float(depth[0, 0, -1, -1]) == 1
    statistics = policy._policy.statistics
    expected = statistics.action_mean.numpy().reshape(-1).copy()
    expected[[6, 13]] = np.clip(expected[[6, 13]], 0, 1)
    np.testing.assert_allclose(actions[0], expected)
    np.testing.assert_allclose(call['proprio'].numpy(),
        statistics.normalize_state(__import__('torch').zeros(1, 14)).numpy())


def test_inference_without_an_initial_observation_is_rejected(official_adapter):
    with pytest.raises(RuntimeError, match='first observation'):
        official_adapter.policy.infer(official_adapter.observation, record_observation=False)


def test_official_source_files_match_recorded_public_revision():
    manifest = json.loads((ROOT / 'third_party/prismwam/OFFICIAL_SOURCE.json').read_text())
    assert manifest['repository'] == 'https://github.com/THUSI-Lab/PrismWAM'
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((ROOT / 'third_party/prismwam/src' / name).read_bytes()).hexdigest() == expected
