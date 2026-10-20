"""RoboDojo observation adapter for the public PrismWAM deployment API."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


def checkpoint_for_variant(variant: str) -> Path:
    if str(variant).lower().replace('-', '') not in {'5b', 'prismwam5b'}:
        raise ValueError(f'Unknown public PrismWAM variant {variant!r}; expected prismwam5b')
    if not os.environ.get('PRISMWAM_CHECKPOINT'):
        raise ValueError('Set PRISMWAM_CHECKPOINT or pass --checkpoint; see README.md')
    return Path(os.environ['PRISMWAM_CHECKPOINT']).expanduser().resolve()


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    return f'stat:{stat.st_size}:{stat.st_mtime_ns}'


def _action_stats_path(data, config_dir: Path | None = None) -> Path:
    for value in (data.get('action_stats_cache_path'), data.get('vae_cache_action_stats_path')):
        if value:
            candidate = Path(str(value)).expanduser()
            if not candidate.is_absolute() and config_dir is not None:
                candidate = config_dir / candidate
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError('PrismWAM action normalization stats are unavailable')


def _remap_local_assets(model_cfg, config_dir: Path) -> None:
    for key in ('depth_vae_checkpoint_path', 'rgb_vae_checkpoint_path'):
        value = model_cfg.get(key)
        if not value:
            raise ValueError(f'PrismWAM config requires {key}')
        source = Path(str(value)).expanduser()
        if not source.is_absolute():
            source = config_dir / source
        if not source.is_file() or not os.access(source, os.R_OK):
            raise FileNotFoundError(f'PrismWAM {key} missing or unreadable: {source}; see README.md')
        model_cfg[key] = str(source.resolve())


def _checkpoint_config_path(checkpoint: Path) -> Path:
    explicit = os.environ.get('PRISMWAM_CONFIG')
    config = Path(explicit).expanduser().resolve() if explicit else checkpoint.parent / 'config.yaml'
    if not config.is_file():
        raise FileNotFoundError(f'Set PRISMWAM_CONFIG to an existing public inference YAML; not found: {config}')
    return config


def _load_policy_config(path: Path):
    from omegaconf import OmegaConf
    path = Path(path)
    cfg = OmegaConf.load(path)
    if not str(cfg.model.get('_target_', '')).startswith('prismwam.'):
        raise ValueError('Expected a public prismwam model factory')
    statistics = cfg.get('deployment', {}).get('statistics_path')
    # Portable deployment metadata is authoritative over training cache paths.
    if statistics:
        cfg.data.train.action_stats_cache_path = statistics
    _remap_local_assets(cfg.model, path.parent)
    return cfg


def _depth_mm(value: Any, *, maximum: float = 1500.0) -> np.ndarray:
    """RoboDojo's distance-to-image-plane annotator reports metres."""
    depth = np.asarray(value, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or min(depth.shape) < 1:
        raise ValueError(f'Expected nonempty HxW depth image, got {depth.shape}')
    return np.nan_to_num(depth * 1000.0, nan=maximum, posinf=maximum, neginf=0.0)


def _rgb(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] < 3 or image.dtype != np.uint8:
        raise ValueError(f'Expected HxWx3 uint8 RGB image, got {image.shape}/{image.dtype}')
    return np.array(image[..., :3], copy=True, order='C')


class PrismWAMPolicy:
    """Fixed public policy weights with caller-owned native-action RGB history."""

    CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')

    def __init__(self, checkpoint: Path, *, device: str = 'cuda:0',
                 replan_steps: int | None = None, inference_steps: int = 10):
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f'PrismWAM checkpoint not found: {checkpoint}')
        config = _checkpoint_config_path(checkpoint)
        if not os.environ.get('PRISMWAM_ASSETS_ROOT'):
            raise ValueError('Set PRISMWAM_ASSETS_ROOT to the downloaded public assets; see README.md')
        assets = Path(os.environ['PRISMWAM_ASSETS_ROOT']).expanduser().resolve()
        os.environ['PRISMWAM_WEIGHTS_ROOT'] = str(assets)
        source = Path(__file__).resolve().parents[1] / 'third_party/prismwam/src'
        if source.is_dir() and str(source) not in sys.path:
            sys.path.insert(0, str(source))
        import torch
        from omegaconf import OmegaConf
        from prismwam.deployment.policy import load_policy
        from prismwam.deployment.history import RGBHistoryBuffer
        from prismwam.hub import TEXT_ENCODER_FILES

        cfg = _load_policy_config(config)
        data = cfg.data.train
        if (data.get('layout') != 'robotwin' or tuple(data.cameras) !=
                ('cam_head', 'cam_left_wrist', 'cam_right_wrist')):
            raise ValueError('Public RoboDojo policy requires the head/left/right robotwin camera layout')
        if (cfg.deployment.get('benchmark') != 'robodojo' or int(data.action_dim) != 14
                or int(cfg.model.get('proprio_dim', 14)) != 14
                or int(cfg.deployment.action_horizon) != 32
                or int(cfg.deployment.replan_steps) != 16):
            raise ValueError('Expected the public RoboDojo 14D, 32-step policy with 16-step replanning')
        if replan_steps not in (None, 16) or inference_steps < 1:
            raise ValueError('RoboDojo PrismWAM requires replan_steps=16 and positive inference_steps')
        if cfg.deployment.get('text_cache_path'):
            raise ValueError('Use the public RoboDojo config with online text encoding')
        text_root = assets / str(cfg.model.model_id)
        for name in TEXT_ENCODER_FILES:
            if not (text_root / name).is_file():
                raise FileNotFoundError(f'Missing public text asset: {text_root/name}; run download_prismwam.py')
        stats = _action_stats_path(data, config.parent)
        cfg.deployment.statistics_path = str(stats)
        cfg.deployment.num_inference_steps = int(inference_steps)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        self.video_size = tuple(int(v) for v in data.video_size)
        self.depth_min, self.depth_max = float(data.depth_min), float(data.depth_max)
        if (len(self.video_size) != 2 or min(self.video_size) < 3
                or not np.isfinite([self.depth_min, self.depth_max]).all()
                or self.depth_min >= self.depth_max):
            raise ValueError('Invalid public policy image size or depth range')
        self._policy = load_policy(resolved, checkpoint, device=device, dtype=torch.bfloat16,
                                   statistics_path=stats, evaluation=resolved['deployment'])
        self.history = RGBHistoryBuffer.from_config(resolved)
        self.checkpoint, self.config_path = checkpoint, config
        self.metadata = dict(backend='prismwam/PrismWAMPolicy', checkpoint=str(checkpoint),
            checkpoint_fingerprint=_fingerprint(checkpoint), config=str(config),
            action_horizon=32, action_dim=14, action_stats=str(stats), replan_steps=16,
            execution_horizon=16, rgb_memory=cfg.model.get('rgb_memory') is not None,
            supports_observe=True, inferences=0)
        if self.metadata['rgb_memory']:
            self.metadata['rgb_memory_config'] = resolved['model']['rgb_memory']
        self.reset()

    def _mosaic(self, frames):
        """Public RoboDojo layout: head above the two wrists, bilinear resize."""
        import torch
        from torch.nn.functional import interpolate
        height, width = self.video_size
        top = round(height * 2 / 3)
        sizes = ((top, width), (height - top, width // 2), (height - top, width - width // 2))
        resized = [interpolate(frame[None], size=size, mode='bilinear', align_corners=False)[0]
                   for frame, size in zip(frames, sizes)]
        return torch.cat((resized[0], torch.cat(resized[1:], dim=-1)), dim=-2).contiguous()

    def _images(self, observation):
        import torch
        from prismwam.deployment.observations import normalize_depth
        rgbs, depths = [], []
        for camera in self.CAMERAS:
            rgb = _rgb(observation[camera])
            depth = _depth_mm(observation[camera + '_depth'], maximum=self.depth_max)
            if rgb.shape[:2] != depth.shape:
                raise ValueError(f'RGB/depth must be pixel-aligned: {camera}')
            # Preserve the public simulation evaluator's float32 arithmetic.
            rgbs.append(torch.from_numpy(rgb).permute(2, 0, 1).float() * (2.0 / 255.0) - 1.0)
            depths.append(normalize_depth(torch.from_numpy(depth), minimum=self.depth_min, maximum=self.depth_max))
        return self._mosaic(rgbs), self._mosaic(depths)

    def observe(self, observation: dict[str, Any]) -> None:
        """Append exactly one frame after each executed control step."""
        rgb, _ = self._images(observation)
        self.history.append(rgb, step=self.history.step + 1)

    def reset(self) -> None:
        self.history.reset()
        self._policy.reset()

    def infer(self, observation: dict[str, Any], *, record_observation: bool = True):
        states = np.asarray(observation['states'], dtype=np.float32)
        if states.shape != (14,) or not np.isfinite(states).all():
            raise ValueError(f'Expected finite 14D joint state, got {states.shape}')
        instruction = str(observation.get('instruction', '')).strip()
        if not instruction:
            raise ValueError('PrismWAM requires a nonempty task instruction')
        rgb, depth = self._images(observation)
        if record_observation:
            self.history.append(rgb, step=self.history.step + 1)
        inputs = dict(input_image=depth[None], rgb_input_image=rgb[None],
                      rgb_history_images=self.history.tensor(), proprio=states[None],
                      prompt=self._policy.format_prompt(instruction))
        # Explicit observed history is authoritative, including short RSI interventions.
        self._policy.reset()
        result = self._policy.predict(inputs, seed=0)
        if result['normalized'] or result['replan_steps'] != 16:
            raise ValueError('PrismWAM must return physical actions with the recorded replan interval')
        actions = np.asarray(result['action'], dtype=np.float32)
        if actions.shape != (32, 14) or not np.isfinite(actions).all():
            raise ValueError(f'PrismWAM returned invalid action array {actions.shape}')
        actions = actions.copy()
        actions[:, [6, 13]] = np.clip(actions[:, [6, 13]], 0.0, 1.0)
        index = int(self.metadata['inferences'])
        self.metadata['inferences'] = index + 1
        return actions, dict(prediction_id=f'prismwam-{index}', inference_index=index,
                             action_horizon=len(actions), checkpoint=str(self.checkpoint))


def write_identity(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + '\n')
