import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from hybrid_rollout.robodojo.io import InputError
from robodoctor.camera_config import enable_camera_depth
from robodoctor.session import RoboDojoSession
from test_grounded import Sim, started, move_args
from test_recovery import policy, ready, tool_call, completed


def wrist_session(*, aliases=False, channel=False):
    value = RoboDojoSession.__new__(RoboDojoSession)
    value.episode_id, value.step_id = 'depth-test', 7
    value.poisoned = value.finished = value.terminated = value.truncated = False
    raw = dict(vision={})
    for arm, depth in [('left', .123456), ('right', .654321)]:
        name = f'{arm}_camera' if aliases else f'cam_{arm}_wrist'
        data = np.full((3, 5, 1) if channel else (3, 5), depth)
        raw['vision'][name] = dict(depth=data, color=np.zeros((3, 5, 3), np.uint8))
    value._capture_wrist_depth(raw)
    return value, raw


@pytest.mark.parametrize('aliases', [False, True])
@pytest.mark.parametrize('channel', [False, True])
def test_both_wrists_use_copied_same_frame_metric_depth(aliases, channel):
    value, raw = wrist_session(aliases=aliases, channel=channel)
    # Subsequent mutation of the native capture buffer cannot change the cache.
    for data in raw['vision'].values():
        data['depth'][...] = 99
    for arm, expected in [('left', .123456), ('right', .654321)]:
        result = value.dispatch('get_depth', dict(episode_id='depth-test', step_id=7, arm=arm, x=4, y=2))
        assert result['ok'] and result['depth_m'] == expected
        assert result['camera'] == f'cam_{arm}_wrist'
        assert result['image_size'] == [5, 3]
        assert result['depth_type'] == 'distance_to_image_plane'
        assert result['units'] == 'm' and result['step_id'] == 7 and result['physical_steps'] == 0
    assert value.step_id == 7


@pytest.mark.parametrize('x,y', [(-1, 0), (5, 0), (0, 3), (0, -1), (1.5, 0), (True, 0), (0, False)])
def test_depth_rejects_invalid_pixels(x, y):
    value, _ = wrist_session()
    result = value.get_depth('left', x, y)
    assert not result['ok'] and result['physical_steps'] == 0 and 'depth_m' not in result


@pytest.mark.parametrize('invalid', [0., -1., np.nan, np.inf, -np.inf])
def test_depth_holes_do_not_sample_neighbours(invalid):
    value, _ = wrist_session()
    value.wrist_depth['cam_right_wrist'][1, 2] = invalid
    assert not value.get_depth('right', 2, 1)['ok']
    assert value.get_depth('right', 1, 1)['ok']


def test_depth_rejects_wrong_arm_stale_tick_and_finished_episode():
    value, _ = wrist_session()
    assert not value.get_depth('both', 1, 1)['ok']
    args = dict(episode_id='depth-test', step_id=6, arm='left', x=1, y=1)
    with pytest.raises(ValueError, match='Stale'):
        value.dispatch('get_depth', args)
    args['step_id'] = 7
    value.finished = True
    with pytest.raises(ValueError, match='finished'):
        value.dispatch('get_depth', args)


@pytest.mark.parametrize('bad', ['missing_camera', 'missing_depth', 'mismatched_shape'])
def test_wrist_depth_capture_fails_on_missing_or_unaligned_depth(bad):
    value, raw = wrist_session()
    if bad == 'missing_camera':
        del raw['vision']['cam_left_wrist']
    elif bad == 'missing_depth':
        del raw['vision']['cam_left_wrist']['depth']
    else:
        raw['vision']['cam_left_wrist']['depth'] = np.ones((5, 3))
    with pytest.raises(ValueError, match='cam_left_wrist'):
        value._capture_wrist_depth(raw)


@pytest.mark.parametrize('aliases', [False, True])
def test_all_camera_depth_enabled_without_losing_custom_annotators(aliases):
    names = ['cam_head', 'left_camera', 'right_camera'] if aliases else ['cam_head', 'cam_left_wrist', 'cam_right_wrist']
    cfg = OmegaConf.create(dict(camera={**{n:{} for n in names}, 'annotator':{
        'common': {'enabled':True, 'rgb_capture':{'type':'rgb', 'device':'cpu'}},
        names[1]: {'enabled':False, 'rgb_capture':{'type':'rgb', 'device':'cpu'},
                   'normals_capture':{'type':'normals', 'device':'cpu'}}}},
        eval_cfg={'observation':{'vision':{'depth':False, 'approximate_depth':True}}}))
    enable_camera_depth(cfg)
    for name in names:
        annotator = cfg.camera.annotator[name]
        assert annotator.enabled and annotator.distance_to_image_plane_capture.type == 'distance_to_image_plane'
        assert annotator.rgb_capture.type == 'rgb'
    assert cfg.camera.annotator[names[1]].normals_capture.type == 'normals'
    assert cfg.eval_cfg.observation.vision.depth and not cfg.eval_cfg.observation.vision.approximate_depth


@pytest.mark.parametrize('wrist_annotators', [False, True])
def test_depth_setup_before_robot_manager_adds_wrist_geometry(wrist_annotators):
    # At enable_camera_depth time, native camera_config.yml only has head
    # geometry. RobotManager._load_camera_cfg adds the wrist mounts later.
    rgb = dict(enabled=True, rgb_capture=dict(type='rgb', device='cpu'))
    annotators = dict(common=rgb, cam_head=rgb)
    if wrist_annotators:
        annotators.update(cam_left_wrist=dict(rgb, normals_capture=dict(type='normals', device='cpu')),
                          cam_right_wrist=rgb)
    cfg = OmegaConf.create(dict(camera=dict(cam_head=dict(camera=dict(type='Gemini_345Lg')),
        annotator=annotators), eval_cfg=dict(observation=dict(vision=dict(depth=False)))))
    OmegaConf.set_struct(cfg, True)
    enable_camera_depth(cfg)
    assert OmegaConf.is_struct(cfg)
    assert 'cam_left_wrist' not in cfg.camera and 'cam_right_wrist' not in cfg.camera
    for name in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
        capture = cfg.camera.annotator[name]
        assert capture.distance_to_image_plane_capture.type == 'distance_to_image_plane'
        assert capture.rgb_capture.type == 'rgb'
    if wrist_annotators:
        assert cfg.camera.annotator.cam_left_wrist.normals_capture.type == 'normals'


def test_client_depth_preserves_grounding_prepositioning_and_budgets(tmp_path):
    client = started(tmp_path)
    client.act(**move_args(client))
    client.grounding(client.request['request_id'], 2, 3)
    grounding = client.groundings.copy()
    request_id = client.request['request_id']
    calls_before = len(client.sim.calls)
    for arm in ('left', 'right'):
        result = client.get_depth(request_id, arm, 4, 5)
        assert result['depth_m'] == .123 and result['request_id'] == request_id
        assert result['image_path'].endswith(f'cam_{arm}_wrist.png')
    assert client.sim.calls[calls_before:] == ['get_depth', 'get_depth']
    assert client.groundings == grounding
    assert client.tick == 1 and len(client.history) == 1 and client.counters['depth_calls'] == 2
    assert json.loads((client.output/'depth_0002.json').read_text())['arm'] == 'right'
    client.act(**move_args(client))
    with pytest.raises(InputError, match='latest'):
        client.get_depth(request_id, 'left', 4, 5)


def test_client_depth_error_is_recoverable_without_changing_stage(tmp_path):
    class Hole(Sim):
        def request(self, op, **args):
            if op == 'get_depth':
                return dict(ok=False, error='No valid depth at this wrist pixel', physical_steps=0)
            return super().request(op, **args)
    client = started(tmp_path, Hole)
    with pytest.raises(InputError, match='No valid depth'):
        client.get_depth(client.request['request_id'], 'left', 0, 0)
    assert client.phase == 'act' and client.tick == 0 and client.counters['depth_calls'] == 0


def test_codex_routes_get_depth_without_moving_before_next_act(policy, tmp_path):
    client = ready(tmp_path)
    def depth(t):
        return dict(id=99, method='item/tool/call', params=dict(threadId=t.thread, turnId=t.turn,
            callId='wrist-depth', tool='get_depth', arguments=dict(
                request_id=client.request['request_id'], arm='left', x=2, y=3)))
    policy.transport.events = [depth, tool_call(client, terminal=True), completed()]
    policy.run(client)
    response = policy.transport.replies[0][1]
    assert response['success'] and len(response['contentItems']) == 1
    assert json.loads(response['contentItems'][0]['text'])['depth_m'] == .123
    assert client.tick == 2 and client.counters['depth_calls'] == 1
