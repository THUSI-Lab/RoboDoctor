import copy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hybrid_rollout.robodojo.io import InputError
from robodoctor.geometry import organized_cloud, ground_pixel, pose7, solve_ik, segment_distance
from robodoctor.client import GroundedTools
from robodoctor.camera_config import enable_camera_depth
from robodoctor.schema import tool_specs


def test_pixel_projection_actual_fy_usd_and_rotated_base():
    depth = np.full((3, 5), 2.)
    k = np.array([[4., 0, 2], [0, 8., 1], [0, 0, 1]])
    camera = pose7([1, 2, 3, 1, 0, 0, 0])
    cloud = organized_cloud(depth, k, camera)
    np.testing.assert_allclose(cloud[2, 4], [2, 1.75, 1])
    base = pose7([1, 1, 1, np.sqrt(.5), 0, 0, np.sqrt(.5)])
    point, world = ground_pixel(cloud, 4, 2, base)
    np.testing.assert_allclose(point, [.75, -1, 0], atol=1e-12)
    np.testing.assert_allclose(world, [2, 1.75, 1])


@pytest.mark.parametrize('depth', [0., -1., np.nan, np.inf])
def test_depth_holes_do_not_pick_neighbour(depth):
    cloud = organized_cloud(np.array([[1., depth]]), np.diag([2., 2., 1.]), np.eye(4))
    with pytest.raises(ValueError, match='valid depth'):
        ground_pixel(cloud, 1, 0, np.eye(4))


@pytest.mark.parametrize('pixel', [(-1, 0), (2, 0), (0, 2), (True, 0), (.5, 0)])
def test_pixel_bounds(pixel):
    with pytest.raises(ValueError, match='Pixel'):
        ground_pixel(np.ones((2, 2, 3)), *pixel, np.eye(4))


class CartesianFK:
    def matrix(self, q):
        t = np.eye(4)
        t[:3, 3] = q[:3]
        t[:3, :3] = Rotation.from_rotvec(q[3:]).as_matrix()
        return t


def test_ik_reachability_and_quaternion_order():
    fk = CartesianFK()
    target = pose7([.2, .3, .4, np.cos(.1), 0, 0, np.sin(.1)])
    solution = solve_ik(fk, np.zeros(6), np.tile([-1, 1], (6, 1)), target)
    np.testing.assert_allclose(fk.matrix(solution), target, atol=1e-6)
    target[0, 3] = 3
    with pytest.raises(ValueError, match='reachable'):
        solve_ik(fk, np.zeros(6), np.tile([-1, 1], (6, 1)), target)
    with pytest.raises(ValueError, match='unit'):
        pose7([0, 0, 0, 2, 0, 0, 0])


def test_native_near_zero_joint_noise_does_not_collapse_ik_trust_region():
    goal = pose7([.3, -.1, .2, 1, 0, 0, 0])
    solved = solve_ik(CartesianFK(), np.full(6, 1e-12), np.tile([-1, 1], (6, 1)), goal)
    np.testing.assert_allclose(CartesianFK().matrix(solved), goal, atol=1e-6)


def test_path_crossing_and_depth_config():
    assert segment_distance(np.array([-1., 0, 0]), np.array([1., 0, 0]), np.zeros(3)) == 0
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(dict(camera=dict(cam_head={}, cam_left_wrist={}, cam_right_wrist={}, annotator=dict(common=dict(
        enabled=True, rgb_capture=dict(type='rgb', device='cpu')))),
        eval_cfg=dict(observation=dict(vision=dict(depth=False, approximate_depth=True)))))
    enable_camera_depth(cfg)
    assert cfg.eval_cfg.observation.vision.depth
    assert cfg.camera.annotator.cam_head.distance_to_image_plane_capture.type == 'distance_to_image_plane'
    assert cfg.camera.annotator.cam_head.rgb_capture.type == 'rgb'


class Sim:
    """Stateful fake separating IK validation, command submission and measured motion."""
    converge = True
    terminal = False
    reject = False
    def __init__(self, *args, **kwargs):
        self.tick = 0
        self.calls = []
        self.eef_targets = []
        self.positions = np.zeros((2, 3))
        self.quaternions = np.array([[1., 0, 0, 0]]*2)
        self.states = np.zeros(14)
        self.states[[6, 13]] = 1
    def request(self, op, **args):
        self.calls.append(op)
        if op == 'metadata':
            return dict(task='build_tower', instruction='Build a tower.', max_episode_steps=200, control_dt=.04)
        if op == 'reset':
            return dict(episode_id='test', step_id=0)
        if op == 'teacher_observation':
            return dict(instruction='Build a tower.', remaining_steps=200-self.tick,
                states=self.states.copy(), eef_positions=self.positions.copy(),
                eef_quaternions_wxyz=self.quaternions.copy(),
                **{key:np.zeros((8, 8, 3), np.uint8) for key in GroundedTools.camera_keys})
        if op == 'grounding':
            return dict(ok=True, grounding_id='g', position=[0, 0, -.3],
                        frame='environment_origin', physical_steps=0)
        if op == 'get_depth':
            return dict(ok=True, camera=f'cam_{args["arm"]}_wrist', arm=args['arm'],
                        depth_m=.123, physical_steps=0, step_id=self.tick)
        if op == 'prepare_targets':
            return dict(ok=not self.reject, error='unreachable', physical_steps=0)
        if op == 'eef_joint_target':
            self.target = copy.deepcopy(args['targets'])
            self.eef_targets.append(self.target)
            return dict(action=np.zeros(14), diagnostics={})
        if op == 'chunk_step':
            self.tick += 1
            for i, arm in enumerate(('left', 'right')):
                if self.converge:
                    self.positions[i] = self.target[arm]['position']
                    self.quaternions[i] = self.target[arm]['quaternion_wxyz']
                self.states[7*i+6] = float(not self.target[arm]['gripper_closed'])
            return dict(step_id=self.tick, steps=[dict(valid=True, executed_action=self.states.copy(),
                obs=dict(states=self.states.copy()), terminated=self.terminal, truncated=False, success=False)])
        if op == 'finish_pilot':
            return dict(step_id=self.tick, success=False, terminated=self.terminal, truncated=False,
                        reason=args['reason'])
        return {}


def started(tmp_path, sim=Sim):
    client = GroundedTools(tmp_path, 'build_tower', rpc_factory=sim, max_decisions=0)
    args = client.next_call(); args.pop('tool')
    client.start(**args)
    return client


def act_args(client):
    args = client.next_call(); args.pop('tool')
    args['response'] = dict(request_id=client.request['request_id'], mode='eef', steps=3,
        stop_on_reach=False, reason='left: refine the visible grasp, right: hold measured pose and preserve gripper state',
        target={arm:dict(position=client.request['current_eef'][arm]['position'].copy(),
                         quaternion_wxyz=client.request['current_eef'][arm]['quaternion_wxyz'].copy(),
                         gripper_closed=bool(client.obs['states'][7*i+6] < .5))
                for i, arm in enumerate(('left', 'right'))})
    return args


def move_args(client):
    client.grounding(client.request['request_id'], 3, 4)
    args = act_args(client)
    args['response'].update(steps=100, stop_on_reach=True)
    args['response']['target']['left']['position'] = [.2, 0, 0]
    return args


def test_ground_move_arrival_and_stale_request(tmp_path):
    client = started(tmp_path)
    args = move_args(client)
    assert client.tick == 0
    packet = client.act(**args)
    assert packet['movement']['reached'] and client.tick == 1
    assert set(packet['movement']['arms']) == {'left', 'right'}
    assert 'current_eef_base' not in packet
    assert packet['available_tools'] == ['grounding', 'get_depth', 'robodojo_act', 'robodojo_joint_act']
    assert client.counters['predictions'] == 0 and client.student is None
    assert client.sim.calls.count('reset') == 1
    with pytest.raises(InputError, match='request_id'):
        client.act(**dict(act_args(client), response=args['response']))
    assert not client.groundings
    result = client.grounding(client.request['request_id'], 3, 4)
    assert result['frame'] == 'environment_origin'


@pytest.mark.parametrize('terminal', [False, True])
def test_nonarrival_never_claims_success(tmp_path, terminal):
    class Stuck(Sim):
        converge = False
    Stuck.terminal = terminal
    client = started(tmp_path, Stuck)
    args = move_args(client); args['response']['steps'] = 2
    packet = client.act(**args)
    assert not packet['movement']['reached']
    assert packet['movement']['status'] == ('terminal' if terminal else 'step_budget')
    assert client.tick == (1 if terminal else 2)


def test_rejected_ik_executes_neither_arm_and_can_be_corrected(tmp_path):
    class Unreachable(Sim):
        reject = True
    client = started(tmp_path, Unreachable)
    args = move_args(client)
    with pytest.raises(InputError, match='unreachable'):
        client.act(**args)
    assert client.tick == 0 and not client.history and not client.sim.eef_targets
    client.sim.reject = False
    assert client.act(**args)['movement']['reached']


def test_tools_have_eef_and_joint_motion_and_shared_grounding_frame():
    specs = {s['name']:s for s in tool_specs()}
    assert set(specs) == {'robodojo_start', 'grounding', 'get_depth', 'robodojo_act', 'robodojo_joint_act'}
    assert 'arm' not in specs['grounding']['inputSchema']['properties']
    response = specs['robodojo_act']['inputSchema']['properties']['response']
    assert response['properties']['steps']['maximum'] == 150
    assert set(response['properties']['target']['properties']) == {'left', 'right'}


@pytest.mark.parametrize('arm', ['left', 'right'])
@pytest.mark.parametrize('translation,angle', [(.2, 0), (0, np.pi/2), (.2, np.pi)])
def test_act_large_targets_reach_ik_without_clipping(tmp_path, arm, translation, angle):
    client = started(tmp_path)
    args = act_args(client)
    target = args['response']['target'][arm]
    target['position'][2] += translation
    target['quaternion_wxyz'] = [np.cos(angle/2), 0, np.sin(angle/2), 0]
    packet = client.act(**args)
    assert packet['step_id'] == 3
    assert len(client.sim.eef_targets) == 3
    assert all(t == args['response']['target'] for t in client.sim.eef_targets)
    assert packet['movement']['arms'][arm]['reached']


def test_dual_arm_coarse_approach_and_grippers_share_one_ack(tmp_path):
    client = started(tmp_path)
    args = move_args(client)
    args['response']['target']['right'].update(position=[-.3, .1, .2], gripper_closed=True)
    packet = client.act(**args)
    assert client.tick == 1 and packet['movement']['reached']
    np.testing.assert_allclose(client.obs['eef_positions'], [[.2,0,0],[-.3,.1,.2]])
    np.testing.assert_equal(client.obs['states'][[6,13]], [1,0])
    assert len(client.sim.eef_targets) == 1 and len(client.history) == 1


def test_gripper_only_action_executes_even_if_poses_already_reached(tmp_path):
    client = started(tmp_path)
    args = act_args(client)
    args['response'].update(stop_on_reach=True)
    args['response']['target']['left']['gripper_closed'] = True
    packet = client.act(**args)
    assert client.tick == 1 and client.obs['states'][6] == 0
    assert packet['movement']['reached']


def test_early_stop_waits_for_both_arms(tmp_path):
    class SlowerRight(Sim):
        def request(self, op, **args):
            previous = self.positions[1].copy()
            result = super().request(op, **args)
            if op == 'chunk_step' and self.tick < 3:
                self.positions[1] = previous
            return result
    client = started(tmp_path, SlowerRight)
    args = move_args(client)
    args['response']['target']['right']['position'] = [-.3, 0, .2]
    packet = client.act(**args)
    assert client.tick == 3 and packet['movement']['reached']
    assert len(client.sim.eef_targets) == 3


def test_smoke_requires_grounding_and_forbids_grasp_commands(tmp_path):
    client = started(tmp_path)
    client.smoke_approach = True
    with pytest.raises(InputError, match='Ground'):
        client.act(**act_args(client))
    args = move_args(client)
    args['response']['target']['right']['gripper_closed'] = True
    with pytest.raises(InputError, match='open grippers'):
        client.act(**args)
    assert client.tick == 0


@pytest.mark.parametrize('bad', [dict(position=[np.nan, 0, 0]), dict(position=[np.inf, 0, 0]),
    dict(position=[0, 0]), dict(quaternion_wxyz=[1, 0, 0]), dict(quaternion_wxyz=[2, 0, 0, 0]),
    dict(quaternion_wxyz=[np.nan, 0, 0, 0]), dict(gripper_closed=1)])
def test_act_invalid_pose_rejected_before_execution(tmp_path, bad):
    client = started(tmp_path)
    args = act_args(client)
    args['response']['target']['right'].update(bad)
    with pytest.raises(InputError):
        client.act(**args)
    assert client.tick == 0 and not client.history and 'prepare_targets' not in client.sim.calls


@pytest.mark.parametrize('bad', [dict(request_id='stale'), dict(mode='joint'), dict(actions=[[0]*14]),
    dict(target={'left': {}}), dict(steps=True), dict(steps=151), dict(steps=0),
    dict(stop_on_reach=1), dict(reason='')])
def test_act_invalid_protocol_rejected_before_execution(tmp_path, bad):
    client = started(tmp_path)
    args = act_args(client); args['response'].update(bad)
    with pytest.raises(InputError):
        client.act(**args)
    assert client.tick == 0 and 'chunk_step' not in client.sim.calls


@pytest.mark.parametrize('reason', [
    'Move both arms.', 'left: approach the object', 'right: hold, left: approach',
    'left: approach, right hold', 'left: , right: hold', 'left: approach, right: ',
    'left: approach\n, right: hold', 'left: approach, right: hold\n', None,
])
def test_act_reason_format_rejected_before_ik(tmp_path, reason):
    client = started(tmp_path)
    args = act_args(client)
    args['response']['reason'] = reason
    with pytest.raises(InputError, match='left:.*right:'):
        client.act(**args)
    assert client.tick == 0 and not client.history
    assert 'prepare_targets' not in client.sim.calls


def test_act_reason_format_preserved_in_history_and_schema(tmp_path):
    import re
    from robodoctor.schema import tool_specs
    client = started(tmp_path)
    args = act_args(client)
    reason = 'left: fine adjustment toward the visible edge, keep gripper open, right: hold measured pose and preserve gripper state'
    args['response']['reason'] = reason
    client.act(**args)
    assert client.history[-1]['response']['reason'] == reason
    spec = next(s for s in tool_specs() if s['name'] == 'robodojo_act')
    reason_schema = spec['inputSchema']['properties']['response']['properties']['reason']
    assert re.fullmatch(reason_schema['pattern'], reason)
    assert not re.fullmatch(reason_schema['pattern'], 'Move both arms.')


@pytest.mark.parametrize('terminal', [True, False])
def test_act_preserves_terminal_and_decision_budget(tmp_path, terminal):
    client = started(tmp_path)
    client.act(**move_args(client))
    client.sim.terminal = terminal
    client.max_decisions = 2
    packet = client.act(**act_args(client))
    assert client.phase == 'done' and packet['rollout_finished']
    assert client.history[-1]['executed_steps'] == (1 if terminal else 3)
    assert packet['result']['reason'] == ('terminal' if terminal else 'decision_budget')


def test_stagnation_stops_long_approach(tmp_path):
    class Stuck(Sim):
        converge = False
    client = started(tmp_path, Stuck)
    packet = client.act(**move_args(client))
    assert packet['movement']['status'] == 'stalled' and client.tick == 20


def test_large_target_keeps_shared_local_ik_increment_bounds():
    from hybrid_rollout.robodojo.robodojo_server.kinematics import ArmFK
    fk = CartesianFK()
    target = dict(position=[.4, .3, .2], quaternion_wxyz=[0, 0, 1, 0])
    joints, diagnostics = ArmFK.bounded_target(fk, np.zeros(6), np.tile([-.04,.04], (6,1)), np.eye(4), target)
    assert np.linalg.norm(diagnostics['bounded_task_delta'][:3]) <= .02 + 1e-9
    assert np.linalg.norm(diagnostics['bounded_task_delta'][3:]) <= .1 + 1e-9
    assert np.max(np.abs(joints)) <= .04 + 1e-9
