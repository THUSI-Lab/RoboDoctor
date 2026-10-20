from copy import deepcopy
from pathlib import Path
import os

import numpy as np
import pytest

from hybrid_rollout.robodojo.io import InputError
from robodoctor.geometry import pose, pose7, solve_ik
from robodoctor.joint_motion import NumericalMotion
from robodoctor.client import GroundedTools
from robodoctor.hybrid import HybridTools
from test_session import session
from test_grounded import Sim, started
from test_hybrid import Student, plan, infer, assessment


def numerical():
    value = session()
    for robot, key in zip(value.kinematics.manager.robot_list, value.kinematics.manager.robot_key):
        robot.arm_joints_name = [f'joint{i}' for i in range(1, 7)]
        key.data.soft_joint_vel_limits = np.ones((1, 6))
    value.motion = NumericalMotion(value.kinematics, .04)
    return value


def joint_targets(delta=.1):
    return dict(left=dict(qpos=[delta, 0, 0, 0, 0, 0], gripper_closed=True),
                right=dict(qpos=[0]*6, gripper_closed=False))


def test_delta_anchored_once_limits_and_measured_velocity_tracking():
    value = numerical()
    q = {'left':np.full(6, .2), 'right':np.full(6, -.2)}
    value.kinematics.manager.get_joint = lambda robot: {0:q[robot.robot_name]}
    goals = joint_targets(.1)
    result = value.prepare_joints(goals, 'delta')
    assert result['ok'] and result['physical_steps'] == 0
    np.testing.assert_allclose(result['joint_targets']['left'], [.3,.2,.2,.2,.2,.2])
    first = value.motion.target(goals, value.step_id)
    assert first['action'][0] == pytest.approx(.24)
    assert first['action'][6] == 0 and first['action'][13] == 1
    np.testing.assert_allclose(first['action'][7:13], q['right'])
    q['left'][0] = .24
    assert value.motion.target(goals, value.step_id+1)['action'][0] == pytest.approx(.28)
    q['left'][0] = .28
    assert value.motion.target(goals, value.step_id+2)['action'][0] == pytest.approx(.30)
    q['left'][0] = .30
    assert value.motion.target(goals, value.step_id+3)['action'][0] == pytest.approx(.30)


@pytest.mark.parametrize('bad', [[2,0,0,0,0,0], [np.nan]*6, [0]*5])
def test_joint_bad_second_arm_rejects_both_without_reusing_old_goal(bad):
    value = numerical()
    goals = joint_targets()
    assert value.prepare_joints(goals, 'absolute')['ok']
    goals['right']['qpos'] = bad
    assert not value.prepare_joints(goals, 'absolute')['ok']
    assert value.motion.plan is None and value.step_id == 3
    with pytest.raises(ValueError):
        value.motion.target(joint_targets(), value.step_id)


def test_numerical_ik_rotated_bases_hold_and_unreachable_atomicity():
    value = numerical()
    goals = {arm:dict(pose(value.kinematics.root(arm) @ pose7([.3,0,0,1,0,0,0])),
                      gripper_closed=False) for arm in ('left','right')}
    result = value.prepare_targets(goals)
    assert result['ok'] and result['planner'] == 'numerical_ik'
    for q in result['joint_targets'].values():
        np.testing.assert_allclose(q, [.3,0,0,0,0,0], atol=1e-6)
    assert all(not p.checked for p in value.kinematics.manager.planner.values())
    goals['right']['position'] = [100,0,0]
    assert not value.prepare_targets(goals)['ok'] and value.motion.plan is None
    hold = {arm:dict(pose(value.kinematics.root(arm)), gripper_closed=True) for arm in ('left','right')}
    value.kinematics.manager.get_joint = lambda robot:{0:np.full(6,.2)}
    assert value.prepare_targets(hold)['ok']
    assert value.motion.target(hold, value.step_id)['action'][0] == .2


class JointSim(Sim):
    """Real numerical target generation over a simple measurable FK fixture."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.native = numerical()
        self.native.kinematics.root = lambda arm: np.eye(4)
        self.native.kinematics.manager.get_joint = lambda robot:{0:self.states[
            0 if robot.robot_name == 'left' else 7:6 if robot.robot_name == 'left' else 13].copy()}

    def request(self, op, **args):
        if op == 'metadata':
            return dict(super().request(op, **args), supports_joint_control=True, motion_planner='numerical_ik')
        if op == 'joint_state':
            return dict(arms=self.native.motion.joint_state(), physical_steps=0)
        if op == 'prepare_joints':
            self.calls.append(op)
            return self.native.motion.prepare_joints(args['targets'], args['coordinate_mode'], self.tick)
        if op == 'eef_joint_target':
            self.calls.append(op)
            return self.native.motion.target(args['targets'], self.tick)
        if op == 'fk_preview':
            return dict(measured_fk_check=dict(passed=True), trajectory=[dict(index=i) for i in range(50)])
        if op == 'chunk_step':
            self.calls.append(op)
            self.tick += 1
            action = args['actions'][0]
            if self.converge:
                self.states = action.copy()
            for i, arm in enumerate(('left','right')):
                fk = pose(self.native.kinematics.fk[arm].matrix(self.states[7*i:7*i+6]))
                self.positions[i] = fk['position']
                self.quaternions[i] = fk['quaternion_wxyz']
            return dict(step_id=self.tick, steps=[dict(valid=True, executed_action=action.copy(),
                obs=dict(states=self.states.copy()), terminated=self.terminal, truncated=False, success=False)])
        return super().request(op, **args)


def arguments(client, *, preview=False):
    paths = GroundedTools.next_call(client)
    return dict(observation_path=paths['observation_path'], output_dir=paths['output_dir'], preview_only=preview,
        response=dict(request_id=client.request['request_id'], mode='joint', coordinate_mode='delta',
            steps=5, stop_on_reach=True, target=joint_targets(),
            reason='left: small joint retreat into visible clear space, right: hold measured joints and keep gripper open'))


def test_preview_no_steps_then_execution_measured_arrival_and_stale_rejection(tmp_path):
    client = started(tmp_path, JointSim)
    assert 'current_joints' in client.request
    args = arguments(client, preview=True)
    preview = client.joint_act(**args)
    assert preview['physical_steps'] == 0 and client.tick == 0 and not client.history
    assert not Path(args['output_dir']).exists()
    packet = client.joint_act(**dict(args, preview_only=False))
    assert packet['movement']['reached'] and client.tick == 3
    assert packet['movement']['arms']['left']['joint_error_rad'] < .01
    assert client.counters['gpt_joint_steps'] == 3 and client.counters['gpt_eef_steps'] == 0
    assert client.history[0]['source'] == 'gpt_joint'
    np.testing.assert_allclose(client.obs['states'][:6], [.1,0,0,0,0,0], atol=1e-7)
    with pytest.raises(InputError):
        client.joint_act(**dict(arguments(client), response=args['response']))


@pytest.mark.parametrize('terminal', [False, True])
def test_blocked_joint_command_does_not_claim_arrival_and_obeys_termination(tmp_path, terminal):
    class Stuck(JointSim):
        converge = False
    Stuck.terminal = terminal
    client = started(tmp_path, Stuck)
    args = arguments(client); args['response']['steps'] = 40
    packet = client.joint_act(**args)
    assert not packet['movement']['reached']
    assert client.tick == (1 if terminal else 20)
    assert packet['movement']['status'] == ('terminal' if terminal else 'stalled')


def test_joint_recovery_preserves_hybrid_gate_and_discards_old_prediction(tmp_path):
    client = HybridTools(tmp_path, 'build_tower', Student(), rpc_factory=JointSim, max_decisions=0)
    call = client.next_call(); call.pop('tool'); client.start(**call)
    with pytest.raises(InputError, match='update_plan'):
        client.joint_act(**arguments(client))
    plan(client); prediction = infer(client)
    with pytest.raises(InputError, match='intervene'):
        client.joint_act(**arguments(client))
    client.review_pi05(client.request['request_id'], prediction['prediction_id'], assessment('intervene'))
    preview = client.joint_act(**arguments(client, preview=True))
    assert preview['physical_steps'] == 0 and client.proposal is not None
    client.joint_act(**arguments(client))
    assert client.proposal is None and client.review is None
    assert client.counters['recovery_steps'] == client.counters['gpt_joint_steps'] == 3
    assert client.history[-1]['discarded_prediction']['prediction_id'] == prediction['prediction_id']
    assert client.next_call()['tool'] == 'update_plan'


def test_joint_hold_still_applies_gripper_and_invalid_limits_do_not_move(tmp_path):
    client = started(tmp_path, JointSim)
    args = arguments(client)
    args['response']['target']['right']['qpos'][0] = 2
    with pytest.raises(InputError, match='limits'):
        client.joint_act(**args)
    assert not client.history and client.tick == 0 and not Path(args['output_dir']).exists()
    args['response']['target'] = joint_targets(0)
    packet = client.joint_act(**args)
    assert client.tick == 1 and packet['movement']['reached']
    assert client.obs['states'][6] == 0 and client.obs['states'][13] == 1


def test_actual_x5_urdf_ik_roundtrip():
    from hybrid_rollout.robodojo.robodojo_server.kinematics import ArmFK
    path = Path(os.environ['ROBODOJO_SOURCE'])/'third_party/curobo/robot/x5_v2/X5A.urdf'
    fk = ArmFK(path, [f'joint{i}' for i in range(1,7)])
    target_q = np.array([.1,.4,-.5,.2,.3,-.2])
    goal = fk.matrix(target_q)
    q = solve_ik(fk, target_q+np.array([.02,-.02,.03,0,-.02,.01]), np.tile([-3.14,3.14],(6,1)), goal)
    np.testing.assert_allclose(fk.matrix(q), goal, atol=1e-5)
