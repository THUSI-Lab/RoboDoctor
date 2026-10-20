from copy import deepcopy

import numpy as np
import pytest

from robodoctor.geometry import pose, pose7
from test_session import session


def targets(value):
    return {arm: dict(pose(value.kinematics.root(arm) @ pose7([.3, 0, 0, 1, 0, 0, 0])),
                      gripper_closed=(arm == 'left')) for arm in ('left', 'right')}


def test_worker_timeout_rejects_both_targets_without_motion():
    value = session()
    def timeout(**kwargs):
        raise RuntimeError('cuRobo solve_ik_to_joint exceeded 120s')
    value.kinematics.manager.planner['right'].solve_ik_to_joint = timeout
    result = value.prepare_targets(targets(value))
    assert not result['ok'] and 'exceeded 120s' in result['error']
    assert result['physical_steps'] == 0 and value.step_id == 3
    assert value.motion.plan is None


def test_native_ik_and_planned_path_are_executed_at_control_rate():
    value = session()
    goals = targets(value)
    assert value.prepare_targets(goals)['ok']
    for planner in value.kinematics.manager.planner.values():
        assert len(planner.checked) == 1
    # .01-second native samples, .04-second controls: first command is sample 4.
    first = value.motion.target(goals, value.step_id)
    value.episode_id = 'motion-test'
    value.poisoned = False
    assert value.dispatch('eef_joint_target', dict(episode_id=value.episode_id,
        step_id=value.step_id, targets=goals)) == first
    np.testing.assert_allclose(first['action'], [.12, 0, 0, 0, 0, 0, 0, .12, 0, 0, 0, 0, 0, 1], atol=1e-12)
    assert first['diagnostics']['left']['method'] == 'curobo_native_ik_plan_joint'
    # Reading a proposal without an ACK must not consume a trajectory point.
    assert value.motion.target(goals, value.step_id) == first
    value.step_id += 1
    second = value.motion.target(goals, value.step_id)
    assert second['action'][0] == pytest.approx(.24)
    value.step_id += 10
    last = value.motion.target(goals, value.step_id)
    assert last['action'][0] == pytest.approx(.3)
    assert last['action'][7] == pytest.approx(.3)


@pytest.mark.parametrize('failure', ['fail_ik', 'fail_plan'])
def test_either_arm_failure_invalidates_entire_plan_without_motion(failure):
    value = session()
    goals = targets(value)
    assert value.prepare_targets(goals)['ok']
    setattr(value.kinematics.manager.planner['right'], failure, True)
    result = value.prepare_targets(goals)
    assert not result['ok'] and result['error'].startswith('right:')
    assert result['physical_steps'] == 0 and value.step_id == 3
    with pytest.raises(ValueError, match='Prepare'):
        value.motion.target(goals, value.step_id)


def test_gripper_hold_preserves_measured_joints_and_needs_no_ik():
    value = session()
    goals = {arm: dict(pose(value.kinematics.root(arm)), gripper_closed=True)
             for arm in ('left', 'right')}
    manager = value.kinematics.manager
    manager.get_joint = lambda robot: {0: np.full(6, .2)}
    assert value.prepare_targets(goals)['ok']
    assert all(not p.checked for p in manager.planner.values())
    command = value.motion.target(goals, value.step_id)['action']
    np.testing.assert_allclose(command, [*[.2]*6, 0, *[.2]*6, 0])


def test_changed_targets_need_a_new_plan_and_do_not_replay_previous_one():
    value = session()
    goals = targets(value)
    assert value.prepare_targets(goals)['ok']
    revised = deepcopy(goals)
    revised['left']['position'][0] += .01
    with pytest.raises(ValueError, match='Prepare'):
        value.motion.target(revised, value.step_id)
    assert value.prepare_targets(revised)['ok']
    value.motion.target(revised, value.step_id)


def test_fractional_time_sampling_and_unequal_trajectory_lengths():
    value = session()
    goals = targets(value)
    right = value.kinematics.manager.planner['right']
    right.dt = .03
    right.plan_joint = lambda start, end: dict(status='Success', position=np.linspace(start, end, 3))
    assert value.prepare_targets(goals)['ok']
    first = value.motion.target(goals, value.step_id)['action']
    assert first[0] == pytest.approx(.12)
    assert first[7] == pytest.approx(.2)
    second = value.motion.target(goals, value.step_id + 1)['action']
    assert second[0] == pytest.approx(.24)
    assert second[7] == pytest.approx(.3)


@pytest.mark.parametrize('bad', [np.empty((0, 6)), np.zeros((4, 7)), np.full((3, 6), np.nan)])
def test_invalid_native_trajectory_never_reaches_execution(bad):
    value = session()
    goals = targets(value)
    value.kinematics.manager.planner['left'].plan_joint = lambda *a: dict(status='Success', position=bad)
    assert not value.prepare_targets(goals)['ok']
    assert value.motion.plan is None


def test_planned_detour_is_not_stopped_by_endpoint_stagnation(tmp_path):
    from test_grounded import Sim, started, move_args

    class Detour(Sim):
        def request(self, op, **args):
            self.converge = self.tick >= 45
            result = super().request(op, **args)
            if op == 'eef_joint_target':
                result['diagnostics'] = {arm: dict(trajectory_sample=min(self.tick + 1, 30),
                    trajectory_samples=31) for arm in ('left', 'right')}
            return result

    client = started(tmp_path, Detour)
    packet = client.act(**move_args(client))
    assert packet['movement']['status'] == 'reached' and client.tick == 46
