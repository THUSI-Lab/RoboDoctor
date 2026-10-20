from types import SimpleNamespace

import numpy as np
import pytest

from robodoctor.geometry import pose7, pose
from robodoctor.session import RoboDojoSession
from robodoctor.motion import CuroboMotion
from test_grounded import CartesianFK


def session():
    value = RoboDojoSession.__new__(RoboDojoSession)
    robots = {arm:SimpleNamespace(arm_joint_indices=list(range(6)), robot_name=arm,
        ee_link_name='link6') for arm in ('left','right')}
    keys = [SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=np.array([[[-1.,1.]]*6]))) for _ in robots]
    manager = SimpleNamespace(robot_list=list(robots.values()), robot_key=keys, get_joint=lambda r:{0:np.zeros(6)})
    roots = {'left':pose7([.3,-.2,.1,np.sqrt(.5),0,0,np.sqrt(.5)]),
             'right':pose7([-.3,.2,.1,np.sqrt(.5),np.sqrt(.5),0,0])}
    value.world_from_env = pose7([5,6,7,np.sqrt(.5),0,np.sqrt(.5),0])
    value.kinematics = SimpleNamespace(robots=robots, manager=manager,
        fk={arm:CartesianFK() for arm in robots}, root=lambda arm:roots[arm])
    manager.get_real_endpose = lambda robot: {0: np.r_[
        pose(roots[robot.robot_name])['position'], pose(roots[robot.robot_name])['quaternion_wxyz']]}
    manager.planner = {arm:FakePlanner() for arm in robots}
    value.motion = CuroboMotion(value.kinematics, .04)
    value.world_bases = {arm:value.world_from_env @ roots[arm] for arm in robots}
    value.step_id = 3
    value.groundings = {}
    return value


class FakePlanner:
    dt = .01
    ee_link = 'link6'

    def __init__(self):
        self.checked = []
        self.fail_ik = self.fail_plan = False

    def solve_ik_to_joint(self, curr_joint_pos, target_ee_pose, real_robot_pose):
        goal = np.linalg.inv(pose7(real_robot_pose)) @ pose7(target_ee_pose)
        self.checked.append(goal)
        if self.fail_ik or np.linalg.norm(goal[:3, 3]) > 1:
            return dict(status='Fail')
        return dict(status='Success', joint_value=np.r_[goal[:3, 3], [0, 0, 0]])

    def plan_joint(self, start_joint_pos, goal_joint_pos):
        if self.fail_plan:
            return dict(status='Fail')
        return dict(status='Success', position=np.linspace(start_joint_pos, goal_joint_pos, 11))


def test_dual_targets_convert_each_env_pose_to_its_own_rotated_base():
    value = session()
    targets = {arm:dict(pose(value.kinematics.root(arm) @ pose7([.3,0,0,1,0,0,0])),
                       gripper_closed=False) for arm in ('left','right')}
    result = value.prepare_targets(targets)
    assert result['ok'] and result['physical_steps'] == 0
    checked = [p.checked[0] for p in value.kinematics.manager.planner.values()]
    assert len(checked) == 2
    for goal in checked:
        np.testing.assert_allclose(goal, pose7([.3,0,0,1,0,0,0]), atol=1e-9)
    # .3 in each base is reachable; world coordinates must not enter arm IK.
    targets['right']['position'] = [30,0,0]
    assert not value.prepare_targets(targets)['ok']
    assert not value.prepare_targets({'left':targets['left']})['ok']


def test_grounding_uses_environment_frame_not_world_or_either_arm_base():
    value = session()
    value.head_cloud = np.zeros((3,4,3))
    value.head_cloud[1,2] = (value.world_from_env @ [.1,.2,.3,1])[:3]
    result = value.grounding(2, 1)
    assert result['ok'] and result['physical_steps'] == 0 and value.step_id == 3
    assert result['frame'] == 'environment_origin' and 'arm' not in result
    np.testing.assert_allclose(result['position'], [.1,.2,.3], atol=1e-9)
    np.testing.assert_allclose(result['world_axes_in_env'], value.world_from_env[:3,:3].T)
    assert result['image_size'] == [4,3]
    assert not value.grounding(5, 0)['ok']


def test_uses_optical_child_intrinsics_and_same_raw_depth():
    value = session()
    for robot in value.kinematics.robots.values():
        robot.base_link = 'base'
    value.kinematics.manager.get_link_pose = lambda *a, **k:{0:np.array([0,0,0,1,0,0,0])}
    calls = []
    def world_pose(camera_axes):
        calls.append(camera_axes)
        return np.array([1,2,3]), np.array([1,0,0,0])
    camera = SimpleNamespace(get_world_pose=world_pose,
        get_intrinsics_matrix=lambda:np.array([[2,0,1],[0,4,1],[0,0,1]]))
    value.env = SimpleNamespace(camera_manager=SimpleNamespace(camera_names=[['cam_head']], cameras=[[camera]]))
    raw = dict(vision=dict(cam_head=dict(depth=np.ones((2,3)),color=np.zeros((2,3,3)))),
               state=dict(left_ee_pose=[0,0,0,1,0,0,0],right_ee_pose=[0,0,0,1,0,0,0]))
    for arm in ('left', 'right'):
        raw['vision'][f'cam_{arm}_wrist'] = dict(depth=np.ones((2,3)), color=np.zeros((2,3,3)))
    value._capture_geometry(raw)
    np.testing.assert_allclose(value.world_from_env, np.linalg.inv(value.kinematics.root('left')), atol=1e-9)
    assert calls == ['usd']
    np.testing.assert_allclose(value.head_cloud[0,2], [1.5,2.25,2])
