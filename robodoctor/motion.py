"""Execute the native RoboDojo cuRobo trajectories at the observation control rate."""
from copy import deepcopy

import numpy as np

from .geometry import array, pose, pose7, pose_error


class CuroboMotion:
    def __init__(self, kinematics, control_dt):
        self.kinematics = kinematics
        self.control_dt = float(control_dt)
        self.plan = None

    def prepare(self, targets, step_id):
        # Publish a plan only after BOTH arms pass; never reuse a failed plan.
        self.plan = None
        arm = 'targets'
        try:
            if not isinstance(targets, dict) or set(targets) != {'left', 'right'}:
                raise ValueError('Explicit left/right targets required')
            paths = {}
            manager = self.kinematics.manager
            for arm in ('left', 'right'):
                target = targets[arm]
                p, q = array(target['position']), array(target['quaternion_wxyz'])
                if p.shape != (3,) or q.shape != (4,):
                    raise ValueError('EEF position/quaternion must have 3/4 components')
                goal = pose7(np.r_[p, q])
                if type(target['gripper_closed']) is not bool:
                    raise ValueError('Explicit boolean gripper required')
                robot = self.kinematics.robots[arm]
                planner = manager.planner[robot.robot_name]
                if planner.ee_link != robot.ee_link_name:
                    raise ValueError('cuRobo tool frame differs from the EEF target link')
                joints = array(manager.get_joint(robot)[0])
                measured = pose7(manager.get_real_endpose(robot)[0])
                distance, angle = pose_error(measured, goal)
                if distance <= 1e-6 and angle <= 1e-6:
                    # A held arm/gripper-only call keeps its measured qpos.
                    path, dt = joints[None, :], self.control_dt
                else:
                    root = pose(self.kinematics.root(arm))
                    solved = planner.solve_ik_to_joint(
                        curr_joint_pos=joints,
                        target_ee_pose=np.r_[p, q].tolist(),
                        real_robot_pose=[*root['position'], *root['quaternion_wxyz']])
                    if solved['status'] != 'Success':
                        raise ValueError('cuRobo IK could not solve the requested pose')
                    goal_joints = array(solved['joint_value'])
                    if goal_joints.shape != (6,) or not np.isfinite(goal_joints).all():
                        raise ValueError('Invalid cuRobo IK joint solution')
                    result = planner.plan_joint(joints, goal_joints)
                    if result['status'] != 'Success':
                        raise ValueError('cuRobo could not plan a path to the requested pose')
                    path, dt = array(result['position']), float(planner.dt)
                if (path.ndim != 2 or path.shape[1] != 6 or len(path) == 0
                        or not np.isfinite(path).all() or not np.isfinite(dt) or dt <= 0):
                    raise ValueError('Invalid cuRobo joint trajectory or sample interval')
                paths[arm] = (path.copy(), dt)
            self.plan = dict(targets=deepcopy(targets), paths=paths, start_step=step_id)
            return dict(ok=True, frame='environment_origin', physical_steps=0, planner='curobo')
        except (ValueError, TypeError, KeyError, RuntimeError) as error:
            return dict(ok=False, error=f'{arm}: {error}', physical_steps=0)

    def target(self, targets, step_id):
        if self.plan is None or targets != self.plan['targets']:
            raise ValueError('Prepare both cuRobo targets before execution')
        elapsed = (step_id - self.plan['start_step'] + 1) * self.control_dt
        if elapsed <= 0:
            raise ValueError('Stale cuRobo trajectory step')
        action, diagnostics = [], {}
        for arm in ('left', 'right'):
            path, dt = self.plan['paths'][arm]
            # Native planner samples at physics dt (typically .004 s), while
            # each control ACK spans .04 s. Sample by time, not one row per ACK.
            sample = min(elapsed / dt, len(path) - 1)
            lo, hi = int(np.floor(sample)), int(np.ceil(sample))
            joints = path[lo] + (sample - lo) * (path[hi] - path[lo])
            action.extend([*joints.tolist(), float(not targets[arm]['gripper_closed'])])
            diagnostics[arm] = dict(method='curobo_native_ik_plan_joint',
                trajectory_sample=float(sample), trajectory_samples=len(path),
                physical_tracking_verified=False)
        return dict(action=action, diagnostics=diagnostics, physical_steps=0)
