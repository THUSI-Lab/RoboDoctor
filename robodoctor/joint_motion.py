"""Measured-state numerical IK and explicit joint targets; no collision planner."""
from copy import deepcopy

import numpy as np

from .geometry import array, pose, pose7, pose_error, solve_ik


class NumericalMotion:
    def __init__(self, kinematics, control_dt):
        self.kinematics = kinematics
        self.control_dt = float(control_dt)
        self.plan = None

    def joint_state(self):
        result = {}
        manager = self.kinematics.manager
        for arm, robot in self.kinematics.robots.items():
            data = manager.robot_key[manager.robot_list.index(robot)].data
            indices = robot.arm_joint_indices
            limits = array(data.soft_joint_pos_limits[0, indices])
            velocity = array(data.soft_joint_vel_limits[0, indices])
            qpos = array(manager.get_joint(robot)[0])
            if (qpos.shape != (6,) or limits.shape != (6, 2) or velocity.shape != (6,)
                    or not np.isfinite(np.r_[qpos, limits.ravel(), velocity]).all()
                    or np.any(limits[:, 0] >= limits[:, 1]) or np.any(velocity <= 0)):
                raise ValueError(f'{arm}: Invalid native joint state/limits')
            result[arm] = dict(qpos=qpos.tolist(), joint_names=list(robot.arm_joints_name),
                position_limits_rad=limits.tolist(), velocity_limits_rad_s=velocity.tolist(),
                gripper_opening=None, units='rad')
        return result

    def prepare(self, targets, step_id):
        return self._prepare(targets, step_id, mode='eef')

    def prepare_joints(self, targets, coordinate_mode, step_id):
        return self._prepare(targets, step_id, mode='joint', coordinate_mode=coordinate_mode)

    def _prepare(self, targets, step_id, mode, coordinate_mode=None):
        self.plan = None
        arm = 'targets'
        try:
            if not isinstance(targets, dict) or set(targets) != {'left', 'right'}:
                raise ValueError('Explicit left/right targets required')
            if mode == 'joint' and coordinate_mode not in ('absolute', 'delta'):
                raise ValueError('coordinate_mode must be absolute or delta')
            state = self.joint_state()
            joints, poses = {}, {}
            for arm in ('left', 'right'):
                t = targets[arm]
                if type(t['gripper_closed']) is not bool:
                    raise ValueError('Explicit boolean gripper required')
                current = array(state[arm]['qpos'])
                limits = array(state[arm]['position_limits_rad'])
                root = self.kinematics.root(arm)
                fk = self.kinematics.fk[arm]
                if mode == 'eef':
                    goal = pose7(np.r_[array(t['position']), array(t['quaternion_wxyz'])])
                    measured = pose7(self.kinematics.manager.get_real_endpose(self.kinematics.robots[arm])[0])
                    distance, angle = pose_error(measured, goal)
                    q = current.copy() if distance <= 1e-6 and angle <= 1e-6 else solve_ik(
                        fk, current, limits, np.linalg.inv(root) @ goal)
                else:
                    value = array(t['qpos'])
                    if value.shape != (6,) or not np.isfinite(value).all():
                        raise ValueError('qpos must contain six finite joint angles in radians')
                    # Anchor delta ONCE to the current measured state, never per ACK.
                    q = current + value if coordinate_mode == 'delta' else value
                if np.any(q < limits[:, 0] - 1e-7) or np.any(q > limits[:, 1] + 1e-7):
                    raise ValueError('Joint target is outside native position limits; no clipping')
                joints[arm] = q.copy()
                poses[arm] = dict(pose(root @ fk.matrix(q)), gripper_closed=t['gripper_closed'])
            self.plan = dict(targets=deepcopy(targets), joints=joints, start_step=step_id, mode=mode)
            return dict(ok=True, physical_steps=0, planner='numerical_ik' if mode == 'eef' else 'direct_joint',
                frame='environment_origin', joint_targets={a:q.tolist() for a,q in joints.items()},
                target_eef=poses, collision_checked=False,
                note='FK endpoints only; no obstacle/self/inter-arm collision planning or straight Cartesian path guarantee.')
        except (ValueError, TypeError, KeyError, RuntimeError) as error:
            return dict(ok=False, error=f'{arm}: {error}', physical_steps=0)

    def target(self, targets, step_id):
        if self.plan is None or targets != self.plan['targets'] or step_id < self.plan['start_step']:
            raise ValueError('Prepare the current targets before execution')
        state = self.joint_state()
        action, diagnostics = [], {}
        for arm in ('left', 'right'):
            current = array(state[arm]['qpos'])
            delta = self.plan['joints'][arm] - current
            bound = array(state[arm]['velocity_limits_rad_s']) * self.control_dt
            fraction = min(1., 1. / max(float(np.max(np.abs(delta) / bound)), 1.))
            command = current + fraction * delta
            action.extend([*command.tolist(), float(not targets[arm]['gripper_closed'])])
            diagnostics[arm] = dict(method='numerical_ik_joint_tracking' if self.plan['mode'] == 'eef'
                else 'direct_joint_tracking', target_qpos=self.plan['joints'][arm].tolist(),
                measured_qpos=current.tolist(), max_joint_error_rad=float(np.max(np.abs(delta))),
                physical_tracking_verified=False, collision_checked=False)
        return dict(action=action, diagnostics=diagnostics, physical_steps=0)
