"""One RoboDojo episode, native action stepping and success/timeout checks."""
import hashlib
from pathlib import Path
import uuid
import numpy as np
from PIL import Image

from hybrid_rollout.robodojo.io import write_json, require, sha256
from hybrid_rollout.robodojo.robodojo_server.decision_log import record_decision
from hybrid_rollout.robodojo.robodojo_server.kinematics import DualKinematics
from .geometry import array, pose, transform, organized_cloud, ground_pixel
from .joint_motion import NumericalMotion
from .camera_config import CAMERA_ALIASES

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def register_native_evaluation(env):
    """Mirror EvalEnv.run_eval's setup, without entering its policy loop.

    Native reset clears the checks. With no registered conditions the native
    RewardManager reports success immediately, so fail closed on that state.
    These remain simulator-only predicates, never teacher planning inputs.
    """
    env.run_reward()
    if hasattr(env, 'get_score'):
        env.get_score()
    names = ('check_list', 'final_check_list', 'trigger_check_list')
    counts = {name: [len(group) for group in getattr(env.reward_manager, name)] for name in names}
    require(all(sum(counts[name][i] for name in names) > 0 for i in range(env.num_envs)),
            'Native task completion conditions are empty; refusing vacuous success')
    support_envs = []
    if getattr(env, 'interact', False) and hasattr(env, 'query_support_arm_traj'):
        support_envs = list(env.get_running_env_idx_list())
        for env_idx in support_envs:
            env.query_support_arm_traj(env_idx=env_idx)
    return dict(registered=True, source='native EvalEnv.run_eval preamble',
                condition_group_counts=counts, initial_support_arm_query_envs=support_envs)


class RoboDojoSession:
    def __init__(self, env, output, task, evaluation_identity=None):
        self.env, self.output = env, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode_id, self.step_id = None, 0
        self.terminated = self.truncated = self.success = False
        self.poisoned, self.finished = True, False
        self.video_writer, self.video_frames = None, 0
        self.source, self.control_epoch = 'student', 0
        self.evaluation_identity = evaluation_identity
        self.finish_reason = None
        self.metadata = dict(task=task, instruction=task, simulator='RoboDojo',
            robot_adapter='dual_arx_x5', control_dt=1/env.obs_manager.collect_freq,
            max_episode_steps=int(env.step_lim), action_dim=14, action_horizon=150,
            frame='environment_origin', eef_link='link6', no_rollback=True,
            supports_fk_preview=True, supports_codex_decision_recording=True,
            cameras=list(CAMERAS), gripper_semantics='continuous_0_closed_1_open')
        # Our recorder owns exactly one frame per actual control ACK. Disable
        # only upstream duplicate video capture, not its native task checks.
        env._stream_vision = lambda *args, **kwargs: None
        self.groundings = {}
        self.metadata.update(supports_grounding=True, supports_dual_arm_absolute=True, supports_get_depth=True,
                             control_version='grounded_direct_v2', motion_planner='numerical_ik',
                             supports_joint_control=True)

    def _check_identity(self, episode_id, step_id):
        require(not self.poisoned and episode_id == self.episode_id and step_id == self.step_id,
                'Stale episode/tick or uncertain simulator state')

    def _observe(self, record=False):
        raw = self.env.get_obs()
        self._capture_geometry(raw)
        state = raw['state']
        states = np.concatenate([np.r_[state[f'{arm}_arm_joint_state'], state[f'{arm}_ee_joint_state']]
                                 for arm in ('left', 'right')]).astype(np.float32)
        require(states.shape == (14,) and np.isfinite(states).all(), 'Invalid native dual-arm state')
        images = {}
        depths = {}
        for key, names in CAMERA_ALIASES.items():
            source = next((n for n in names if n in raw['vision']), None)
            require(source is not None, f'Missing {key}; native cameras={list(raw["vision"])}')
            vision = raw['vision'][source]
            images[key] = np.asarray(vision['color'], np.uint8)[..., :3]
            require('depth' in vision, f'Missing metric depth for {key}')
            depth = np.asarray(vision['depth'], np.float32)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            require(depth.ndim == 2 and depth.shape == images[key].shape[:2],
                    f'Invalid depth/RGB alignment for {key}')
            depths[key] = depth.copy()
        poses = np.asarray([state[f'{arm}_ee_pose'] for arm in ('left', 'right')], np.float32)
        self.obs = dict(**images, **{f'{key}_depth': value for key, value in depths.items()},
            states=states, eef_positions=poses[:, :3],
            eef_quaternions_wxyz=poses[:, 3:], instruction=raw['instruction'],
            remaining_steps=max(0, self.metadata['max_episode_steps']-self.step_id))
        self.obs['eef_base_poses'] = np.asarray([
            [*self.base_eef[arm]['position'], *self.base_eef[arm]['quaternion_wxyz']]
            for arm in ('left', 'right')])
        if record:
            directory = self.episode_dir/'observations'
            directory.mkdir(exist_ok=True)
            np.savez_compressed(directory/f'{self.step_id:06d}.npz', **self.obs)
            if self.video_writer is None:
                import imageio.v2 as imageio
                self.video_writer = imageio.get_writer(str(self.output/'sensors.mp4'),
                    fps=1/self.metadata['control_dt'], codec='libx264', pixelformat='yuv420p',
                    macro_block_size=2, output_params=['-movflags', '+frag_keyframe+empty_moov',
                                                      '-frag_duration', '1000000'])
            frames = [np.asarray(Image.fromarray(images[k]).resize((640, 360))) for k in CAMERAS]
            self.video_writer.append_data(np.concatenate(frames, axis=1))
            self.video_frames += 1
        return self.obs

    def _capture_geometry(self, raw):
        """Cache geometry from the exact same sensor capture as returned RGB."""
        self._capture_wrist_depth(raw)
        names = self.env.camera_manager.camera_names[0]
        name = next(n for n in ('cam_head', 'cam_high', 'head_camera', 'top_camera') if n in names)
        camera = self.env.camera_manager.cameras[0][names.index(name)]
        vision = raw['vision'][name]
        if 'depth' not in vision:
            raise ValueError('Head camera metric depth is required; launch robodoctor.server')
        # The optical child may have a nonidentity transform. The manager's
        # parent-Xform extrinsics and fx=fy approximation are insufficient.
        position, quaternion = camera.get_world_pose(camera_axes='usd')
        self.head_cloud = organized_cloud(vision['depth'], array(camera.get_intrinsics_matrix()),
                                         transform(array(position), array(quaternion)))
        if self.head_cloud.shape[:2] != np.asarray(vision['color']).shape[:2]:
            raise ValueError('Head depth/RGB resolution mismatch')
        self.world_bases, self.base_eef = {}, {}
        for arm, robot in self.kinematics.robots.items():
            value = self.kinematics.manager.get_link_pose(robot, robot.base_link, is_relative=False)[0]
            self.world_bases[arm] = transform(value[:3], value[3:])
            value = raw['state'][f'{arm}_ee_pose']
            self.base_eef[arm] = pose(np.linalg.inv(self.kinematics.root(arm)) @ transform(value[:3], value[3:]))
        self.world_from_env = self.world_bases['left'] @ np.linalg.inv(self.kinematics.root('left'))

    def _capture_wrist_depth(self, raw):
        self.wrist_depth = {}
        for key in ('cam_left_wrist', 'cam_right_wrist'):
            name = next((n for n in CAMERA_ALIASES[key] if n in raw['vision']), None)
            if name is None or 'depth' not in raw['vision'][name]:
                raise ValueError(f'{key} metric depth is required; launch robodoctor.server')
            vision = raw['vision'][name]
            depth = array(vision['depth'])
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if depth.ndim != 2 or depth.shape != np.asarray(vision['color']).shape[:2]:
                raise ValueError(f'{key} depth/RGB resolution mismatch')
            self.wrist_depth[key] = depth.copy()

    def get_depth(self, arm, x, y):
        try:
            if arm not in ('left', 'right'):
                raise ValueError('arm must be left or right')
            camera = f'cam_{arm}_wrist'
            depth = self.wrist_depth[camera]
            height, width = depth.shape
            if type(x) is not int or type(y) is not int or not (0 <= x < width and 0 <= y < height):
                raise ValueError(f'Pixel must be integer (x,y) within {width}x{height}')
            value = float(depth[y, x])
            if not np.isfinite(value) or value <= 0:
                raise ValueError('No valid depth at this wrist pixel; select another visible surface pixel')
        except ValueError as error:
            return dict(ok=False, error=str(error), physical_steps=0)
        return dict(ok=True, arm=arm, camera=camera, pixel=[x, y], image_size=[width, height],
                    pixel_origin='top_left', depth_m=value, units='m',
                    depth_type='distance_to_image_plane', frame=f'{camera}_optical',
                    source='wrist_camera_metric_depth', step_id=self.step_id, physical_steps=0)

    def grounding(self, x, y):
        try:
            point, world = ground_pixel(self.head_cloud, x, y, self.world_from_env)
        except ValueError as error:
            return dict(ok=False, error=str(error), physical_steps=0)
        identifier = uuid.uuid4().hex
        self.groundings[identifier] = dict(position=point, step_id=self.step_id)
        return dict(ok=True, grounding_id=identifier, pixel=[x, y],
                    image_size=[self.head_cloud.shape[1], self.head_cloud.shape[0]],
                    position=point.tolist(), frame='environment_origin', units='m',
                    world_axes_in_env=self.world_from_env[:3, :3].T.tolist(),
                    source='head_camera_metric_depth_organized_point_cloud',
                    step_id=self.step_id, physical_steps=0)

    def prepare_targets(self, targets):
        """Plan BOTH absolute EEF targets before executing either arm."""
        return self.motion.prepare(targets, self.step_id)

    def prepare_joints(self, targets, coordinate_mode):
        return self.motion.prepare_joints(targets, coordinate_mode, self.step_id)

    def joint_state(self):
        state = self.motion.joint_state()
        for i, arm in enumerate(('left', 'right')):
            state[arm]['gripper_opening'] = float(self.obs['states'][7*i+6])
        return dict(arms=state, physical_steps=0)

    def reset(self, seed, source, policy_version):
        require(self.episode_id is None, 'One fresh episode only; no rollback/reset')
        if self.evaluation_identity:
            require(seed == self.evaluation_identity['layout_id'], 'Reset seed differs from the frozen layout_id')
            actual_layout = self.env.seed_manager.seed_info[seed]['scene_layout']
            require(sha256(actual_layout) == self.evaluation_identity['layout_sha256'],
                    'Native layout changed before reset')
        try:
            self.env.reset(seed=[seed])
        except Exception as error:
            # Only the native named exception is classified as an invalid layout;
            # infrastructure/model errors are never silently removed from evaluation.
            write_json(self.output/'evaluation_outcome.json', dict(
                evaluation_case=self.evaluation_identity, complete=False, valid_for_success_rate=False,
                status='invalid_native_layout' if type(error).__name__ == 'UnStableError' else 'reset_error',
                error_type=type(error).__name__, error=str(error)))
            raise
        registration = register_native_evaluation(self.env)
        write_json(self.output/'native_evaluation.json', registration)
        self.episode_id = uuid.uuid4().hex
        self.episode_dir = self.output/self.episode_id
        self.episode_dir.mkdir()
        self.poisoned = False
        self.kinematics = DualKinematics(self.env)
        self.motion = NumericalMotion(self.kinematics, self.metadata['control_dt'])
        write_json(self.output/'fk_validation.json', self.kinematics.check())
        self._observe(record=True)
        fingerprints = {}
        for key, value in self.obs.items():
            arr = np.asarray(value)
            fingerprints[key] = dict(shape=list(arr.shape), dtype=str(arr.dtype),
                sha256=hashlib.sha256(arr.tobytes()).hexdigest())
        write_json(self.output/'initial_observation_fingerprint.json', dict(
            evaluation_case=self.evaluation_identity, fields=fingerprints,
            scope='Recorded RGB, proprio, robot EEF and instruction; not a complete physics state',
            bitwise_physics_reproducibility_guaranteed=False))
        self.metadata['instruction'] = self.obs['instruction']
        identity = hashlib.sha256(self.obs['states'].tobytes()).hexdigest()
        result = dict(episode_id=self.episode_id, step_id=0, initial_state_hash=identity,
            instruction=self.obs['instruction'], metadata=dict(self.metadata, layout_id=seed,
            eval_seed=getattr(self.env, 'eval_seed', None), evaluation_case=self.evaluation_identity,
            initial_state_hash_scope='robot_proprio_only_not_complete_scene',
            policy_version=policy_version, native_reset_settling_not_counted_as_policy_controls=True))
        write_json(self.output/'reset.json', result)
        return result

    def chunk_step(self, actions, **kwargs):
        require(not self.finished and not self.terminated and not self.truncated, 'Episode finished')
        a = np.asarray(actions, np.float32)
        require(a.ndim == 2 and a.shape[1] == 14 and len(a) >= 1 and np.isfinite(a).all(),
                'Expected at least one finite 14D absolute action')
        a = a[:16]
        require(np.all((a[:, [6, 13]] >= 0) & (a[:, [6, 13]] <= 1)), 'Invalid gripper opening')
        rows = []
        for action in a:
            command = {key: value for arm, offset in (('left', 0), ('right', 7))
                for key, value in ((f'{arm}_arm_joint_state', action[offset:offset+6]),
                                   (f'{arm}_ee_joint_state', action[offset+6:offset+7]))}
            before = int(self.env.take_action_cnt[0])
            self.env.take_action(command)
            require(int(self.env.take_action_cnt[0]) == before+1, 'Native action was not executed exactly once')
            self.step_id += 1
            ended = bool(self.env.end_flag[0])
            self.success = bool(ended and self.env.success[0])
            self.truncated = bool(ended and not self.success and self.step_id >= self.env.step_lim)
            self.terminated = bool(ended and not self.truncated)
            obs = self._observe(record=True)
            row = dict(valid=True, executed_action=action.copy(), obs=dict(states=obs['states']),
                observation_path=str(self.episode_dir/'observations'/f'{self.step_id:06d}.npz'),
                step_id=self.step_id, terminated=self.terminated, truncated=self.truncated, success=self.success,
                source=self.source, control_epoch=self.control_epoch, **kwargs)
            rows.append(row)
            write_json(self.episode_dir/f'action_{self.step_id-1:06d}.json',
                dict(row, executed_action=action.tolist(), obs=dict(states=obs['states'].tolist())))
            if ended:
                self._write_summary('terminal')
                break
        return dict(episode_id=self.episode_id, step_id=self.step_id, steps=rows)

    def _write_summary(self, reason):
        if self.episode_id is None:
            return
        if self.video_writer is not None:
            self.video_writer.close(); self.video_writer = None
        if self.finish_reason is not None:
            reason = self.finish_reason  # Do not overwrite a budget/error outcome during server cleanup.
        complete = self.terminated or self.truncated
        invalid = 0 in getattr(self.env, 'unstable_envs', set())
        eligible = complete and not invalid and not self.poisoned
        score = None
        if eligible:
            score = 1.0 if self.success else (float(self.env.reward_manager.get_score()[0])/100
                     if hasattr(self.env, 'get_score') else 0.0)
        write_json(self.output/'evaluation_outcome.json', dict(
            evaluation_case=self.evaluation_identity, complete=complete,
            valid_for_success_rate=eligible, native_success=self.success if eligible else None,
            native_score=score, status=('invalid_native_layout' if invalid else
                'native_completed' if eligible else 'budget_censored' if reason == 'decision_budget' else 'incomplete'),
            reason=reason, native_control_steps=self.step_id,
            native_step_limit=self.metadata['max_episode_steps']))
        write_json(self.output/'summary.json', dict(episode_id=self.episode_id, step_id=self.step_id,
            success=self.success, terminated=self.terminated, truncated=self.truncated,
            complete=self.terminated or self.truncated, reason=reason, video_frames=self.video_frames,
            control_dt=self.metadata['control_dt'], no_rollback=True,
            evaluation_case=self.evaluation_identity))

    def dispatch(self, op, args):
        if op == 'metadata':
            return self.metadata
        if op == 'reset':
            return self.reset(**args)
        self._check_identity(args['episode_id'], args['step_id'])
        data = {k: v for k, v in args.items() if k not in ('episode_id', 'step_id')}
        if op == 'teacher_observation':
            return self.obs  # Cached exact post-ACK observation; no hidden step/render.
        if op in ('grounding', 'get_depth', 'prepare_targets', 'prepare_joints', 'joint_state'):
            require(not self.finished and not self.terminated and not self.truncated, 'Episode finished')
            return getattr(self, op)(**data)
        if op == 'record_codex_decision':
            return record_decision(self, args)
        if op == 'begin_combination':
            write_json(self.output/'combination.json', data)
            return dict(physical_steps=0)
        if op == 'switch_control_source':
            require(data['source'] in ('student', 'gpt_eef', 'gpt_joint'), 'Invalid source')
            self.source = data['source']; self.control_epoch += 1
            write_json(self.episode_dir/f'source_{self.control_epoch:04d}.json', dict(step_id=self.step_id, **data))
            return dict(physical_steps=0, source=self.source, control_epoch=self.control_epoch)
        if op == 'fk_preview':
            return self.kinematics.preview(data['actions'])
        if op == 'eef_joint_target':
            return self.motion.target(data['targets'], self.step_id)
        if op == 'chunk_step':
            return self.chunk_step(**data)
        if op == 'finish_pilot':
            self.finished = True
            self.finish_reason = data['reason']
            self._write_summary(data['reason'])
            return dict(episode_id=self.episode_id, step_id=self.step_id, success=self.success,
                terminated=self.terminated, truncated=self.truncated, video_frames=self.video_frames,
                control_dt=self.metadata['control_dt'], reason=data['reason'],
                evaluation_case=self.evaluation_identity)
        raise ValueError('Unsupported no-rollback operation: '+op)
