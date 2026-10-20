"""Grounding, absolute pre-operation movement, then dual-arm EEF control."""
import json
import numpy as np

from hybrid_rollout.robodojo.io import InputError, write_json
from hybrid_rollout.robodojo.robodojo_server.gpt_only_client import GPTOnlyTools
from hybrid_rollout.robodojo.robodojo_server.decision_log import decision_event, emit
from .geometry import pose_error, transform
from .validation import validate_action
from .debug_video import RecordedDebugVideo


class GroundedTools(GPTOnlyTools):
    method = 'gpt_only'
    control_version = 'grounded_direct_v2'

    def __init__(self, *args, smoke_approach=False, reasoning_effort='xhigh', **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = RecordedDebugVideo(self.output)
        self.smoke_approach = smoke_approach
        self.reasoning_effort = reasoning_effort
        self.groundings = {}
        self.counters.update(grounding_calls=0, depth_calls=0)

    def next_call(self):
        call = super().next_call()
        if call and self.phase == 'act':
            call['response'] = ('Choose dual-arm EEF targets; no per-call translation or rotation cap; steps 1..150. '
                                'For joint recovery use robodojo_joint_act with these same paths.')
        return call

    def _fresh(self, request_id):
        if self.phase != 'act' or request_id != self.request['request_id']:
            raise InputError('Use request_id from the latest active observation')

    def _observation(self, directory, *, result=None):
        packet = super()._observation(directory, result=result)
        for item in packet['images']:
            height, width = self.obs[item['camera']].shape[:2]
            item.update(width=width, height=height, pixel_origin='top_left')
        packet.update(control_version=self.control_version,
                      workflow='grounding (environment_origin) -> robodojo_act (dual-arm coarse/fine)',
                      available_tools=[] if self.phase == 'done' else ['grounding', 'get_depth', 'robodojo_act', 'robodojo_joint_act'])
        if self.meta.get('supports_joint_control') and self.phase != 'done':
            packet['current_joints'] = self._rpc('joint_state')['arms']
        if self.phase != 'done':
            self.request.update(packet)
            write_json(packet['request_path'], self.request)
        write_json(self.observation_path, packet)
        self.groundings.clear()
        return packet

    def start(self, **arguments):
        packet = super().start(**arguments)
        self.run.update(control_version=self.control_version, action_space='dual_arm_eef_and_joint', action_horizon=150,
                        motion_planner=self.meta.get('motion_planner', 'numerical_ik'),
                        teacher_reasoning_effort=self.reasoning_effort,
                        eef_target_delta_limited=False,
                        absolute_pose_format='x,y,z,qw,qx,qy,qz', absolute_frame='environment_origin')
        write_json(self.output/'run.json', self.run)
        return packet

    def finish(self, reason):
        result = super().finish(reason)
        result['control_version'] = self.control_version
        write_json(self.output/'result.json', result)
        return result

    def close(self):
        if self.sim is not None:
            self.sim.close()
        if self.episode is not None:
            if (self.output.parent/'sim/summary.json').is_file():
                self.recorder.finalize()
            else:
                # A lost RPC can leave the writer running. The owning launcher
                # renders only after stopping simulation and recovering frames.
                print(json.dumps(dict(event='debug_video', status='deferred',
                    reason='Waiting for simulator shutdown and sensor finalization')), flush=True)

    def grounding(self, request_id, x, y):
        self._fresh(request_id)
        if type(x) is not int or type(y) is not int:
            raise InputError('Specify integer native-image head pixel (x,y)')
        result = self._rpc('grounding', x=x, y=y)
        if not result['ok']:
            raise InputError(result['error'])
        self.groundings[result['grounding_id']] = result
        self.counters['grounding_calls'] += 1
        packet = dict(result, request_id=request_id, next_call=self.next_call(),
                      suggested_next_tool='robodojo_act',
                      note='Surface sample, not object center or gripper TCP. Offset away from the object before moving.')
        write_json(self.output/f'grounding_{self.counters["grounding_calls"]:04d}.json', packet)
        return packet

    def get_depth(self, request_id, arm, x, y):
        self._fresh(request_id)
        if arm not in ('left', 'right') or type(x) is not int or type(y) is not int:
            raise InputError('Specify left/right wrist camera and integer native-image pixel (x,y)')
        result = self._rpc('get_depth', arm=arm, x=x, y=y)
        if not result['ok']:
            raise InputError(result['error'])
        self.counters['depth_calls'] += 1
        packet = dict(result, request_id=request_id, observation_path=str(self.observation_path),
                      image_path=str(self.observation_path.parent/f'{result["camera"]}.png'),
                      next_call=self.next_call(),
                      note='Positive camera-axis depth, not finger-to-object gap or a environment-frame motion delta. '
                           'Use with current wrist RGB to refine the grasp; re-query after motion.')
        write_json(self.output/f'depth_{self.counters["depth_calls"]:04d}.json', packet)
        return packet

    def joint_act(self, *, observation_path, response, output_dir, preview_only):
        if type(preview_only) is not bool:
            raise InputError('preview_only must be boolean')
        arguments = dict(observation_path=observation_path, response=response, output_dir=output_dir)
        if preview_only:
            self._check('robodojo_act', arguments)
            try:
                validate_action(response, self.request, joint=True)
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                raise InputError(str(error)) from error
            result = self._rpc('prepare_joints', targets=response['target'], coordinate_mode=response['coordinate_mode'])
            if not result['ok']:
                raise InputError(result['error'])
            return dict(result, request_id=self.request['request_id'], preview_only=True,
                        next_call=self.next_call())
        # Dynamic dispatch preserves HybridTools' plan and intervention checks.
        return self.act(_joint=True, **arguments)

    def _observe_control_steps(self, steps):
        """Let controllers consume the exact post-control observations."""
        pass

    def act(self, *, _joint=False, **arguments):
        response = arguments.get('response')
        directory = self._check('robodojo_act', arguments)
        try:
            validate_action(response, self.request, joint=_joint)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise InputError(str(error)) from error
        if self.smoke_approach:
            if _joint:
                raise InputError('Smoke approach requires the grounded EEF arrival check')
            if not self.groundings:
                raise InputError('Ground the visible target on this observation before a smoke approach')
            for arm in ('left', 'right'):
                if response['target'][arm]['gripper_closed']:
                    raise InputError('Smoke approach preserves open grippers; no grasping')
        prepared = (self._rpc('prepare_joints', targets=response['target'], coordinate_mode=response['coordinate_mode'])
                    if _joint else self._rpc('prepare_targets', targets=response['target']))
        if not prepared['ok']:
            raise InputError(prepared['error'])
        source = 'gpt_joint' if _joint else 'gpt_eef'
        if self.source != source:
            self._rpc('switch_control_source', source=source, reason=response['reason'])
            self.source = source
        decision = len(self.history)
        response = dict(response, evaluation_method=self.method, frame='joint_radians' if _joint else 'environment_origin')
        if _joint:
            response['resolved_joint_targets'] = prepared['joint_targets']
            response['target_eef'] = prepared['target_eef']
        write_json(self.output/f'response_{decision:03d}.json', response)
        receipt = self._rpc('record_codex_decision', decision=decision, prediction_id=None, response=response)
        emit(dict(decision_event(episode_id=self.episode, step_id=self.tick, decision=decision,
            prediction_id=None, response=response), simulator_receipt=receipt))
        start, trace, errors = self.tick, [], []
        eef_targets = prepared['target_eef'] if _joint else response['target']
        goals = {arm: transform(t['position'], t['quaternion_wxyz']) for arm, t in eef_targets.items()}
        status = 'step_budget'
        measured = self.obs
        # Always execute at least one ACK: an already-reached pose may carry a
        # new gripper command. Arrival only describes link6 poses, not a grasp.
        for _ in range(response['steps']):
            proposal = self.target_proposal(response['target'])
            result = self._rpc('chunk_step', actions=np.asarray([proposal['action']], np.float32),
                               teacher_request_id=self.request['request_id'])
            self._observe_control_steps(result['steps'])
            for row in result['steps']:
                row['edited_target'], row['ik_diagnostics'] = response['target'], proposal['diagnostics']
            trace.extend(result['steps'])
            self.tick = result['step_id']
            measured = self._rpc('teacher_observation')
            arm_errors = {arm: pose_error(transform(measured['eef_positions'][i],
                measured['eef_quaternions_wxyz'][i]), goals[arm])
                for i, arm in enumerate(('left', 'right'))}
            joint_errors = {arm: float(np.max(np.abs(np.asarray(prepared['joint_targets'][arm])
                - measured['states'][7*i:7*i+6]))) for i, arm in enumerate(('left', 'right'))} if _joint else {}
            arrived = (all(v <= .01 for v in joint_errors.values()) if _joint
                       else all(d <= .005 and a <= .03 for d, a in arm_errors.values()))
            if any(r['terminated'] or r['truncated'] for r in result['steps']):
                status = 'terminal'
                break
            if response.get('stop_on_reach', True) and arrived:
                status = 'reached'
                break
            # A planned detour can temporarily increase endpoint error. Apply
            # endpoint-stagnation detection after both trajectories finish.
            planning = any(d.get('trajectory_sample', float('inf')) < d.get('trajectory_samples', 0)-1
                           for d in proposal['diagnostics'].values())
            if planning:
                errors.clear()
            else:
                errors.append(sum(joint_errors.values()) if _joint else sum(d + .2*a for d, a in arm_errors.values()))
            if response.get('stop_on_reach', True) and len(errors) >= 20 and errors[-20] - min(errors[-19:]) < 1e-4:
                status = 'stalled'
                break
        valid = [r for r in trace if r['valid']]
        arms = {arm: dict(position_error_m=d, rotation_error_rad=a, reached=d <= .005 and a <= .03)
                for arm, (d, a) in arm_errors.items()}
        if _joint:
            for arm in arms:
                arms[arm].update(joint_error_rad=joint_errors[arm], reached=joint_errors[arm] <= .01,
                                 target_qpos=prepared['joint_targets'][arm])
        movement = dict(status=status, reached=all(v['reached'] for v in arms.values()), arms=arms,
                        frame='environment_origin', executed_steps=len(valid),
                        position_error_m=max(d for d, a in arm_errors.values()),
                        rotation_error_rad=max(a for d, a in arm_errors.values()),
                        arrival_scope='joint_angles_only_not_grasp_success' if _joint else 'link6_poses_only_not_grasp_success')
        write_json(self.output/f'edit_{decision:03d}.json', [dict(target=r['edited_target'],
            diagnostics=r['ik_diagnostics'], valid=r['valid'],
            executed_action=np.asarray(r['executed_action']).tolist()) for r in trace])
        executed = np.asarray([r['executed_action'] for r in valid], np.float32).reshape(-1, 14)
        np.savez_compressed(self.output/f'execution_{decision:03d}.npz', actions=executed,
            states=np.asarray([r['obs']['states'] for r in valid], np.float32).reshape(-1, 14))
        self.counters[source+'_steps'] += len(valid)
        done = bool(valid and (valid[-1]['terminated'] or valid[-1]['truncated']))
        self.history.append(dict(decision=decision, start_tick=start, end_tick=self.tick,
            response=response, evaluation_method=self.method, simulator_decision_receipt=receipt,
            executed_steps=len(valid), source=self.source, source_detail='joint' if _joint else 'eef', prediction_id=None,
            executed_gripper_closed=self.gripper_history(executed), terminal=done, movement=movement,
            native_success=bool(valid and valid[-1]['success'])))
        write_json(self.output/'history.json', self.history)
        write_json(self.output/'progress.json', dict(episode_id=self.episode, step_id=self.tick,
            evaluation_method=self.method, **self.counters))
        exhausted = self.max_decisions > 0 and len(self.history) >= self.max_decisions
        smoke_done = self.smoke_approach and status == 'reached'
        reason = 'terminal' if done else 'smoke_approach_complete' if smoke_done else 'decision_budget'
        final = self.finish(reason) if done or exhausted or smoke_done else None
        packet = self._observation(directory, result=final)
        packet['movement'] = movement
        write_json(self.observation_path, packet)
        return packet
