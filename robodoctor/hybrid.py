"""Agent-owned subtask planning and short, reviewed pi05 execution prefixes."""
from copy import deepcopy
import json
from pathlib import Path
import re
import uuid

import numpy as np

from hybrid_rollout.robodojo.io import InputError, write_json
from hybrid_rollout.robodojo.robodojo_server.client import RoboDojoTools
from hybrid_rollout.robodojo.robodojo_server.decision_log import decision_event, emit
from hybrid_rollout.robodojo.robodojo_server.proposal_diagnostics import action_diagnostics
from .client import GroundedTools
from .schema import ACT_REASON_PATTERN
from .memory import write_plan_todo


def validate_plan(subtasks):
    if not isinstance(subtasks, list) or not subtasks:
        raise InputError('Provide a nonempty full subtask plan')
    entries = {}
    for task in subtasks:
        if not isinstance(task, dict):
            raise InputError('Each subtask must be an object')
        for key in ('id', 'src', 'dst', 'cond', 'next_action', 'evidence'):
            if not isinstance(task.get(key), str) or not task[key].strip():
                raise InputError(f'Each subtask needs nonempty {key}')
        if task['id'] in entries:
            raise InputError('Subtask IDs must be unique')
        if task.get('status') not in ('pending', 'ready', 'in_progress', 'blocked', 'done'):
            raise InputError('Invalid subtask status')
        if task.get('arm_plan') not in ('single', 'handoff', 'coordinated_dual'):
            raise InputError('Invalid arm_plan')
        arms = task.get('arms_used')
        if not isinstance(arms, list) or not arms or any(a not in ('left', 'right') for a in arms) or len(set(arms)) != len(arms):
            raise InputError('Specify unique arms_used: left and/or right')
        if task['arm_plan'] == 'single' and len(arms) != 1:
            raise InputError('A single-arm subtask must name exactly one arm')
        if task['arm_plan'] != 'single' and set(arms) != {'left', 'right'}:
            raise InputError('Handoff/coordinated_dual must name both arms')
        expected_count = 2 if task['arm_plan'] == 'coordinated_dual' else 1
        if type(task.get('simultaneous_arms')) is not int or task['simultaneous_arms'] != expected_count:
            raise InputError('simultaneous_arms is 2 for coordinated_dual, otherwise 1')
        reach = task.get('reachability')
        if not isinstance(reach, dict) or set(reach) != {'left.src', 'left.dst', 'right.src', 'right.dst'} or any(v not in ('yes', 'no', 'unknown') for v in reach.values()):
            raise InputError('Record both arms src/dst reachability as yes/no/unknown')
        deps = task.get('depends_on')
        if not isinstance(deps, list) or any(not isinstance(d, str) for d in deps) or len(set(deps)) != len(deps):
            raise InputError('depends_on must contain unique subtask IDs')
        entries[task['id']] = task
    visiting, visited = set(), set()
    def visit(identifier):
        if identifier not in entries or identifier in visiting:
            raise InputError('Dependencies must exist and be acyclic')
        if identifier in visited:
            return
        visiting.add(identifier)
        for dependency in entries[identifier]['depends_on']:
            visit(dependency)
        visiting.remove(identifier)
        visited.add(identifier)
    for identifier in entries:
        visit(identifier)
    for task in subtasks:
        if task['status'] in ('ready', 'in_progress', 'done') and any(entries[d]['status'] != 'done' for d in task['depends_on']):
            raise InputError('Ready/in_progress/done subtasks require completed dependencies')
    return deepcopy(subtasks)


class HybridTools(GroundedTools):
    method = 'pi05_plus_gpt'
    control_version = 'grounded_hybrid_v1'

    def __init__(self, output, task, student, **kwargs):
        super().__init__(output, task, **kwargs)
        self.student = student
        self.plan, self.plan_step, self.plan_revision = [], None, 0
        self.proposal = self.review = None
        self.notes_path = self.output/'codex_workspace/agent/NOTES.md'
        self.todo_path = self.notes_path.with_name('TODO.md')

    def next_call(self):
        if self.phase != 'act':
            return super().next_call()
        request_id = getattr(self, 'request', {}).get('request_id')
        if not self.plan or self.plan_step != self.tick:
            return dict(tool='update_plan', request_id=request_id,
                        response='Reconcile observed progress and publish the full subtask plan')
        if self.proposal is None:
            return dict(tool='pi05_infer', request_id=request_id,
                observation_path=str(self.observation_path), output_dir=str(self.output/'proposals'/f'{len(self.history):03d}'))
        if self.review is None:
            return dict(tool='review_pi05', request_id=request_id,
                        prediction_id=self.proposal['prediction']['prediction_id'])
        if self.review['assessment']['decision'] == 'intervene':
            return GroundedTools.next_call(self)
        return dict(tool='pi05_execute', request_id=request_id,
            prediction_id=self.proposal['prediction']['prediction_id'],
            output_dir=str(self.output/'observations'/f'{len(self.history)+1:03d}'),
            response=f'Execute at most {self.review["assessment"]["safe_steps"]} reviewed steps')

    def _check(self, name, arguments):
        if name != 'robodojo_act':
            return super()._check(name, arguments)
        if self.phase != 'act':
            raise InputError('Motion requires an active observation')
        expected = GroundedTools.next_call(self)
        for key in ('observation_path', 'output_dir'):
            if arguments.get(key) != expected[key]:
                raise InputError(f'{key} must equal {expected[key]!r}')
        directory = Path(arguments['output_dir'])
        if directory.exists() or not directory.resolve().is_relative_to(self.output):
            raise InputError('Output must be a fresh directory under this rollout root')
        return directory

    def _planning_current(self):
        if not self.plan or self.plan_step != self.tick:
            raise InputError('Call update_plan with the latest observation before inference or motion')

    def _observation(self, directory, *, result=None):
        self.proposal = self.review = None
        packet = super()._observation(directory, result=result)
        packet.update(evaluation_method=self.method, workflow='plan -> infer -> identify/replan -> review -> execute/correct',
            plan_path=str(self.output/'plan.json'), plan_revision=self.plan_revision,
            plan_update_required=self.phase != 'done',
            available_tools=[] if self.phase == 'done' else
            ['update_plan', 'grounding', 'get_depth', 'pi05_infer', 'review_pi05', 'pi05_execute', 'robodojo_act', 'robodojo_joint_act'])
        if self.phase != 'done':
            self.request.update(packet)
            packet['next_call'] = self.next_call()
            self.request['next_call'] = packet['next_call']
            write_json(packet['request_path'], self.request)
        self.observation_packet = packet
        write_json(self.observation_path, packet)
        return packet

    def start(self, **arguments):
        # Preserve the reference checkpoint identity and simulator reset contract.
        self.source = 'student'
        RoboDojoTools.start(self, **arguments)
        self.phase = 'act'
        self.run.update(evaluation_method=self.method, control_version=self.control_version,
            teacher_reasoning_effort=self.reasoning_effort, pi05_enabled=True,
            pi05_inference_calls=0, gate_policy='subtask-evidence-review', motion_planner=self.meta.get('motion_planner', 'numerical_ik'),
            absolute_frame='environment_origin', student_execution_horizon=15)
        self.run.pop('gate_instruction_sha256', None)  # Do not claim the reference gate was used.
        write_json(self.output/'run.json', self.run)
        # start() already wrote the synchronized initial observation; add its
        # request without taking another observation or physical step.
        packet = self.observation_packet
        self.request = dict(packet, request_id=uuid.uuid4().hex,
            request_path=str(self.output/'request_000.json'), plan_update_required=True)
        packet = self.request.copy()
        packet['next_call'] = self.next_call()
        self.request.update(packet)
        write_json(self.output/'request_000.json', self.request)
        write_json(self.observation_path, packet)
        return packet

    def update_plan(self, request_id, subtasks, reason):
        self._fresh(request_id)
        if not isinstance(reason, str) or not reason.strip():
            raise InputError('Explain plan changes using current observed evidence')
        plan = validate_plan(subtasks)
        self.plan, self.plan_step = plan, self.tick
        self.plan_revision += 1
        self.review = None  # A changed assignment requires a new proposal review.
        packet = dict(episode_id=self.episode, step_id=self.tick, request_id=request_id,
            revision=self.plan_revision, reason=reason, subtasks=plan, physical_steps=0)
        (self.output/'plans').mkdir(exist_ok=True)
        write_json(self.output/'plan.json', packet)
        write_json(self.output/'plans'/f'{self.plan_revision:04d}.json', packet)
        write_plan_todo(self.todo_path, plan, self.plan_revision, self.tick)
        return dict(packet, todo_path=str(self.todo_path), notes_path=str(self.notes_path), next_call=self.next_call(),
                    next_action='Infer a fresh pi05 proposal, or review the existing same-observation proposal')

    def infer(self, request_id, observation_path, output_dir):
        self._fresh(request_id)
        self._planning_current()
        if observation_path != str(self.observation_path):
            raise InputError('Use the current observation_path')
        if self.proposal is not None:
            raise InputError('Review/execute or correct the existing proposal before requesting another')
        directory = Path(output_dir)
        expected = self.output/'proposals'/f'{len(self.history):03d}'
        if directory != expected or directory.exists():
            raise InputError(f'Use fresh output_dir {expected}')
        directory.mkdir(parents=True)
        actions, prediction = self.student.infer(self.obs, directory/'actions.npz')
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (50, 14) or not np.isfinite(actions).all() or np.any((actions[:, [6, 13]] < 0) | (actions[:, [6, 13]] > 1)):
            raise RuntimeError('Invalid pi05 actions; expected finite H50x14 absolute joints/openings')
        fk = self._rpc('fk_preview', actions=actions)
        if not fk['measured_fk_check']['passed'] or len(fk['trajectory']) != 50:
            raise RuntimeError('FK preview does not match the measured robot')
        self.counters['predictions'] += 1
        self.run['pi05_inference_calls'] = self.counters['predictions']
        write_json(self.output/'run.json', self.run)
        self.proposal = dict(actions=actions.copy(), prediction=prediction, fk=fk,
                             request_id=request_id, step_id=self.tick)
        packet = dict(request_id=request_id, step_id=self.tick, **prediction,
            proposal_path=str(directory/'actions.npz'), images=self.observation_packet['images'],
            current_eef=self.request['current_eef'], student_eef_trajectory=fk['trajectory'],
            fk_check=fk['measured_fk_check'], frame='environment_origin',
            action_diagnostics=action_diagnostics(actions), physical_steps=0,
            execution_limit=15, plan_path=str(self.output/'plan.json'),
            next_call=self.next_call(),
            next_action='Use RGB + grounding + FK to identify each arm subtask; update plan if needed, then review_pi05')
        write_json(directory/'proposal.json', packet)
        self.request.update(prediction_id=prediction['prediction_id'],
                            proposal_path=packet['proposal_path'], student_eef_trajectory=fk['trajectory'])
        write_json(self.request['request_path'], self.request)
        (self.output/f'proposal_{len(self.history):03d}.npz').hardlink_to(directory/'actions.npz')
        return packet

    def review_pi05(self, request_id, prediction_id, assessment):
        self._fresh(request_id)
        self._planning_current()
        if self.proposal is None or prediction_id != self.proposal['prediction']['prediction_id']:
            raise InputError('Review the current prediction_id')
        if not isinstance(assessment, dict) or assessment.get('decision') not in ('allow', 'intervene'):
            raise InputError('Assessment decision must be allow or intervene')
        if not isinstance(assessment.get('reason'), str) or not re.fullmatch(ACT_REASON_PATTERN, assessment['reason']):
            raise InputError('Use reason format left: ..., right: ...')
        steps = assessment.get('safe_steps')
        if type(steps) is not int or not 0 <= steps <= 15:
            raise InputError('safe_steps must be 0..15')
        allow = assessment['decision'] == 'allow'
        for field in ('failure_detected', 'path_unsafe', 'dependency_violation'):
            if type(assessment.get(field)) is not bool:
                raise InputError(f'Explicit boolean {field} required')
            if allow and assessment[field]:
                raise InputError('Failure, unsafe path or dependency violation requires intervention')
        if (allow and steps == 0) or (not allow and steps != 0):
            raise InputError('allow requires 1..15 safe steps; intervene requires zero')
        entries = {t['id']:t for t in self.plan}
        selected = []
        for arm in ('left', 'right'):
            item = assessment.get(arm)
            if not isinstance(item, dict) or item.get('confidence') not in ('low', 'medium', 'high') or not isinstance(item.get('evidence'), str) or not item['evidence'].strip():
                raise InputError('Identify both arms with confidence and observed evidence')
            refs = item.get('grounding_ids')
            if not isinstance(refs, list) or any(not isinstance(r, str) or r not in self.groundings for r in refs):
                raise InputError('Grounding references must come from this observation')
            identifier = item.get('subtask_id')
            if identifier is not None and identifier not in entries:
                raise InputError('Identified subtask must exist in the current plan')
            if allow:
                if identifier is None or item['confidence'] == 'low':
                    raise InputError('Unknown/low-confidence intent cannot be released; inspect or intervene')
                task = entries[identifier]
                if task['status'] not in ('ready', 'in_progress') or arm not in task['arms_used']:
                    raise InputError('Selected subtask must be ready/in_progress and assigned to that arm')
                selected.append(task)
        if allow and selected[0]['id'] == selected[1]['id'] and selected[0]['arm_plan'] != 'coordinated_dual':
            raise InputError('Two arms acting on one subtask require coordinated_dual; split handoffs into dependencies')
        self.review = dict(request_id=request_id, prediction_id=prediction_id,
                           plan_revision=self.plan_revision, assessment=deepcopy(assessment))
        (self.output/'reviews').mkdir(exist_ok=True)
        write_json(self.output/'reviews'/f'{len(self.history):03d}_{self.plan_revision:04d}.json', self.review)
        return dict(self.review, physical_steps=0, next_call=self.next_call(),
                    next_action='pi05_execute' if allow else 'robodojo_act')

    def act(self, **arguments):
        self._planning_current()
        if self.proposal is not None and (self.review is None or self.review['assessment']['decision'] != 'intervene'):
            raise InputError('Review the pi05 proposal with decision=intervene before direct correction')
        review = deepcopy(self.review)
        prediction = self.proposal['prediction'] if self.proposal else None
        packet = super().act(**arguments)
        record = self.history[-1]
        record.update(plan_revision=self.plan_revision, policy_review=review,
                      discarded_prediction=prediction, discarded_student_steps=50 if prediction else 0)
        self.counters['recovery_steps'] += record['executed_steps']
        write_json(self.output/'history.json', self.history)
        # super().act may finalize before these audit additions.
        if self.phase == 'done':
            result = json.loads((self.output/'result.json').read_text())
            result.update(self.counters)
            write_json(self.output/'result.json', result)
            packet['result'] = result
        packet['counters'] = dict(self.counters)
        write_json(self.observation_path, packet)
        return packet

    def execute(self, request_id, prediction_id, steps, output_dir):
        self._fresh(request_id)
        self._planning_current()
        if self.proposal is None or self.review is None or prediction_id != self.proposal['prediction']['prediction_id']:
            raise InputError('A current reviewed pi05 proposal is required')
        assessment = self.review['assessment']
        if assessment['decision'] != 'allow' or self.review['plan_revision'] != self.plan_revision:
            raise InputError('Only a current allow review can execute pi05')
        if type(steps) is not int or not 1 <= steps <= assessment['safe_steps']:
            raise InputError('Execute only the reviewed safe prefix, at most 15 steps')
        directory = self._check('robodojo_act', dict(observation_path=str(self.observation_path), output_dir=output_dir))
        decision, start = len(self.history), self.tick
        response = dict(request_id=request_id, mode='student', steps=steps, reason=assessment['reason'],
                        assessment=deepcopy(assessment), plan_revision=self.plan_revision, evaluation_method=self.method)
        write_json(self.output/f'response_{decision:03d}.json', response)
        receipt = self._rpc('record_codex_decision', decision=decision, prediction_id=prediction_id, response=response)
        emit(dict(decision_event(episode_id=self.episode, step_id=self.tick, decision=decision,
            prediction_id=prediction_id, response=response), simulator_receipt=receipt))
        if self.source != 'student':
            self._rpc('switch_control_source', source='student', reason=assessment['reason'])
            self.source = 'student'
        result = self._rpc('chunk_step', actions=self.proposal['actions'][:steps].copy(),
                           student_prediction_id=prediction_id)
        self.tick = result['step_id']
        valid = [r for r in result['steps'] if r['valid']]
        executed = np.asarray([r['executed_action'] for r in valid], np.float32).reshape(-1, 14)
        np.savez_compressed(self.output/f'execution_{decision:03d}.npz', actions=executed,
            states=np.asarray([r['obs']['states'] for r in valid], np.float32).reshape(-1, 14))
        self.counters['student_steps'] += len(valid)
        done = bool(valid and (valid[-1]['terminated'] or valid[-1]['truncated']))
        self.history.append(dict(decision=decision, start_tick=start, end_tick=self.tick, response=response,
            executed_steps=len(valid), source='student', source_detail='student', prediction_id=prediction_id,
            selected_student_steps=steps, discarded_student_steps=50-len(valid),
            assessment=deepcopy(assessment), plan_revision=self.plan_revision, simulator_decision_receipt=receipt,
            executed_gripper_closed=self.gripper_history(executed), terminal=done,
            native_success=bool(valid and valid[-1]['success']), evaluation_method=self.method))
        write_json(self.output/'history.json', self.history)
        write_json(self.output/'progress.json', dict(episode_id=self.episode, step_id=self.tick,
            evaluation_method=self.method, **self.counters))
        exhausted = self.max_decisions > 0 and len(self.history) >= self.max_decisions
        final = self.finish('terminal' if done else 'decision_budget') if done or exhausted else None
        return self._observation(directory, result=final)

    def finish(self, reason):
        result = RoboDojoTools.finish(self, reason)
        result.update(evaluation_method=self.method, control_version=self.control_version,
            pi05_enabled=True, pi05_inference_calls=self.counters['predictions'], plan_revision=self.plan_revision)
        write_json(self.output/'result.json', result)
        return result
