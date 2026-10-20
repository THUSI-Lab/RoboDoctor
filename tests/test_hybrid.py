from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_rollout.robodojo.io import InputError
from robodoctor.client import GroundedTools
from robodoctor.hybrid import HybridTools, validate_plan
from robodoctor.schema import tool_specs
from test_grounded import Sim


class Student:
    metadata = dict(checkpoint_sha256='test-checkpoint', config='test-config',
                    checkpoint='/test/checkpoint', action_horizon=50, action_dim=14)

    def __init__(self):
        self.inputs = []

    def infer(self, observation, output):
        self.inputs.append(deepcopy(observation))
        actions = np.zeros((50, 14), dtype=np.float32)
        actions[:, 0] = -.2 + np.arange(50)*.001
        actions[:, 7] = .2 + np.arange(50)*.001
        actions[:, [6, 13]] = 1
        np.savez_compressed(output, actions=actions)
        return actions, dict(prediction_id=f'prediction-{len(self.inputs)}', inference_index=len(self.inputs)-1)


class HybridSim(Sim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source, self.executions = 'student', []
        self.fail_transport = False

    def request(self, op, **kwargs):
        if op == 'reset':
            self.source = kwargs['source']
        if op == 'switch_control_source':
            self.source = kwargs['source']
        if op == 'fk_preview':
            self.calls.append(op)
            return dict(measured_fk_check=dict(passed=True), trajectory=[dict(index=i, **{
                arm:dict(position=row[offset:offset+3].tolist(), quaternion_wxyz=[1,0,0,0],
                         gripper_closed=False, gripper_opening=1.)
                for arm, offset in [('left', 0), ('right', 7)]}) for i,row in enumerate(kwargs['actions'])])
        if op == 'chunk_step':
            self.executions.append(kwargs['actions'].copy())
            if self.fail_transport:
                raise TimeoutError('uncertain ACK')
            if self.source == 'student':
                self.calls.append(op)
                rows = []
                for action in kwargs['actions']:
                    self.tick += 1
                    self.states = action.copy()
                    rows.append(dict(valid=True, executed_action=action.copy(), obs=dict(states=action.copy()),
                                     terminated=self.terminal, truncated=False, success=False))
                    if self.terminal:
                        break
                return dict(step_id=self.tick, steps=rows)
        return super().request(op, **kwargs)


@pytest.fixture
def client(tmp_path):
    value = HybridTools(tmp_path, 'build_tower', Student(), rpc_factory=HybridSim, max_decisions=0)
    call = value.next_call()
    value.start(**{k:v for k,v in call.items() if k != 'tool'})
    return value


def tasks():
    return [dict(id=arm, src=f'{arm} object', dst='tray', cond='stable placement', status='ready',
                 arm_plan='single', arms_used=[arm], simultaneous_arms=1,
                 reachability={f'{a}.{p}':'unknown' for a in ('left', 'right') for p in ('src', 'dst')},
                 depends_on=[], next_action='approach', evidence='visible on table') for arm in ('left', 'right')]


def plan(client, subtasks=None):
    return client.update_plan(client.request['request_id'], subtasks or tasks(), 'Updated from current three-camera observation')


def infer(client):
    return client.infer(client.request['request_id'], str(client.observation_path),
                        str(client.output/'proposals'/f'{len(client.history):03d}'))


def assessment(decision='allow'):
    return dict(decision=decision, safe_steps=3 if decision == 'allow' else 0,
        failure_detected=False, path_unsafe=False, dependency_violation=False,
        reason='left: left approach is clear in RGB and FK, right: right approach is clear in RGB and FK',
        **{arm:dict(subtask_id=arm, confidence='high', evidence='grounded object and wrist alignment',
                   grounding_ids=[]) for arm in ('left', 'right')})


def review(client, decision='allow', value=None):
    return client.review_pi05(client.request['request_id'], client.proposal['prediction']['prediction_id'],
                              value or assessment(decision))


def execute(client, steps=3):
    return client.execute(client.request['request_id'], client.proposal['prediction']['prediction_id'], steps,
                           str(client.output/'observations'/f'{len(client.history)+1:03d}'))


def direct(client):
    return dict(observation_path=str(client.observation_path),
        output_dir=str(client.output/'observations'/f'{len(client.history)+1:03d}'),
        response=dict(request_id=client.request['request_id'], mode='eef', steps=1, stop_on_reach=False,
            reason='left: correct the left grasp using grounding, right: hold measured pose with gripper open',
            target={arm:dict(position=[0,0,.2], quaternion_wxyz=[1,0,0,0], gripper_closed=False)
                    for arm in ('left','right')}))


def test_plan_first_single_reset_and_current_request(client):
    assert client.sim.calls.count('reset') == 1
    assert client.next_call()['tool'] == 'update_plan'
    assert client.run['pi05_enabled'] and client.run['evaluation_method'] == 'pi05_plus_gpt'
    with pytest.raises(InputError, match='update_plan'):
        infer(client)
    with pytest.raises(InputError, match='update_plan'):
        client.act(**direct(client))
    packet = plan(client)
    assert packet['next_call']['tool'] == 'pi05_infer'
    assert packet['next_call']['request_id'] == client.request['request_id']
    assert 'Current plan revision 1' in client.todo_path.read_text()
    assert not client.notes_path.exists()  # The controller archives revisions, not update_plan.


def test_infer_is_read_only_fk_all_samples_and_three_images(client):
    plan(client)
    packet = infer(client)
    assert len(packet['student_eef_trajectory']) == 50
    assert len(packet['images']) == 3 and packet['frame'] == 'environment_origin'
    assert client.tick == 0 and not client.sim.executions
    assert packet['next_call']['tool'] == 'review_pi05'
    assert client.student.inputs[0]['instruction'] == 'Build a tower.'
    with pytest.raises(InputError, match='existing proposal'):
        infer(client)
    with pytest.raises(InputError, match='reviewed'):
        execute(client)


def test_reorder_independent_subtasks_then_execute_exact_prefix(client):
    plan(client)
    infer(client)
    original = client.proposal['actions'].copy()
    client.grounding(client.request['request_id'], 2, 3)
    reordered = tasks()[::-1]
    reordered[0]['status'] = 'in_progress'
    plan(client, reordered)
    value = assessment()
    value['right']['grounding_ids'] = ['g']
    review(client, value=value)
    packet = execute(client)
    np.testing.assert_array_equal(client.sim.executions[0], original[:3])
    assert client.counters['student_steps'] == 3 and client.tick == 3
    assert client.proposal is None and client.review is None
    assert packet['next_call']['tool'] == 'update_plan'
    assert packet['next_call']['request_id'] == packet['request_id']
    assert client.history[-1]['discarded_student_steps'] == 47
    with pytest.raises(InputError, match='update_plan'):
        infer(client)
    assert not client.groundings  # Old pixel evidence is invalid after motion.
    plan(client)
    infer(client)
    np.testing.assert_array_equal(client.student.inputs[-1]['states'], original[2])


def test_plan_changes_invalidate_review_not_prediction(client):
    plan(client)
    infer(client)
    review(client)
    prediction = client.proposal
    plan(client)
    assert client.proposal is prediction and client.review is None
    with pytest.raises(InputError):
        execute(client)


@pytest.mark.parametrize('failure', ['path_unsafe', 'dependency_violation', 'failure_detected'])
def test_failure_cannot_be_allowed(client, failure):
    plan(client)
    infer(client)
    value = assessment()
    value[failure] = True
    with pytest.raises(InputError, match='requires intervention'):
        review(client, value=value)
    assert not client.sim.executions


def test_dependency_and_grounding_identity_rejections(client):
    value = tasks()
    value[1].update(status='pending', depends_on=['left'])
    plan(client, value)
    infer(client)
    with pytest.raises(InputError, match='ready/in_progress'):
        review(client)
    decision = assessment('intervene')
    decision['right']['grounding_ids'] = ['stale-grounding']
    with pytest.raises(InputError, match='Grounding references'):
        review(client, value=decision)
    decision['right']['grounding_ids'] = []
    decision['dependency_violation'] = True
    assert review(client, value=decision)['next_action'] == 'robodojo_act'


def test_correction_discards_proposal_switches_source_and_reinfers(client):
    plan(client)
    infer(client)
    with pytest.raises(InputError, match='intervene'):
        client.act(**direct(client))
    review(client, 'intervene')
    packet = client.act(**direct(client))
    assert client.source == client.sim.source == 'gpt_eef'
    assert client.proposal is None and client.tick == 1
    assert client.counters['recovery_steps'] == 1
    assert client.history[-1]['discarded_prediction']['prediction_id'] == 'prediction-1'
    assert packet['previous_result']['policy_review']['assessment']['decision'] == 'intervene'
    plan(client)
    infer(client)
    review(client)
    execute(client)
    assert client.sim.source == 'student' and client.tick == 4
    assert client.sim.calls.count('reset') == 1


def test_reject_long_or_stale_prefix(client):
    plan(client)
    infer(client)
    review(client)
    with pytest.raises(InputError, match='safe prefix'):
        execute(client, 4)
    with pytest.raises(InputError, match='reviewed'):
        client.execute(client.request['request_id'], 'old-prediction', 1, 'unused')
    assert not client.sim.executions


def test_terminal_native_ends_prefix_and_preserves_pi05_identity(client):
    plan(client)
    infer(client)
    review(client)
    client.sim.terminal = True
    packet = execute(client)
    assert packet['rollout_finished'] and client.tick == 1
    result = json.loads((client.output/'result.json').read_text())
    assert result['pi05_enabled'] and result['pi05_inference_calls'] == 1
    assert result['control_version'] == 'grounded_hybrid_v1'
    assert client.history[-1]['discarded_student_steps'] == 49


def test_uncertain_ack_is_not_retried_or_marked_recoverable(client):
    plan(client)
    infer(client)
    review(client)
    client.sim.fail_transport = True
    with pytest.raises(TimeoutError, match='uncertain'):
        execute(client)
    assert len(client.sim.executions) == 1 and not client.history


@pytest.mark.parametrize('kind', ['cycle', 'missing', 'premature', 'duplicate'])
def test_invalid_dependency_graph(kind):
    value = tasks()
    if kind == 'cycle':
        value[0]['depends_on'] = ['right']
        value[1]['depends_on'] = ['left']
    elif kind == 'missing':
        value[0]['depends_on'] = ['absent']
    elif kind == 'premature':
        value[0]['depends_on'] = ['right']
    else:
        value[1]['id'] = 'left'
    with pytest.raises(InputError):
        validate_plan(value)


def test_hybrid_workspace_tools_and_prompt_snapshot(tmp_path, monkeypatch):
    from robodoctor import runner
    from test_recovery import Transport
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a,**k:SimpleNamespace(stdout='test-codex'))
    worker = runner.CodexPolicy(tmp_path/'controller/codex_workspace', '/fake/codex',
        method='pi05_plus_gpt', transport_factory=Transport)
    try:
        assert worker.controller_version == 'grounded_hybrid_v1'
        tools = tool_specs('pi05_plus_gpt')
        assert len(tools) == 9
        import jsonschema
        for tool in tools:
            jsonschema.Draft202012Validator.check_schema(tool['inputSchema'])
        assert (worker.agent_workspace/'context/direct_control.md').exists()
        assert (worker.agent_workspace/'.agents/skills/robodojo-hybrid-rollout/gate_prompt.md').exists()
    finally:
        worker.close()


def test_agent_hybrid_tool_loop_and_413_keeps_review_without_replay(client, tmp_path, monkeypatch):
    from robodoctor import runner
    from test_recovery import Transport, error_event, completed, PAYLOAD_ERROR
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a,**k:SimpleNamespace(stdout='test-codex'))
    monkeypatch.setattr(runner.time, 'sleep', lambda *_:None)
    worker = runner.CodexPolicy(tmp_path/'codex_workspace', '/fake/codex',
        method='pi05_plus_gpt', transport_factory=Transport)
    def call(name, args, terminal=False):
        def event(transport):
            client.sim.terminal = terminal
            return dict(id=len(transport.replies)+1, method='item/tool/call', params=dict(
                threadId=transport.thread, turnId=transport.turn, callId=str(len(transport.replies)),
                tool=name, arguments=args()))
        return event
    worker.transport.events = [
        call('update_plan', lambda:dict(request_id=client.request['request_id'], subtasks=tasks(), reason='initial plan')),
        call('pi05_infer', lambda:dict(request_id=client.request['request_id'],
            observation_path=str(client.observation_path), output_dir=str(client.output/'proposals/000'))),
        call('review_pi05', lambda:dict(request_id=client.request['request_id'],
            prediction_id=client.proposal['prediction']['prediction_id'], assessment=assessment())),
        error_event(), completed(PAYLOAD_ERROR),
        call('pi05_execute', lambda:dict(request_id=client.request['request_id'],
            prediction_id=client.proposal['prediction']['prediction_id'], steps=3,
            output_dir=str(client.output/'observations/001')), terminal=True), completed()]
    try:
        worker.run(client)
        assert all(r['success'] for _,r in worker.transport.replies)
        assert worker.transport.threads == 2
        assert client.counters['predictions'] == len(client.sim.executions) == 1
        assert client.sim.calls.count('reset') == 1 and client.phase == 'done'
    finally:
        worker.close()


def test_hybrid_video_with_hold_policy_and_correction_without_fake_proposals(client):
    import imageio.v2 as imageio
    from robodoctor.debug_video import RecordedDebugVideo
    # First a hold/direct action without any policy proposal.
    plan(client)
    client.act(**direct(client))
    assert not (client.output/'proposal_000.npz').exists()
    plan(client)
    infer(client)
    review(client)
    execute(client, 2)
    plan(client)
    infer(client)
    review(client, 'intervene')
    client.sim.terminal = True
    client.act(**direct(client))
    sim = client.output.parent/'sim'
    sim.mkdir(exist_ok=True)
    with imageio.get_writer(str(sim/'sensors.mp4'), fps=25, codec='libx264') as writer:
        for tick in range(client.tick+1):
            writer.append_data(np.full((32,96,3), tick*20, dtype=np.uint8))
    result = RecordedDebugVideo(client.output).render()
    assert result['frames'] == client.tick+1 == 5
    assert result['layout_version'] == 'hybrid_plan_v1'
    with imageio.get_reader(result['video_path']) as reader:
        assert reader.count_frames() == 5
    assert not (client.output/'proposal_000.npz').exists()
