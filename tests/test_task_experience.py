import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robodoctor.task_experience import TaskExperience, validate_task_experience
from robodoctor import runner
from test_recovery import Transport, completed, policy, ready, tool_call


def store(tmp_path, source=None):
    controller = tmp_path/'controller'
    agent = controller/'codex_workspace/agent'
    agent.mkdir(parents=True)
    (agent/'workspace.json').write_text('{}')
    (agent/'AGENTS.md').write_text('Direct control\n')
    (agent/'TODO.md').write_text('Placement unconfirmed; grip slipped.\n')
    (agent/'NOTES.md').write_text('Step 3: gripper closed but object remained on table.\n')
    return TaskExperience(controller, agent, source=source, iteration=3)


def episode():
    return SimpleNamespace(task='build_tower', seed=2, episode='ep', tick=3, phase='done')


@pytest.mark.parametrize('source_kind', ['missing', 'empty', 'wrong_suffix'])
def test_bad_input_experience_rejected(tmp_path, source_kind):
    path = tmp_path/('experience.json' if source_kind == 'wrong_suffix' else 'experience.md')
    if source_kind != 'missing':
        path.write_text('' if source_kind == 'empty' else 'Task reference')
    with pytest.raises(ValueError, match='nonempty Markdown'):
        validate_task_experience(path)


def test_prior_experience_is_snapshotted_and_added_to_planning_prompt(tmp_path, monkeypatch):
    prior = tmp_path/'iter2.md'
    prior.write_text('Prior RSI lesson: retreat before approaching again.\n')
    original = prior.read_bytes()
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='test-codex'))
    worker = runner.CodexPolicy(tmp_path/'controller/codex_workspace', '/fake/codex',
                               task_experience=prior, iteration=3, transport_factory=Transport)
    assert str(worker.task_experience.prior) in worker.prompt
    assert 'Read it before planning motion' in worker.prompt
    assert 'reobserve and re-ground' in worker.prompt
    assert worker.task_experience.prior.read_bytes() == original
    worker.task_experience.prior.write_text('Changed local reference\n')
    assert prior.read_bytes() == original
    metadata = json.loads((worker.agent_workspace/'workspace.json').read_text())
    assert metadata['iteration'] == 3
    assert metadata['task_experience_output'].endswith('controller/task_experience.md')


def test_fallback_preserves_prior_and_current_evidence_without_claiming_success(tmp_path):
    prior = tmp_path/'iter2.md'
    prior.write_text('Prior lesson: avoid a shallow pinch.\n')
    experience = store(tmp_path, prior)
    path = experience.save(episode())
    text = path.read_text()
    assert 'recorded evidence fallback' in text
    assert 'Prior lesson: avoid a shallow pinch' in text
    assert 'Placement unconfirmed; grip slipped' in text
    assert 'object remained on table' in text
    assert 'Native success: unknown' in text
    assert not path.with_suffix('.md.tmp').exists()


def test_model_summary_has_host_native_outcome_including_failure(tmp_path):
    experience = store(tmp_path)
    sim = tmp_path/'sim'
    sim.mkdir()
    (sim/'evaluation_outcome.json').write_text(json.dumps(dict(
        complete=True, valid_for_success_rate=True, native_success=False, native_score=0.2)))
    (experience.controller/'result.json').write_text('{"reason":"native_timeout"}')
    experience.draft.write_text('## Improvements\nUse a deeper grip and verify with a test lift.\n')
    text = experience.save(episode()).read_text()
    assert 'Summary source: model' in text
    assert 'Use a deeper grip' in text
    assert 'Native success: False' in text
    assert 'Native score (0–1): 0.2' in text
    assert 'Finish reason: native_timeout' in text


def test_summary_turn_cannot_move_robot_or_overwrite_execution_evidence(policy, tmp_path):
    client = ready(tmp_path)
    policy.transport.events = [tool_call(client, terminal=True), completed()]
    policy.run(client)
    original = (policy.workspace/'call_0000_request.json').read_bytes()
    physical_calls = list(client.sim.calls)
    tick = client.tick
    def write_summary(transport):
        policy.task_experience.draft.write_text('Verified task lessons: check support before releasing.\n')
        return dict(method='item/completed', params=dict(threadId=transport.thread, turnId=transport.turn,
                    item=dict(type='fileChange', id='experience-write', status='completed')))
    policy.transport.events = [tool_call(client, arguments={'forbidden': True}), write_summary, completed()]
    policy.finalize_task_experience(client)
    assert client.tick == tick
    assert client.sim.calls == physical_calls
    assert (policy.workspace/'call_0000_request.json').read_bytes() == original
    assert not policy.transport.replies[-1][1]['success']
    assert 'Task experience generation cannot call simulator services' in json.dumps(policy.transport.replies[-1])
    assert (policy.workspace/'experience_call_0000_request.json').exists()
    assert 'check support before releasing' in policy.task_experience.output.read_text()
    prompts = [params['input'][0]['text'] for method, params in policy.transport.requests if method == 'turn/start']
    assert 'Do not call any simulator service' in prompts[-1]


@pytest.mark.parametrize('failure', ['transport', 'interrupt', 'pending_turn'])
def test_summary_failure_or_interruption_keeps_fallback(policy, tmp_path, failure):
    client = ready(tmp_path)
    policy.transport.events = [tool_call(client, terminal=True), completed()]
    policy.run(client)
    if failure == 'pending_turn':
        policy.last_run_completed = False
        turns = policy.transport.turns
        policy.finalize_task_experience(client)
        assert policy.transport.turns == turns
    else:
        def interrupt(transport):
            raise KeyboardInterrupt('stop')
        policy.transport.events = [interrupt if failure == 'interrupt' else RuntimeError('transport closed')]
        if failure == 'interrupt':
            with pytest.raises(KeyboardInterrupt):
                policy.finalize_task_experience(client)
        else:
            policy.finalize_task_experience(client)
            assert (policy.workspace/'task_experience_failure.json').exists()
    text = policy.task_experience.output.read_text()
    assert 'recorded evidence fallback' in text
    assert 'Native success: unknown' in text


def test_summary_keeps_usage_from_all_previous_threads(policy, tmp_path):
    client = ready(tmp_path)
    policy.thread_usage['previous-thread'] = dict(total=dict(inputTokens=100, outputTokens=10))
    policy.transport.events = [tool_call(client, terminal=True), completed()]
    policy.run(client)
    def usage(transport):
        return dict(method='thread/tokenUsage/updated', params=dict(threadId=transport.thread,
            tokenUsage=dict(total=dict(inputTokens=250, outputTokens=25))))
    policy.transport.events = [usage, completed()]
    policy.finalize_task_experience(client)
    report = json.loads((policy.workspace.parent/'token_usage.json').read_text())
    assert report['usage']['total'] == dict(inputTokens=350, outputTokens=35)
