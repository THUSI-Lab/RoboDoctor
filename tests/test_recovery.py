import json
from types import SimpleNamespace

import pytest

from robodoctor import runner
from robodoctor.recovery import closed_failed_turn, oversized_request, recovery_message
from test_grounded import started, move_args, act_args


PAYLOAD_ERROR = dict(message='unexpected status 413 Payload Too Large: <html>cloudflare</html>',
                     codexErrorInfo='other')


class Transport:
    def __init__(self, argv, workspace):
        self.process = SimpleNamespace(pid=123)
        self.requests, self.replies, self.events, self.probes = [], [], [], []
        self.threads = self.turns = 0

    def request(self, method, params, timeout):
        self.requests.append((method, params))
        if method == 'thread/start':
            self.threads += 1
            self.thread = f'thread-{self.threads}'
            return dict(model=runner.MODEL, reasoningEffort=params['config']['model_reasoning_effort'],
                        thread=dict(id=self.thread))
        if method == 'turn/start':
            self.turns += 1
            self.turn = f'turn-{self.turns}'
            return dict(turn=dict(id=self.turn))
        if method == 'thread/read':
            return self.probes.pop(0)(self)
        return {}

    def next_message(self, timeout):
        assert self.events, 'Unexpected wait: recovery did not finish'
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event(self)

    def reply(self, identifier, result):
        self.replies.append((identifier, result))

    def notify(self, *args):
        pass

    def close(self):
        pass


@pytest.fixture
def policy(tmp_path, monkeypatch, request):
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='test-codex'))
    monkeypatch.setattr(runner.time, 'sleep', lambda delay: None)
    monkeypatch.delenv('CODEX_MAX_TOTAL_TOKENS', raising=False)
    return runner.CodexPolicy(tmp_path/'controller/codex_workspace', '/fake/codex',
                              transport_factory=Transport, effort=getattr(request, 'param', 'xhigh'))


def error_event(error=PAYLOAD_ERROR, retry=False):
    return lambda t: dict(method='error', params=dict(threadId=t.thread, turnId=t.turn,
                                                     error=error, willRetry=retry))


def completed(error=None):
    return lambda t: dict(method='turn/completed', params=dict(threadId=t.thread,
        turn=dict(id=t.turn, status='failed' if error else 'completed', error=error)))


def tool_call(client, *, terminal=False, arguments=None):
    def event(t):
        client.sim.terminal = terminal
        return dict(id=len(t.replies)+1, method='item/tool/call',
            params=dict(threadId=t.thread, turnId=t.turn, callId=str(len(t.replies)),
                        tool='robodojo_act', arguments=arguments or act_args(client)))
    return event


def ready(tmp_path):
    (tmp_path/'rollout').mkdir()
    client = started(tmp_path/'rollout')
    client.act(**move_args(client))
    return client


def test_413_feedback_new_thread_keeps_episode_and_never_replays(policy, tmp_path):
    client = ready(tmp_path)
    episode = client.episode
    policy.transport.events = [tool_call(client), error_event(), completed(PAYLOAD_ERROR),
                               tool_call(client, terminal=True), completed()]
    policy.run(client)
    assert client.phase == 'done' and client.tick == 5 and client.episode == episode
    assert client.sim.calls.count('reset') == 1
    assert len(client.sim.eef_targets) == 5  # Initial approach + three ACKs before error + one after.
    assert len(policy.transport.replies) == 2
    assert len(client.history) == 3  # Initial approach and two model-chosen actions.
    threads = [p for m, p in policy.transport.requests if m == 'thread/start']
    assert len(threads) == 2 and threads[0] == threads[1]
    turns = [p for m, p in policy.transport.requests if m == 'turn/start']
    feedback = turns[1]['input'][0]['text']
    assert 'cannot be sent as constructed' in feedback
    assert 'Do not repeat that request' in feedback
    assert '413 Payload Too Large' in feedback and '"step_id": 4' in feedback
    assert 'data:image' not in feedback and len(feedback) < 8000
    worker = json.loads((policy.workspace/'worker.json').read_text())
    assert worker['thread_ids'] == ['thread-1', 'thread-2']
    record = json.loads((policy.workspace/'model_error_0000.json').read_text())
    assert record['physical_actions_replayed'] == 0 and record['status'] == 'continue_sent'


@pytest.mark.parametrize('error', [dict(message='503 service unavailable'),
    dict(message='network error'), dict(message='Model server internal error')])
def test_other_model_errors_return_feedback_in_same_thread(policy, tmp_path, error):
    client = ready(tmp_path)
    policy.transport.events = [error_event(error), completed(error),
                               tool_call(client, terminal=True), completed()]
    policy.run(client)
    assert policy.transport.threads == 1 and policy.transport.turns == 2
    turns = [p for m, p in policy.transport.requests if m == 'turn/start']
    assert error['message'] in turns[1]['input'][0]['text']
    assert client.tick == 2 and client.sim.calls.count('reset') == 1


def test_appserver_retry_does_not_start_competing_turn(policy, tmp_path):
    client = ready(tmp_path)
    policy.transport.events = [error_event(retry=True), tool_call(client, terminal=True), completed()]
    policy.run(client)
    assert policy.transport.turns == 1 and policy.transport.threads == 1
    assert not list(policy.workspace.glob('model_error_*.json'))


def test_failed_completion_without_error_notification_recovers(policy, tmp_path):
    client = ready(tmp_path)
    policy.transport.events = [completed(PAYLOAD_ERROR), tool_call(client, terminal=True), completed()]
    policy.run(client)
    assert policy.transport.threads == 2 and client.tick == 2


def test_missing_completion_waits_until_probe_confirms_failed_idle(policy, tmp_path):
    client = ready(tmp_path)
    def probe(t, status):
        assert t.turns == 1 and client.tick == 1
        return dict(thread=dict(id=t.thread, status=dict(type=status),
            turns=[dict(id=t.turn, status='failed', error=PAYLOAD_ERROR)]))
    policy.transport.probes = [lambda t: probe(t, 'active'), lambda t: probe(t, 'idle')]
    policy.transport.events = [error_event(), TimeoutError(), TimeoutError(),
                               tool_call(client, terminal=True), completed()]
    policy.run(client)
    assert not policy.transport.probes and policy.transport.turns == 2


def test_repeated_provider_errors_do_not_hit_old_retry_cap(policy, tmp_path):
    client = ready(tmp_path)
    error = dict(message='503 unavailable')
    policy.transport.events = [completed(error)] * 22 + [tool_call(client, terminal=True), completed()]
    policy.run(client)
    assert policy.transport.turns == 23 and client.tick == 2
    last = json.loads((policy.workspace/'model_error_0021.json').read_text())
    assert last['delay_seconds'] == 60


def test_native_done_does_not_trigger_paid_recovery(policy, tmp_path):
    client = ready(tmp_path)
    policy.transport.events = [tool_call(client, terminal=True), error_event()]
    policy.run(client)
    assert policy.transport.turns == 1 and client.phase == 'done'


def test_tokens_accumulate_across_threads_without_limit(policy, tmp_path, monkeypatch):
    monkeypatch.setenv('CODEX_MAX_TOTAL_TOKENS', '100')
    client = ready(tmp_path)
    def usage(count):
        return lambda t: dict(method='thread/tokenUsage/updated', params=dict(threadId=t.thread,
            tokenUsage=dict(total=dict(totalTokens=count))))
    policy.transport.events = [usage(60), completed(PAYLOAD_ERROR), usage(50),
                               tool_call(client, terminal=True), completed()]
    policy.run(client)
    saved = json.loads((policy.workspace.parent/'token_usage.json').read_text())
    assert saved['usage']['total']['totalTokens'] == 110
    assert saved['configured_limit'] is None
    assert client.phase == 'done' and client.tick == 2


def test_uncertain_physical_error_is_not_replayed_or_treated_as_model_error(policy, tmp_path):
    client = ready(tmp_path)
    calls = []
    def uncertain(**arguments):
        calls.append(arguments)
        raise RuntimeError('Simulator ACK lost')
    client.act = uncertain
    policy.transport.events = [tool_call(client)]
    with pytest.raises(RuntimeError, match='ACK lost'):
        policy.run(client)
    assert len(calls) == 1 and policy.transport.turns == 1


@pytest.mark.parametrize('error', [PAYLOAD_ERROR,
    {'codexErrorInfo': {'responseStreamDisconnected': {'httpStatusCode': 413}}},
    {'message': 'context_length_exceeded'}, {'message': 'Request Entity Too Large'},
    {'message': '{"error":{"message":"Exceeded maximum number of images (50) allowed in the request."}}'}])
def test_oversized_detection(error):
    assert oversized_request(error)


def test_image_limit_feedback_new_thread_keeps_episode(policy, tmp_path):
    client = ready(tmp_path)
    image_error = {'message': 'Exceeded maximum number of images (50) allowed in the request.'}
    policy.transport.events = [completed(image_error), tool_call(client, terminal=True), completed()]
    policy.run(client)
    assert policy.transport.threads == 2
    assert client.sim.calls.count('reset') == 1
    assert client.phase == 'done'


def test_image_budget_rotates_thread_and_injects_latest_three(policy, tmp_path):
    policy.latest_image_items = [
        {'camera': 'cam_high', 'path': 'high.png'},
        {'camera': 'cam_left_wrist', 'path': 'left.png'},
        {'camera': 'cam_right_wrist', 'path': 'right.png'},
    ]
    policy.thread_image_count = runner.THREAD_IMAGE_ROTATE_AT
    assert policy._maybe_rotate_thread('image_budget')
    assert policy.transport.threads == 2
    policy._turn('Continue')
    turns = [p for m, p in policy.transport.requests if m == 'turn/start']
    assert [item['path'] for item in turns[-1]['input'] if item['type'] == 'localImage'] == [
        'high.png', 'left.png', 'right.png']
    assert policy.thread_image_count == 3


def test_small_error_feedback_even_for_large_html(tmp_path):
    client = ready(tmp_path)
    message = recovery_message(client, {'message': '413 ' + 'x'*100000},
                               tmp_path/'error.json', fresh_thread=True)
    assert len(message) < 8000 and 'cannot be sent as constructed' in message


def test_probe_rejects_unrelated_and_active_turns():
    for status, thread, turn in [('active', 't', 'u'), ('idle', 'old', 'u'), ('idle', 't', 'old')]:
        assert closed_failed_turn(dict(thread=dict(id=thread, status=dict(type=status),
            turns=[dict(id=turn, status='failed')])), 't', 'u') is None


@pytest.mark.parametrize('policy', ['medium'], indirect=True)
def test_medium_effort_applies_to_config_turns_and_recovery(policy, tmp_path):
    client = ready(tmp_path)
    policy.transport.events = [completed(PAYLOAD_ERROR), tool_call(client, terminal=True), completed()]
    policy.run(client)
    launch = json.loads((policy.workspace/'launch.json').read_text())
    worker = json.loads((policy.workspace/'worker.json').read_text())
    assert launch['config']['model_reasoning_effort'] == 'medium'
    assert worker['reasoning_effort'] == 'medium'
    assert policy.transport.threads == 2
    for method, params in policy.transport.requests:
        if method == 'thread/start':
            assert params['config']['model_reasoning_effort'] == 'medium'
        elif method == 'turn/start':
            assert params['effort'] == 'medium'
