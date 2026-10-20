import http.client
import json

import pytest

from robodoctor.launch import prepare_codex, verify_smoke
from test_grounded import started, move_args


@pytest.mark.parametrize('owned_student', [False, True])
def test_hybrid_launch_passes_identity_and_cleans_only_owned_processes(tmp_path, monkeypatch, owned_student):
    from robodoctor import launch
    from types import SimpleNamespace
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    output = tmp_path/'run'
    argv = ['launch', '--output', str(output), '--evaluation-method', 'pi05_plus_gpt',
            '--checkpoint', str(checkpoint), '--gpu', '2', '--student-gpu', '3']
    if owned_student:
        argv += ['--student-python', '/fake/openpi-python']
    monkeypatch.setattr(launch.sys, 'argv', argv)
    monkeypatch.setattr(launch, 'prepare_codex', lambda *a,**k: ({}, {'provider':'test'}))
    monkeypatch.setattr(launch.subprocess, 'run', lambda *a,**k: SimpleNamespace(stdout='fake-codex'))
    monkeypatch.setattr(launch.signal, 'signal', lambda *a: None)
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, *args): pass
        def setsockopt(self, *args): pass
    monkeypatch.setattr(launch.socket, 'socket', lambda *a,**k:Socket())
    monkeypatch.setattr(launch, 'student_ready', lambda port:True)
    calls, stopped = [], []
    def popen(command, **kwargs):
        process = SimpleNamespace(pid=len(calls)+100, returncode=0, poll=lambda:None)
        calls.append((command, kwargs, process))
        if command[0].endswith('run_server.sh'):
            kwargs['stdout'].write('{"event": "ready"}\n')
        return process
    monkeypatch.setattr(launch.subprocess, 'Popen', popen)
    monkeypatch.setattr(launch, 'wait_for_controller', lambda *a:None)
    monkeypatch.setattr(launch, 'stop_process', lambda p: stopped.append(p.pid) if p else None)
    launch.main()
    assert len(calls) == (3 if owned_student else 2)
    command = calls[-1][0]
    assert command[command.index('--evaluation-method')+1] == 'pi05_plus_gpt'
    assert command[command.index('--checkpoint')+1] == str(checkpoint)
    assert set(stopped) == {p.pid for _,_,p in calls}
    if owned_student:
        assert calls[0][0][0] == '/fake/openpi-python'
        assert calls[0][1]['env']['CUDA_VISIBLE_DEVICES'] == '3'
    assert calls[-2][1]['env']['CUDA_VISIBLE_DEVICES'] == '2'


@pytest.mark.parametrize('status,body,expected', [(200, b'OK\n', True), (503, b'OK\n', False),
                                                (200, b'not OpenPI', False)])
def test_student_readiness_uses_health_request(monkeypatch, status, body, expected):
    from robodoctor import launch
    from types import SimpleNamespace
    calls = []
    class Connection:
        def __init__(self, host, port, timeout):
            assert (host, port, timeout) == ('127.0.0.1', 18830, 1)
        def request(self, method, path): calls.append((method, path))
        def getresponse(self): return SimpleNamespace(status=status, read=lambda n:body)
        def close(self): calls.append('closed')
    monkeypatch.setattr(launch.http.client, 'HTTPConnection', Connection)
    assert launch.student_ready(18830) is expected
    assert calls == [('GET', '/healthz'), 'closed']


@pytest.mark.parametrize('error', [ConnectionRefusedError(), TimeoutError(),
                                  http.client.BadStatusLine('bad HTTP')])
def test_student_readiness_retries_startup_errors_and_closes(monkeypatch, error):
    from robodoctor import launch
    closed = []
    class Connection:
        def __init__(self, *args, **kwargs): pass
        def request(self, *args): raise error
        def close(self): closed.append(True)
    monkeypatch.setattr(launch.http.client, 'HTTPConnection', Connection)
    assert launch.student_ready(18830) is False
    assert closed == [True]


def final_records(root, **changes):
    result = dict(episode_id='episode', step_id=1300, video_frames=1301, reason='terminal')
    result.update(changes)
    for name in ('controller/result.json', 'sim/summary.json'):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))


def test_wait_allows_simulator_exit_during_video_rendering(tmp_path, monkeypatch):
    from robodoctor import launch
    from types import SimpleNamespace
    final_records(tmp_path)
    states = iter([None, None, 0])
    ticks = iter([0, 10, 11, 12])
    monkeypatch.setattr(launch.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(launch.time, 'sleep', lambda _: None)
    launch.wait_for_controller(SimpleNamespace(poll=lambda: next(states)),
        SimpleNamespace(poll=lambda: 0), tmp_path, run_timeout=1)


@pytest.mark.parametrize('evidence', ['missing', 'mismatched_episode', 'missing_frames', 'partial_json'])
def test_wait_still_rejects_early_simulator_exit(tmp_path, evidence):
    from robodoctor import launch
    from types import SimpleNamespace
    if evidence != 'missing':
        final_records(tmp_path)
        path = tmp_path/'sim/summary.json'
        if evidence == 'partial_json':
            path.write_text('{')
        else:
            data = json.loads(path.read_text())
            data['episode_id' if evidence == 'mismatched_episode' else 'video_frames'] = 'incorrect'
            path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match='Simulator exited'):
        launch.wait_for_controller(SimpleNamespace(poll=lambda: None),
            SimpleNamespace(poll=lambda: 0), tmp_path, run_timeout=10)


def test_wait_finalization_has_separate_timeout(tmp_path, monkeypatch):
    from robodoctor import launch
    from types import SimpleNamespace
    final_records(tmp_path)
    ticks = iter([0, 1, 3])
    monkeypatch.setattr(launch.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(TimeoutError, match='debug video timeout'):
        launch.wait_for_controller(SimpleNamespace(poll=lambda: None),
            SimpleNamespace(poll=lambda: 0), tmp_path, run_timeout=100, finalize_timeout=1)


def test_wait_still_limits_active_rollout(tmp_path, monkeypatch):
    from robodoctor import launch
    from types import SimpleNamespace
    ticks = iter([0, 2])
    monkeypatch.setattr(launch.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(TimeoutError, match='rollout wall-time'):
        launch.wait_for_controller(SimpleNamespace(poll=lambda: None),
            SimpleNamespace(poll=lambda: None), tmp_path, run_timeout=1)


def test_isolated_config_does_not_copy_credentials_or_parent_session(tmp_path):
    source = tmp_path/'source.toml'
    source.write_text('model_provider="openai"\n[model_providers.openai]\n'
                      'base_url="https://api.openai.com/v1"\nexperimental_bearer_token="synthetic-secret"\n')
    original = source.read_bytes()
    env, identity = prepare_codex(tmp_path/'home', auth_profile='openai', source_config=source,
                                 environ={'CODEX_HOME':'/parent', 'CODEX_THREAD_ID':'parent-thread'})
    assert env['OPENAI_API_KEY'] == 'synthetic-secret'
    assert env['CODEX_HOME'] == str(tmp_path/'home')
    assert 'CODEX_THREAD_ID' not in env
    assert identity['provider'] == 'openai'
    assert source.read_bytes() == original
    assert 'synthetic-secret' not in (tmp_path/'home/config.toml').read_text()
    assert (tmp_path/'home').stat().st_mode & 0o777 == 0o700


def test_source_provider_mismatch_rejects_without_home(tmp_path):
    source = tmp_path/'source.toml'
    source.write_text('model_provider="other"\n')
    with pytest.raises(ValueError, match='match'):
        prepare_codex(tmp_path/'home', auth_profile='openai', source_config=source, environ={})
    assert not (tmp_path/'home').exists()


def test_medium_effort_in_generated_config_and_rollout_metadata(tmp_path):
    from robodoctor.client import GroundedTools
    from test_grounded import Sim
    home = tmp_path/'home'
    prepare_codex(home, auth_profile='openai', source_config=None,
                  environ={'OPENAI_API_KEY':'synthetic-secret'}, effort='medium')
    assert 'model_reasoning_effort = "medium"' in (home/'config.toml').read_text()
    output = tmp_path/'controller'
    output.mkdir()
    client = GroundedTools(output, 'build_tower', rpc_factory=Sim, reasoning_effort='medium')
    args = client.next_call(); args.pop('tool')
    client.start(**args)
    assert json.loads((output/'run.json').read_text())['teacher_reasoning_effort'] == 'medium'


def test_smoke_stops_at_arrival_without_manipulation(tmp_path):
    client = started(tmp_path)
    client.smoke_approach = True
    packet = client.act(**move_args(client))
    assert packet['movement']['reached'] and client.phase == 'done'
    assert packet['result']['reason'] == 'smoke_approach_complete'
    assert client.tick == 1 and not packet['result']['complete']


def test_smoke_proof_rejects_tool_call_without_measured_arrival(tmp_path):
    workspace = tmp_path/'controller/codex_workspace'
    workspace.mkdir(parents=True)
    (workspace/'worker.json').write_text(json.dumps(dict(thread_id='t',pid=123)))
    (workspace/'call_0000_request.json').write_text(json.dumps(dict(tool='grounding')))
    (workspace/'call_0000_result.json').write_text(json.dumps(dict(ok=True)))
    (tmp_path/'controller/result.json').write_text(json.dumps(dict(reason='decision_budget')))
    with pytest.raises(RuntimeError, match='did not reach'):
        verify_smoke(tmp_path)
    assert not json.loads((tmp_path/'smoke_result.json').read_text())['passed']


import socket

import pytest

from robodoctor.launch import require_free_port


def test_port_check_rejects_an_active_listener():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        with pytest.raises(OSError):
            require_free_port(listener.getsockname()[1])


def test_port_check_allows_reuse_after_a_server_closes():
    with socket.socket() as listener, socket.socket() as client:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        listener.listen()
        client.connect(('127.0.0.1', port))
        connection, _ = listener.accept()
        with connection:
            connection.shutdown(socket.SHUT_WR)
            assert client.recv(1) == b''
            client.shutdown(socket.SHUT_WR)
            assert connection.recv(1) == b''
    # The server's closed connection remains in TIME_WAIT. The earlier
    # plain bind rejected this port before the reusable HTTP server started.
    with socket.socket() as probe:
        with pytest.raises(OSError):
            probe.bind(('127.0.0.1', port))
    require_free_port(port)


@pytest.mark.parametrize('with_harness', [False, True])
def test_rsi_launch_owns_prismwam_and_forwards_experience_and_config_effort(tmp_path, monkeypatch, with_harness):
    from robodoctor import launch
    from types import SimpleNamespace
    checkpoint = tmp_path/'weights.pt'
    checkpoint.touch()
    config = tmp_path/'source.toml'
    config.write_text('model="gpt-6-astra"\nmodel_reasoning_effort="medium"\n'
                      'model_provider="test"\naction_chunk_length=32\n'
                      '[model_providers.test]\nbase_url="https://example.invalid/v1"\n')
    experience = tmp_path/'experience.md'
    experience.write_text('# Experience\n')
    output = tmp_path/'run'
    extra = []
    if with_harness:
        from test_pipeline import fake_harness
        experience.write_text('Experience through iter2\n')
        snapshot = fake_harness(None, experience=experience, output=tmp_path/'harness')
        extra = ['--harness-dir', str(snapshot)]
    monkeypatch.setattr(launch.sys, 'argv', ['launch', '--output', str(output),
        '--codex-config', str(config), '--evaluation-method', 'rsi_policy_doctor',
        '--checkpoint', str(checkpoint), '--intervention-experience', str(experience), '--iteration', '2', *extra])
    monkeypatch.setattr(launch, 'prepare_codex', lambda *a, **k: ({}, {'provider': 'test'}))
    monkeypatch.setattr(launch.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='test-codex'))
    monkeypatch.setattr(launch.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(launch, 'require_free_port', lambda port: None)
    monkeypatch.setattr(launch, 'student_ready', lambda port: True)
    calls, stopped = [], []
    def popen(command, **kwargs):
        process = SimpleNamespace(pid=100+len(calls), returncode=0, poll=lambda: None)
        calls.append(command)
        if command[0].endswith('run_server.sh'):
            kwargs['stdout'].write('{"event": "ready"}\n')
        return process
    monkeypatch.setattr(launch.subprocess, 'Popen', popen)
    monkeypatch.setattr(launch, 'wait_for_controller', lambda *a: None)
    monkeypatch.setattr(launch, 'stop_process', lambda p: stopped.append(p.pid) if p else None)
    launch.main()
    assert calls[0][calls[0].index('-m')+1] == 'robodoctor.prismwam_server'
    command = calls[-1]
    for flag, value in [('--student-backend', 'prismwam'), ('--iteration', '2'), ('--effort', 'medium'),
                        ('--action-chunk-length', '32'), ('--intervention-experience', str(experience))]:
        assert command[command.index(flag)+1] == value
    assert sorted(stopped) == [100, 101, 102]
    metadata = json.loads((output/'launch.json').read_text())
    assert metadata['evaluation_method'] == 'rsi_policy_doctor'
    assert metadata['codex_config'] == str(config)
    if with_harness:
        assert command[command.index('--harness-dir')+1] == str(snapshot)
        assert metadata['harness_dir'] == str(snapshot)
    else:
        assert '--harness-dir' not in command


def test_direct_launcher_rejects_rsi_harness_before_creating_output(tmp_path, monkeypatch):
    from robodoctor import launch
    output = tmp_path/'run'
    monkeypatch.setattr(launch.sys, 'argv', ['launch', '--output', str(output),
                                          '--harness-dir', str(tmp_path/'snapshot')])
    with pytest.raises(SystemExit):
        launch.main()
    assert not output.exists()


def test_direct_launch_forwards_task_experience_without_starting_a_student(tmp_path, monkeypatch):
    from robodoctor import launch
    from types import SimpleNamespace
    experience = tmp_path/'iter3.md'
    experience.write_text('Prior direct task lessons\n')
    output = tmp_path/'run'
    monkeypatch.setattr(launch.sys, 'argv', ['launch', '--output', str(output),
        '--task-experience', str(experience), '--iteration', '4'])
    monkeypatch.setattr(launch, 'prepare_codex', lambda *a, **k: ({}, {'provider': 'test'}))
    monkeypatch.setattr(launch.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='test-codex'))
    monkeypatch.setattr(launch.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(launch, 'require_free_port', lambda port: None)
    calls = []
    def popen(command, **kwargs):
        calls.append(command)
        if command[0].endswith('run_server.sh'):
            kwargs['stdout'].write('{"event": "ready"}\n')
        return SimpleNamespace(pid=len(calls), returncode=0, poll=lambda: None)
    monkeypatch.setattr(launch.subprocess, 'Popen', popen)
    monkeypatch.setattr(launch, 'wait_for_controller', lambda *a: None)
    monkeypatch.setattr(launch, 'stop_process', lambda *a: None)
    launch.main()
    assert len(calls) == 2
    command = calls[-1]
    assert command[command.index('--task-experience')+1] == str(experience)
    assert command[command.index('--iteration')+1] == '4'
    metadata = json.loads((output/'launch.json').read_text())
    assert metadata['task_experience'] == str(experience)
    assert metadata['task_experience_output'] == str(output/'controller/task_experience.md')


def test_invalid_environment_key_is_rejected_without_echoing_its_value(tmp_path):
    from robodoctor.codex_config import prepare_codex
    value = 'not-an-environment-variable'
    source = {'model_provider': 'test', 'model_providers': {
        'test': {'base_url': 'https://example.invalid/v1', 'env_key': value}}}
    with pytest.raises(ValueError, match='environment variable name') as error:
        prepare_codex(tmp_path / 'home', None, effort='medium', source=source, environ={})
    assert value not in str(error.value)
    assert not (tmp_path / 'home').exists()
