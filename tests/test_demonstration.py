import json
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from robodoctor.demonstration import prepare, validate
from robodoctor import runner
from test_recovery import Transport


@pytest.fixture
def video(tmp_path):
    path = tmp_path/'demo.mp4'
    with imageio.get_writer(str(path), fps=4, macro_block_size=16) as writer:
        for i in range(9):
            writer.append_data(np.full((32, 48, 3), i*25, dtype=np.uint8))
    return path


def test_video_samples_include_endpoints_and_are_auditable(tmp_path, video):
    manifest, prompt = prepare(tmp_path/'example', 'visual-1-shot', video=video, frames=3)
    assert [f['frame'] for f in manifest['frames']] == [0, 4, 8]
    assert [f['seconds'] for f in manifest['frames']] == [0, 1, 2]
    assert manifest['duration_seconds'] == 2.25
    assert (tmp_path/'example/demonstration.mp4').read_bytes() == video.read_bytes()
    assert 'NOT current' in prompt
    assert len(manifest['sha256']) == 64
    assert json.loads((tmp_path/'example/manifest.json').read_text()) == manifest


@pytest.mark.parametrize('mode,details,video', [
    ('0-shot', 'example.md', None), ('textual-1-shot', None, None),
    ('visual-1-shot', None, None), ('textual-1-shot', 'example.md', 'example.mp4'),
])
def test_invalid_modes_fail_without_launching(mode, details, video):
    with pytest.raises(ValueError):
        validate(mode, details, video)


def test_bad_inputs(tmp_path):
    text = tmp_path/'example.md'
    text.write_text(' ')
    with pytest.raises(ValueError):
        validate('textual-1-shot', text)
    text.write_text('抓取红色物体')
    with pytest.raises(ValueError, match='direct control'):
        validate('textual-1-shot', text, method='pi05_plus_gpt')
    with pytest.raises(ValueError, match='between'):
        validate('0-shot', frames=100)
    corrupt = tmp_path/'broken.mp4'
    corrupt.write_bytes(b'invalid video')
    with pytest.raises((OSError, ValueError)):
        prepare(tmp_path/'bad', 'visual-1-shot', video=corrupt)


@pytest.mark.parametrize('mode', ['0-shot', 'textual-1-shot', 'visual-1-shot'])
def test_policy_receives_example_and_attaches_video_only_once(tmp_path, monkeypatch, video, mode):
    details = tmp_path/'task.md'
    details.write_text('先放红色物体，再放蓝色物体。')
    real_run = runner.subprocess.run
    monkeypatch.setattr(runner.subprocess, 'run', lambda command, **kwargs:
        SimpleNamespace(stdout='test-codex') if command[0] == '/fake/codex' else real_run(command, **kwargs))
    worker = runner.CodexPolicy(tmp_path/'controller/codex_workspace', '/fake/codex',
        transport_factory=Transport, control_mode=mode,
        task_details=details if mode == 'textual-1-shot' else None,
        demo_video=video if mode == 'visual-1-shot' else None, demo_frames=3)
    assert f'DIRECT CONTROL MODE: {mode}' in worker.prompt
    if mode == 'textual-1-shot':
        assert details.read_text() in worker.thread_params['developerInstructions']
    worker._turn('Begin')
    worker._turn('Continue')
    turns = [p for m, p in worker.transport.requests if m == 'turn/start']
    assert len([i for i in turns[0]['input'] if i['type'] == 'localImage']) == (3 if mode == 'visual-1-shot' else 0)
    assert len(turns[1]['input']) == 1
    assert json.loads((worker.workspace/'worker.json').read_text())['control_mode'] == mode


@pytest.mark.parametrize('mode', ['textual-1-shot', 'visual-1-shot'])
def test_launcher_forwards_example_arguments(tmp_path, monkeypatch, video, mode):
    from robodoctor import launch
    details = tmp_path/'task.md'
    details.write_text('Put red before blue.')
    flag, source = ('--task-details', details) if mode == 'textual-1-shot' else ('--demo-video', video)
    monkeypatch.setattr(launch.sys, 'argv', ['launch', '--output', str(tmp_path/'run'),
        '--control-mode', mode, flag, str(source), '--demo-frames', '3'])
    monkeypatch.setattr(launch, 'prepare_codex', lambda *a, **k: ({}, {'provider': 'test'}))
    monkeypatch.setattr(launch.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='fake-codex'))
    monkeypatch.setattr(launch.signal, 'signal', lambda *a: None)
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, *args): pass
        def setsockopt(self, *args): pass
    monkeypatch.setattr(launch.socket, 'socket', lambda *a, **k: Socket())
    calls = []
    def popen(command, **kwargs):
        calls.append(command)
        if command[0].endswith('run_server.sh'):
            kwargs['stdout'].write('{"event": "ready"}\n')
        return SimpleNamespace(pid=123, returncode=0, poll=lambda: None)
    monkeypatch.setattr(launch.subprocess, 'Popen', popen)
    monkeypatch.setattr(launch, 'wait_for_controller', lambda *a: None)
    monkeypatch.setattr(launch, 'stop_process', lambda *a: None)
    launch.main()
    command = calls[-1]
    assert command[command.index('--control-mode')+1] == mode
    assert command[command.index(flag)+1] == str(source.resolve())
    assert command[command.index('--demo-frames')+1] == '3'
