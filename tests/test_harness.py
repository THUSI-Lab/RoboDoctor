import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robodoctor.harness import evolve_harness
from robodoctor.harness_snapshot import PROMPT_FILES, prompt_hashes, validate_harness
from robodoctor.pipeline import build_parser
from robodoctor import runner
from test_recovery import Transport as PolicyTransport


class Editor:
    def __init__(self, argv, workspace, *, env):
        self.workspace = workspace
        self.env = env
        self.argv = argv
        self.closed = False
        self.mode = 'success'
        self.requests = []

    def request(self, method, params, timeout):
        assert 0 < timeout <= 900
        self.requests.append((method, params))
        if method == 'thread/start':
            assert params['dynamicTools'] == []
            assert 'direct-control prompts' in params['developerInstructions']
            return dict(model=params['model'], reasoningEffort=params['config']['model_reasoning_effort'],
                        thread=dict(id='editor-thread'))
        if method == 'turn/start':
            assert (self.workspace/'iter2_experience.md').read_text() == 'Iter2: stale proposal after recovery.'
            for name in PROMPT_FILES:
                if self.mode == 'unchanged' and name == 'SKILL.md':
                    continue
                path = self.workspace/'agent'/name
                path.write_text(path.read_text() + '\nVerify fresh inference after recovery (iter2 evidence).\n')
            if self.mode == 'empty':
                (self.workspace/'agent/SKILL.md').write_text('')
            if self.mode == 'extra':
                (self.workspace/'agent/unexpected.txt').write_text('outside edit scope')
            if self.mode == 'evidence':
                (self.workspace/'iter2_experience.md').write_text('tampered')
            if self.mode == 'symlink':
                path = self.workspace/'agent/SKILL.md'
                path.unlink()
                path.symlink_to(self.workspace/'iter2_experience.md')
            return dict(turn=dict(id='editor-turn'))
        return {}

    def notify(self, *args):
        pass

    def next_message(self, timeout):
        if self.mode == 'timeout':
            raise TimeoutError('test timeout')
        if self.mode == 'interrupt':
            raise KeyboardInterrupt('test interrupt')
        return dict(method='turn/completed', params=dict(threadId='editor-thread',
            turn=dict(id='editor-turn', status='failed' if self.mode == 'failed' else 'completed')))

    def close(self):
        self.closed = True


@pytest.fixture
def setup_editor(tmp_path, monkeypatch):
    for name in ('ROLLOUT_AUTH_PROFILE', 'ROLLOUT_API_GATE_URL', 'ROLLOUT_OPENAI_API_KEY_FILE'):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path/'config.toml'
    config.write_text('model = "gpt-6-astra"\nmodel_provider = "test"\n'
                      'model_reasoning_effort = "medium"\n'
                      '[model_providers.test]\nbase_url = "https://example.invalid/v1"\n'
                      'env_key = "TEST_MODEL_KEY"\n')
    args = build_parser().parse_args(['--codex-config', str(config)])
    evidence = tmp_path/'experience.md'
    evidence.write_text('Iter2: stale proposal after recovery.')
    editors = []
    def run(mode='success'):
        def factory(*a, **kw):
            editor = Editor(*a, **kw)
            editor.mode = mode
            editors.append(editor)
            return editor
        return evolve_harness(args, experience=evidence, output=tmp_path/'harness',
            transport_factory=factory, environ=dict(TEST_MODEL_KEY='test-only-credential'))
    return run, editors


def test_editor_publishes_two_rsi_prompts_in_isolated_home(setup_editor, tmp_path):
    run, editors = setup_editor
    source = Path(runner.__file__).parent
    original = {name: prompt_hashes(source/name) for name in ('prompts', 'rsi_prompts')}
    snapshot = run()
    editor = editors[0]
    assert editor.closed
    assert editor.env['CODEX_HOME'] == str(tmp_path/'harness/codex_home')
    assert editor.env['TEST_MODEL_KEY'] == 'test-only-credential'
    assert validate_harness(snapshot) == prompt_hashes(snapshot)
    state = json.loads((tmp_path/'harness/evolution.json').read_text())
    assert state['status'] == 'completed'
    assert all(state['baseline'][name] != state['files'][name] for name in PROMPT_FILES)
    for name in ('prompts', 'rsi_prompts'):
        assert prompt_hashes(source/name) == original[name]
    for path in (tmp_path/'harness').rglob('*'):
        if path.is_file():
            assert 'test-only-credential' not in path.read_text()


@pytest.mark.parametrize('mode,error', [('unchanged', ValueError), ('empty', ValueError), ('extra', ValueError),
    ('evidence', ValueError), ('symlink', ValueError), ('failed', RuntimeError),
    ('timeout', TimeoutError), ('interrupt', KeyboardInterrupt)])
def test_failed_edit_never_publishes_and_always_closes(setup_editor, tmp_path, mode, error):
    run, editors = setup_editor
    with pytest.raises(error):
        run(mode)
    assert editors[0].closed
    assert not (tmp_path/'harness/prompts').exists()
    state = json.loads((tmp_path/'harness/evolution.json').read_text())
    assert state['status'] == ('interrupted' if mode == 'interrupt' else 'failed')


def test_rsi_injects_edited_pair_and_direct_retains_its_own_prompts(setup_editor, tmp_path, monkeypatch):
    run, _ = setup_editor
    snapshot = run()
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='test-codex'))
    rsi = runner.CodexPolicy(tmp_path/'rsi', '/fake/codex', method='rsi_policy_doctor',
                            harness_dir=snapshot, transport_factory=PolicyTransport)
    direct = runner.CodexPolicy(tmp_path/'direct', '/fake/codex', transport_factory=PolicyTransport)
    try:
        for name in PROMPT_FILES:
            text = (snapshot/name).read_text()
            assert text in rsi.prompt
            assert (rsi.agent_workspace/name).read_text() == text
        direct_root = Path(runner.__file__).parent/'prompts'
        assert (direct_root/'SKILL.md').read_text() in direct.prompt
        assert (direct_root/'context/teacher_context.md').read_text() in direct.prompt
        assert 'Verify fresh inference after recovery (iter2 evidence).' not in direct.prompt
        assert json.loads((rsi.workspace/'harness.json').read_text())['files'] == prompt_hashes(snapshot)
    finally:
        rsi.close()
        direct.close()
    with pytest.raises(ValueError, match='direct uses its own prompts'):
        runner.CodexPolicy(tmp_path/'invalid', '/fake/codex', harness_dir=snapshot)
    assert not (tmp_path/'invalid').exists()
    (snapshot/'SKILL.md').write_text('changed after publication')
    with pytest.raises(ValueError, match='changed'):
        runner.CodexPolicy(tmp_path/'tampered', '/fake/codex', method='rsi_policy_doctor',
                          harness_dir=snapshot)
    assert not (tmp_path/'tampered').exists()
