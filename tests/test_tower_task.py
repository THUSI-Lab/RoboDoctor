import json
from pathlib import Path
import subprocess
import sys

import pytest

from robodoctor.tower_test import verify_tower_test
from test_grounded import started, move_args


def write(root, name, value):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def completed_run(root, *, success=True):
    write(root, 'controller/run.json', dict(task='build_tower', pi05_enabled=False))
    write(root, 'controller/codex_workspace/worker.json', dict(thread_id='independent-thread', pid=1234,
        codex_version='test-cli', codex_executable='/test/codex', smoke_approach=False, tower_test=True))
    terminal = dict(episode_id='episode', step_id=45, complete=True, success=success,
                    terminated=success, truncated=not success, reason='terminal')
    write(root, 'controller/result.json', dict(terminal, pi05_inference_calls=0))
    write(root, 'sim/summary.json', terminal)
    write(root, 'sim/evaluation_outcome.json', dict(complete=True, valid_for_success_rate=True,
        native_success=success, native_score=1.0 if success else .4, reason='terminal', native_control_steps=45))
    write(root, 'controller/history.json', [dict(source_detail='move_to_abs', executed_steps=40),
                                           dict(source_detail='eef', executed_steps=5)])
    tools = [('grounding', dict(ok=True)),
             ('move_to_abs', dict(movement=dict(reached=True))),
             ('robodojo_act', dict(previous_result=dict(executed_steps=5)))]
    for i, (tool, result) in enumerate(tools):
        write(root, f'controller/codex_workspace/call_{i:04d}_request.json', dict(tool=tool))
        write(root, f'controller/codex_workspace/call_{i:04d}_result.json', result)
    # Verifier checks artifact presence; video decoding is a separate GPU-run check.
    (root/'sim/sensors.mp4').write_bytes(b'fixture-video-artifact')


def test_full_task_requires_native_success_and_complete_workflow(tmp_path):
    completed_run(tmp_path)
    report = verify_tower_test(tmp_path)
    assert report['passed'] and report['exit_code'] == 0
    assert report['execution_verified'] and report['physical_steps'] == 45


def test_full_task_accepts_unified_dual_arm_coarse_and_fine_calls(tmp_path):
    completed_run(tmp_path)
    write(tmp_path, 'controller/history.json', [dict(source_detail='eef', executed_steps=40),
                                              dict(source_detail='eef', executed_steps=5)])
    write(tmp_path, 'controller/codex_workspace/call_0001_request.json', dict(tool='robodojo_act'))
    write(tmp_path, 'controller/codex_workspace/call_0001_result.json', dict(
        movement=dict(reached=True), previous_result=dict(executed_steps=40)))
    report = verify_tower_test(tmp_path)
    assert report['passed'] and report['tool_sequence'] == ['grounding','robodojo_act','robodojo_act']


def test_native_timeout_and_partial_score_are_failed_task_not_success(tmp_path):
    completed_run(tmp_path, success=False)
    report = verify_tower_test(tmp_path)
    assert report['status'] == 'native_failure' and report['exit_code'] == 2
    assert report['execution_verified'] and not report['passed']
    assert report['native_score'] == .4


@pytest.mark.parametrize('reason', ['decision_budget', 'smoke_approach_complete', 'controller_error'])
def test_early_exit_is_incomplete_even_when_controller_claims_success(tmp_path, reason):
    completed_run(tmp_path)
    result = json.loads((tmp_path/'controller/result.json').read_text())
    result.update(reason=reason, complete=False)
    write(tmp_path, 'controller/result.json', result)
    report = verify_tower_test(tmp_path)
    assert report['status'] == 'incomplete' and report['exit_code'] == 1


@pytest.mark.parametrize('change,check', [
    ('wrong_task', 'task_is_build_tower'), ('smoke', 'full_task_mode'),
    ('missing_thread', 'independent_codex'), ('bad_step_count', 'step_accounting'),
    ('different_episode', 'same_episode'), ('invalid_layout', 'valid_native_evaluation'),
    ('missing_video', 'video_recorded'), ('rejected_act', 'grounded_approach_then_manipulation'),
    ('no_physical_act', 'physical_manipulation'), ('launcher_error', 'launcher_completed'),
])
def test_missing_or_inconsistent_evidence_cannot_pass(tmp_path, change, check):
    completed_run(tmp_path)
    files = {
        'wrong_task': ('controller/run.json', {'task':'other'}),
        'smoke': ('controller/codex_workspace/worker.json', {'smoke_approach':True}),
        'missing_thread': ('controller/codex_workspace/worker.json', {'thread_id':None}),
        'bad_step_count': ('sim/summary.json', {'step_id':44}),
        'different_episode': ('sim/summary.json', {'episode_id':'other'}),
        'invalid_layout': ('sim/evaluation_outcome.json', {'valid_for_success_rate':False}),
        'rejected_act': ('controller/codex_workspace/call_0002_result.json', {'error_type':'recoverable_tool_input'}),
    }
    if change in files:
        name, updates = files[change]
        data = json.loads((tmp_path/name).read_text()); data.update(updates)
        write(tmp_path, name, data)
    elif change == 'missing_video':
        (tmp_path/'sim/sensors.mp4').unlink()
    elif change == 'no_physical_act':
        write(tmp_path, 'controller/history.json', [dict(source_detail='move_to_abs',executed_steps=45)])
    else:
        write(tmp_path, 'launch_failure.json', dict(error='wall-time limit'))
    report = verify_tower_test(tmp_path)
    assert not report['passed'] and check in report['failed_checks']


def test_missing_artifacts_still_produce_diagnostic_report(tmp_path):
    report = verify_tower_test(tmp_path, execution_error='Simulator startup failed')
    assert not report['passed'] and report['artifact_errors']
    assert json.loads((tmp_path/'tower_test_result.json').read_text())['exit_code'] == 1


def test_complete_mode_continues_after_approach_until_native_termination(tmp_path):
    client = started(tmp_path)
    arrival = client.act(**move_args(client))
    assert arrival['movement']['reached'] and not arrival['rollout_finished']
    assert client.phase == 'act'
    # Native termination occurs on the following actual bounded EEF action.
    client.sim.terminal = True
    args = client.next_call(); args.pop('tool')
    args['response'] = dict(request_id=client.request['request_id'], mode='eef', steps=1,
        reason='left: complete final manipulation of the visible tower, right: hold measured pose and preserve gripper state',
        target={arm:dict(position=client.request['current_eef'][arm]['position'],
                         quaternion_wxyz=[1,0,0,0], gripper_closed=True) for arm in ('left','right')})
    packet = client.act(**args)
    assert packet['rollout_finished'] and packet['result']['complete']
    assert packet['result']['reason'] == 'terminal' and client.tick == 2


def test_cli_rejects_smoke_and_full_tower_modes_before_launch():
    result = subprocess.run([sys.executable, '-m', 'robodoctor.launch', '--tower-test', '--smoke-approach'],
                            capture_output=True, text=True)
    assert result.returncode == 2 and 'not allowed' in result.stderr


def test_full_tower_arguments_without_starting_a_process(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / 'run_codex.sh'
    script.write_text((root / script.name).read_text())
    fake = tmp_path / 'python'
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    fake.chmod(0o755)
    (tmp_path / 'runtime_env.sh').write_text(f'export ROBODOJO_PYTHON="{fake}"\n')
    arguments = ['--task', 'build_tower', '--tower-test', '--max-decisions', '0',
                 '--run-timeout', '14400', '--gpu', '2']
    result = subprocess.run(['bash', str(script), *arguments],
                            capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == ['-m', 'robodoctor.launch', *arguments]
