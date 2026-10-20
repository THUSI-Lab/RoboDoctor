"""Offline verification of a complete, independently controlled tower episode."""
import argparse
import json
from pathlib import Path

from hybrid_rollout.robodojo.io import write_json


def verify_tower_test(root, *, execution_error=None):
    root = Path(root).resolve()
    errors = []

    def read(relative, kind=dict):
        try:
            data = json.loads((root/relative).read_text())
            if not isinstance(data, kind):
                raise ValueError(f'expected {kind.__name__}')
            return data
        except (OSError, ValueError) as error:
            errors.append(dict(path=str(relative), error=str(error)))
            return kind()

    run = read('controller/run.json')
    worker = read('controller/codex_workspace/worker.json')
    result = read('controller/result.json')
    summary = read('sim/summary.json')
    native = read('sim/evaluation_outcome.json')
    history = read('controller/history.json', list)
    # Only successful recorded service calls count. Rejected input is evidence,
    # but must not be mistaken for actual grounding, arrival or manipulation.
    sequence, accepted = [], []
    grounded = approached = False
    for path in sorted((root/'controller/codex_workspace').glob('call_*_request.json')):
        request = read(path.relative_to(root))
        response = read(path.with_name(path.name.replace('_request', '_result')).relative_to(root))
        tool = request.get('tool')
        sequence.append(tool)
        if response.get('error') or response.get('error_type'):
            continue
        if tool == 'grounding' and response.get('ok') is True:
            accepted.append(tool)
            grounded = True
        elif tool == 'move_to_abs' and response.get('movement', {}).get('reached') is True:
            accepted.append(tool)
            approached = grounded
        elif tool == 'robodojo_act' and response.get('previous_result', {}).get('executed_steps', 0) > 0:
            if approached:
                accepted.append(tool)
            elif grounded and response.get('movement', {}).get('reached'):
                accepted.append('move_to_abs')  # Historical verifier name for a reached coarse move.
                approached = True
    ordered = iter(accepted)
    workflow = all(any(value == tool for value in ordered)
                   for tool in ('grounding', 'move_to_abs', 'robodojo_act'))
    rows = [row for row in history if isinstance(row, dict)]
    executed = sum(row.get('executed_steps', 0) for row in rows)
    video = root/'sim/sensors.mp4'
    checks = dict(
        task_is_build_tower=run.get('task') == 'build_tower',
        independent_codex=bool(worker.get('thread_id') and worker.get('pid')
                               and worker.get('codex_version') and worker.get('codex_executable')),
        full_task_mode=worker.get('smoke_approach') is False and worker.get('tower_test') is True,
        no_pi05=run.get('pi05_enabled') is False and result.get('pi05_inference_calls') == 0,
        grounded_approach_then_manipulation=workflow,
        physical_manipulation=any(row.get('source_detail') == 'eef' and row.get('executed_steps', 0) > 0
                                  for row in rows),
        same_episode=bool(result.get('episode_id')) and result.get('episode_id') == summary.get('episode_id'),
        step_accounting=bool(executed > 0 and executed == result.get('step_id')
                             == summary.get('step_id') == native.get('native_control_steps')),
        native_terminal=(result.get('complete') is True and summary.get('complete') is True
                         and native.get('complete') is True
                         and result.get('reason') == summary.get('reason') == native.get('reason') == 'terminal'
                         and bool(result.get('terminated') or result.get('truncated'))),
        valid_native_evaluation=native.get('valid_for_success_rate') is True,
        video_recorded=video.is_file() and video.stat().st_size > 0,
        artifact_files_readable=not errors,
        controller_completed=execution_error is None and not (root/'controller/failure.json').exists(),
        launcher_completed=not (root/'launch_failure.json').exists(),
    )
    success = (result.get('success') is True and summary.get('success') is True
               and native.get('native_success') is True and result.get('truncated') is False)
    verified = all(checks.values())
    passed = verified and success
    status = ('passed' if passed else 'native_failure' if verified else
              'incomplete' if not checks['native_terminal'] else 'invalid_evidence')
    report = dict(schema='robodoctor.tower_task_test.v1', task='build_tower', passed=passed,
                  status=status, execution_verified=verified, native_success=native.get('native_success'),
                  native_score=native.get('native_score'), checks=checks,
                  failed_checks=[name for name, value in checks.items() if not value],
                  execution_error=execution_error, artifact_errors=errors,
                  episode_id=result.get('episode_id'), physical_steps=result.get('step_id', 0),
                  decisions=len(history), tool_sequence=sequence, worker=worker,
                  native_result=result, native_evaluation=native,
                  video_path=str(video), debug_video_path=result.get('debug_video_path'),
                  exit_code=0 if passed else 2 if status == 'native_failure' else 1)
    write_json(root/'tower_test_result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    if not args.run_root.is_dir():
        parser.error('run-root must be an existing run directory')
    report = verify_tower_test(args.run_root)
    print(json.dumps(dict(status=report['status'], passed=report['passed'],
                          native_score=report['native_score'], failed_checks=report['failed_checks'])))
    raise SystemExit(report['exit_code'])


if __name__ == '__main__':
    main()
