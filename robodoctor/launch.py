"""Own one simulator, controller and independent Codex CLI with isolated config."""
from .demonstration import add_arguments, validate
from .task_experience import validate_task_experience
from .harness_snapshot import validate_harness
import argparse
import http.client
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time
import uuid

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from .codex_config import prepare_codex as _prepare_codex, read_config
from hybrid_rollout.robodojo.io import write_json


def require_free_port(port):
    """Reject active listeners while allowing reuse after an earlier rollout."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))


def student_ready(port):
    """Use OpenPI's HTTP health endpoint without opening an inference session."""
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
    try:
        connection.request('GET', '/healthz')
        response = connection.getresponse()
        return response.status == 200 and response.read(16).strip() == b'OK'
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def prepare_codex(home, *, config_path=None, source_config=None, auth_profile=None,
                  key_file=None, environ=None, effort="xhigh", model=None, api_gate_url=None):
    config_path = config_path or source_config
    source = None
    if auth_profile:
        from hybrid_rollout.robodojo.codex_backend.profiles import profile, config_text
        identity = profile(auth_profile)
        if identity['auth_mode'] != 'api':
            raise ValueError('Independent launcher requires an API auth profile')
        environment = os.environ if environ is None else environ
        if config_path and not key_file and not environment.get('OPENAI_API_KEY'):
            source = tomllib.loads(Path(config_path).read_text())
            provider = source.get('model_providers', {}).get(source.get('model_provider'), {})
            if source.get('model_provider') != identity['provider'] or provider.get('base_url') != identity['base_url']:
                raise ValueError('Source config provider/base_url must match the selected auth profile')
        else:
            source = tomllib.loads(config_text(auth_profile))
    environment, identity = _prepare_codex(home, config_path, effort=effort, model=model,
        key_file=key_file, environ=environ, source=source, api_gate_url=api_gate_url)
    if auth_profile:
        environment['ROLLOUT_AUTH_PROFILE'] = auth_profile
    return environment, identity


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=25)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def rollout_finalized(output):
    """finish_pilot ACK and matching simulator summary permit offline cleanup."""
    try:
        result = json.loads((output/'controller/result.json').read_text())
        summary = json.loads((output/'sim/summary.json').read_text())
    except (OSError, ValueError):
        return False
    return (isinstance(result, dict) and isinstance(summary, dict)
            and bool(result.get('episode_id'))
            and type(result.get('step_id')) is int and result['step_id'] >= 0
            and all(result.get(key) == summary.get(key)
                    for key in ('episode_id', 'step_id', 'reason', 'video_frames'))
            and result.get('video_frames') == result['step_id'] + 1)


def wait_for_controller(controller, sim, output, run_timeout, finalize_timeout=900):
    deadline = time.monotonic() + run_timeout
    finalize_deadline = None
    while controller.poll() is None:
        if finalize_deadline is None and rollout_finalized(output):
            finalize_deadline = time.monotonic() + finalize_timeout
            print('Physical rollout finalized; waiting for controller cleanup and debug video.', flush=True)
        sim_status = sim.poll()
        if sim_status is not None and (sim_status != 0 or finalize_deadline is None):
            raise RuntimeError('Simulator exited while controller was active')
        if finalize_deadline is not None:
            if time.monotonic() > finalize_deadline:
                raise TimeoutError('Controller cleanup/debug video timeout; recorded episode is preserved')
        elif time.monotonic() > deadline:
            raise TimeoutError('Codex rollout wall-time limit reached')
        time.sleep(1)


def verify_smoke(root):
    """Require proof that the independent model called both new tools."""
    worker = json.loads((root/'controller/codex_workspace/worker.json').read_text())
    calls = []
    for path in sorted((root/'controller/codex_workspace').glob('call_*_request.json')):
        request = json.loads(path.read_text())
        result_path = path.with_name(path.name.replace('_request', '_result'))
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        calls.append((request, result))
    clicks = [result for request, result in calls if request['tool'] == 'grounding' and result.get('ok')]
    movements = [result['movement'] for request, result in calls
                 if request['tool'] in ('robodojo_act', 'move_to_abs') and result.get('movement', {}).get('reached')]
    result = json.loads((root/'controller/result.json').read_text())
    passed = bool(clicks and movements and result.get('reason') == 'smoke_approach_complete'
                  and worker.get('thread_id') and worker.get('pid'))
    report = dict(passed=passed, scope='Independent Codex CLI selects the pixel and absolute pose in real RoboDojo',
                  worker=worker, tool_sequence=[request['tool'] for request, _ in calls],
                  selected_points=clicks, movement=movements[-1] if movements else None, native_result=result)
    if passed:
        from .smoke import annotate_video
        report['video'] = annotate_video(root, clicks, movements[-1])
    write_json(root/'smoke_result.json', report)
    if not passed:
        raise RuntimeError('Codex smoke did not reach a grounded pre-operation pose; see smoke_result.json')
    return report


def build_parser(*, add_help=True):
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, add_help=add_help)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--task', default='build_tower')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-seed', type=int, default=0)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--sim-port', type=int, default=19253)
    parser.add_argument('--evaluation-method', choices=('gpt_only', 'pi05_plus_gpt', 'rsi_policy_doctor'), default='gpt_only')
    parser.add_argument('--intervention-experience', type=Path,
                        default=Path(os.environ['INTERVENTION_EXPERIENCE']) if os.environ.get('INTERVENTION_EXPERIENCE') else None)
    parser.add_argument('--task-experience', type=Path, help='Prior task experience Markdown for direct control')
    parser.add_argument('--harness-dir', type=Path, help='Validated run-local RSI prompt snapshot; RSI only')
    parser.add_argument('--iteration', type=int, default=int(os.environ.get('RSI_ITERATION', '0')))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--student-backend', choices=('auto', 'prismwam', 'pi05'), default='auto',
                        help='Hybrid policy backend. auto selects PrismWAM for a .pt checkpoint and legacy Pi05 for a directory.')
    parser.add_argument('--student-model', choices=('prismwam5b',),
                        default=os.environ.get('PRISMWAM_MODEL', 'prismwam5b'))
    parser.add_argument('--student-port', type=int, default=18830)
    parser.add_argument('--student-python', help='Python used for the owned PrismWAM/Pi05 policy server')
    parser.add_argument('--student-gpu', default='1')
    parser.add_argument('--codex', default=os.environ.get('CODEX_BIN', 'codex'))
    parser.add_argument('--codex-config', '--source-config', dest='source_config', type=Path,
                        default=Path(os.environ.get('CODEX_CONFIG', str(root/'codex.toml'))))
    parser.add_argument('--auth-profile', default=os.environ.get('ROLLOUT_AUTH_PROFILE'))
    parser.add_argument('--api-gate-url', default=os.environ.get('ROLLOUT_API_GATE_URL'))
    parser.add_argument('--key-file', default=os.environ.get('ROLLOUT_OPENAI_API_KEY_FILE'))
    parser.add_argument('--max-decisions', type=int, default=100)
    parser.add_argument('--effort', choices=('low', 'medium', 'high', 'xhigh', 'max'),
                        default=None, help='Override model_reasoning_effort from codex.toml')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke-approach', action='store_true')
    mode.add_argument('--tower-test', action='store_true', help='Verify complete native tower task success')
    parser.add_argument('--startup-timeout', type=float, default=900)
    parser.add_argument('--run-timeout', type=float, default=1800)
    add_arguments(parser)
    return parser


def main():
    root = Path(__file__).resolve().parent.parent
    parser = build_parser()
    args = parser.parse_args()
    args.codex = shutil.which(args.codex) or args.codex
    try:
        validate(args.control_mode, args.task_details, args.demo_video, args.demo_frames, args.evaluation_method)
        validate_task_experience(args.task_experience)
        if args.harness_dir:
            validate_harness(args.harness_dir, args.evaluation_method)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    hybrid = args.evaluation_method != 'gpt_only'
    if hybrid and args.checkpoint is None and os.environ.get('PRISMWAM_CHECKPOINT'):
        args.checkpoint = Path(os.environ['PRISMWAM_CHECKPOINT'])
    source = read_config(args.source_config)
    model = source.get('model', 'gpt-6-astra')
    if not isinstance(model, str) or not model.strip():
        parser.error('Source config model must be a non-empty string')
    args.effort = args.effort or source.get('model_reasoning_effort', 'medium')
    if args.effort not in ('low', 'medium', 'high', 'xhigh', 'max'):
        parser.error('Source config model_reasoning_effort must be low, medium, high, xhigh or max')
    backend = args.student_backend
    if backend == 'auto' and args.checkpoint is not None:
        backend = 'prismwam' if args.checkpoint.suffix == '.pt' else 'pi05'
    if hybrid and args.checkpoint is None:
        parser.error('Hybrid requires --checkpoint pointing to a PrismWAM .pt file or legacy Pi05 checkpoint directory')
    if hybrid and args.checkpoint is not None:
        if args.evaluation_method == 'pi05_plus_gpt' and backend != 'pi05':
            parser.error('pi05_plus_gpt requires the pi05 backend')
        if backend == 'prismwam' and not args.checkpoint.is_file():
            parser.error('PrismWAM hybrid requires --checkpoint pointing to an existing .pt file')
        if backend == 'pi05' and not args.checkpoint.is_dir():
            parser.error('Legacy Pi05 hybrid requires --checkpoint pointing to a checkpoint directory')
    if hybrid and (args.smoke_approach or args.tower_test):
        parser.error('Hybrid/RSI uses native task termination, not the direct-only verifier')
    if not hybrid and (args.checkpoint is not None or args.student_python or args.intervention_experience):
        parser.error('Policy options require hybrid/RSI control')
    if args.evaluation_method != 'rsi_policy_doctor' and args.intervention_experience:
        parser.error('--intervention-experience requires rsi_policy_doctor')
    if args.evaluation_method != 'gpt_only' and args.task_experience:
        parser.error('--task-experience requires direct control (gpt_only)')
    if hybrid and args.student_port == args.sim_port:
        parser.error('Student and simulator ports must differ')
    if args.max_decisions < 0 or min(args.startup_timeout, args.run_timeout) <= 0:
        parser.error('Budgets/timeouts must be valid positive values (max-decisions=0 is unlimited)')
    if (args.smoke_approach or args.tower_test) and args.task != 'build_tower':
        parser.error('Tower tests require --task build_tower')
    run_id = datetime.now(timezone.utc).strftime('codex_%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:6]
    output = (args.output or root/'results'/run_id).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'logs').mkdir()
    (output/'sim/native_runtime').mkdir(parents=True)
    home = root/'runtime/codex_homes'/run_id
    environment, identity = prepare_codex(home, config_path=args.source_config, key_file=args.key_file,
                                         effort=args.effort, model=model, auth_profile=args.auth_profile,
                                         api_gate_url=args.api_gate_url)
    require_free_port(args.sim_port)
    version = subprocess.run([args.codex, '--version'], env=environment, capture_output=True,
                             text=True, check=True).stdout.strip()
    write_json(output/'launch.json', dict(task=args.task, seed=args.seed, eval_seed=args.eval_seed,
        control_mode=args.control_mode, task_details=str(args.task_details.resolve()) if args.task_details else None,
        demo_video=str(args.demo_video.resolve()) if args.demo_video else None, demo_frames=args.demo_frames,
        evaluation_method=args.evaluation_method, checkpoint=str(args.checkpoint.resolve()) if hybrid else None,
        student_backend=backend if hybrid else None, student_model=args.student_model if hybrid else None,
        student_port=args.student_port if hybrid else None,
        student_python=(args.student_python or (
            os.environ.get('PRISMWAM_PYTHON') or sys.executable
            if backend == 'prismwam' else None)) if hybrid else None,
        student_gpu=args.student_gpu if hybrid and (backend == 'prismwam' or args.student_python) else None,
        gpu=args.gpu, codex_executable=str(Path(args.codex).resolve()), codex_version=version,
        model=model, reasoning_effort=args.effort, provider=identity['provider'],
        codex_home=str(home), smoke_approach=args.smoke_approach, max_decisions=args.max_decisions,
        tower_test=args.tower_test,
        codex_config=str(args.source_config), source_config=str(args.source_config), sim_port=args.sim_port,
        intervention_experience=str(args.intervention_experience.resolve()) if args.intervention_experience else None,
        task_experience=str(args.task_experience.resolve()) if args.task_experience else None,
        task_experience_output=str(output/'controller/task_experience.md') if not hybrid else None,
        iteration=args.iteration,
        harness_dir=str(args.harness_dir.resolve()) if args.harness_dir else None,
        api_gate_url=args.api_gate_url))
    print(f'Independent {version}: {model} / {args.effort} / {identity["provider"]}\nArtifacts: {output}', flush=True)
    sim = controller = student = None
    status = 1
    execution_error = None
    def terminate(signum, frame):
        raise KeyboardInterrupt('Stopping owned rollout processes')
    signal.signal(signal.SIGTERM, terminate)
    try:
        owned_student = hybrid and (backend == 'prismwam' or args.student_python)
        if owned_student:
            require_free_port(args.student_port)
            if backend == 'prismwam' and not args.student_python:
                args.student_python = os.environ.get('PRISMWAM_PYTHON') or sys.executable
            policy_env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.student_gpu,
                              XLA_PYTHON_CLIENT_PREALLOCATE='false')
            with (output/'logs/student.log').open('w') as log:
                if backend == 'prismwam':
                    policy_env['PYTHONPATH'] = ':'.join([str(root), str(root/'third_party/prismwam/src'),
                        policy_env.get('PYTHONPATH', '')])
                    student_command = [args.student_python, '-m', 'robodoctor.prismwam_server',
                        '--variant', args.student_model, '--checkpoint', str(args.checkpoint.resolve()),
                        '--gpu', args.student_gpu, '--device', f'cuda:0', '--port', str(args.student_port),
                        '--identity-output', str(output/'student_identity.json')]
                else:
                    if os.environ.get('OPENPI_SOURCE'):
                        policy_source = Path(os.environ['OPENPI_SOURCE']).expanduser().resolve()
                        policy_env['PYTHONPATH'] = ':'.join([str(policy_source/'src'),
                            str(policy_source/'packages/openpi-client/src'), policy_env.get('PYTHONPATH', '')])
                    student_command = [args.student_python, '-m',
                        'hybrid_rollout.robodojo.pi05_server.server', '--checkpoint', str(args.checkpoint.resolve()),
                        '--port', str(args.student_port), '--identity-output', str(output/'student_identity.json')]
                student = subprocess.Popen(student_command, cwd=root, env=policy_env,
                                           stdout=log, stderr=subprocess.STDOUT)
            write_json(output/'processes.json', dict(student_pid=student.pid))
            deadline = time.monotonic()+args.startup_timeout
            while True:
                if student.poll() is not None:
                    raise RuntimeError('student policy server exited during startup; see logs/student.log')
                if student_ready(args.student_port):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('student policy startup timeout')
                time.sleep(1)
        sim_env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
        with (output/'logs/sim.log').open('w') as log:
            sim = subprocess.Popen([str(root/'run_server.sh'), '--task', args.task,
                '--output', str(output/'sim'), '--port', str(args.sim_port), '--eval-seed', str(args.eval_seed)],
                cwd=output/'sim/native_runtime', env=sim_env, stdout=log, stderr=subprocess.STDOUT)
        write_json(output/'processes.json', dict(sim_pid=sim.pid, student_pid=student.pid if student else None))
        deadline = time.monotonic()+args.startup_timeout
        while '"event": "ready"' not in (output/'logs/sim.log').read_text(errors='replace'):
            if sim.poll() is not None:
                raise RuntimeError(f'Simulator exited during startup (exit code {sim.returncode}); see logs/sim.log')
            if time.monotonic() > deadline:
                raise TimeoutError('Simulator startup timeout')
            time.sleep(1)
        command = [sys.executable, '-m', 'robodoctor.runner', '--output', str(output/'controller'),
                   '--task', args.task, '--codex', args.codex, '--sim-port', str(args.sim_port),
                   '--seed', str(args.seed), '--max-decisions', str(args.max_decisions), '--effort', args.effort,
                   '--model', model,
                   '--evaluation-method', args.evaluation_method, '--iteration', str(args.iteration)]
        if args.harness_dir:
            command += ['--harness-dir', str(args.harness_dir.resolve())]
        if hybrid:
            if args.intervention_experience:
                command += ['--intervention-experience', str(args.intervention_experience.resolve())]
            command += ['--checkpoint', str(args.checkpoint.resolve()), '--student-port', str(args.student_port),
                        '--student-backend', backend]
            if source.get('action_chunk_length') is not None:
                command += ['--action-chunk-length', str(int(source['action_chunk_length']))]
        command += ['--control-mode', args.control_mode, '--demo-frames', str(args.demo_frames)]
        if args.task_experience:
            command += ['--task-experience', str(args.task_experience.resolve())]
        if args.task_details:
            command += ['--task-details', str(args.task_details.resolve())]
        if args.demo_video:
            command += ['--demo-video', str(args.demo_video.resolve())]
        if args.smoke_approach:
            command.append('--smoke-approach')
        if args.tower_test:
            command.append('--tower-test')
        with (output/'logs/controller.log').open('w') as log:
            controller = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        write_json(output/'processes.json', dict(sim_pid=sim.pid, controller_pid=controller.pid,
                                               student_pid=student.pid if student else None))
        wait_for_controller(controller, sim, output, args.run_timeout)
        if controller.returncode:
            raise RuntimeError(f'Controller exited {controller.returncode}; see logs/controller.log')
        if args.smoke_approach:
            report = verify_smoke(output)
            print(f'Smoke passed: {report["movement"]}\nVideo: {report["video"]["path"]}', flush=True)
        status = 0
    except BaseException as error:
        execution_error = f'{type(error).__name__}: {error}'
        write_json(output/'launch_failure.json', dict(error=execution_error))
        raise
    finally:
        stop_process(controller)
        stop_process(sim)
        stop_process(student)
        if execution_error and (output/'controller/run.json').is_file():
            try:
                from .debug_video import recover_sensors, RecordedDebugVideo
                recover_sensors(output/'controller')
                RecordedDebugVideo(output/'controller', retry_incomplete=True).finalize()
            except Exception as error:
                print(f'Offline video recovery failed: {error}', flush=True)
        if args.tower_test:
            from .tower_test import verify_tower_test
            report = verify_tower_test(output, execution_error=execution_error)
            status = report['exit_code']
            print(f'Tower test: {report["status"]}; native success={report["native_success"]}; '
                  f'report: {output}/tower_test_result.json', flush=True)
        (output/'exit_status.txt').write_text(str(status)+'\n')
    if status:
        raise SystemExit(status)


if __name__ == '__main__':
    main()
