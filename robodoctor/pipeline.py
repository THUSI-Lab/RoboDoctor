"""Sequential episodes with three RSI gate trials and a configurable control tail."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import uuid

from .launch import build_parser as launch_parser, stop_process
from .harness import evolve_harness
from .harness_snapshot import validate_harness


def threshold_value(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise argparse.ArgumentTypeError('threshold must be finite and between 0 and 100 (percent)')
    return value


def read_outcome(output):
    path = Path(output)/'sim/evaluation_outcome.json'
    try:
        outcome = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'Missing or unreadable native evaluation: {path}') from exc
    if not isinstance(outcome, dict):
        raise ValueError(f'Invalid evaluation object: {path}')
    score = outcome.get('native_score')
    if (outcome.get('complete') is not True or outcome.get('valid_for_success_rate') is not True
            or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError(f'Incomplete/invalid native evaluation cannot determine the gate: {path}')
    return outcome


def mean_score(scores):
    if len(scores) != 3 or any(type(s) not in (int, float) or not math.isfinite(s)
                               or not 0 <= s <= 1 for s in scores):
        raise ValueError('Gate requires three valid native scores for iter0–2')
    return math.fsum(scores)/3


def tail_method(scores, threshold=30):
    threshold = threshold_value(threshold)
    return 'gpt_only' if mean_score(scores)*100 < threshold else 'rsi_policy_doctor'


def run_episode(command, log_path):
    """Let the launcher terminate its own simulator, policy and Codex on interruption."""
    with Path(log_path).open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            return process.wait()
        except BaseException:
            # A launcher can need separate time to stop each of its owned children.
            process.terminate()
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                stop_process(process)
            raise


class Pipeline:
    def __init__(self, args, *, episode_runner=run_episode, harness_runner=evolve_harness):
        self.args = args
        self.output = Path(args.output).resolve()
        self.episode_runner = episode_runner
        self.harness_runner = harness_runner
        self.total_iterations = 3 + args.tail_iterations
        self.state = dict(task=args.task, seed=None if args.iteration_seeds else args.seed,
                          iteration_seeds=[self.iteration_seed(i) for i in range(3 if args.direct_tail else 0, self.total_iterations)],
                          eval_seed=args.eval_seed,
                          start_iteration=3 if args.direct_tail else 0,
                          gate_iterations=3, tail_iterations=args.tail_iterations,
                          threshold=args.threshold, score_statistic='mean', score_unit='percent', iterations=[],
                          gate=None, status='pending', harness=dict(status='pending'))
        if args.direct_tail:
            self.state['harness'] = dict(status='skipped', reason='direct_control')
            self.state['gate'] = dict(source='user_confirmed', threshold=args.threshold,
                                     tail_method='gpt_only', direct_fallback=True)

    def iteration_seed(self, iteration):
        if self.args.iteration_seeds:
            return self.args.iteration_seeds[iteration-3]
        return self.args.seed if self.args.seed is not None else iteration % 3

    def command(self, iteration, method, experience_input=None):
        args = self.args
        command = [sys.executable, '-m', 'robodoctor.launch', '--task', args.task,
                   '--seed', str(self.iteration_seed(iteration)), '--eval-seed', str(args.eval_seed),
                   '--output', str(self.output/f'iter{iteration}'),
                   '--evaluation-method', method, '--control-mode', '0-shot',
                   '--iteration', str(iteration), '--codex-config', str(args.source_config.resolve()),
                   '--codex', args.codex, '--gpu', args.gpu, '--sim-port', str(args.sim_port),
                   '--max-decisions', str(args.max_decisions),
                   '--startup-timeout', str(args.startup_timeout), '--run-timeout', str(args.run_timeout)]
        for flag, value in [('--effort', args.effort), ('--key-file', args.key_file),
                            ('--auth-profile', args.auth_profile), ('--api-gate-url', args.api_gate_url)]:
            if value is not None:
                command.extend([flag, str(value)])
        if method == 'rsi_policy_doctor':
            if iteration >= 3:
                command.extend(['--harness-dir', str(self.output/'harness/prompts')])
            command.extend(['--checkpoint', str(args.checkpoint.resolve()),
                            '--student-backend', args.student_backend, '--student-model', args.student_model,
                            '--student-port', str(args.student_port), '--student-gpu', args.student_gpu])
            if args.student_python:
                command.extend(['--student-python', args.student_python])
            if experience_input:
                command.extend(['--intervention-experience', str(experience_input)])
        elif experience_input:
            command.extend(['--task-experience', str(experience_input)])
        return command

    def save(self):
        path = self.output/'pipeline.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.state, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        temporary.replace(path)

    def run(self):
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output/'logs').mkdir()
        (self.output/'experience').mkdir()
        self.state['status'] = 'running'
        self.save()
        previous_experience = None
        method = 'gpt_only' if self.args.direct_tail else 'rsi_policy_doctor'
        try:
            for iteration in range(3 if self.args.direct_tail else 0, self.total_iterations):
                experience_input = None
                if previous_experience:
                    # Each iteration gets a writable copy; earlier snapshots remain immutable.
                    experience_input = self.output/'experience'/f'input_iter{iteration}.md'
                    shutil.copyfile(previous_experience, experience_input)
                command = self.command(iteration, method, experience_input)
                record = dict(iteration=iteration, seed=self.iteration_seed(iteration),
                              evaluation_method=method, status='running',
                              command=command, experience_input=str(experience_input) if experience_input else None)
                if iteration >= 3 and method == 'rsi_policy_doctor':
                    hashes = validate_harness(self.output/'harness/prompts')
                    if hashes != self.state['harness']['files']:
                        raise ValueError('Shared RSI Harness changed between tail iterations')
                    record['harness_files'] = hashes
                self.state['iterations'].append(record)
                self.save()
                print(f'iter{iteration}: {method}; artifacts: {self.output/f"iter{iteration}"}', flush=True)
                code = self.episode_runner(command, self.output/'logs'/f'iter{iteration}.log')
                record['returncode'] = code
                if code:
                    raise RuntimeError(f'iter{iteration} launcher exited with {code}; see logs/iter{iteration}.log')
                outcome = read_outcome(self.output/f'iter{iteration}')
                record.update(native_score=outcome['native_score'], native_success=outcome.get('native_success'),
                              evaluation_outcome=str(self.output/f'iter{iteration}/sim/evaluation_outcome.json'))
                name = 'intervention_experience.md' if method == 'rsi_policy_doctor' else 'task_experience.md'
                experience = self.output/f'iter{iteration}/controller'/name
                if not experience.is_file() or not experience.read_text(encoding='utf-8').strip():
                    raise ValueError(f'iter{iteration} has no final experience: {experience}')
                previous_experience = self.output/'experience'/f'iter{iteration}.md'
                shutil.copyfile(experience, previous_experience)
                record['experience_output'] = str(previous_experience)
                record['experience_type'] = 'intervention' if method == 'rsi_policy_doctor' else 'task'
                record['status'] = 'completed'
                if iteration == 2:
                    scores = [row['native_score'] for row in self.state['iterations']]
                    method = tail_method(scores, self.args.threshold)
                    if method == 'gpt_only':
                        # RSI experience stays archived; the first direct episode starts fresh.
                        previous_experience = None
                    average = mean_score(scores)
                    self.state['gate'] = dict(scores=scores, score=average*100, native_score=average,
                                              statistic='mean', score_unit='percent',
                                              threshold=self.args.threshold, tail_method=method,
                                              direct_fallback=method == 'gpt_only')
                    print(f'Gate: mean(iter0–2)={average*100:g}%, threshold={self.args.threshold:g}%; '
                          f'iter3–{self.total_iterations-1}: {method}', flush=True)
                    if method == 'gpt_only':
                        self.state['harness'] = dict(status='skipped', reason='direct_control')
                    else:
                        self.state['harness'] = dict(status='running',
                            experience=str(previous_experience), output=str(self.output/'harness'))
                        self.save()
                        print('Harness: improving RSI prompts from iter2 experience', flush=True)
                        try:
                            self.harness_runner(self.args, experience=previous_experience,
                                                output=self.output/'harness')
                            hashes = validate_harness(self.output/'harness/prompts')
                            self.state['harness'].update(status='completed', files=hashes,
                                prompts=str(self.output/'harness/prompts'))
                        except BaseException as exc:
                            self.state['harness'].update(
                                status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                error=f'{type(exc).__name__}: {exc}')
                            raise
                self.save()
            self.state['status'] = 'completed'
        except BaseException as exc:
            self.state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                              error=f'{type(exc).__name__}: {exc}')
            if self.state['iterations'] and self.state['iterations'][-1]['status'] == 'running':
                self.state['iterations'][-1]['status'] = self.state['status']
            raise
        finally:
            self.save()
        return self.state


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, parents=[launch_parser(add_help=False)])
    parser.add_argument('--threshold', type=threshold_value, default=30,
                        help='Direct fallback if mean native_score(iter0–2) * 100 is below this percentage (default: 30)')
    parser.add_argument('--tail-iterations', type=int, default=6,
                        help='Episodes after the three RSI gate trials (default: 6; nine total)')
    parser.add_argument('--harness-timeout', type=float, default=900,
                        help='Deadline in seconds for the RSI-only Codex Harness editing stage (default: 900)')
    parser.add_argument('--dry-run', action='store_true', help='Print both possible iteration paths; start no processes')
    parser.add_argument('--direct-tail', action='store_true',
                        help='Run only the direct tail from iter3 with zero experience; iter0–2 are confirmed below threshold')
    parser.add_argument('--iteration-seeds', type=int, nargs='+', metavar='SEED',
                        help='One seed per direct-tail iteration (default: repeating 0, 1, 2)')
    for action in parser._actions:
        if action.dest == 'seed':
            action.help = 'Use this seed for every iteration; default seeds repeat 0, 1, 2'
    parser.set_defaults(evaluation_method='rsi_policy_doctor', max_decisions=0, run_timeout=14400,
                        iteration=0, intervention_experience=None, seed=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.tail_iterations < 1:
        parser.error('--tail-iterations must be positive')
    if not math.isfinite(args.harness_timeout) or args.harness_timeout <= 0:
        parser.error('--harness-timeout must be finite and positive')
    if args.harness_dir:
        parser.error('Pipeline generates its own RSI Harness after iter2; do not supply --harness-dir')
    if args.iteration_seeds and len(args.iteration_seeds) != args.tail_iterations:
        parser.error('--iteration-seeds must provide one seed per tail iteration')
    if args.direct_tail:
        args.evaluation_method = 'gpt_only'
    if ((not args.direct_tail and args.evaluation_method != 'rsi_policy_doctor') or args.control_mode != '0-shot'
            or args.intervention_experience or args.task_experience or args.iteration != 0 or args.task_details or args.demo_video
            or args.smoke_approach or args.tower_test):
        parser.error('Pipeline starts at iter0 (RSI) or iter3 (--direct-tail) without prior experience or examples')
    if args.iteration_seeds and not args.direct_tail:
        parser.error('--iteration-seeds requires --direct-tail')
    if not args.direct_tail:
        if args.checkpoint is None and os.environ.get('PRISMWAM_CHECKPOINT'):
            args.checkpoint = Path(os.environ['PRISMWAM_CHECKPOINT'])
        if args.checkpoint is None:
            parser.error('Pipeline requires --checkpoint (or PRISMWAM_CHECKPOINT)')
        backend = args.student_backend
        if backend == 'auto':
            backend = 'prismwam' if args.checkpoint.suffix == '.pt' else 'pi05'
        if (backend == 'prismwam' and not args.checkpoint.is_file()
                or backend == 'pi05' and not args.checkpoint.is_dir()):
            parser.error('Checkpoint must be a PrismWAM .pt file or Pi05 checkpoint directory')
    if args.max_decisions < 0 or min(args.startup_timeout, args.run_timeout) <= 0:
        parser.error('Invalid budgets/timeouts')
    if not args.direct_tail and args.sim_port == args.student_port:
        parser.error('Student and simulator ports must differ')
    if args.output is None:
        run_id = datetime.now(timezone.utc).strftime('pipeline_%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:6]
        args.output = Path(__file__).resolve().parent.parent/'results'/run_id
    pipeline = Pipeline(args)
    if args.dry_run:
        paths = {}
        for tail in (('gpt_only',) if args.direct_tail else ('gpt_only', 'rsi_policy_doctor')):
            paths[tail] = [pipeline.command(i, 'rsi_policy_doctor' if i < 3 else tail,
                          pipeline.output/'experience'/f'input_iter{i}.md'
                          if i > 0 and not (i == 3 and tail == 'gpt_only') else None)
                          for i in range(3 if args.direct_tail else 0, pipeline.total_iterations)]
        print(json.dumps(dict(score_statistic='mean', score_unit='percent', threshold=args.threshold,
            harness=dict(gpt_only='skipped', rsi_policy_doctor=dict(
                after_iteration=2, experience=str(pipeline.output/'experience/iter2.md'),
                prompts=str(pipeline.output/'harness/prompts'), timeout=args.harness_timeout)),
            paths=paths), indent=2))
        return
    def terminate(signum, frame):
        raise KeyboardInterrupt('Stopping this pipeline and its owned rollout')
    signal.signal(signal.SIGTERM, terminate)
    try:
        state = pipeline.run()
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as exc:
        print(f'Pipeline stopped: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
    print(f'Completed {len(state["iterations"])} iterations. Gate: {state["gate"]}. '
          f'Report: {pipeline.output/"pipeline.json"}', flush=True)


if __name__ == '__main__':
    main()
