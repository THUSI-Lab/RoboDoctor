import json
from pathlib import Path
import shutil

import pytest

from robodoctor.pipeline import Pipeline, build_parser, main, read_outcome, tail_method
from robodoctor.harness_snapshot import prompt_hashes


def fake_harness(args, *, experience, output):
    assert experience.read_text() == 'Experience through iter2\n'
    snapshot = output/'prompts'
    shutil.copytree(Path(__file__).parents[1]/'robodoctor/rsi_prompts', snapshot)
    for name in ('SKILL.md', 'context/teacher_context.md'):
        with (snapshot/name).open('a') as stream:
            stream.write('\nHarness revision from iter2 evidence.\n')
    (snapshot/'manifest.json').write_text(json.dumps(dict(
        status='completed', evaluation_method='rsi_policy_doctor', files=prompt_hashes(snapshot))))
    return snapshot


def arguments(tmp_path, threshold=30):
    checkpoint = tmp_path/'weights.pt'
    checkpoint.touch()
    return build_parser().parse_args([
        '--task', 'build_tower', '--seed', '2', '--eval-seed', '4',
        '--threshold', str(threshold), '--output', str(tmp_path/'run'),
        '--checkpoint', str(checkpoint), '--codex-config', str(tmp_path/'config.toml'),
        '--effort', 'medium', '--gpu', '2', '--student-gpu', '3'])


def flag(command, name):
    return command[command.index(name)+1]


def evaluation(output, score, **overrides):
    path = output/'sim/evaluation_outcome.json'
    path.parent.mkdir(parents=True)
    outcome = dict(complete=True, valid_for_success_rate=True, native_score=score, native_success=score == 1)
    outcome.update(overrides)
    path.write_text(json.dumps(outcome))


@pytest.mark.parametrize('scores,threshold,tail', [
    ([0.1, 0.2, 0.3], 30, 'gpt_only'),
    ([0.8, 0.0, 0.0], 30, 'gpt_only'),
    ([0.8, 0.1, 0.2], 30, 'rsi_policy_doctor'),
    ([0.3, 0.3, 0.3], 30, 'rsi_policy_doctor'),
    ([0.2999, 0.3, 0.3], 30, 'gpt_only'),
    ([1, 0, 0], 30, 'rsi_policy_doctor'),
    ([0, 0, 0], 0, 'rsi_policy_doctor'),
    ([1, 1, 1], 100, 'rsi_policy_doctor'),
])
def test_mean_gate_and_strict_percent_threshold(scores, threshold, tail):
    assert tail_method(scores, threshold) == tail


@pytest.mark.parametrize('scores', [[0.2], [None, 0.2, 0.3], [True, 0.2, 0.3],
                                  [float('nan'), 0.2, 0.3], [1.1, 0.2, 0.3]])
def test_gate_never_treats_missing_or_invalid_scores_as_zero(scores):
    with pytest.raises(ValueError):
        tail_method(scores)


@pytest.mark.parametrize('scores,tail', [([0.1, 0.2, 0.3], 'gpt_only'),
                                       ([0.8, 0.0, 0.0], 'gpt_only'),
                                       ([0.8, 0.1, 0.2], 'rsi_policy_doctor'),
                                       ([0.3, 0.3, 0.3], 'rsi_policy_doctor')])
@pytest.mark.parametrize('tail_iterations', [3, 6])
def test_pipeline_carries_rsi_and_direct_task_experience(tmp_path, scores, tail, tail_iterations):
    args = arguments(tmp_path)
    args.tail_iterations = tail_iterations
    seen = []
    harness_calls = []
    def harness(*a, **kw):
        assert len(seen) == 3
        harness_calls.append(kw)
        return fake_harness(*a, **kw)
    def runner(command, log):
        iteration = int(flag(command, '--iteration'))
        method = flag(command, '--evaluation-method')
        output = Path(flag(command, '--output'))
        seen.append(method)
        if iteration >= 3 and method == 'rsi_policy_doctor':
            assert flag(command, '--harness-dir') == str(args.output/'harness/prompts')
            assert len(harness_calls) == 1
        else:
            assert '--harness-dir' not in command
        assert flag(command, '--seed') == '2'
        assert flag(command, '--eval-seed') == '4'
        assert flag(command, '--effort') == 'medium'
        assert flag(command, '--control-mode') == '0-shot'
        assert flag(command, '--codex-config') == str(tmp_path/'config.toml')
        evaluation(output, scores[iteration] if iteration < 3 else 0.4)
        input_option = '--intervention-experience' if method == 'rsi_policy_doctor' else '--task-experience'
        if iteration and not (iteration == 3 and method == 'gpt_only'):
            source = Path(flag(command, input_option))
            assert source.read_text() == f'Experience through iter{iteration-1}\n'
            source.write_text(f'Experience through iter{iteration}\n')
        else:
            assert input_option not in command
        if method == 'rsi_policy_doctor':
            assert '--checkpoint' in command
            assert '--task-experience' not in command
            name = 'intervention_experience.md'
        else:
            for option in ('--checkpoint', '--student-python', '--student-backend', '--intervention-experience'):
                assert option not in command
            name = 'task_experience.md'
        target = output/'controller'/name
        target.parent.mkdir()
        target.write_text(f'Experience through iter{iteration}\n')
        return 0
    state = Pipeline(args, episode_runner=runner, harness_runner=harness).run()
    assert len(harness_calls) == (1 if tail == 'rsi_policy_doctor' else 0)
    if tail == 'rsi_policy_doctor':
        assert all(row['harness_files'] == state['harness']['files'] for row in state['iterations'][3:])
    else:
        assert state['harness'] == dict(status='skipped', reason='direct_control')
        assert not (args.output/'harness').exists()
    assert seen == ['rsi_policy_doctor']*3+[tail]*tail_iterations
    assert state['status'] == 'completed'
    assert state['gate']['score'] == pytest.approx(sum(scores)/3*100)
    assert state['gate']['native_score'] == pytest.approx(sum(scores)/3)
    assert state['gate']['statistic'] == 'mean'
    assert state['gate']['threshold'] == 30
    assert state['gate']['score_unit'] == 'percent'
    assert state['gate']['tail_method'] == tail
    if tail == 'gpt_only':
        assert state['iterations'][3]['experience_input'] is None
        assert not (args.output/'experience/input_iter3.md').exists()
    assert [row['iteration'] for row in state['iterations']] == list(range(3+tail_iterations))
    assert json.loads((args.output/'pipeline.json').read_text()) == state
    for iteration in range(3+tail_iterations):
        assert (args.output/'experience'/f'iter{iteration}.md').read_text() == f'Experience through iter{iteration}\n'
        assert state['iterations'][iteration]['experience_type'] == (
            'task' if iteration >= 3 and tail == 'gpt_only' else 'intervention')


@pytest.mark.parametrize('overrides', [{'complete': False}, {'valid_for_success_rate': False},
                                     {'native_score': None}, {'native_score': float('inf')},
                                     {'native_score': True}])
def test_invalid_native_outcome_is_rejected(tmp_path, overrides):
    evaluation(tmp_path, 0.5, **overrides)
    with pytest.raises(ValueError, match='Incomplete/invalid'):
        read_outcome(tmp_path)


@pytest.mark.parametrize('failure', ['launcher', 'evaluation', 'experience', 'interrupt'])
def test_failed_iteration_stops_and_persists_evidence(tmp_path, failure):
    args = arguments(tmp_path)
    calls = []
    def runner(command, log):
        calls.append(command)
        if failure == 'launcher':
            return 9
        if failure == 'interrupt':
            raise KeyboardInterrupt('test stop')
        evaluation(Path(flag(command, '--output')), None if failure == 'evaluation' else 0.2)
        return 0
    with pytest.raises((ValueError, RuntimeError, KeyboardInterrupt)):
        Pipeline(args, episode_runner=runner).run()
    state = json.loads((args.output/'pipeline.json').read_text())
    assert len(calls) == 1
    assert state['gate'] is None
    assert state['status'] == ('interrupted' if failure == 'interrupt' else 'failed')
    assert state['iterations'][0]['status'] == state['status']


def test_dry_run_has_both_paths_and_creates_no_artifacts(tmp_path, capsys):
    checkpoint = tmp_path/'weights.pt'
    checkpoint.touch()
    output = tmp_path/'dry'
    main(['--dry-run', '--checkpoint', str(checkpoint), '--output', str(output)])
    plan = json.loads(capsys.readouterr().out)
    assert plan['score_statistic'] == 'mean'
    assert plan['threshold'] == 30
    assert plan['score_unit'] == 'percent'
    assert not output.exists()
    for tail, commands in plan['paths'].items():
        assert [flag(c, '--evaluation-method') for c in commands] == ['rsi_policy_doctor']*3+[tail]*6
        assert [flag(c, '--seed') for c in commands] == ['0', '1', '2']*3
        for iteration, command in enumerate(commands):
            assert ('--harness-dir' in command) == (iteration >= 3 and tail == 'rsi_policy_doctor')
            if iteration and not (iteration == 3 and tail == 'gpt_only'):
                option = '--task-experience' if iteration >= 3 and tail == 'gpt_only' else '--intervention-experience'
                assert flag(command, option) == str(output/'experience'/f'input_iter{iteration}.md')
            else:
                assert '--task-experience' not in command
                assert '--intervention-experience' not in command


@pytest.mark.parametrize('scores', [[0.1, 0.2, 0.3], [0.8, 0.1, 0.2]])
def test_default_seeds_repeat_in_both_pipeline_branches(tmp_path, scores):
    args = arguments(tmp_path)
    args.seed = build_parser().parse_args([]).seed
    seen = []

    def runner(command, log):
        iteration = int(flag(command, '--iteration'))
        seen.append(int(flag(command, '--seed')))
        output = Path(flag(command, '--output'))
        evaluation(output, scores[iteration] if iteration < 3 else 0.4)
        method = flag(command, '--evaluation-method')
        name = 'intervention_experience.md' if method == 'rsi_policy_doctor' else 'task_experience.md'
        experience = output / 'controller' / name
        experience.parent.mkdir()
        experience.write_text(f'Experience through iter{iteration}\n')
        return 0

    state = Pipeline(args, episode_runner=runner, harness_runner=fake_harness).run()
    assert seen == [0, 1, 2]*3
    assert state['iteration_seeds'] == seen
    assert [row['seed'] for row in state['iterations']] == seen
    assert json.loads((args.output / 'pipeline.json').read_text())['iteration_seeds'] == seen


@pytest.mark.parametrize('threshold', ['nan', 'inf', '-0.1', '100.1'])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--threshold', threshold])


def test_direct_tail_skips_rsi_and_carries_only_direct_experience(tmp_path):
    args = build_parser().parse_args(['--direct-tail', '--tail-iterations', '3', '--iteration-seeds', '0', '1', '2',
                                     '--output', str(tmp_path/'run')])
    seen = []
    def runner(command, log):
        iteration = int(flag(command, '--iteration'))
        seen.append(iteration)
        assert flag(command, '--seed') == str(iteration-3)
        assert flag(command, '--evaluation-method') == 'gpt_only'
        assert '--checkpoint' not in command and '--intervention-experience' not in command
        if iteration == 3:
            assert '--task-experience' not in command
        else:
            assert Path(flag(command, '--task-experience')).read_text() == f'Direct iter{iteration-1}\n'
        output = Path(flag(command, '--output'))
        evaluation(output, 0.2)
        experience = output/'controller/task_experience.md'
        experience.parent.mkdir()
        experience.write_text(f'Direct iter{iteration}\n')
        return 0
    state = Pipeline(args, episode_runner=runner).run()
    assert seen == [3, 4, 5]
    assert state['status'] == 'completed'
    assert state['gate']['source'] == 'user_confirmed'
    assert [row['seed'] for row in state['iterations']] == [0, 1, 2]
    assert not (args.output/'iter0').exists()
    assert not (args.output/'experience/input_iter3.md').exists()


@pytest.mark.parametrize('seed_args,expected', [([], ['0', '1', '2']),
                                             (['--iteration-seeds', '4', '5', '6'], ['4', '5', '6']),
                                             (['--seed', '7'], ['7', '7', '7'])])
def test_direct_tail_dry_run_does_not_require_a_policy_checkpoint(tmp_path, monkeypatch, capsys,
                                                                seed_args, expected):
    monkeypatch.delenv('PRISMWAM_CHECKPOINT', raising=False)
    main(['--direct-tail', '--tail-iterations', '3', '--dry-run', '--output', str(tmp_path/'run'), *seed_args])
    plan = json.loads(capsys.readouterr().out)
    assert list(plan['paths']) == ['gpt_only']
    commands = plan['paths']['gpt_only']
    assert [flag(c, '--iteration') for c in commands] == ['3', '4', '5']
    assert [flag(c, '--seed') for c in commands] == expected
    assert '--task-experience' not in commands[0]


def test_iteration_seeds_are_rejected_for_full_pipeline():
    with pytest.raises(SystemExit):
        main(['--iteration-seeds', '0', '1', '2', '--dry-run'])


@pytest.mark.parametrize('tail_iterations', ['0', '-1'])
def test_invalid_tail_length_is_rejected(tail_iterations):
    with pytest.raises(SystemExit):
        main(['--direct-tail', '--dry-run', '--tail-iterations', tail_iterations])


def test_direct_tail_seed_count_matches_tail_length():
    with pytest.raises(SystemExit):
        main(['--direct-tail', '--dry-run', '--tail-iterations', '6', '--iteration-seeds', '0', '1', '2'])


def test_nine_episode_dry_run_preserves_gate_and_seed_schedule(tmp_path, capsys):
    checkpoint = tmp_path/'weights.pt'
    checkpoint.touch()
    main(['--dry-run', '--tail-iterations', '6', '--checkpoint', str(checkpoint),
          '--output', str(tmp_path/'run')])
    plan = json.loads(capsys.readouterr().out)
    for tail, commands in plan['paths'].items():
        assert [flag(c, '--iteration') for c in commands] == [str(i) for i in range(9)]
        assert [flag(c, '--seed') for c in commands] == ['0', '1', '2']*3
        assert [flag(c, '--evaluation-method') for c in commands] == ['rsi_policy_doctor']*3+[tail]*6


@pytest.mark.parametrize('error', [RuntimeError('editing failed'), TimeoutError('deadline'),
                                  KeyboardInterrupt('stop')])
def test_harness_failure_stops_before_iter3(tmp_path, error):
    args = arguments(tmp_path)
    seen = []
    def episode(command, log):
        i = int(flag(command, '--iteration'))
        seen.append(i)
        output = Path(flag(command, '--output'))
        evaluation(output, 1)
        (output/'controller').mkdir()
        (output/'controller/intervention_experience.md').write_text(f'Experience through iter{i}\n')
        return 0
    def harness(*a, **kw):
        raise error
    with pytest.raises(type(error)):
        Pipeline(args, episode_runner=episode, harness_runner=harness).run()
    state = json.loads((args.output/'pipeline.json').read_text())
    assert seen == [0, 1, 2]
    assert state['iterations'][-1]['status'] == 'completed'
    assert state['harness']['status'] == state['status']
    assert state['status'] == ('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_invalid_harness_deadline(value):
    with pytest.raises(SystemExit):
        main(['--direct-tail', '--dry-run', '--harness-timeout', value])
