from copy import deepcopy
import json
import numpy as np
import pytest
from hybrid_rollout.robodojo.io import InputError
from robodoctor.rsi import RSITools
from robodoctor.experience import ExperienceStore
from robodoctor.rsi_schema import rsi_specs
from test_grounded import Sim, GroundedTools

class Policy:
    metadata = dict(backend='prismwam/PrismWAMPolicy', checkpoint='/ckpt.pt', checkpoint_sha256='x'*64,
                    action_horizon=4, action_dim=14, execution_horizon=2)
    def __init__(self): self.calls=[]
    def infer(self, obs, output):
        self.calls.append(deepcopy(obs)); actions=np.zeros((4,14),np.float32); actions[:,[6,13]]=1
        np.savez_compressed(output, actions=actions); return actions, dict(prediction_id=f'p{len(self.calls)}')
    def reset(self): self.calls.append('reset')

class RSISim(Sim):
    def request(self, op, **args):
        if op == 'fk_preview':
            actions=np.asarray(args['actions'])
            return dict(measured_fk_check=dict(passed=True), trajectory=[dict(index=i,
                left=dict(position=row[:3].tolist(), quaternion_wxyz=[1,0,0,0], gripper_closed=False, gripper_opening=1),
                right=dict(position=row[7:10].tolist(), quaternion_wxyz=[1,0,0,0], gripper_closed=False, gripper_opening=1)) for i,row in enumerate(actions)])
        if op in ('record_codex_decision','begin_combination','switch_control_source'): return {}
        if op == 'chunk_step' and 'student_prediction_id' in args:
            self.tick += len(args['actions'])
            return dict(step_id=self.tick, steps=[dict(valid=True, executed_action=np.asarray(action),
                obs=dict(states=self.states.copy()), terminated=False, truncated=False, success=False)
                for action in args['actions']])
        return super().request(op, **args)

def started(tmp_path, experience=None):
    p=Policy(); c=RSITools(tmp_path,'build_tower',p,experience_path=experience,rpc_factory=RSISim,max_decisions=0)
    args=c.next_call(); args.pop('tool'); return c,p,c.start(**args)

def rollout(c):
    args=c.next_call(); return c.rollout(args['request_id'],args['observation_path'],args['output_dir'])

def direct(c, steps):
    return dict(observation_path=str(c.observation_path),output_dir=str(c.output/'observations'/f'{len(c.history)+1:03d}'),response=dict(
        request_id=c.request['request_id'],mode='eef',steps=steps,stop_on_reach=False,
        reason='left: fine correction from RGB-D evidence, right: hold measured pose and preserve gripper state',
        target={a:dict(position=[0,0,0],quaternion_wxyz=[1,0,0,0],gripper_closed=False) for a in ('left','right')}))

def test_iter0_rollout_and_policy_prefix(tmp_path):
    c,p,_=started(tmp_path); packet=rollout(c)
    assert packet['prediction_id']=='p1' and packet['fk_check']['passed']
    args=c.next_call(); out=c.continue_policy(args['request_id'],packet['prediction_id'],2,args['output_dir'],
        'Both arm FK paths remain clear', 'Current RGB-D shows no obstacle along the first two FK rows')
    assert c.counters['student_steps']==2 and out['next_call']['tool']=='policy_rollout' and len(p.calls)==1

def test_prismwam_replan_executes_sixteen_step_prefix(tmp_path):
    class SixteenStepPolicy(Policy):
        metadata = dict(Policy.metadata, action_horizon=32, execution_horizon=16)

        def infer(self, obs, output):
            self.calls.append(deepcopy(obs))
            actions = np.zeros((32, 14), np.float32)
            actions[:, [6, 13]] = 1
            np.savez_compressed(output, actions=actions)
            return actions, dict(prediction_id='p1')

    policy = SixteenStepPolicy()
    controller = RSITools(tmp_path, 'build_tower', policy,
                          rpc_factory=RSISim, max_decisions=0)
    start_args = controller.next_call(); start_args.pop('tool')
    controller.start(**start_args)
    packet = rollout(controller)
    assert len(packet['action_chunk']) == 32
    assert len(packet['student_eef_trajectory']) == 32
    assert packet['execution_limit'] == 16
    args = controller.next_call()
    controller.continue_policy(args['request_id'], packet['prediction_id'], 16,
                               args['output_dir'], 'Clear path', 'RGB-D and FK show clear path')
    assert controller.counters['student_steps'] == 16

def test_intervention_is_fine_and_restarts_policy(tmp_path):
    c,p,_=started(tmp_path); rollout(c)
    with pytest.raises(InputError,match='1..5'): c.act(**direct(c,6))
    out=c.act(**direct(c,2))
    assert out['next_call']['tool']=='policy_rollout' and 'reset' in p.calls

@pytest.mark.parametrize('joint', [False, True])
@pytest.mark.parametrize('stop', ['budget', 'reached', 'terminal'])
def test_intervention_preserves_and_appends_exact_rgb_history(tmp_path, joint, stop):
    class ObservingPolicy(Policy):
        def __init__(self):
            super().__init__()
            self.observations = []

        def observe(self, obs):
            self.observations.append(deepcopy(obs))

    class ObservingSim(RSISim):
        def request(self, op, **args):
            if op == 'teacher_observation':
                obs = super().request(op, **args)
                for camera in GroundedTools.camera_keys:
                    obs[camera][:] = self.tick
                    obs[camera + '_depth'] = np.full((8, 8), self.tick + .5, np.float32)
                return obs
            if op == 'prepare_joints':
                self.target = {arm: dict(position=[0, 0, 0], quaternion_wxyz=[1, 0, 0, 0],
                                        gripper_closed=target['gripper_closed'])
                               for arm, target in args['targets'].items()}
                return dict(ok=True, target_eef=deepcopy(self.target),
                            joint_targets={arm: target['qpos'] for arm, target in args['targets'].items()})
            if op == 'eef_joint_target' and joint:
                return dict(action=self.states.copy(), diagnostics={})
            if op == 'chunk_step':
                rows = []
                for action in args['actions']:
                    result = super().request(op, **dict(args, actions=[action]))
                    obs = self.request('teacher_observation')
                    path = tmp_path / f'physical_step_{self.tick:03d}.npz'
                    np.savez_compressed(path, **obs)
                    for row in result['steps']:
                        row['observation_path'] = str(path)
                    rows.extend(result['steps'])
                return dict(step_id=self.tick, steps=rows)
            return super().request(op, **args)

    policy = ObservingPolicy()
    controller = RSITools(tmp_path, 'build_tower', policy,
                          rpc_factory=ObservingSim, max_decisions=0)
    args = controller.next_call(); args.pop('tool')
    controller.start(**args)
    packet = rollout(controller)
    args = controller.next_call()
    controller.continue_policy(args['request_id'], packet['prediction_id'], 2,
                               args['output_dir'], 'Clear path', 'RGB-D and FK show clear path')
    rollout(controller)
    with pytest.raises(InputError, match='1..5'):
        controller.act(**direct(controller, 6))
    assert len(policy.observations) == 3

    arguments = direct(controller, 5)
    arguments['response']['stop_on_reach'] = stop == 'reached'
    controller.sim.terminal = stop == 'terminal'
    if joint:
        arguments['response'].update(mode='joint', coordinate_mode='absolute',
            target={arm: dict(qpos=[0] * 6, gripper_closed=False) for arm in ('left', 'right')})
        output = controller.joint_act(**arguments, preview_only=False)
    else:
        output = controller.act(**arguments)

    final_tick = 7 if stop == 'budget' else 3
    assert controller.tick == final_tick
    assert output['movement']['executed_steps'] == final_tick - 2
    assert 'reset' not in policy.calls
    assert len(policy.observations) == final_tick + 1
    for tick, obs in enumerate(policy.observations):
        for camera in GroundedTools.camera_keys:
            np.testing.assert_array_equal(obs[camera], np.full((8, 8, 3), tick, np.uint8))
            np.testing.assert_array_equal(obs[camera + '_depth'], np.full((8, 8), tick + .5, np.float32))
    if stop != 'terminal':
        assert output['next_call']['tool'] == 'policy_rollout'
        rollout(controller)
        assert len(policy.observations) == final_tick + 1
        np.testing.assert_array_equal(policy.calls[-1]['cam_high'], policy.observations[-1]['cam_high'])

def experience_record():
    return dict(event='missed contact',skill='fine grounded approach',trigger='policy FK enters object side',
      policy_state={'observation':{'rgbd':'object side visible'},'action':[0,1],'fk':{'risk':'contact'}},
      intervention={'timing':'before contact','tools':['grounding','get_depth','robodojo_act'],
                    'steps':2,'technique':'fine approach','object_interaction':'grasp correction'},
      stop_condition='fingers clear in fresh RGB-D',outcome='recovered')

def test_experience_optional_and_skill_update(tmp_path):
    c,_,_=started(tmp_path); assert 'update_experience' in c.observation_packet['available_tools']
    assert c.experience.path == tmp_path/'intervention_experience.md'
    assert c.experience.path.exists()
    assert ExperienceStore(c.experience.path).document['skills'] == []
    path=tmp_path/'experience.md'; c,_,_=started(tmp_path/'with_exp',path)
    result=c.update_experience(c.request['request_id'],experience_record())
    text=path.read_text()
    assert result['experience_version']==1 and result['experience']['skills'][0]['skill']=='fine grounded approach'
    assert '## Policy diagnosis skills' in text
    assert '## Intervention operation information' in text
    assert not text.lstrip().startswith('{')
    revision=tmp_path/'with_exp'/'experience_revisions'/'0001.md'
    assert revision.exists() and revision.read_text().startswith('# RSI Intervention Experience')
    assert (tmp_path/'with_exp'/'intervention_experience.md').read_text() == text

def policy_success_record():
    return dict(record_type='policy_success', event='board grasp verified after policy lift',
        skill='left arm board edge grasp', trigger='reuse only for a similarly exposed board edge',
        policy_state=dict(observation={'rgbd': 'board moves with left fingers after lift', 'tick': 2},
                          action={'executed_prefix': [0, 1]}, fk={'measured': 'left EEF above board'}),
        policy_interaction=dict(target_object='flat board', arm='left', interaction_type='grasp',
            contact_region='exposed short edge; qualitative location', approach_direction='from above',
            eef_orientation='fingers straddle edge; qualitative, angle not measured',
            gripper_sequence='open during approach, close at edge, hold during lift',
            motion_sequence='approach, close, lift',
            success_evidence='fresh wrist/head RGB shows board following fingers after lift',
            applicability='exposed edge, clear approach, similar board pose'),
        outcome='autonomous policy grasp succeeded; whole task still ongoing')

def test_successful_policy_method_survives_failed_intervention_update(tmp_path):
    controller, policy, _ = started(tmp_path)
    packet = rollout(controller)
    proposal = controller.proposal['actions'].copy()
    args = controller.next_call()
    controller.continue_policy(args['request_id'], packet['prediction_id'], 2,
        args['output_dir'], 'Clear approach', 'RGB-D and FK support the approach')
    result = controller.update_experience(controller.request['request_id'], policy_success_record())
    assert result['physical_steps'] == 0
    assert controller.tick == 2 and controller.counters['student_steps'] == 2
    assert len(policy.calls) == 1
    snapshot_path = tmp_path/'intervention_experience.md'
    first_text = snapshot_path.read_text()
    assert 'Successful policy interaction' in first_text and 'Contact region' in first_text
    assert 'No intervention performed' in first_text

    # A later iteration adds a failed correction without replacing the verified
    # autonomous method, and another load/save keeps both sets of evidence.
    store = ExperienceStore(snapshot_path, iteration=1, task='build_tower')
    correction = experience_record()
    correction.update(record_type='intervention', outcome='modified grasp remains unverified')
    correction['intervention'].update(policy_reference='D0001: left arm board edge grasp',
        failure_evidence='fingers missed edge in fresh wrist RGB',
        adaptation='shift contact inward and rotate EEF to align fingers with the edge')
    store.update(correction)
    reloaded = ExperienceStore(snapshot_path)
    assert [s['record_type'] for s in reloaded.document['skills']] == ['policy_success', 'intervention']
    baseline = reloaded.document['skills'][0]['diagnosis_markdown']
    assert 'board following fingers' in baseline and 'exposed short edge' in baseline
    reloaded.save()
    text = snapshot_path.read_text()
    assert baseline in text and 'shift contact inward' in text and 'remains unverified' in text
    assert text.count('\n## ') == 2
    next_packet = rollout(controller)
    np.testing.assert_array_equal(next_packet['action_chunk'], proposal)

@pytest.mark.parametrize('invalid', ['missing_geometry', 'missing_success', 'takeover'])
def test_policy_success_requires_method_and_autonomous_evidence(tmp_path, invalid):
    store = ExperienceStore(tmp_path/'experience.md')
    record = policy_success_record()
    if invalid == 'missing_geometry':
        del record['policy_interaction']['eef_orientation']
    elif invalid == 'missing_success':
        del record['policy_interaction']['success_evidence']
    else:
        record['intervention'] = experience_record()['intervention']
    with pytest.raises(ValueError):
        store.update(record)
    assert store.document['version'] == 0 and not store.path.exists()

@pytest.mark.parametrize('inherited', [False, True])
def test_native_timeout_saves_experience_without_a_skill_update(tmp_path, inherited):
    class TimeoutSim(RSISim):
        def request(self, op, **args):
            result = super().request(op, **args)
            if op == 'metadata':
                result['max_episode_steps'] = 2
            if op == 'teacher_observation':
                result['remaining_steps'] = max(0, 2 - self.tick)
            if op == 'chunk_step':
                result['steps'][-1]['truncated'] = self.tick >= 2
            if op == 'finish_pilot':
                result['truncated'] = self.tick >= 2
            return result

    source = tmp_path/'inherited.md' if inherited else None
    if source:
        store = ExperienceStore(source, iteration=2, task='build_tower')
        store.update(experience_record())
    controller = RSITools(tmp_path/'run', 'build_tower', Policy(),
                          experience_path=source, iteration=3, rpc_factory=TimeoutSim,
                          max_decisions=0)
    args = controller.next_call(); args.pop('tool')
    controller.start(**args)
    packet = rollout(controller)
    args = controller.next_call()
    terminal = controller.continue_policy(args['request_id'], packet['prediction_id'], 2,
        args['output_dir'], 'Clear path', 'RGB-D and FK show clear path')
    assert controller.phase == 'done' and terminal['rollout_finished']
    result = json.loads((controller.output/'result.json').read_text())
    assert result['truncated'] and not result['success']
    assert result['experience_version'] == (1 if inherited else 0)
    output = controller.output/'intervention_experience.md'
    assert result['experience_output_path'] == str(output)
    snapshot = ExperienceStore(output).document
    assert snapshot['version'] == (1 if inherited else 0)
    assert snapshot['latest_iteration'] == 3
    assert len(snapshot['skills']) == int(inherited)
    if inherited:
        assert output.read_text() == source.read_text()
        assert snapshot['skills'][0]['skill'] == 'fine grounded approach'

def test_failed_final_rpc_still_saves_current_experience(tmp_path):
    controller, _, _ = started(tmp_path)
    controller.update_experience(controller.request['request_id'], experience_record())
    controller.experience.path.unlink()
    original = controller.sim.request
    def disconnected(op, **args):
        if op == 'finish_pilot':
            raise TimeoutError('simulator unavailable')
        return original(op, **args)
    controller.sim.request = disconnected
    with pytest.raises(TimeoutError, match='simulator unavailable'):
        controller.finish('controller_error')
    snapshot = ExperienceStore(controller.experience.path).document
    assert snapshot['version'] == 1
    assert snapshot['skills'][0]['skill'] == 'fine grounded approach'

@pytest.mark.parametrize('write_fails', [False, True])
def test_cleanup_saves_experience_and_always_closes_simulator(tmp_path, monkeypatch, write_fails):
    controller, _, _ = started(tmp_path)
    controller.experience.path.unlink()
    closed = []
    monkeypatch.setattr(controller.sim, 'close', lambda: closed.append(True), raising=False)
    if write_fails:
        def unavailable(*args):
            raise OSError('experience storage unavailable')
        monkeypatch.setattr(controller.experience, 'save', unavailable)
        with pytest.raises(OSError, match='experience storage unavailable'):
            controller.close()
    else:
        controller.close()
        assert ExperienceStore(controller.experience.path).document['version'] == 0
    assert closed == [True]

def test_rsi_tools_replace_old_gate_contract():
    assert {x['name'] for x in rsi_specs()} == {'robodojo_start','policy_rollout','policy_continue','update_experience','robodojo_act','robodojo_joint_act','grounding','get_depth'}


def test_merged_codex_runner_dispatches_rsi_tools_and_preserves_rsi_memory(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from robodoctor import runner
    from test_recovery import Transport, completed
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='test-codex'))
    monkeypatch.setenv('RSI_CODEX_PROVIDER', 'test_provider')
    controller, policy, _ = started(tmp_path)
    worker = runner.CodexPolicy(tmp_path/'codex_workspace', '/fake/codex',
                               method='rsi_policy_doctor', effort='medium', experience_enabled=True,
                               transport_factory=Transport)
    assert worker.thread_params['modelProvider'] == 'test_provider'
    assert worker.memory.path.name == 'CURRENT.md'
    assert 'update_experience' in {s['name'] for s in worker.thread_params['dynamicTools']}
    assert not (worker.workspace/'demonstration').exists()
    def tool(name, arguments):
        def event(transport):
            return dict(id=len(transport.replies)+1, method='item/tool/call',
                        params=dict(threadId=transport.thread, turnId=transport.turn,
                                    callId=str(len(transport.replies)), tool=name, arguments=arguments()))
        return event
    def proposal():
        arguments = controller.next_call()
        arguments.pop('tool')
        return arguments
    def intervention():
        controller.sim.terminal = True
        return direct(controller, 1)
    worker.transport.events = [tool('policy_rollout', proposal), tool('robodojo_act', intervention), completed()]
    worker.run(controller)
    assert len(worker.transport.replies) == 2
    assert all(reply['success'] for _, reply in worker.transport.replies)
    assert controller.phase == 'done'
    assert controller.counters['predictions'] == 1
