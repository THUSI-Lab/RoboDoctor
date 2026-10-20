"""Additional tools for planning and reviewing learned-policy intent."""
from hybrid_rollout.robodojo.skill.schema import _obj
from .schema import ACT_REASON_PATTERN


def hybrid_specs():
    from .schema import tool_specs
    specs = tool_specs('gpt_only')
    text = {'type': 'string', 'minLength': 1}
    strings = dict(type='array', items=text)
    enum = lambda *values: dict(type='string', enum=list(values))
    def spec(name, description, **fields):
        return dict(type='function', name=name, description=description, inputSchema=_obj(fields))
    subtask = _obj(dict(id=text, status=enum('pending', 'ready', 'in_progress', 'blocked', 'done'),
        src=text, dst=text, cond=text, arm_plan=enum('single', 'handoff', 'coordinated_dual'),
        arms_used=dict(type='array', items=enum('left', 'right'), minItems=1, maxItems=2, uniqueItems=True),
        simultaneous_arms=dict(type='integer', minimum=1, maximum=2),
        reachability=_obj({k:enum('yes', 'no', 'unknown') for k in ('left.src', 'left.dst', 'right.src', 'right.dst')}),
        depends_on=strings, next_action=text, evidence=text))
    identity = _obj(dict(subtask_id={'type': ['string', 'null']}, confidence=enum('low', 'medium', 'high'),
                         evidence=text, grounding_ids=strings))
    assessment = _obj(dict(decision=enum('allow', 'intervene'),
        safe_steps=dict(type='integer', minimum=0, maximum=15), left=identity, right=identity,
        failure_detected={'type':'boolean'}, path_unsafe={'type':'boolean'}, dependency_violation={'type':'boolean'},
        reason=dict(type='string', pattern=ACT_REASON_PATTERN)))
    specs.extend([
        spec('update_plan', 'Publish a complete current subtask plan before the first inference/motion and after each action. '
             'Reconcile actual RGB progress and pi05 FK intent; independent tasks may be reordered, causal prerequisites cannot. '
             'Record assignments/reachability and observed evidence. Replaces the complete TODO.md subtask list; the host archives TODO revisions in NOTES.md and retains plan history. '
             'Invalidates the previous review, but keeps a same-observation pi05 proposal. No physical steps.',
             request_id=text, subtasks=dict(type='array', items=subtask, minItems=1), reason=text),
        spec('pi05_infer', 'Predict H50 absolute dual-arm joints from the current three RGB cameras, robot state and task instruction. '
             'Returns all 50 environment_origin FK link6 poses and gripper openings; predicts robot kinematics, not future objects. '
             'No execution. Requires an up-to-date plan. output_dir must be controller/proposals/NNN where NNN is the next decision index.',
             request_id=text, observation_path=text, output_dir=text),
        spec('review_pi05', 'Identify each arm current intended subtask using three cameras, FK path/gripper sequence and grounding. '
             'Update plan first if pi05 chose another valid independent subtask. Report actual failure, path risk and dependency violation. '
             'Allow only a safe 1..15-step prefix; near contact use 1..5. Unknown intent, failures or unsafe routes require intervene with safe_steps=0. '
             'Use real current grounding_ids; no need to ground an idle arm. Both arms require evidence; assign an explicit hold subtask if idle. '
             'Intervention uses robodojo_act or robodojo_joint_act. No physical steps.',
             request_id=text, prediction_id=text, assessment=assessment),
        spec('pi05_execute', 'Execute only the reviewed safe prefix without editing policy joints. '
             'Return fresh three-camera observation, invalidate the entire remaining prediction and review, then update the plan. '
             'No automatic policy continuation. Native termination stops execution. Use next_call.output_dir for the new observation.',
             request_id=text, prediction_id=text, steps=dict(type='integer', minimum=1, maximum=15), output_dir=text),
    ])
    for tool in specs:
        if tool['name'] not in ('robodojo_act', 'robodojo_joint_act'):
            continue
        tool['description'] += (' In hybrid mode this is a direct intervention/observation-hold tool. '
        'Update the plan first; if a pi05 proposal exists, review it with decision=intervene before acting. '
        'After correction reobserve, update the plan, and infer fresh pi05 actions; never resume an old suffix.')
    return specs
