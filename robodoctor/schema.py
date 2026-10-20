from hybrid_rollout.robodojo.skill.schema import _obj, direct_response_schema


ACT_REASON_PATTERN = r'^left: \S(?:[^\r\n]*\S)?, right: \S(?:[^\r\n]*\S)?$'


def tool_specs(method='gpt_only'):
    if method == 'rsi_policy_doctor':
        from .rsi_schema import rsi_specs
        return rsi_specs()
    if method == 'pi05_plus_gpt':
        from .hybrid_schema import hybrid_specs
        return hybrid_specs()
    if method != 'gpt_only':
        raise ValueError('Grounded direct control does not use pi05')
    string = {'type': 'string'}
    arm = {'type': 'string', 'enum': ['left', 'right']}
    def spec(name, description, **properties):
        return dict(type='function', name=name, description=description, inputSchema=_obj(properties))
    response = direct_response_schema()
    response['properties']['reason'].update(pattern=ACT_REASON_PATTERN, description=(
        'One English line, exactly: left: <ID> <reason>, right: <ID> <reason>. Start each arm description with its TODO subtask ID, then purpose and visible evidence; use the same ID for a coordinated subtask, or HOLD for an unassigned idle arm. '
        'Describe coarse positioning, fine adjustment, gripper operation or holding for EACH arm. '
        'Briefly summarize path clearance or collision risk and why coarse/fine is appropriate; '
        'for simultaneous motion include whether both swept paths stay safely separated. '
        'For an idle arm explicitly state holding the measured pose and preserving its gripper state. '
        'Example: left: A coarse approach above the grounded object through visible clear space with gripper open, '
        'right: HOLD hold measured pose and keep gripper open to stay clear.'))
    response['properties']['steps']['maximum'] = 150
    response['properties']['steps']['description'] = (
        'Control-step budget, 1..150, shared by both arms. Use a large budget ONLY when confident the entire grasp/movement swept path is obstacle-free. '
        'Possible obstacles, occlusion or uncertain clearance on either route require a small budget (typically 1..5), short clear waypoints and reinspection. '
        'Possible collision between the arms, grippers or held objects also requires fine control with nearby safe targets and stop_on_reach=false; detour or move sequentially. '
        'Also use short budgets for fine corrections or gripper settling.')
    response['properties']['stop_on_reach'] = {'type': 'boolean', 'description':
        'true: coarse travel, stop when both link6 poses arrive or tracking stalls. '
        'false: execute the selected steps for fine motion or gripper settling; native termination still stops execution. '
        'Pose arrival does not verify grasp or finger closure.'}
    response['required'].append('stop_on_reach')
    for side in ('left', 'right'):
        response['properties']['target']['properties'][side]['properties']['gripper_closed']['description'] = (
            'true closes, false opens. Applied from the FIRST control step, together with arm motion; '
            'not deferred until pose arrival. For gripper-only control hold the current measured pose, '
            'use stop_on_reach=false and a short step budget. Preserve this state while carrying.')
    joint_target = _obj(dict(qpos=dict(type='array', items=dict(type='number'), minItems=6, maxItems=6),
                             gripper_closed=dict(type='boolean')))
    joint_response = _obj(dict(request_id=string, mode=dict(type='string', enum=['joint']),
        coordinate_mode=dict(type='string', enum=['absolute', 'delta']),
        steps=dict(type='integer', minimum=1, maximum=150), stop_on_reach=dict(type='boolean'),
        reason=dict(response['properties']['reason']), target=_obj(dict(left=joint_target, right=joint_target))))
    return [
        spec('robodojo_start', 'Start once; return synchronized RGB and measured link6 poses in environment_origin.',
             task=string, output_dir=string),
        spec('grounding', 'Select an original-resolution head-camera pixel (x right, y down). Return surface XYZ in environment_origin, metres; shared by both arms; no motion. Coarse localization only: this is not a required grasp point or EEF target. Choose the actual grasp using fresh head/own-wrist RGB and wrist get_depth.',
             request_id=string, x={'type': 'integer', 'minimum': 0}, y={'type': 'integer', 'minimum': 0}),
        spec('get_depth', 'Read metric depth at an original-resolution wrist RGB pixel. '
             'arm selects cam_left_wrist or cam_right_wrist. Returns positive camera-axis depth_m from the current observation; '
             'not finger clearance or environment XYZ. No motion. Re-query after motion.',
             request_id=string, arm=arm, x={'type': 'integer', 'minimum': 0}, y={'type': 'integer', 'minimum': 0}),
        spec('robodojo_act', 'Unified absolute dual-arm IK control for coarse approach, transfer and fine manipulation. '
             'Always supply BOTH left/right link6 positions, unit wxyz orientations and boolean gripper_closed in environment_origin. '
             'No per-call target displacement/angle cap. Both arms execute together; steps=1..150. '
             'The total episode control-step budget is finite: prefer useful simultaneous work whenever safe. '
             'Before each act, actively choose useful work for the other arm instead of defaulting to idle or fully sequential pick-and-place. '
             'Both arms share each executed step; fewer calls alone do not save steps. Serialize only conflicting/dependent segments, '
             'preserve fine control near contact and reserve steps for release, retraction and required arm return. '
             'BEFORE OUTPUTTING AN ACTION: assess the full trajectories of both arms, wrists, fingers and held objects, including rotations and gripper motion. '
             'Check scene obstacles and mutual collisions throughout travel, accounting for unequal progress or one arm arriving first. '
             'For every planned simultaneous dual-arm move, require target.left.position[0] < target.right.position[0] in the current dual-X5 environment_origin frame '
             '(left is negative X, right is positive X). Equal or reversed X order is forbidden even with front-to-back Y separation or different Z heights. '
             'If this ordering check fails, execute SEQUENTIALLY, never simultaneously in either coarse or fine control: '
             'move one arm while copying the waiting arm\'s current measured pose and gripper state, confirm arrival at a safe destination/clearance waypoint '
             'and clearance of the second route, then hold the first arm at its new measured pose while moving the second. '
             'Choose order and additional clearance waypoints from the scene; a stationary arm remains an obstacle. '
             'Smaller steps do not waive the ordering rule. Correct endpoint ordering does not replace full swept-path collision checks. '
             'Simultaneous COARSE travel is allowed only when both swept paths are confidently obstacle-free and safely separated. '
             'Possible collisions or uncertain clearance require FINE safe segments (nearby targets, 1..5 steps, stop_on_reach=false), '
             'with detours or one arm holding while the other clears the shared space. Never follow a predicted colliding route merely with smaller steps. '
             'Grounding is only a coarse reference, not a required grasp/contact point or final EEF coordinate. '
             'For pre-grasp, leave a larger scene-dependent stand-off; then independently refine grasp position, orientation and insertion depth from fresh head/own-wrist RGB, wrist get_depth and current_eef. '
             'Fine targets may differ from grounding XYZ; a new grounding call is not required for every fine correction. '
             'For reset/return from clutter, fine-retreat until the arm/fingers/held object are clear, then coarse through clear space. '
             'COARSE: ground a pre-grasp/lift/transfer waypoint and use stop_on_reach=true. Use large steps ONLY if confident the full swept path is obstacle-free; '
             'if obstacles may exist or visibility/clearance is uncertain, use small steps (typically 1..5) and short clear waypoints, then reobserve. '
             'check both measured arrivals (5 mm / 0.03 rad). Detour around obstacles and preserve gripper state while carrying. '
             'If one arm can safely complete an object\'s operation, do not introduce a handoff. Each arm has limited, orientation-dependent reach. '
             'Approximate visual operating boundaries in the head-camera image: LEFT arm uses the extension of the visible base\'s RIGHT edge; '
             'RIGHT arm uses the extension of the base\'s LEFT edge. These are rough guides, not exact IK limits or reachability guarantees. '
             'An operation beyond the current arm\'s boundary, or an otherwise unreachable required pose, requires a supported handoff: '
             'place the object on stable support in a clear central area reachable by both arms, '
             'verify support, release and retract clear, then reobserve/re-ground and grasp with the other arm. '
             'Verify the new grasp with a small test lift; check both approach paths and do not assume central space is reachable or safe. '
             'EFFICIENCY EXAMPLE: when routes do not interfere, pick a left-side object with the left arm and a right-side object with the right arm concurrently. '
             'One arm may transfer toward its destination while the other moves an object needing handoff toward a clear shared central staging area outside the first route. '
             'Verify support before releasing the staged object and retract; after the receiving arm places its own object and is free, re-ground and grasp the staged object. '
             'FINE: near contact or placement, use wrist RGB/get_depth, small absolute pose corrections and usually 1..5 steps with stop_on_reach=false; '
             'also use fine when approaching easily toppled targets or nearby unstable objects, even along an apparently clear route. '
             'inspect after each call and avoid pushing through unexpected contact. '
             'For intentional pushing/toppling, close the gripper before contact in a hold-pose call; use the closed gripper as the contact point and fine control for contact and withdrawal. '
             'GRIPPER: gripper_closed=true closes, false opens, beginning on the first step. Reach the grasp pose before closing in a separate short hold-pose call. '
             'For irregular objects seek deeper finger-pad engagement where clear, broad stable flat edges and balanced contact near the estimated center of mass; '
             'prefer flat regions because the broad gripper pad faces can make larger effective contact with them, helping stabilize the grasp against slipping or rotation. '
             'avoid tip-only pinches and table contact. Confirm with a small test lift. '
             'For placement, release directly above the target at a suitable modest height if landing impact will not affect the goal; '
             'exact bottom contact is unnecessary then. Otherwise lower onto support first. Verify landing and settling. '
             'Returns per-arm errors, arrival and fresh RGB; arrival is not grasp success. '
             'Uses this repository numerical IK seeded from measured joints, within native joint limits, then velocity-limited joint tracking. '
             'Both arms must solve before any motion. No cuRobo planning and no collision checking. '
             'Does not guarantee a straight Cartesian path; use clear waypoints and hold unused arms explicitly. '
             'On IK failure or stalled motion, inspect the scene and use robodojo_joint_act for a small joint-space escape, then retry IK from the new observation.',
             observation_path=string, response=response, output_dir=string),
        spec('robodojo_joint_act', 'Direct dual-arm qpos control, bypassing IK and cuRobo. '
             'qpos has six radians in each arm current_joints.joint_names order; gripper_closed is separate. '
             'coordinate_mode=absolute sets joint angles; delta adds the six offsets ONCE to current measured qpos, never repeatedly per step. '
             'Supply both arms. Hold with zero deltas or current absolute qpos and preserve gripper state. '
             'preview_only=true returns resolved joint targets and environment_origin FK endpoints without moving or consuming steps; '
             'output_dir remains unused and request_id unchanged. Inspect the preview before executing with preview_only=false. '
             'Native joint position limits are enforced without clipping; tracking obeys native joint velocity limits. '
             'No obstacle/self/inter-arm collision checking; FK endpoints do not certify the swept path. '
             'Use for small escape/reconfiguration after failed IK or visibly stalled motion: move away from contact, never force through a blockage. '
             'Typically adjust one or a few joints, use 1..5 steps, inspect fresh RGB and measured qpos, then retry EEF IK. '
             'For simultaneous motion, predicted left EEF X must be less than right EEF X; otherwise move sequentially. '
             'stop_on_reach=true stops at both measured joint errors <=0.01 rad or tracking stall; false uses the budget. '
             'Both modes obey native termination. Gripper commands apply on the first step. '
             'An accepted command is not movement or grasp success. Use the same observation_path/output_dir as robodojo_act.',
             observation_path=string, response=joint_response, output_dir=string, preview_only=dict(type='boolean')),
    ]
