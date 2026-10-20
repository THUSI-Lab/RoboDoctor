"""Codex dynamic tools for RSI hybrid control."""
from hybrid_rollout.robodojo.skill.schema import _obj


def direct_response_schema():
    number = {"type": "number"}
    target = _obj(dict(position={"type": "array", "items": number, "minItems": 3, "maxItems": 3},
                       quaternion_wxyz={"type": "array", "items": number, "minItems": 4, "maxItems": 4},
                       gripper_closed={"type": "boolean"}))
    return _obj(dict(request_id={"type": "string"}, mode={"type": "string", "enum": ["eef"]},
                     steps={"type": "integer", "minimum": 1, "maximum": 5}, reason={"type": "string"},
                     stop_on_reach={"type": "boolean"}, target=_obj(dict(left=target, right=target))))


def _spec(name, description, **fields):
    return dict(type="function", name=name, description=description,
                inputSchema=_obj(fields))


def rsi_specs():
    text = {"type": "string", "minLength": 1}
    integer = {"type": "integer"}
    response = direct_response_schema()
    interaction = {"type": "object", "additionalProperties": False,
        "description": "Observed autonomous policy method, used as a baseline only during fine intervention. "
                       "Use object-relative geometry; label qualitative/unknown values and cite actual success evidence.",
        "properties": {"target_object": text, "arm": {"type": "string", "enum": ["left", "right", "both"]},
                       "interaction_type": text, "contact_region": text, "approach_direction": text,
                       "eef_orientation": text, "gripper_sequence": text, "motion_sequence": text,
                       "success_evidence": text, "applicability": text},
        "required": ["target_object", "arm", "interaction_type", "contact_region", "approach_direction",
                     "eef_orientation", "gripper_sequence", "motion_sequence", "success_evidence", "applicability"]}
    record = {"type": "object", "additionalProperties": False, "properties": {
        "event": text, "skill": text, "trigger": text,
        "record_type": {"type": "string", "enum": ["policy_success", "policy_failure", "intervention"]},
        "policy_interaction": interaction,
        "policy_state": {"type": "object", "description":
                         "Source-linked RGB-D/robot observation, executed policy prefix or risky proposal, "
                         "and measured/preview FK. Distinguish execution evidence from predictions.",
                         "properties": {"observation": {"type": "object"}, "action": {}, "fk": {"type": "object"}},
                         "required": ["observation", "action", "fk"]},
        "intervention": {"type": "object", "description":
                         "Fine intervention evidence, including the successful policy baseline and the "
                         "specific adaptation. If no baseline is available, explicitly say so.",
                         "properties": {"timing": text, "tools": {"type": "array", "items": text},
                                        "technique": text, "object_interaction": text, "steps": integer,
                                        "policy_reference": text, "adaptation": text, "failure_evidence": text},
                         "required": ["timing", "tools", "technique", "object_interaction"]},
        "stop_condition": text, "outcome": text, "notes": {"type": "string"},
        "skill_id": {"type": "string"},
    }, "required": ["event", "skill", "trigger", "policy_state", "outcome"],
        "description": "policy_success requires policy_interaction and actual autonomous success evidence; "
                       "omit intervention and stop_condition when no takeover occurred. Other records require "
                       "intervention and stop_condition. Omitted record_type keeps the legacy diagnosis format."}
    return [
        _spec("robodojo_start", "Start one fresh RoboDojo episode and return the first RGB-D observation.",
              task=text, output_dir=text),
        _spec("policy_rollout", "Run the selected pi05 or PrismWAM policy on the current observation. "
              "This is read-only: it returns the full action chunk and measured FK preview. "
              "Call it at iter 0 and after every policy chunk or intervention.",
              request_id=text, observation_path=text, output_dir=text),
        _spec("policy_continue", "Execute a contiguous prefix of the current policy chunk without editing it. "
              "Choose the longest prefix whose RGB-D and FK remain safe; this is policy execution, "
              "not intervention. The next observation invalidates the unused suffix.",
              request_id=text, prediction_id=text, steps=dict(type="integer", minimum=1, maximum=32),
              output_dir=text, reason=text, evidence=text),
        _spec("update_experience", "Record reusable verified autonomous policy object interactions, "
              "policy failures and intervention recoveries. For policy_success capture the object/arm, "
              "grasp/contact region, approach, EEF orientation, gripper/motion sequence, success evidence "
              "and applicability in policy_interaction; no fictitious intervention is needed. "
              "Write to the active Markdown intervention experience. The document has two blocks: "
              "Policy diagnosis skills and Intervention operation information. Decide the update timing yourself. "
              "With no input path, the run creates a new .md document from an empty skill set. "
              "policy_state must include observation/action/fk. Failure/intervention records also require "
              "intervention timing/tools/technique/object_interaction and stop_condition; record the "
              "policy_reference, adaptation and failure_evidence when applicable. Successful methods and "
              "adaptations guide fine intervention only; policy_continue remains unchanged.",
              request_id=text, record=record),
        _spec("robodojo_act", "Fine intervention only. Execute 1 to 5 measured EEF steps using current "
              "RGB, grounding and optional wrist depth. Use this when the policy chunk is unsafe, "
              "failing, or needs a contact-level correction. Re-ground an applicable recorded successful "
              "policy method; after failure revise the grasp/contact location, EEF orientation or other "
              "relevant parameter using fresh evidence. Do not use for ordinary policy execution.",
              observation_path=text, response=response, output_dir=text),
        _spec("robodojo_joint_act", "Fine intervention in joint space when EEF IK is unreliable. "
              "Execute 1 to 5 measured steps and then rerun policy_rollout.",
              observation_path=text, response={"type": "object", "additionalProperties": False,
                  "properties": {"request_id": text, "mode": {"type": "string", "enum": ["joint"]},
                      "coordinate_mode": {"type": "string", "enum": ["absolute", "delta"]},
                      "steps": {"type": "integer", "minimum": 1, "maximum": 5},
                      "stop_on_reach": {"type": "boolean"}, "reason": text,
                      "target": {"type": "object", "additionalProperties": False, "required": ["left", "right"],
                          "properties": {a: {"type": "object", "additionalProperties": False,
                              "required": ["qpos", "gripper_closed"],
                              "properties": {"qpos": {"type": "array", "items": {"type": "number"}, "minItems": 6, "maxItems": 6},
                                  "gripper_closed": {"type": "boolean"}}} for a in ("left", "right")}}},
                  "required": ["request_id", "mode", "coordinate_mode", "steps", "stop_on_reach", "reason", "target"]},
              output_dir=text, preview_only={"type": "boolean"}),
        _spec("grounding", "Project a native head-camera pixel through metric depth into environment_origin. "
              "Use the returned grounding_id in your intervention reasoning.", request_id=text,
              x={"type": "integer"}, y={"type": "integer"}),
        _spec("get_depth", "Read one metric wrist-camera depth pixel for a grasp or contact correction.",
              request_id=text, arm={"type": "string", "enum": ["left", "right"]},
              x={"type": "integer"}, y={"type": "integer"}),
    ]
