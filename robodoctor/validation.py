"""EEF and joint tool input checks; native joint limits are enforced server-side."""
import re
import numpy as np

from hybrid_rollout.robodojo.robodojo_server.validation import validate_public_language
from .geometry import pose7
from .schema import ACT_REASON_PATTERN


def validate_action(response, request, *, joint=False):
    if not isinstance(response, dict) or response.get('request_id') != request['request_id']:
        raise ValueError('Response must match the current observation request_id')
    reason = response.get('reason')
    if not isinstance(reason, str) or re.fullmatch(ACT_REASON_PATTERN, reason) is None:
        raise ValueError('reason must be one English line in the format '
                         '"left: <left-arm purpose and evidence>, right: <right-arm purpose and evidence>"; '
                         'both descriptions are required, including holding for an idle arm')
    validate_public_language(response)
    if response.get('mode') != ('joint' if joint else 'eef'):
        raise ValueError('Use robodojo_act for mode=eef or robodojo_joint_act for mode=joint')
    if response.get('actions') not in (None, []):
        raise ValueError('Use explicit dual-arm targets, not raw actions arrays')
    steps = response.get('steps')
    if type(steps) is not int or not 1 <= steps <= 150:
        raise ValueError('Expected 1..150 execution steps')
    if type(response.get('stop_on_reach', True)) is not bool:
        raise ValueError('stop_on_reach must be boolean')
    targets = response.get('target')
    if not isinstance(targets, dict) or set(targets) != {'left', 'right'}:
        raise ValueError('Action requires explicit left and right targets')
    if joint and response.get('coordinate_mode') not in ('absolute', 'delta'):
        raise ValueError('coordinate_mode must be absolute or delta')
    for target in targets.values():
        if joint:
            q = np.asarray(target['qpos'], dtype=float)
            if q.shape != (6,) or not np.isfinite(q).all():
                raise ValueError('qpos must contain six finite joint angles in radians')
        else:
            p = np.asarray(target['position'], dtype=float)
            q = np.asarray(target['quaternion_wxyz'], dtype=float)
            if p.shape != (3,) or q.shape != (4,):
                raise ValueError('EEF position/quaternion must have 3/4 components')
            pose7(np.r_[p, q])
        if type(target['gripper_closed']) is not bool:
            raise ValueError('Explicit boolean gripper required')
