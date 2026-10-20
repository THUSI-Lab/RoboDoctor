"""Calibrated RGB-D grounding; poses are metres and scalar-first quaternions."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from hybrid_rollout.robodojo.robodojo_server.action_edit_kinematics import pose, transform


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def pose7(value):
    value = array(value)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError('pose must be [x,y,z,qw,qx,qy,qz], all finite')
    if abs(np.linalg.norm(value[3:]) - 1) > 1e-4:
        raise ValueError('pose quaternion must be unit wxyz')
    return transform(value[:3], value[3:])


def organized_cloud(depth, intrinsics, world_from_camera_usd):
    """Unproject image-plane depth into H x W x 3 world points.

    Isaac USD camera axes: +X right, +Y up, -Z forward. Image axes:
    +x right, +y down. Use the actual optical Camera prim, not its parent.
    Invalid samples remain NaN and must never silently select a neighbour.
    """
    depth = array(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    k, extrinsic = array(intrinsics), array(world_from_camera_usd)
    if depth.ndim != 2 or k.shape != (3, 3) or extrinsic.shape != (4, 4):
        raise ValueError('Expected HxW depth, 3x3 intrinsics and 4x4 camera pose')
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or np.allclose(k, np.eye(3)):
        raise ValueError('Missing or invalid camera calibration')
    if not np.isfinite(extrinsic).all():
        raise ValueError('Invalid camera extrinsics')
    y, x = np.indices(depth.shape)
    rays = np.stack((x, y, np.ones_like(x)), axis=-1) @ np.linalg.inv(k).T
    valid = np.isfinite(depth) & (depth > 0)
    points = rays * np.where(valid, depth, np.nan)[..., None]
    points *= [1, -1, -1]
    return points @ extrinsic[:3, :3].T + extrinsic[:3, 3]


def ground_pixel(cloud, x, y, world_from_base):
    height, width, _ = cloud.shape
    if type(x) is not int or type(y) is not int or not (0 <= x < width and 0 <= y < height):
        raise ValueError(f'Pixel must be integer (x,y) within {width}x{height}')
    point = cloud[y, x]
    if not np.isfinite(point).all():
        raise ValueError('No valid depth at this pixel; select another visible surface pixel')
    base_point = (np.linalg.inv(world_from_base) @ np.r_[point, 1])[:3]
    return base_point, point


def pose_error(actual, target):
    return (float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
            float(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()))


def segment_distance(start, end, point):
    delta = end - start
    t = np.clip(np.dot(point - start, delta) / max(np.dot(delta, delta), 1e-15), 0, 1)
    return float(np.linalg.norm(start + t * delta - point))


def solve_ik(fk, joints, limits, target):
    """Robot-only bounded numerical IK, seeded from measured joints; no physics."""
    limits, joints = array(limits), array(joints)
    def residual(q):
        actual = fk.matrix(q)
        rotation = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
        return np.r_[target[:3, 3] - actual[:3, 3], .2 * rotation]
    seed = np.clip(joints, limits[:, 0], limits[:, 1])
    # Native reset leaves ~1e-12 joint noise. TRF initializes its trust radius
    # from ||x0||: almost-zero (unlike exactly-zero) can stop on ftol immediately.
    seed[np.abs(seed) < 1e-8] = 0
    seed = np.clip(seed, limits[:, 0], limits[:, 1])
    solved = least_squares(residual, seed,
                           bounds=(limits[:, 0], limits[:, 1]), max_nfev=200,
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    distance, angle = pose_error(fk.matrix(solved.x), target)
    if distance > .005 or angle > .03:
        raise ValueError(f'IK did not find a reachable pose: {distance:.4f} m, {angle:.4f} rad')
    return solved.x
