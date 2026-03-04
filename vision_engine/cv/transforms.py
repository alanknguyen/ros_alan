"""
vision_engine/cv/transforms.py — Coordinate Frame Transforms & Calibration Math

Provides utilities for:
  - Quaternion ↔ Euler angle conversion
  - Rotation matrix ↔ quaternion conversion
  - OptiTrack Y-up → Z-up (robotics) frame conversion
  - Rigid body transform application (4×4 homogeneous)
  - SVD-based rigid registration for OptiTrack↔robot calibration (Arun's method)

Quaternion convention: (x, y, z, w) — Hamilton convention, matching OptiTrack/ROS.

Coordinate frame conventions:
  - OptiTrack (Motive default): Y-up, right-handed. X=right, Y=up, Z=towards cameras.
  - Robotics (Z-up):           Z-up, right-handed. X=forward, Y=left, Z=up.
  - Conversion: x_zup = x_yup, y_zup = -z_yup, z_zup = y_yup
"""

import numpy as np
from typing import Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Quaternion Utilities
# ──────────────────────────────────────────────────────────────────────────────

def quaternion_to_euler(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """
    Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw) in radians.

    Uses the ZYX (yaw-pitch-roll) convention, which is standard in robotics.

    Parameters
    ----------
    qx, qy, qz, qw : float
        Quaternion components (Hamilton convention).

    Returns
    -------
    (roll, pitch, yaw) : tuple of float
        Euler angles in radians. Roll=rotation about X, Pitch=about Y, Yaw=about Z.
    """
    # Roll (X-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation) — clamped to avoid NaN at gimbal lock
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (Z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """
    Convert Euler angles (roll, pitch, yaw) in radians to quaternion (x, y, z, w).

    Uses ZYX convention (yaw applied first, then pitch, then roll).

    Parameters
    ----------
    roll : float
        Rotation about X-axis in radians.
    pitch : float
        Rotation about Y-axis in radians.
    yaw : float
        Rotation about Z-axis in radians.

    Returns
    -------
    (qx, qy, qz, qw) : tuple of float
        Quaternion components.
    """
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """
    Convert quaternion (x, y, z, w) to a 3×3 rotation matrix.

    Parameters
    ----------
    qx, qy, qz, qw : float
        Quaternion components.

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        Rotation matrix.
    """
    # Normalize quaternion
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm < 1e-10:
        return np.eye(3)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert a 3×3 rotation matrix to quaternion (x, y, z, w).

    Uses Shepperd's method for numerical stability.

    Parameters
    ----------
    R : np.ndarray, shape (3, 3)
        Rotation matrix (must be orthogonal with det=+1).

    Returns
    -------
    (qx, qy, qz, qw) : tuple of float
        Quaternion components.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return qx, qy, qz, qw


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions q1 * q2.

    Parameters
    ----------
    q1, q2 : np.ndarray, shape (4,)
        Quaternions as (x, y, z, w).

    Returns
    -------
    q : np.ndarray, shape (4,)
        Product quaternion (x, y, z, w).
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Frame Conversions: OptiTrack Y-up ↔ Robotics Z-up
# ──────────────────────────────────────────────────────────────────────────────

# The rotation matrix that converts from Y-up to Z-up:
#   x_zup =  x_yup
#   y_zup = -z_yup
#   z_zup =  y_yup
#
# As a 3×3 rotation matrix R_yup_to_zup:
#   [1   0   0 ]     [x_yup]     [x_zup]
#   [0   0  -1 ]  ×  [y_yup]  =  [y_zup]
#   [0   1   0 ]     [z_yup]     [z_zup]
R_YUP_TO_ZUP = np.array([
    [1.0,  0.0,  0.0],
    [0.0,  0.0, -1.0],
    [0.0,  1.0,  0.0],
])

# The corresponding quaternion for a 90° rotation about X-axis:
#   This rotates Y→Z and Z→-Y, which is exactly R_YUP_TO_ZUP.
#   q = (sin(45°), 0, 0, cos(45°)) = (√2/2, 0, 0, √2/2)
Q_YUP_TO_ZUP = np.array([np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2])


def position_yup_to_zup(pos_yup: np.ndarray) -> np.ndarray:
    """
    Convert a position vector from OptiTrack Y-up frame to robotics Z-up frame.

    Parameters
    ----------
    pos_yup : np.ndarray, shape (3,)
        Position (x, y, z) in Y-up frame.

    Returns
    -------
    pos_zup : np.ndarray, shape (3,)
        Position (x, y, z) in Z-up frame.
    """
    return R_YUP_TO_ZUP @ pos_yup


def quaternion_yup_to_zup(quat_yup: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion from OptiTrack Y-up frame to robotics Z-up frame.

    The rotation is pre-multiplied: q_zup = q_frame_change * q_yup.

    Parameters
    ----------
    quat_yup : np.ndarray, shape (4,)
        Quaternion (x, y, z, w) in Y-up frame.

    Returns
    -------
    quat_zup : np.ndarray, shape (4,)
        Quaternion (x, y, z, w) in Z-up frame.
    """
    return quaternion_multiply(Q_YUP_TO_ZUP, quat_yup)


# ──────────────────────────────────────────────────────────────────────────────
# Homogeneous Transforms (4×4)
# ──────────────────────────────────────────────────────────────────────────────

def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Construct a 4×4 homogeneous transform matrix from rotation + translation.

    Parameters
    ----------
    rotation : np.ndarray, shape (3, 3)
        Rotation matrix.
    translation : np.ndarray, shape (3,)
        Translation vector.

    Returns
    -------
    T : np.ndarray, shape (4, 4)
        Homogeneous transform matrix.
    """
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def apply_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply a 4×4 homogeneous transform to a set of 3D points.

    Parameters
    ----------
    T : np.ndarray, shape (4, 4)
        Homogeneous transform matrix.
    points : np.ndarray, shape (N, 3) or (3,)
        3D point(s) to transform.

    Returns
    -------
    transformed : np.ndarray, same shape as input
        Transformed point(s).
    """
    single = points.ndim == 1
    if single:
        points = points.reshape(1, 3)

    # Homogeneous coordinates: append 1s
    ones = np.ones((points.shape[0], 1))
    pts_h = np.hstack([points, ones])  # (N, 4)

    # Transform
    result = (T @ pts_h.T).T[:, :3]  # (N, 3)

    if single:
        return result.flatten()
    return result


def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    Invert a 4×4 homogeneous transform matrix efficiently.

    For a rigid transform [R|t], the inverse is [R^T | -R^T @ t].
    This is faster and more numerically stable than np.linalg.inv().

    Parameters
    ----------
    T : np.ndarray, shape (4, 4)
        Homogeneous transform matrix.

    Returns
    -------
    T_inv : np.ndarray, shape (4, 4)
        Inverse transform.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# ──────────────────────────────────────────────────────────────────────────────
# SVD-Based Rigid Registration (Arun's Method)
# ──────────────────────────────────────────────────────────────────────────────

def compute_rigid_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Compute the best-fit rigid transform (rotation + translation) that maps
    source_points to target_points using SVD-based point registration.

    This implements Arun's method (1987) for least-squares rigid body
    registration. Given N ≥ 3 point correspondences, it finds the rotation R
    and translation t that minimizes:

        Σ ||target_i - (R @ source_i + t)||²

    Used for OptiTrack↔robot frame calibration: record the same physical point
    in both frames, then compute the transform between them.

    Parameters
    ----------
    source_points : np.ndarray, shape (N, 3)
        Points in the source frame (e.g., OptiTrack world frame).
    target_points : np.ndarray, shape (N, 3)
        Corresponding points in the target frame (e.g., robot base frame).
        Must have the same number of points as source_points.

    Returns
    -------
    T : np.ndarray, shape (4, 4)
        4×4 homogeneous transform matrix mapping source → target.
    rms_error : float
        Root-mean-square registration error in meters.

    Raises
    ------
    ValueError
        If fewer than 3 point correspondences are provided, or if arrays
        have different lengths.

    Notes
    -----
    The algorithm:
        1. Compute centroids of both point sets
        2. Center the points (subtract centroids)
        3. Compute cross-covariance matrix H = Σ source'_i @ target'_i^T
        4. SVD decompose: U, S, Vt = svd(H)
        5. Rotation: R = V @ U^T (with determinant correction for reflections)
        6. Translation: t = centroid_target - R @ centroid_source

    Reference:
        Arun, K.S., Huang, T.S., Blostein, S.D. (1987).
        "Least-Squares Fitting of Two 3-D Point Sets."
        IEEE PAMI, 9(5), 698-700.
    """
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)

    if source_points.shape != target_points.shape:
        raise ValueError(
            f"Point arrays must have same shape. "
            f"Got source={source_points.shape}, target={target_points.shape}"
        )
    if source_points.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 point correspondences for rigid registration. "
            f"Got {source_points.shape[0]}."
        )

    # Step 1: Compute centroids
    centroid_source = source_points.mean(axis=0)
    centroid_target = target_points.mean(axis=0)

    # Step 2: Center the points
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target

    # Step 3: Cross-covariance matrix
    H = source_centered.T @ target_centered  # (3, 3)

    # Step 4: SVD
    U, S, Vt = np.linalg.svd(H)

    # Step 5: Rotation (with reflection correction)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    # Step 6: Translation
    t = centroid_target - R @ centroid_source

    # Compute RMS error
    transformed = (R @ source_points.T).T + t
    errors = np.linalg.norm(transformed - target_points, axis=1)
    rms_error = float(np.sqrt(np.mean(errors ** 2)))

    T = make_transform(R, t)
    return T, rms_error


# ──────────────────────────────────────────────────────────────────────────────
# Convenience
# ──────────────────────────────────────────────────────────────────────────────

def euler_degrees_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles in degrees. Convenience wrapper.

    Returns
    -------
    (roll_deg, pitch_deg, yaw_deg) : tuple of float
    """
    r, p, y = quaternion_to_euler(qx, qy, qz, qw)
    return np.degrees(r), np.degrees(p), np.degrees(y)
