"""
vision_engine/cv/camera_model.py — Pinhole Camera Model

Implements the standard pinhole camera model for:
  - Projecting 3D world points onto 2D image pixels (for annotations)
  - Unprojecting 2D pixels + depth back to 3D world points (for point clouds)
  - Generating OpenGL-compatible projection matrices (for the 3D renderer)

Camera Coordinate Convention
----------------------------
  - X = right
  - Y = down
  - Z = forward (into the scene)

This follows the standard optical frame convention used by OpenCV, ROS
(camera_color_optical_frame), and most camera SDKs.

Intrinsic Parameters
--------------------
The pinhole model uses 4 intrinsic parameters:
  - fx, fy: Focal lengths in pixels (horizontal, vertical)
  - cx, cy: Principal point (optical center) in pixels

The intrinsic matrix K:
    K = [fx   0  cx]
        [ 0  fy  cy]
        [ 0   0   1]

Extrinsic Parameters
--------------------
An optional 4×4 extrinsic matrix defines the world-to-camera transform.
If set, project() transforms world points to camera frame before projection.
If not set, points are assumed to already be in camera frame.
"""

import numpy as np
from typing import Optional, Tuple


class PinholeCamera:
    """
    Pinhole camera model with projection and unprojection.

    Parameters
    ----------
    fx : float
        Horizontal focal length in pixels.
    fy : float
        Vertical focal length in pixels.
    cx : float
        Principal point X coordinate in pixels.
    cy : float
        Principal point Y coordinate in pixels.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    extrinsic : np.ndarray or None, shape (4, 4)
        World-to-camera transform (optional). If provided, project() accepts
        world-frame points and transforms them to camera frame automatically.
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        extrinsic: Optional[np.ndarray] = None,
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height

        # Intrinsic matrix (3×3)
        self.K = np.array([
            [fx,  0.0, cx],
            [0.0, fy,  cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Extrinsic matrix (4×4 world→camera, optional)
        self.extrinsic = (
            np.asarray(extrinsic, dtype=np.float64) if extrinsic is not None else None
        )

    @classmethod
    def from_config(cls, config: dict) -> "PinholeCamera":
        """
        Create a PinholeCamera from a config dictionary.

        Expected keys: fx, fy, cx, cy, width, height.

        Parameters
        ----------
        config : dict
            Camera configuration (e.g., from scene_config.yaml "camera" section).

        Returns
        -------
        PinholeCamera
        """
        return cls(
            fx=float(config["fx"]),
            fy=float(config["fy"]),
            cx=float(config["cx"]),
            cy=float(config["cy"]),
            width=int(config["width"]),
            height=int(config["height"]),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Projection: 3D → 2D
    # ──────────────────────────────────────────────────────────────────────

    def project(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points onto the image plane.

        If an extrinsic matrix is set, input points are treated as world-frame
        coordinates and transformed to camera frame first. Otherwise, points
        are assumed to already be in camera frame.

        Parameters
        ----------
        points_3d : np.ndarray, shape (N, 3) or (3,)
            3D points to project.

        Returns
        -------
        pixels : np.ndarray, shape (N, 2) or (2,)
            Projected pixel coordinates (u, v). May be outside image bounds
            for points behind or beside the camera.
        valid : np.ndarray, shape (N,) or scalar bool
            True for points in front of the camera (z > 0) and within image bounds.
        """
        single = points_3d.ndim == 1
        pts = points_3d.reshape(-1, 3).astype(np.float64)

        # Transform to camera frame if extrinsic is set
        if self.extrinsic is not None:
            ones = np.ones((pts.shape[0], 1))
            pts_h = np.hstack([pts, ones])  # (N, 4)
            pts_cam = (self.extrinsic @ pts_h.T).T[:, :3]  # (N, 3)
        else:
            pts_cam = pts

        # Perspective projection
        z = pts_cam[:, 2]
        # Avoid division by zero for points at or behind camera
        z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)

        u = self.fx * pts_cam[:, 0] / z_safe + self.cx
        v = self.fy * pts_cam[:, 1] / z_safe + self.cy

        pixels = np.stack([u, v], axis=1)

        # Validity: in front of camera and within image bounds
        valid = (
            (z > 0)
            & (u >= 0) & (u < self.width)
            & (v >= 0) & (v < self.height)
        )

        if single:
            return pixels[0], valid[0]
        return pixels, valid

    # ──────────────────────────────────────────────────────────────────────
    # Unprojection: 2D + Depth → 3D
    # ──────────────────────────────────────────────────────────────────────

    def unproject(self, u: float, v: float, depth: float) -> np.ndarray:
        """
        Unproject a pixel coordinate + depth to a 3D point.

        The returned point is in camera frame (if no extrinsic) or world frame
        (if extrinsic is set, via inverse transform).

        Parameters
        ----------
        u : float
            Pixel X coordinate.
        v : float
            Pixel Y coordinate.
        depth : float
            Depth in meters at this pixel.

        Returns
        -------
        point_3d : np.ndarray, shape (3,)
            3D point in camera frame (or world frame if extrinsic is set).
        """
        # Camera frame coordinates
        x_cam = (u - self.cx) * depth / self.fx
        y_cam = (v - self.cy) * depth / self.fy
        z_cam = depth

        pt_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64)

        # Transform to world frame if extrinsic is set
        if self.extrinsic is not None:
            from cv.transforms import invert_transform, apply_transform
            T_cam_to_world = invert_transform(self.extrinsic)
            return apply_transform(T_cam_to_world, pt_cam)

        return pt_cam

    def unproject_batch(
        self,
        us: np.ndarray,
        vs: np.ndarray,
        depths: np.ndarray,
    ) -> np.ndarray:
        """
        Batch unproject multiple pixels + depths to 3D points.

        Parameters
        ----------
        us : np.ndarray, shape (N,)
            Pixel X coordinates.
        vs : np.ndarray, shape (N,)
            Pixel Y coordinates.
        depths : np.ndarray, shape (N,)
            Depth values in meters.

        Returns
        -------
        points_3d : np.ndarray, shape (N, 3)
            3D points in camera frame (or world frame if extrinsic is set).
        """
        x_cam = (us - self.cx) * depths / self.fx
        y_cam = (vs - self.cy) * depths / self.fy
        z_cam = depths

        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        if self.extrinsic is not None:
            from cv.transforms import invert_transform, apply_transform
            T_cam_to_world = invert_transform(self.extrinsic)
            return apply_transform(T_cam_to_world, pts_cam)

        return pts_cam

    # ──────────────────────────────────────────────────────────────────────
    # OpenGL Projection Matrix
    # ──────────────────────────────────────────────────────────────────────

    def get_opengl_projection_matrix(
        self,
        near: float = 0.01,
        far: float = 10.0,
    ) -> np.ndarray:
        """
        Compute a 4×4 OpenGL projection matrix from camera intrinsics.

        This allows the OpenGL renderer to use the same projection as the
        physical camera, enabling accurate overlay alignment.

        Parameters
        ----------
        near : float
            Near clipping plane distance in meters.
        far : float
            Far clipping plane distance in meters.

        Returns
        -------
        proj : np.ndarray, shape (4, 4)
            Column-major OpenGL projection matrix.

        Notes
        -----
        The standard OpenGL projection from pinhole intrinsics:
            [2*fx/w    0      1-2*cx/w         0         ]
            [  0     2*fy/h   2*cy/h-1         0         ]
            [  0       0    -(f+n)/(f-n)  -2*f*n/(f-n)   ]
            [  0       0        -1              0         ]
        """
        w, h = self.width, self.height
        proj = np.zeros((4, 4), dtype=np.float64)

        proj[0, 0] = 2.0 * self.fx / w
        proj[1, 1] = 2.0 * self.fy / h
        proj[0, 2] = 1.0 - 2.0 * self.cx / w
        proj[1, 2] = 2.0 * self.cy / h - 1.0
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2.0 * far * near / (far - near)
        proj[3, 2] = -1.0

        return proj

    def set_extrinsic(self, extrinsic: np.ndarray) -> None:
        """
        Set or update the world-to-camera extrinsic transform.

        Parameters
        ----------
        extrinsic : np.ndarray, shape (4, 4)
            World-to-camera homogeneous transform.
        """
        self.extrinsic = np.asarray(extrinsic, dtype=np.float64)
