"""
vision_engine/cv/depth_processor.py — RGB-D Depth Processing & Point Cloud Generation

Processes depth images from an RGB-D sensor to:
  - Look up depth at specific pixel coordinates
  - Generate colored 3D point clouds for OpenGL rendering
  - Filter out invalid depth values (NaN, zero, out-of-range)

Point clouds are returned as numpy arrays suitable for direct upload
to OpenGL vertex buffers (positions + colors).
"""

import numpy as np
from typing import Tuple, Optional

from cv.camera_model import PinholeCamera


class DepthProcessor:
    """
    Processes depth images and generates point clouds.

    Parameters
    ----------
    camera : PinholeCamera
        Camera model for pixel ↔ 3D conversion.
    min_depth : float
        Minimum valid depth in meters (default 0.1m).
    max_depth : float
        Maximum valid depth in meters (default 5.0m).
    """

    def __init__(
        self,
        camera: PinholeCamera,
        min_depth: float = 0.1,
        max_depth: float = 5.0,
    ):
        self.camera = camera
        self.min_depth = min_depth
        self.max_depth = max_depth

    def depth_at_pixel(self, depth_image: np.ndarray, u: int, v: int) -> Optional[float]:
        """
        Look up the depth value at a specific pixel.

        Parameters
        ----------
        depth_image : np.ndarray, shape (H, W)
            Depth image in meters (float32).
        u : int
            Pixel X coordinate (column).
        v : int
            Pixel Y coordinate (row).

        Returns
        -------
        depth : float or None
            Depth in meters, or None if the pixel is out of bounds or
            the depth value is invalid (NaN, zero, out of range).
        """
        h, w = depth_image.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None

        d = float(depth_image[v, u])
        if np.isnan(d) or d <= 0 or d < self.min_depth or d > self.max_depth:
            return None

        return d

    def to_point_cloud(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        downsample: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert an RGB-D image pair into a colored 3D point cloud.

        Parameters
        ----------
        rgb_image : np.ndarray, shape (H, W, 3)
            BGR image from camera, dtype uint8.
        depth_image : np.ndarray, shape (H, W)
            Depth image in meters, dtype float32.
        downsample : int
            Skip every N pixels to reduce point count (default 4).
            A 640×480 image with downsample=4 produces ~19,200 points.

        Returns
        -------
        points : np.ndarray, shape (N, 3)
            3D point positions in camera frame (or world frame if camera
            has an extrinsic transform set), dtype float32.
        colors : np.ndarray, shape (N, 3)
            Point colors as RGB floats in [0, 1], dtype float32.
        """
        h, w = depth_image.shape[:2]

        # Generate pixel coordinate grids (downsampled)
        vs, us = np.mgrid[0:h:downsample, 0:w:downsample]
        us = us.flatten().astype(np.float64)
        vs = vs.flatten().astype(np.float64)

        # Sample depths at grid positions
        depths = depth_image[vs.astype(int), us.astype(int)].astype(np.float64)

        # Filter valid depths
        valid = (
            ~np.isnan(depths)
            & (depths > self.min_depth)
            & (depths < self.max_depth)
        )
        us = us[valid]
        vs = vs[valid]
        depths = depths[valid]

        if len(us) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        # Unproject to 3D
        points = self.camera.unproject_batch(us, vs, depths)

        # Sample colors (BGR → RGB, normalize to [0, 1])
        colors_bgr = rgb_image[vs.astype(int), us.astype(int)]  # (N, 3) uint8
        colors_rgb = colors_bgr[:, ::-1].astype(np.float32) / 255.0

        return points.astype(np.float32), colors_rgb
