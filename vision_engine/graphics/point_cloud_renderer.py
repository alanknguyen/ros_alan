"""
vision_engine/graphics/point_cloud_renderer.py — RGB-D Point Cloud Rendering

Renders colored 3D point clouds from RGB-D camera data as GL_POINTS
in the OpenGL scene. Points are circular with configurable size.

Usage
-----
Integrated into the SceneRenderer — enable via config:
    renderer:
      show_point_cloud: true
"""

import numpy as np
import moderngl
from typing import Optional


class PointCloudRenderer:
    """
    Renders a colored 3D point cloud in the OpenGL scene.

    Parameters
    ----------
    ctx : moderngl.Context
        OpenGL context.
    program : moderngl.Program
        Point cloud shader program (from shaders.POINT_CLOUD_*).
    point_size : float
        GL point size in pixels (default 3.0).
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        program: moderngl.Program,
        point_size: float = 3.0,
    ):
        self.ctx = ctx
        self.program = program
        self.point_size = point_size

        self._vao: Optional[moderngl.VertexArray] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._num_points: int = 0

    def update(self, points: np.ndarray, colors: np.ndarray) -> None:
        """
        Upload new point cloud data to GPU.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3), dtype float32
            3D point positions.
        colors : np.ndarray, shape (N, 3), dtype float32
            Point colors RGB in [0, 1].
        """
        if len(points) == 0:
            self._num_points = 0
            return

        # Interleave position + color: (N, 6) float32
        data = np.hstack([
            points.astype(np.float32),
            colors.astype(np.float32),
        ])

        # Recreate buffer (simple approach — fine for ~20k points at 30fps)
        if self._vbo is not None:
            self._vbo.release()
        self._vbo = self.ctx.buffer(data.tobytes())

        if self._vao is not None:
            self._vao.release()
        self._vao = self.ctx.vertex_array(
            self.program,
            [(self._vbo, "3f 3f", "in_position", "in_color")],
        )
        self._num_points = len(points)

    def render(self, view_matrix: np.ndarray, projection_matrix: np.ndarray) -> None:
        """
        Render the point cloud.

        Parameters
        ----------
        view_matrix : np.ndarray, shape (4, 4)
        projection_matrix : np.ndarray, shape (4, 4)
        """
        if self._num_points == 0 or self._vao is None:
            return

        self.program["u_view"].write(view_matrix.astype(np.float32).tobytes())
        self.program["u_projection"].write(projection_matrix.astype(np.float32).tobytes())
        self.program["u_point_size"].value = self.point_size

        self._vao.render(moderngl.POINTS)

    def release(self) -> None:
        """Release GPU resources."""
        if self._vao:
            self._vao.release()
        if self._vbo:
            self._vbo.release()
