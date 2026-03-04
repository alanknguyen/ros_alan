"""
vision_engine/graphics/scene_graph.py — 3D Scene Graph Manager

Manages all renderable objects in the OpenGL scene:
  - Table (ground plane with grid)
  - Workspace bounds (wireframe box)
  - Coordinate axes (RGB=XYZ triad at origin)
  - Tracked rigid bodies (cubes, cylinders, spheres from OptiTrack)
  - Robot gripper indicator
  - Optional point cloud from RGB-D

Each object maintains its own vertex buffer (VBO) and vertex array (VAO).
The scene graph handles creation, updates, and rendering of all objects.
"""

import numpy as np
import moderngl
from typing import Dict, Optional, Tuple

from graphics.primitives import (
    make_cube, make_cylinder, make_sphere,
    make_grid, make_axis_triad, make_wireframe_box,
)
from cv.transforms import quaternion_to_rotation_matrix


class SceneGraph:
    """
    Manages all renderable objects in the 3D scene.

    Parameters
    ----------
    ctx : moderngl.Context
        The OpenGL context.
    phong_program : moderngl.Program
        Shader program for lit solid objects.
    flat_program : moderngl.Program
        Shader program for unlit lines/wireframes.
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        phong_program: moderngl.Program,
        flat_program: moderngl.Program,
    ):
        self.ctx = ctx
        self.phong_program = phong_program
        self.flat_program = flat_program

        # Static scene elements (created once)
        self._grid_vao: Optional[moderngl.VertexArray] = None
        self._grid_vertex_count = 0
        self._axes_vao: Optional[moderngl.VertexArray] = None
        self._bounds_vao: Optional[moderngl.VertexArray] = None
        self._bounds_vertex_count = 0

        # Dynamic objects: name → (vao, vertex_count, model_matrix)
        self._objects: Dict[str, Tuple[moderngl.VertexArray, int, np.ndarray]] = {}

        # Primitive mesh cache: shape_key → (vao, vertex_count)
        self._mesh_cache: Dict[str, Tuple[moderngl.VertexArray, int]] = {}

        # Gripper indicator
        self._gripper_vao: Optional[moderngl.VertexArray] = None
        self._gripper_model: np.ndarray = np.eye(4, dtype=np.float32)
        self._gripper_visible = False

    # ──────────────────────────────────────────────────────────────────────
    # Static Scene Setup
    # ──────────────────────────────────────────────────────────────────────

    def setup_table(self, height: float = 0.7, size: float = 2.0) -> None:
        """
        Create the table grid at the specified height.

        Parameters
        ----------
        height : float
            Table surface height in meters (Z coordinate).
        size : float
            Grid extent in meters.
        """
        grid_data = make_grid(size=size, spacing=0.1, color=(0.35, 0.35, 0.35))
        vbo = self.ctx.buffer(grid_data.tobytes())
        self._grid_vao = self.ctx.vertex_array(
            self.flat_program,
            [(vbo, "3f 3f", "in_position", "in_color")],
        )
        self._grid_vertex_count = len(grid_data)
        self._table_height = height

    def setup_axes(self, length: float = 0.15) -> None:
        """Create coordinate axis triad at origin."""
        axes_data = make_axis_triad(length=length)
        vbo = self.ctx.buffer(axes_data.tobytes())
        self._axes_vao = self.ctx.vertex_array(
            self.flat_program,
            [(vbo, "3f 3f", "in_position", "in_color")],
        )

    def setup_workspace_bounds(
        self,
        x_min: float, x_max: float,
        y_min: float, y_max: float,
        z_min: float, z_max: float,
    ) -> None:
        """Create wireframe workspace bounding box."""
        bounds_data = make_wireframe_box(
            x_min, x_max, y_min, y_max, z_min, z_max,
            color=(0.6, 0.6, 0.0),
        )
        vbo = self.ctx.buffer(bounds_data.tobytes())
        self._bounds_vao = self.ctx.vertex_array(
            self.flat_program,
            [(vbo, "3f 3f", "in_position", "in_color")],
        )
        self._bounds_vertex_count = len(bounds_data)

    def setup_gripper(self) -> None:
        """Create a simple gripper indicator (small wireframe cube)."""
        # Small gray cube to represent the end-effector
        gripper_data = make_cube(color=(0.7, 0.7, 0.7))
        vbo = self.ctx.buffer(gripper_data.tobytes())
        self._gripper_vao = self.ctx.vertex_array(
            self.phong_program,
            [(vbo, "3f 3f 3f", "in_position", "in_normal", "in_color")],
        )

    # ──────────────────────────────────────────────────────────────────────
    # Dynamic Object Updates
    # ──────────────────────────────────────────────────────────────────────

    def _get_or_create_mesh(
        self,
        shape: str,
        color: Tuple[float, float, float],
    ) -> Tuple[moderngl.VertexArray, int]:
        """Get a cached mesh VAO or create one for the given shape+color."""
        key = f"{shape}_{color[0]:.2f}_{color[1]:.2f}_{color[2]:.2f}"

        if key not in self._mesh_cache:
            if shape == "cube":
                data = make_cube(color=color)
            elif shape == "cylinder":
                data = make_cylinder(color=color)
            elif shape == "sphere":
                data = make_sphere(color=color)
            else:
                data = make_cube(color=(0.5, 0.5, 0.5))

            vbo = self.ctx.buffer(data.tobytes())
            vao = self.ctx.vertex_array(
                self.phong_program,
                [(vbo, "3f 3f 3f", "in_position", "in_normal", "in_color")],
            )
            self._mesh_cache[key] = (vao, len(data))

        return self._mesh_cache[key]

    def update_rigid_body(
        self,
        name: str,
        position: np.ndarray,
        quaternion: np.ndarray,
        shape_info: dict,
    ) -> None:
        """
        Update a rigid body's position and orientation in the scene.

        Creates the mesh on first call, then just updates the model matrix.

        Parameters
        ----------
        name : str
            Rigid body name (must match OptiTrack).
        position : np.ndarray, shape (3,)
            World position (x, y, z) in meters.
        quaternion : np.ndarray, shape (4,)
            Orientation as (qx, qy, qz, qw).
        shape_info : dict
            Object info from config: {shape, size/radius/height, color, mass}.
        """
        shape = shape_info.get("shape", "cube")
        color = tuple(shape_info.get("color", [0.5, 0.5, 0.5]))

        vao, vertex_count = self._get_or_create_mesh(shape, color)

        # Build model matrix: translation + rotation + scale
        model = np.eye(4, dtype=np.float32)

        # Rotation from quaternion
        R = quaternion_to_rotation_matrix(
            quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        )
        model[:3, :3] = R.astype(np.float32)

        # Scale based on shape dimensions
        if shape == "cube":
            size = shape_info.get("size", [0.05, 0.05, 0.05])
            model[:3, :3] *= np.array(size, dtype=np.float32)
        elif shape == "cylinder":
            r = shape_info.get("radius", 0.025)
            h = shape_info.get("height", 0.08)
            scale = np.array([r * 2, r * 2, h], dtype=np.float32)
            model[:3, :3] *= scale
        elif shape == "sphere":
            r = shape_info.get("radius", 0.03)
            model[:3, :3] *= r * 2

        # Translation
        model[:3, 3] = np.array(position, dtype=np.float32)

        self._objects[name] = (vao, vertex_count, model)

    def update_gripper(self, position: np.ndarray, is_open: bool) -> None:
        """
        Update the gripper indicator position.

        Parameters
        ----------
        position : np.ndarray, shape (3,)
            End-effector world position.
        is_open : bool
            Gripper open/closed state.
        """
        self._gripper_visible = True
        self._gripper_model = np.eye(4, dtype=np.float32)
        # Scale: small cube representing gripper
        scale = 0.04 if is_open else 0.02
        self._gripper_model[0, 0] = scale
        self._gripper_model[1, 1] = scale
        self._gripper_model[2, 2] = scale
        self._gripper_model[:3, 3] = np.array(position, dtype=np.float32)

    def remove_rigid_body(self, name: str) -> None:
        """Remove a rigid body from the scene."""
        self._objects.pop(name, None)

    # ──────────────────────────────────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────────────────────────────────

    def render(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        camera_position: np.ndarray,
    ) -> None:
        """
        Render all scene objects.

        Parameters
        ----------
        view_matrix : np.ndarray, shape (4, 4)
            Camera view matrix.
        projection_matrix : np.ndarray, shape (4, 4)
            Projection matrix.
        camera_position : np.ndarray, shape (3,)
            Camera world position (for specular lighting).
        """
        # ── Flat shader objects (grid, axes, bounds) ──
        self.flat_program["u_view"].write(view_matrix.astype(np.float32).tobytes())
        self.flat_program["u_projection"].write(projection_matrix.astype(np.float32).tobytes())

        # Table grid
        if self._grid_vao is not None:
            model = np.eye(4, dtype=np.float32)
            model[2, 3] = self._table_height  # Offset to table height
            self.flat_program["u_model"].write(model.tobytes())
            self._grid_vao.render(moderngl.LINES)

        # Coordinate axes
        if self._axes_vao is not None:
            model = np.eye(4, dtype=np.float32)
            self.flat_program["u_model"].write(model.tobytes())
            self._axes_vao.render(moderngl.LINES)

        # Workspace bounds
        if self._bounds_vao is not None:
            model = np.eye(4, dtype=np.float32)
            self.flat_program["u_model"].write(model.tobytes())
            self._bounds_vao.render(moderngl.LINES)

        # ── Phong shader objects (rigid bodies, gripper) ──
        self.phong_program["u_view"].write(view_matrix.astype(np.float32).tobytes())
        self.phong_program["u_projection"].write(projection_matrix.astype(np.float32).tobytes())
        self.phong_program["u_light_dir"].write(
            np.array([-0.3, -0.5, -1.0], dtype=np.float32).tobytes()
        )
        self.phong_program["u_light_color"].write(
            np.array([1.0, 1.0, 1.0], dtype=np.float32).tobytes()
        )
        self.phong_program["u_ambient_color"].write(
            np.array([0.2, 0.2, 0.2], dtype=np.float32).tobytes()
        )
        self.phong_program["u_camera_pos"].write(
            camera_position.astype(np.float32).tobytes()
        )

        # Rigid bodies
        for name, (vao, vertex_count, model) in self._objects.items():
            self.phong_program["u_model"].write(model.tobytes())
            vao.render(moderngl.TRIANGLES)

        # Gripper
        if self._gripper_visible and self._gripper_vao is not None:
            self.phong_program["u_model"].write(self._gripper_model.tobytes())
            self._gripper_vao.render(moderngl.TRIANGLES)
