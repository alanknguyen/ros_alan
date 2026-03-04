"""
vision_engine/graphics/renderer.py — Main OpenGL 3D Scene Renderer

Provides two rendering modes:
  1. Windowed (GLFW): Real-time 3D visualization for human operators
  2. Offscreen (FBO): Headless rendering for VLM snapshot capture

The renderer creates and manages the OpenGL context, compiles shaders,
and orchestrates the scene graph and camera controller.

Usage
-----
    from graphics.renderer import SceneRenderer

    renderer = SceneRenderer(width=1280, height=720)
    renderer.setup_scene(workspace_config)

    while not renderer.should_close():
        renderer.update_scene(snapshot, object_registry)
        renderer.render()

    renderer.shutdown()
"""

import numpy as np
import moderngl
import glfw
from typing import Optional

from graphics.shaders import (
    PHONG_VERTEX, PHONG_FRAGMENT,
    FLAT_VERTEX, FLAT_FRAGMENT,
)
from graphics.scene_graph import SceneGraph
from graphics.camera_controller import CameraController
from cv.scene_state import SceneSnapshot


class SceneRenderer:
    """
    OpenGL 3D scene renderer with GLFW window and offscreen FBO support.

    Parameters
    ----------
    width : int
        Window width in pixels (default 1280).
    height : int
        Window height in pixels (default 720).
    headless : bool
        If True, no GLFW window is created — offscreen FBO only.
    title : str
        Window title.
    background_color : tuple of 3 floats
        Scene background color RGB in [0, 1].
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        headless: bool = False,
        title: str = "Vision Engine — 3D Scene",
        background_color: tuple = (0.15, 0.15, 0.15),
        cs100_model=None,
    ):
        self.width = width
        self.height = height
        self.headless = headless
        self.background_color = background_color
        self._cs100_model = cs100_model

        self._window = None
        self._ctx: Optional[moderngl.Context] = None
        self._scene: Optional[SceneGraph] = None
        self._camera: Optional[CameraController] = None

        # Offscreen FBO for snapshot rendering
        self._fbo: Optional[moderngl.Framebuffer] = None
        self._fbo_width = 640
        self._fbo_height = 480

        # Projection matrix
        self._projection: Optional[np.ndarray] = None

        self._init_gl(title)

    def _init_gl(self, title: str) -> None:
        """Initialize GLFW window and OpenGL context."""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # OpenGL 3.3 Core Profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        if self.headless:
            glfw.window_hint(glfw.VISIBLE, False)

        self._window = glfw.create_window(
            self.width, self.height, title, None, None
        )
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self._window)

        # Create moderngl context from existing GLFW context
        self._ctx = moderngl.create_context()
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.enable(moderngl.CULL_FACE)

        # Compile shader programs
        self._phong_program = self._ctx.program(
            vertex_shader=PHONG_VERTEX,
            fragment_shader=PHONG_FRAGMENT,
        )
        self._flat_program = self._ctx.program(
            vertex_shader=FLAT_VERTEX,
            fragment_shader=FLAT_FRAGMENT,
        )

        # Create scene graph and camera
        self._scene = SceneGraph(self._ctx, self._phong_program, self._flat_program)
        self._camera = CameraController(mode="birds_eye")

        # Perspective projection
        self._update_projection(self.width, self.height)

        # Create offscreen FBO for snapshots
        self._create_fbo(self._fbo_width, self._fbo_height)

        # Set up GLFW callbacks
        glfw.set_key_callback(self._window, self._key_callback)
        glfw.set_mouse_button_callback(self._window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self._window, self._cursor_pos_callback)
        glfw.set_scroll_callback(self._window, self._scroll_callback)
        glfw.set_framebuffer_size_callback(self._window, self._resize_callback)

        print("[Renderer] OpenGL initialized")
        print(f"  Vendor:   {self._ctx.info['GL_VENDOR']}")
        print(f"  Renderer: {self._ctx.info['GL_RENDERER']}")
        print(f"  Version:  {self._ctx.info['GL_VERSION']}")

    def _update_projection(self, width: int, height: int) -> None:
        """Compute perspective projection matrix."""
        import pyrr
        self._projection = np.array(
            pyrr.matrix44.create_perspective_projection_matrix(
                fovy=60.0,
                aspect=width / max(height, 1),
                near=0.01,
                far=50.0,
            ),
            dtype=np.float32,
        )

    def _create_fbo(self, width: int, height: int) -> None:
        """Create offscreen framebuffer for snapshot rendering."""
        color = self._ctx.texture((width, height), 4)
        depth = self._ctx.depth_renderbuffer((width, height))
        self._fbo = self._ctx.framebuffer(color_attachments=[color], depth_attachment=depth)
        self._fbo_width = width
        self._fbo_height = height

    # ──────────────────────────────────────────────────────────────────────
    # Scene Setup
    # ──────────────────────────────────────────────────────────────────────

    def setup_scene(self, workspace_config: dict) -> None:
        """
        Initialize static scene elements from workspace configuration.

        Parameters
        ----------
        workspace_config : dict
            Workspace bounds from scene_config.yaml.
        """
        table_height = workspace_config.get("table_height", 0.7)
        self._scene.setup_table(height=table_height)
        self._scene.setup_axes(length=0.15)
        self._scene.setup_workspace_bounds(
            x_min=workspace_config.get("x_min", 0.4),
            x_max=workspace_config.get("x_max", 1.0),
            y_min=workspace_config.get("y_min", -0.6),
            y_max=workspace_config.get("y_max", 0.6),
            z_min=workspace_config.get("z_min", 0.0),
            z_max=table_height + 0.5,
        )
        self._scene.setup_gripper()

        # Update camera to look at workspace center
        cx = (workspace_config.get("x_min", 0.4) + workspace_config.get("x_max", 1.0)) / 2
        cy = (workspace_config.get("y_min", -0.6) + workspace_config.get("y_max", 0.6)) / 2
        self._camera.workspace_center = np.array([cx, cy, table_height], dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Scene Updates
    # ──────────────────────────────────────────────────────────────────────

    def update_scene(self, snapshot: SceneSnapshot, object_registry: dict) -> None:
        """
        Update the scene from a SceneSnapshot.

        Auto-centers the camera on tracked objects so they are always visible,
        regardless of where they are in OptiTrack/world coordinates.

        Parameters
        ----------
        snapshot : SceneSnapshot
            Current scene state from the aggregator.
        object_registry : dict
            Object definitions from scene_config.yaml "objects" section.
        """
        # Collect valid positions for auto-centering
        valid_positions = []

        # Update rigid bodies
        for name, body in snapshot.rigid_bodies.items():
            if not body.tracking_valid:
                continue

            shape_info = object_registry.get(name, {
                "shape": "sphere",
                "radius": 0.02,
                "color": [0.5, 0.5, 0.5],
                "mass": 0.1,
            })

            # Route CS-100 L-shape objects to dedicated renderer
            if (shape_info.get("render_as") == "cs100_lshape"
                    and self._cs100_model is not None):
                marker_positions = self._cs100_model.compute_marker_positions(
                    body.position, body.quaternion,
                )
                self._scene.update_cs100(marker_positions, shape_info)
            else:
                self._scene.update_rigid_body(
                    name=name,
                    position=body.position,
                    quaternion=body.quaternion,
                    shape_info=shape_info,
                )

            valid_positions.append(body.position)

        # Auto-center camera on tracked objects
        if valid_positions:
            self._camera.auto_center(valid_positions)

        # Update gripper if available
        if snapshot.gripper_position is not None:
            self._scene.update_gripper(
                snapshot.gripper_position,
                snapshot.gripper_open if snapshot.gripper_open is not None else True,
            )

    # ──────────────────────────────────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────────────────────────────────

    def render(self) -> None:
        """Render the scene to the GLFW window."""
        self._ctx.screen.use()
        r, g, b = self.background_color
        self._ctx.clear(r, g, b, 1.0)

        view = self._camera.get_view_matrix()
        cam_pos = self._camera.get_camera_position()

        self._scene.render(view, self._projection, cam_pos)

        glfw.swap_buffers(self._window)
        glfw.poll_events()

    def render_snapshot(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> np.ndarray:
        """
        Render the scene to an offscreen framebuffer and return as image.

        Parameters
        ----------
        width : int or None
            Snapshot width (default: fbo_width from config).
        height : int or None
            Snapshot height (default: fbo_height from config).

        Returns
        -------
        image : np.ndarray, shape (H, W, 3), dtype uint8
            Rendered RGB image.
        """
        w = width or self._fbo_width
        h = height or self._fbo_height

        # Recreate FBO if size changed
        if w != self._fbo_width or h != self._fbo_height:
            self._create_fbo(w, h)

        self._fbo.use()
        r, g, b = self.background_color
        self._ctx.clear(r, g, b, 1.0)

        # Use FBO-sized projection
        import pyrr
        proj = np.array(
            pyrr.matrix44.create_perspective_projection_matrix(
                fovy=60.0, aspect=w / max(h, 1), near=0.01, far=50.0,
            ),
            dtype=np.float32,
        )

        view = self._camera.get_view_matrix()
        cam_pos = self._camera.get_camera_position()
        self._scene.render(view, proj, cam_pos)

        # Read pixels
        raw = self._fbo.color_attachments[0].read()
        image = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
        image = image[::-1, :, :3]  # Flip vertically, drop alpha
        return image.copy()

    # ──────────────────────────────────────────────────────────────────────
    # Camera Control
    # ──────────────────────────────────────────────────────────────────────

    def set_camera_mode(self, mode: str) -> None:
        """Switch camera mode: 'birds_eye', 'robot_view', 'free'."""
        self._camera.set_mode(mode)
        print(f"[Renderer] Camera: {mode}")

    # ──────────────────────────────────────────────────────────────────────
    # Window Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def should_close(self) -> bool:
        """Check if the window should close (user pressed close/Q)."""
        return glfw.window_should_close(self._window)

    def shutdown(self) -> None:
        """Clean up OpenGL resources and close window."""
        if self._window:
            glfw.destroy_window(self._window)
        glfw.terminate()
        print("[Renderer] Shutdown complete.")

    # ──────────────────────────────────────────────────────────────────────
    # GLFW Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _key_callback(self, window, key, scancode, action, mods) -> None:
        """Handle keyboard input."""
        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_1:
            self.set_camera_mode("birds_eye")
        elif key == glfw.KEY_2:
            self.set_camera_mode("robot_view")
        elif key == glfw.KEY_3:
            self.set_camera_mode("free")
        elif key == glfw.KEY_S:
            # Save snapshot
            img = self.render_snapshot()
            import cv2
            filename = "output/snapshot.png"
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"[Renderer] Snapshot saved to {filename}")

    def _mouse_button_callback(self, window, button, action, mods) -> None:
        x, y = glfw.get_cursor_pos(window)
        self._camera.handle_mouse_button(button, action, x, y)

    def _cursor_pos_callback(self, window, x, y) -> None:
        self._camera.handle_mouse_move(x, y)

    def _scroll_callback(self, window, dx, dy) -> None:
        self._camera.handle_scroll(dx, dy)

    def _resize_callback(self, window, width, height) -> None:
        self._ctx.viewport = (0, 0, width, height)
        self._update_projection(width, height)
        self.width = width
        self.height = height
