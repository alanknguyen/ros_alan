"""
vision_engine/graphics/camera_controller.py — 3D Camera Viewpoint Control

Manages camera position and orientation for the OpenGL renderer.
Supports three modes:

  1. Birds-Eye:     Top-down view of the workspace
  2. Robot View:    From behind the robot, looking forward at the table
  3. Free Orbit:    Mouse-controlled orbit camera (drag to rotate, scroll to zoom)

All cameras produce a 4×4 view matrix for OpenGL via pyrr.matrix44.create_look_at.
"""

import numpy as np
import pyrr
from typing import Optional


class CameraController:
    """
    Manages the camera viewpoint for 3D scene rendering.

    Parameters
    ----------
    mode : str
        Initial camera mode: "birds_eye", "robot_view", or "free".
    workspace_center : tuple of 3 floats
        Center of the workspace (x, y, z) — the default look-at target.
    """

    def __init__(
        self,
        mode: str = "birds_eye",
        workspace_center: tuple = (0.7, 0.0, 0.7),
    ):
        self.workspace_center = np.array(workspace_center, dtype=np.float32)

        # Free camera orbit state
        self._orbit_yaw = -90.0       # degrees, horizontal angle
        self._orbit_pitch = 45.0      # degrees, vertical angle
        self._orbit_distance = 1.5    # meters from target
        self._orbit_target = self.workspace_center.copy()

        # Mouse tracking
        self._mouse_pressed = False
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        """
        Switch camera mode.

        Parameters
        ----------
        mode : str
            "birds_eye", "robot_view", or "free"
        """
        self.mode = mode

    def get_view_matrix(self) -> np.ndarray:
        """
        Compute the 4×4 view matrix for the current camera mode.

        Returns
        -------
        view : np.ndarray, shape (4, 4), dtype float32
            OpenGL view matrix.
        """
        if self.mode == "birds_eye":
            return self._birds_eye_view()
        elif self.mode == "robot_view":
            return self._robot_view()
        else:
            return self._free_orbit_view()

    def get_camera_position(self) -> np.ndarray:
        """
        Get the camera's world-space position (needed for specular lighting).

        Returns
        -------
        pos : np.ndarray, shape (3,)
        """
        if self.mode == "birds_eye":
            return np.array([0.7, 0.0, 2.0], dtype=np.float32)
        elif self.mode == "robot_view":
            return np.array([-0.2, 0.0, 1.2], dtype=np.float32)
        else:
            return self._orbit_camera_position()

    # ──────────────────────────────────────────────────────────────────────
    # Preset Views
    # ──────────────────────────────────────────────────────────────────────

    def _birds_eye_view(self) -> np.ndarray:
        """Top-down view looking straight down at the workspace."""
        eye = np.array([0.7, 0.0, 2.0], dtype=np.float32)
        target = self.workspace_center
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # X is "up" in top-down
        return np.array(pyrr.matrix44.create_look_at(eye, target, up), dtype=np.float32)

    def _robot_view(self) -> np.ndarray:
        """View from behind the robot, looking forward towards the table."""
        eye = np.array([-0.2, 0.0, 1.2], dtype=np.float32)
        target = self.workspace_center
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.array(pyrr.matrix44.create_look_at(eye, target, up), dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Free Orbit Camera
    # ──────────────────────────────────────────────────────────────────────

    def _orbit_camera_position(self) -> np.ndarray:
        """Compute camera position from orbit angles and distance."""
        yaw_rad = np.radians(self._orbit_yaw)
        pitch_rad = np.radians(self._orbit_pitch)

        x = self._orbit_distance * np.cos(pitch_rad) * np.cos(yaw_rad)
        y = self._orbit_distance * np.cos(pitch_rad) * np.sin(yaw_rad)
        z = self._orbit_distance * np.sin(pitch_rad)

        return self._orbit_target + np.array([x, y, z], dtype=np.float32)

    def _free_orbit_view(self) -> np.ndarray:
        """Orbit camera view matrix."""
        eye = self._orbit_camera_position()
        target = self._orbit_target
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.array(pyrr.matrix44.create_look_at(eye, target, up), dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Mouse/Scroll Input (for free camera mode)
    # ──────────────────────────────────────────────────────────────────────

    def handle_mouse_button(self, button: int, action: int, x: float, y: float) -> None:
        """
        Handle mouse button press/release.

        Parameters
        ----------
        button : int
            GLFW mouse button (0=left, 1=right, 2=middle).
        action : int
            1=press, 0=release.
        x, y : float
            Cursor position.
        """
        if button == 0:  # Left button
            self._mouse_pressed = (action == 1)
            self._last_mouse_x = x
            self._last_mouse_y = y

    def handle_mouse_move(self, x: float, y: float) -> None:
        """
        Handle mouse movement (orbit when dragging in free mode).

        Parameters
        ----------
        x, y : float
            Current cursor position.
        """
        if not self._mouse_pressed or self.mode != "free":
            self._last_mouse_x = x
            self._last_mouse_y = y
            return

        dx = x - self._last_mouse_x
        dy = y - self._last_mouse_y
        self._last_mouse_x = x
        self._last_mouse_y = y

        # Sensitivity
        self._orbit_yaw += dx * 0.3
        self._orbit_pitch += dy * 0.3

        # Clamp pitch to avoid flipping
        self._orbit_pitch = np.clip(self._orbit_pitch, -89.0, 89.0)

    def handle_scroll(self, dx: float, dy: float) -> None:
        """
        Handle scroll wheel (zoom in free mode).

        Parameters
        ----------
        dx, dy : float
            Scroll amounts (dy is the main scroll axis).
        """
        if self.mode != "free":
            return

        self._orbit_distance -= dy * 0.1
        self._orbit_distance = np.clip(self._orbit_distance, 0.3, 5.0)
