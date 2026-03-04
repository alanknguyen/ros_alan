"""
vision_engine/cv/annotator.py — OpenCV 2D Scene Annotation Overlays

Projects OptiTrack rigid body data onto an image (RGB camera feed or
rendered 3D snapshot) and draws informative overlays:
  - Object name labels
  - Coordinate axis triads at each object
  - Distance to gripper
  - Physics status icons (stable/unstable, graspable/not)
  - Workspace bounds projected onto the image

These annotations help both human operators and VLMs understand the scene.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple

from cv.scene_state import SceneSnapshot
from cv.camera_model import PinholeCamera
from cv.transforms import quaternion_to_rotation_matrix, euler_degrees_from_quaternion


# Color constants (BGR for OpenCV)
COLOR_GREEN = (0, 200, 0)
COLOR_RED = (0, 0, 200)
COLOR_YELLOW = (0, 200, 200)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (200, 200, 0)
COLOR_AXIS_X = (0, 0, 255)   # Red
COLOR_AXIS_Y = (0, 255, 0)   # Green
COLOR_AXIS_Z = (255, 0, 0)   # Blue


class SceneAnnotator:
    """
    Annotates images with projected scene information.

    Parameters
    ----------
    camera : PinholeCamera or None
        Camera model for 3D→2D projection. If None, annotations are
        placed using a simple top-down mapping.
    image_width : int
        Output image width (used when no camera image is available).
    image_height : int
        Output image height.
    """

    def __init__(
        self,
        camera: Optional[PinholeCamera] = None,
        image_width: int = 640,
        image_height: int = 480,
    ):
        self.camera = camera
        self.image_width = image_width
        self.image_height = image_height

    def annotate(
        self,
        image: Optional[np.ndarray],
        snapshot: SceneSnapshot,
        predictions: Optional[Dict[str, dict]] = None,
        workspace: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Draw all annotations on an image.

        If no image is provided, creates a black canvas.

        Parameters
        ----------
        image : np.ndarray or None
            RGB/BGR image to annotate (modified in-place and returned).
        snapshot : SceneSnapshot
            Current scene state.
        predictions : dict or None
            Physics predictions per object (from PhysicsPredictor).
        workspace : dict or None
            Workspace bounds for overlay.

        Returns
        -------
        annotated : np.ndarray
            Annotated image (BGR, uint8).
        """
        if image is None:
            image = np.zeros(
                (self.image_height, self.image_width, 3), dtype=np.uint8
            )
            image[:] = (40, 40, 40)  # Dark gray background

        img = image.copy()

        # Draw each rigid body
        for name, body in snapshot.rigid_bodies.items():
            if not body.tracking_valid:
                continue

            pred = predictions.get(name, {}) if predictions else {}
            self._draw_body(img, name, body, pred, snapshot.gripper_position)

        # Draw gripper position
        if snapshot.gripper_position is not None:
            self._draw_gripper(img, snapshot.gripper_position, snapshot.gripper_open)

        # Draw info bar at bottom
        self._draw_info_bar(img, snapshot, predictions)

        return img

    def _world_to_pixel(self, pos_3d: np.ndarray) -> Optional[Tuple[int, int]]:
        """Project a 3D world point to pixel coordinates."""
        if self.camera is not None:
            pixel, valid = self.camera.project(pos_3d)
            if valid:
                return int(pixel[0]), int(pixel[1])
            return None
        else:
            # Simple top-down projection (no camera):
            # Map workspace X=[0,1.4] → pixel X=[50, width-50]
            # Map workspace Y=[-0.8,0.8] → pixel Y=[50, height-50]
            margin = 50
            w = self.image_width - 2 * margin
            h = self.image_height - 2 * margin

            px = int(margin + (pos_3d[0] / 1.4) * w)
            py = int(margin + (0.5 - pos_3d[1] / 1.6) * h)

            if 0 <= px < self.image_width and 0 <= py < self.image_height:
                return px, py
            return None

    def _draw_body(
        self,
        img: np.ndarray,
        name: str,
        body,
        pred: dict,
        gripper_pos: Optional[np.ndarray],
    ) -> None:
        """Draw annotations for a single rigid body."""
        pixel = self._world_to_pixel(body.position)
        if pixel is None:
            return

        cx, cy = pixel

        # Object marker (circle)
        stable = pred.get("stable", True)
        graspable = pred.get("graspable", True)
        marker_color = COLOR_GREEN if stable else COLOR_YELLOW
        cv2.circle(img, (cx, cy), 12, marker_color, 2)
        cv2.circle(img, (cx, cy), 3, COLOR_WHITE, -1)

        # Name label
        label = name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (cx - tw//2 - 2, cy - 25 - th), (cx + tw//2 + 2, cy - 22), (0, 0, 0), -1)
        cv2.putText(img, label, (cx - tw//2, cy - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

        # Position text
        pos = body.position
        pos_text = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
        cv2.putText(img, pos_text, (cx + 18, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_CYAN, 1)

        # Distance to gripper
        if gripper_pos is not None:
            dist = np.linalg.norm(body.position - gripper_pos)
            dist_text = f"{dist:.2f}m"
            cv2.putText(img, dist_text, (cx + 18, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1)

        # Physics status icons
        status_y = cy + 35
        if stable:
            cv2.putText(img, "stable", (cx - 15, status_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_GREEN, 1)
        else:
            cv2.putText(img, "UNSTABLE", (cx - 20, status_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_RED, 1)

        if not graspable:
            cv2.putText(img, "no-grasp", (cx - 20, status_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_RED, 1)

        # Mini axis triad
        self._draw_axis_triad(img, cx, cy, body.quaternion, length=20)

    def _draw_axis_triad(
        self,
        img: np.ndarray,
        cx: int, cy: int,
        quaternion: np.ndarray,
        length: int = 20,
    ) -> None:
        """Draw a small RGB axis triad at pixel position."""
        R = quaternion_to_rotation_matrix(
            quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        )

        axes = [
            (R[:, 0], COLOR_AXIS_X),  # X axis = red
            (R[:, 1], COLOR_AXIS_Y),  # Y axis = green
            (R[:, 2], COLOR_AXIS_Z),  # Z axis = blue
        ]

        for axis_dir, color in axes:
            # Project axis direction to 2D (simple: use X and Y components)
            ex = int(cx + axis_dir[0] * length)
            ey = int(cy - axis_dir[2] * length)  # Flip Z for screen coords
            cv2.arrowedLine(img, (cx, cy), (ex, ey), color, 1, tipLength=0.3)

    def _draw_gripper(
        self,
        img: np.ndarray,
        position: np.ndarray,
        is_open: Optional[bool],
    ) -> None:
        """Draw gripper indicator."""
        pixel = self._world_to_pixel(position)
        if pixel is None:
            return

        cx, cy = pixel
        size = 15

        # Draw gripper as a diamond
        pts = np.array([
            [cx, cy - size], [cx + size, cy],
            [cx, cy + size], [cx - size, cy],
        ], dtype=np.int32)
        cv2.polylines(img, [pts], True, COLOR_WHITE, 2)

        state_text = "OPEN" if is_open else "CLOSED"
        cv2.putText(img, f"gripper: {state_text}", (cx + 20, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1)

    def _draw_info_bar(
        self,
        img: np.ndarray,
        snapshot: SceneSnapshot,
        predictions: Optional[Dict[str, dict]],
    ) -> None:
        """Draw a status bar at the bottom of the image."""
        h, w = img.shape[:2]
        bar_height = 30
        cv2.rectangle(img, (0, h - bar_height), (w, h), (0, 0, 0), -1)

        n_bodies = sum(1 for b in snapshot.rigid_bodies.values() if b.tracking_valid)
        info = f"t={snapshot.timestamp:.2f} | {n_bodies} objects tracked"

        if predictions:
            n_stable = sum(1 for p in predictions.values() if p.get("stable", True))
            n_graspable = sum(1 for p in predictions.values() if p.get("graspable", False))
            info += f" | {n_stable} stable | {n_graspable} graspable"

        cv2.putText(img, info, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
