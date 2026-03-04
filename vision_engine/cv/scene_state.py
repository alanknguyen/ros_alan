"""
vision_engine/cv/scene_state.py — Scene State Data Structures & Aggregator

Defines the core data structures that represent the state of the physical scene
at a moment in time, and an aggregator that combines data from multiple sources
(OptiTrack, RGB-D camera, robot state) into synchronized snapshots.

Data Flow
---------
    OptiTrack rigid bodies ─┐
    RGB-D camera images ────┼──► SceneStateAggregator ──► SceneSnapshot
    Robot gripper state ────┘                              (frozen at time t)

The SceneSnapshot is the universal data container consumed by:
  - OpenGL renderer (3D visualization)
  - Physics engine (stability/grasp predictions)
  - Scene annotator (2D overlays)
  - Scene-to-language converter (LLM text)

Thread Safety
-------------
The aggregator uses a threading lock to protect internal state. All public
methods are thread-safe for concurrent reads/writes from different sources
(e.g., OptiTrack callback thread + main loop).
"""

import time
import copy
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional

from cv.optitrack_client import RigidBodyState


# ──────────────────────────────────────────────────────────────────────────────
# Scene Snapshot — Immutable state at time t
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SceneSnapshot:
    """
    A frozen snapshot of the entire scene at a specific moment in time.

    This is the primary data structure passed between all vision engine
    components. It captures everything known about the scene at time t.

    Attributes
    ----------
    timestamp : float
        Unix timestamp (seconds since epoch) when this snapshot was captured.

    rigid_bodies : dict[str, RigidBodyState]
        All tracked rigid bodies from OptiTrack, keyed by name.
        Positions/orientations are in the calibrated world frame (Z-up).

    gripper_position : np.ndarray or None
        Robot end-effector position (x, y, z) in world frame.
        None if robot state is not available (standalone mode).

    gripper_open : bool or None
        True if gripper is open, False if closed, None if unknown.

    rgb_image : np.ndarray or None
        RGB camera image, shape (H, W, 3), dtype uint8, BGR format.
        None if no camera is connected.

    depth_image : np.ndarray or None
        Depth image, shape (H, W), dtype float32, values in meters.
        None if no depth sensor is connected.
    """
    timestamp: float
    rigid_bodies: Dict[str, RigidBodyState]
    gripper_position: Optional[np.ndarray] = None
    gripper_open: Optional[bool] = None
    rgb_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None


# ──────────────────────────────────────────────────────────────────────────────
# Scene State Aggregator
# ──────────────────────────────────────────────────────────────────────────────

class SceneStateAggregator:
    """
    Combines data from multiple sources into synchronized SceneSnapshots.

    Each data source updates its portion of the state independently:
      - OptiTrack client pushes rigid body updates (via callback or poll)
      - Camera feeds push RGB and depth images
      - Robot interface pushes gripper state

    When capture_snapshot() is called, the aggregator freezes a deep copy of
    all current data into a SceneSnapshot. This snapshot is immutable and
    safe to pass to downstream components without synchronization concerns.

    Parameters
    ----------
    optitrack_client : OptiTrackClient or None
        If provided, rigid body data is polled from this client.
        If None, rigid bodies must be set manually via set_rigid_bodies().

    calibration_transform : np.ndarray or None
        4×4 homogeneous transform from OptiTrack world frame to the desired
        output frame (e.g., robot base frame). If None, no transform is applied
        (OptiTrack coordinates are used directly after Y-up→Z-up conversion).
    """

    def __init__(self, optitrack_client=None, calibration_transform=None):
        self._optitrack_client = optitrack_client
        self._calibration_transform = calibration_transform

        # Internal state (protected by lock)
        self._lock = threading.Lock()
        self._rigid_bodies: Dict[str, RigidBodyState] = {}
        self._rgb_image: Optional[np.ndarray] = None
        self._depth_image: Optional[np.ndarray] = None
        self._gripper_position: Optional[np.ndarray] = None
        self._gripper_open: Optional[bool] = None

    def capture_snapshot(self) -> SceneSnapshot:
        """
        Capture a frozen snapshot of the current scene state.

        If an OptiTrack client is connected, rigid body data is polled from it.
        If a calibration transform is set, rigid body positions are transformed.

        Returns
        -------
        snapshot : SceneSnapshot
            Deep copy of all current scene data, safe for downstream use.
        """
        # Poll OptiTrack if connected
        if self._optitrack_client is not None:
            bodies = self._optitrack_client.get_rigid_bodies()
            if bodies:
                # Apply calibration transform if available
                if self._calibration_transform is not None:
                    bodies = self._apply_calibration(bodies)
                with self._lock:
                    self._rigid_bodies = bodies

        with self._lock:
            snapshot = SceneSnapshot(
                timestamp=time.time(),
                rigid_bodies=copy.deepcopy(self._rigid_bodies),
                gripper_position=(
                    self._gripper_position.copy()
                    if self._gripper_position is not None
                    else None
                ),
                gripper_open=self._gripper_open,
                rgb_image=(
                    self._rgb_image.copy()
                    if self._rgb_image is not None
                    else None
                ),
                depth_image=(
                    self._depth_image.copy()
                    if self._depth_image is not None
                    else None
                ),
            )

        return snapshot

    # ──────────────────────────────────────────────────────────────────────
    # Data Source Setters (called by camera/robot feeds)
    # ──────────────────────────────────────────────────────────────────────

    def set_rgb_image(self, image: np.ndarray) -> None:
        """
        Update the latest RGB image.

        Parameters
        ----------
        image : np.ndarray, shape (H, W, 3)
            BGR image from camera, dtype uint8.
        """
        with self._lock:
            self._rgb_image = image

    def set_depth_image(self, image: np.ndarray) -> None:
        """
        Update the latest depth image.

        Parameters
        ----------
        image : np.ndarray, shape (H, W)
            Depth values in meters, dtype float32.
        """
        with self._lock:
            self._depth_image = image

    def set_gripper_state(
        self,
        position: Optional[np.ndarray] = None,
        is_open: Optional[bool] = None,
    ) -> None:
        """
        Update robot gripper state.

        Parameters
        ----------
        position : np.ndarray or None, shape (3,)
            End-effector position (x, y, z) in world frame.
        is_open : bool or None
            True if gripper is open, False if closed.
        """
        with self._lock:
            if position is not None:
                self._gripper_position = np.asarray(position, dtype=np.float64)
            if is_open is not None:
                self._gripper_open = is_open

    def set_rigid_bodies(self, bodies: Dict[str, RigidBodyState]) -> None:
        """
        Manually set rigid body states (for testing without OptiTrack).

        Parameters
        ----------
        bodies : dict[str, RigidBodyState]
            Rigid body states keyed by name.
        """
        with self._lock:
            self._rigid_bodies = copy.deepcopy(bodies)

    def set_calibration_transform(self, transform: np.ndarray) -> None:
        """
        Set or update the calibration transform.

        Parameters
        ----------
        transform : np.ndarray, shape (4, 4)
            Homogeneous transform from OptiTrack frame to output frame.
        """
        self._calibration_transform = np.asarray(transform, dtype=np.float64)

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _apply_calibration(
        self, bodies: Dict[str, RigidBodyState]
    ) -> Dict[str, RigidBodyState]:
        """
        Apply the calibration transform to all rigid body positions.

        Transforms positions from OptiTrack world frame to the calibrated
        output frame (e.g., robot base frame).
        """
        from cv.transforms import apply_transform, quaternion_multiply, rotation_matrix_to_quaternion

        T = self._calibration_transform
        R = T[:3, :3]

        # Convert rotation matrix to quaternion for orientation transform
        r_quat = np.array(rotation_matrix_to_quaternion(R))

        calibrated = {}
        for name, state in bodies.items():
            new_pos = apply_transform(T, state.position)
            new_quat = quaternion_multiply(r_quat, state.quaternion)
            # Normalize quaternion
            new_quat = new_quat / np.linalg.norm(new_quat)

            calibrated[name] = RigidBodyState(
                name=state.name,
                id=state.id,
                position=new_pos,
                quaternion=new_quat,
                timestamp=state.timestamp,
                tracking_valid=state.tracking_valid,
            )

        return calibrated
