"""
vision_engine/physics/world.py — PyBullet Physics World Mirror

Mirrors the OptiTrack scene in a PyBullet physics simulation for:
  - Stability analysis (will objects fall/topple?)
  - Grasp feasibility (does the object fit the gripper?)
  - Collision checking (is the path to an object clear?)
  - Drop simulation (where does an object land if released?)

The physics world runs in DIRECT mode (headless, no GUI) for performance.
It is re-synchronized from OptiTrack data before each prediction cycle.

Architecture
------------
    SceneSnapshot → sync_from_snapshot() → PyBullet world mirrors real scene
                  → step_prediction()    → Simulate forward N seconds
                  → check_grasp_feasibility() → Geometric + physics check
                  → check_collision()    → Ray cast along planned path
                  → simulate_drop()      → Release object, observe landing
"""

import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, Optional, Tuple

from cv.scene_state import SceneSnapshot
from cv.transforms import quaternion_to_euler
from physics.body_registry import BodyRegistry


class PhysicsWorld:
    """
    PyBullet physics simulation mirroring the real scene.

    Parameters
    ----------
    gravity : float
        Gravitational acceleration in m/s² (default -9.81, negative = down).
    table_height : float
        Table surface height in meters (default 0.7).
    time_step : float
        Physics simulation time step in seconds (default 1/240).
    """

    def __init__(
        self,
        gravity: float = -9.81,
        table_height: float = 0.7,
        time_step: float = 1.0 / 240.0,
    ):
        self.gravity = gravity
        self.table_height = table_height
        self.time_step = time_step

        # Connect PyBullet in DIRECT mode (headless, no GUI window)
        self._physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, gravity, physicsClientId=self._physics_client)
        p.setTimeStep(time_step, physicsClientId=self._physics_client)

        # Create table as a static collision plane at table_height
        # Using a thin box instead of infinite plane for better collision
        table_half_extents = [1.0, 1.0, 0.01]
        table_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=table_half_extents,
            physicsClientId=self._physics_client,
        )
        self._table_id = p.createMultiBody(
            baseMass=0,  # Static body
            baseCollisionShapeIndex=table_collision,
            basePosition=[0.5, 0.0, table_height - 0.01],
            physicsClientId=self._physics_client,
        )

        # Tracked PyBullet body IDs: rigid_body_name → pybullet_id
        self._body_ids: Dict[str, int] = {}

        print("[Physics] PyBullet world initialized (DIRECT mode)")

    def sync_from_snapshot(
        self,
        snapshot: SceneSnapshot,
        registry: BodyRegistry,
    ) -> None:
        """
        Synchronize the physics world with the current OptiTrack scene.

        Creates new PyBullet bodies for newly-seen rigid bodies, and
        updates positions/orientations for existing ones.

        Parameters
        ----------
        snapshot : SceneSnapshot
            Current scene state from the aggregator.
        registry : BodyRegistry
            Object shape/mass definitions.
        """
        # Track which bodies are still present
        seen_names = set()

        for name, body in snapshot.rigid_bodies.items():
            if not body.tracking_valid:
                continue
            seen_names.add(name)

            # Convert quaternion from (x,y,z,w) to PyBullet's (x,y,z,w) — same format
            quat = [body.quaternion[0], body.quaternion[1],
                    body.quaternion[2], body.quaternion[3]]
            pos = body.position.tolist()

            if name in self._body_ids:
                # Update existing body position/orientation
                p.resetBasePositionAndOrientation(
                    self._body_ids[name], pos, quat,
                    physicsClientId=self._physics_client,
                )
                # Reset velocity (we're tracking, not simulating)
                p.resetBaseVelocity(
                    self._body_ids[name], [0, 0, 0], [0, 0, 0],
                    physicsClientId=self._physics_client,
                )
            else:
                # Create new body
                shape_info = registry.get(name)
                pb_id = self._create_body(name, pos, quat, shape_info)
                if pb_id is not None:
                    self._body_ids[name] = pb_id

        # Remove bodies that are no longer tracked
        for name in list(self._body_ids.keys()):
            if name not in seen_names:
                p.removeBody(self._body_ids[name], physicsClientId=self._physics_client)
                del self._body_ids[name]

    def _create_body(
        self,
        name: str,
        position: list,
        quaternion: list,
        shape_info: dict,
    ) -> Optional[int]:
        """Create a PyBullet collision body from shape info."""
        shape_type = shape_info.get("shape", "sphere")
        mass = shape_info.get("mass", 0.1)

        if shape_type == "cube":
            size = shape_info.get("size", [0.05, 0.05, 0.05])
            half_extents = [s / 2 for s in size]
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=self._physics_client,
            )
        elif shape_type == "cylinder":
            radius = shape_info.get("radius", 0.025)
            height = shape_info.get("height", 0.08)
            collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=radius,
                height=height,
                physicsClientId=self._physics_client,
            )
        elif shape_type == "sphere":
            radius = shape_info.get("radius", 0.02)
            collision = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=radius,
                physicsClientId=self._physics_client,
            )
        else:
            return None

        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision,
            basePosition=position,
            baseOrientation=quaternion,
            physicsClientId=self._physics_client,
        )

        # Set friction
        p.changeDynamics(
            body_id, -1,
            lateralFriction=0.5,
            restitution=0.3,
            physicsClientId=self._physics_client,
        )

        return body_id

    def step_prediction(self, horizon_sec: float = 2.0) -> Dict[str, dict]:
        """
        Simulate physics forward and check object stability.

        Saves current state, steps physics for horizon_sec, records final
        positions, then restores original state.

        Parameters
        ----------
        horizon_sec : float
            How far into the future to simulate (seconds).

        Returns
        -------
        predictions : dict[str, dict]
            Per-object predictions:
                {name: {stable: bool, displacement: float,
                        predicted_pos: (3,), initial_pos: (3,)}}
        """
        # Save current state
        saved_states = {}
        for name, body_id in self._body_ids.items():
            pos, orn = p.getBasePositionAndOrientation(
                body_id, physicsClientId=self._physics_client
            )
            vel, ang_vel = p.getBaseVelocity(
                body_id, physicsClientId=self._physics_client
            )
            saved_states[name] = {
                "pos": list(pos), "orn": list(orn),
                "vel": list(vel), "ang_vel": list(ang_vel),
            }

        # Step physics forward
        num_steps = int(horizon_sec / self.time_step)
        for _ in range(num_steps):
            p.stepSimulation(physicsClientId=self._physics_client)

        # Record predicted positions
        predictions = {}
        for name, body_id in self._body_ids.items():
            pred_pos, pred_orn = p.getBasePositionAndOrientation(
                body_id, physicsClientId=self._physics_client
            )
            initial_pos = np.array(saved_states[name]["pos"])
            final_pos = np.array(pred_pos)
            displacement = float(np.linalg.norm(final_pos - initial_pos))

            predictions[name] = {
                "stable": displacement < 0.02,  # <2cm = stable
                "displacement": displacement,
                "predicted_pos": final_pos,
                "initial_pos": initial_pos,
            }

        # Restore original state
        for name, body_id in self._body_ids.items():
            state = saved_states[name]
            p.resetBasePositionAndOrientation(
                body_id, state["pos"], state["orn"],
                physicsClientId=self._physics_client,
            )
            p.resetBaseVelocity(
                body_id, state["vel"], state["ang_vel"],
                physicsClientId=self._physics_client,
            )

        return predictions

    def check_grasp_feasibility(
        self,
        body_name: str,
        registry: BodyRegistry,
        gripper_width: float = 0.08,
    ) -> dict:
        """
        Check if an object can be grasped by a top-down gripper.

        Parameters
        ----------
        body_name : str
            Name of the rigid body to check.
        registry : BodyRegistry
            Object definitions.
        gripper_width : float
            Maximum gripper opening in meters.

        Returns
        -------
        result : dict
            {graspable: bool, reason: str, suggested_height: float}
        """
        shape_info = registry.get(body_name)
        shape = shape_info.get("shape", "sphere")

        # Determine the object's smallest graspable dimension
        if shape == "cube":
            size = shape_info.get("size", [0.05, 0.05, 0.05])
            min_dim = min(size[0], size[1])  # X-Y plane dimensions
            grasp_height = size[2]
        elif shape == "cylinder":
            radius = shape_info.get("radius", 0.025)
            min_dim = radius * 2
            grasp_height = shape_info.get("height", 0.08)
        elif shape == "sphere":
            radius = shape_info.get("radius", 0.02)
            min_dim = radius * 2
            grasp_height = radius * 2
        else:
            return {"graspable": False, "reason": "unknown shape", "suggested_height": 0.0}

        graspable = min_dim < gripper_width
        reason = "fits gripper" if graspable else f"too wide ({min_dim:.3f}m > {gripper_width:.3f}m)"

        # Suggested grasp height: object top - half grasp depth
        if body_name in self._body_ids:
            pos, _ = p.getBasePositionAndOrientation(
                self._body_ids[body_name],
                physicsClientId=self._physics_client,
            )
            suggested_height = pos[2] + grasp_height / 2
        else:
            suggested_height = self.table_height + grasp_height

        return {
            "graspable": graspable,
            "reason": reason,
            "suggested_height": suggested_height,
            "min_dimension": min_dim,
        }

    def check_collision(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
    ) -> dict:
        """
        Check for collisions along a straight-line path using ray casting.

        Parameters
        ----------
        start_pos : np.ndarray, shape (3,)
            Start position of the path.
        end_pos : np.ndarray, shape (3,)
            End position of the path.

        Returns
        -------
        result : dict
            {clear: bool, hit_object: str or None, hit_position: (3,) or None}
        """
        result = p.rayTest(
            start_pos.tolist(), end_pos.tolist(),
            physicsClientId=self._physics_client,
        )

        if result and result[0][0] >= 0:
            hit_body_id = result[0][0]
            hit_pos = np.array(result[0][3])

            # Find name
            hit_name = None
            for name, bid in self._body_ids.items():
                if bid == hit_body_id:
                    hit_name = name
                    break

            return {
                "clear": False,
                "hit_object": hit_name or "table",
                "hit_position": hit_pos,
            }

        return {"clear": True, "hit_object": None, "hit_position": None}

    def simulate_drop(
        self,
        body_name: str,
        drop_position: np.ndarray,
    ) -> dict:
        """
        Simulate dropping an object from a given position.

        Parameters
        ----------
        body_name : str
            Object to drop.
        drop_position : np.ndarray, shape (3,)
            Position to release the object from.

        Returns
        -------
        result : dict
            {landing_pos: (3,), drop_height: float, settled: bool}
        """
        if body_name not in self._body_ids:
            return {"landing_pos": drop_position, "drop_height": 0.0, "settled": False}

        body_id = self._body_ids[body_name]

        # Save state
        orig_pos, orig_orn = p.getBasePositionAndOrientation(
            body_id, physicsClientId=self._physics_client
        )

        # Move to drop position
        p.resetBasePositionAndOrientation(
            body_id, drop_position.tolist(), [0, 0, 0, 1],
            physicsClientId=self._physics_client,
        )
        p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0],
                           physicsClientId=self._physics_client)

        # Simulate 3 seconds of falling
        for _ in range(int(3.0 / self.time_step)):
            p.stepSimulation(physicsClientId=self._physics_client)

        # Record landing
        land_pos, land_orn = p.getBasePositionAndOrientation(
            body_id, physicsClientId=self._physics_client
        )
        land_vel, _ = p.getBaseVelocity(body_id, physicsClientId=self._physics_client)

        # Check if settled (velocity near zero)
        speed = np.linalg.norm(land_vel)
        settled = speed < 0.01

        # Restore original state
        p.resetBasePositionAndOrientation(
            body_id, list(orig_pos), list(orig_orn),
            physicsClientId=self._physics_client,
        )
        p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0],
                           physicsClientId=self._physics_client)

        return {
            "landing_pos": np.array(land_pos),
            "drop_height": float(drop_position[2] - land_pos[2]),
            "settled": settled,
        }

    def disconnect(self) -> None:
        """Disconnect PyBullet."""
        if self._physics_client >= 0:
            p.disconnect(self._physics_client)
            print("[Physics] Disconnected.")
