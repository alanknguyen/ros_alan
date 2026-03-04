"""
vision_engine/physics/predictions.py — Unified Physics Prediction Engine

Wraps PhysicsWorld to run all prediction types in one call and produce
a structured results dictionary consumed by scene_to_language.py.

Predictions per object:
  - Stability: will the object remain in place?
  - Grasp feasibility: can the gripper pick it up?
  - Distance to gripper: how far is it from the end-effector?
"""

import numpy as np
from typing import Dict, Optional

from cv.scene_state import SceneSnapshot
from physics.world import PhysicsWorld
from physics.body_registry import BodyRegistry


class PhysicsPredictor:
    """
    Runs all physics predictions on the current scene.

    Parameters
    ----------
    world : PhysicsWorld
        The PyBullet physics world.
    registry : BodyRegistry
        Object shape/mass registry.
    gripper_width : float
        Maximum gripper opening in meters (default 0.08).
    prediction_horizon : float
        Stability simulation horizon in seconds (default 2.0).
    """

    def __init__(
        self,
        world: PhysicsWorld,
        registry: BodyRegistry,
        gripper_width: float = 0.08,
        prediction_horizon: float = 2.0,
    ):
        self.world = world
        self.registry = registry
        self.gripper_width = gripper_width
        self.prediction_horizon = prediction_horizon

    def predict_all(self, snapshot: SceneSnapshot) -> Dict[str, dict]:
        """
        Run all predictions on the current scene.

        Parameters
        ----------
        snapshot : SceneSnapshot
            Current scene state.

        Returns
        -------
        predictions : dict[str, dict]
            Per-object predictions with keys:
                - stable (bool)
                - displacement (float, meters)
                - graspable (bool)
                - grasp_reason (str)
                - suggested_grasp_height (float)
                - distance_to_gripper (float or None)
                - min_dimension (float)
        """
        # Sync physics world with current scene
        self.world.sync_from_snapshot(snapshot, self.registry)

        # Run stability prediction
        stability = self.world.step_prediction(self.prediction_horizon)

        # Build unified predictions
        predictions = {}
        for name, body in snapshot.rigid_bodies.items():
            if not body.tracking_valid:
                continue

            # Stability
            stab = stability.get(name, {"stable": True, "displacement": 0.0})

            # Grasp feasibility
            grasp = self.world.check_grasp_feasibility(
                name, self.registry, self.gripper_width
            )

            # Distance to gripper
            dist = None
            if snapshot.gripper_position is not None:
                dist = float(np.linalg.norm(
                    body.position - snapshot.gripper_position
                ))

            predictions[name] = {
                "stable": stab["stable"],
                "displacement": stab["displacement"],
                "graspable": grasp["graspable"],
                "grasp_reason": grasp["reason"],
                "suggested_grasp_height": grasp["suggested_height"],
                "min_dimension": grasp.get("min_dimension", 0.0),
                "distance_to_gripper": dist,
            }

        return predictions
