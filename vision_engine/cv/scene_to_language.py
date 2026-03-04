"""
vision_engine/cv/scene_to_language.py — Scene State → Structured Text for LLM/VLM/VLA

Converts a SceneSnapshot + physics predictions into a structured text
description that is prepended to the LLM/VLM/VLA prompt before action
decision.

The output format is designed to be:
  - Easily parseable by language models
  - Compact (minimize token usage)
  - Complete (all info needed for spatial reasoning)

Output Example
--------------
    SCENE STATE (t=1712345678.42):
    Objects:
      cube_1: pos=(0.412,-0.185,0.732) euler_deg=(0,90,0) color=red stable=yes graspable=yes dist_to_gripper=0.23m
      cylinder_1: pos=(0.550,0.100,0.710) euler_deg=(0,0,45) color=blue stable=yes graspable=yes dist_to_gripper=0.15m
    Robot:
      gripper_pos=(0.600,0.000,0.600) gripper_state=open
    Physics:
      cube_1: stable, no predicted motion
      cylinder_1: may roll if pushed (displacement=0.015m over 2s)
    Workspace: x=[0.4,1.0] y=[-0.6,0.6] table_z=0.7
    Nearest: cylinder_1 (0.15m from gripper)
"""

import numpy as np
from typing import Dict, Optional

from cv.scene_state import SceneSnapshot
from cv.transforms import euler_degrees_from_quaternion


# Color name lookup (approximate RGB float → name)
_COLOR_NAMES = {
    (1.0, 0.0, 0.0): "red",
    (0.0, 1.0, 0.0): "green",
    (0.0, 0.0, 1.0): "blue",
    (1.0, 1.0, 0.0): "yellow",
    (1.0, 0.5, 0.0): "orange",
    (0.5, 0.0, 0.5): "purple",
    (1.0, 1.0, 1.0): "white",
    (0.5, 0.5, 0.5): "gray",
    (0.0, 0.0, 0.0): "black",
}


def _color_name(rgb: list) -> str:
    """Find the closest named color for an RGB triple."""
    if not rgb or len(rgb) < 3:
        return "unknown"
    best_name = "unknown"
    best_dist = float("inf")
    for ref_rgb, name in _COLOR_NAMES.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb, ref_rgb))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


class SceneDescriber:
    """
    Converts scene state + physics predictions to structured text for LLMs.

    Parameters
    ----------
    object_registry : dict
        Object definitions from scene_config.yaml "objects" section.
    workspace : dict or None
        Workspace bounds from scene_config.yaml.
    """

    def __init__(
        self,
        object_registry: Optional[dict] = None,
        workspace: Optional[dict] = None,
    ):
        self.object_registry = object_registry or {}
        self.workspace = workspace or {}

    def describe(
        self,
        snapshot: SceneSnapshot,
        predictions: Optional[Dict[str, dict]] = None,
    ) -> str:
        """
        Generate a structured text description of the scene.

        Parameters
        ----------
        snapshot : SceneSnapshot
            Current scene state.
        predictions : dict or None
            Physics predictions per object.

        Returns
        -------
        description : str
            Structured text for the LLM prompt.
        """
        lines = []
        lines.append(f"SCENE STATE (t={snapshot.timestamp:.2f}):")

        # ── Objects ──
        lines.append("Objects:")
        nearest_name = None
        nearest_dist = float("inf")

        for name, body in sorted(snapshot.rigid_bodies.items()):
            if not body.tracking_valid:
                continue

            p = body.position
            q = body.quaternion
            roll, pitch, yaw = euler_degrees_from_quaternion(q[0], q[1], q[2], q[3])

            # Get color from registry
            obj_info = self.object_registry.get(name, {})
            color = _color_name(obj_info.get("color", [0.5, 0.5, 0.5]))
            shape = obj_info.get("shape", "unknown")

            # Physics predictions
            pred = predictions.get(name, {}) if predictions else {}
            stable_str = "yes" if pred.get("stable", True) else "no"
            graspable_str = "yes" if pred.get("graspable", True) else "no"

            # Distance to gripper
            dist_str = "N/A"
            if snapshot.gripper_position is not None:
                dist = float(np.linalg.norm(body.position - snapshot.gripper_position))
                dist_str = f"{dist:.2f}m"
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_name = name

            line = (
                f"  {name}: shape={shape} pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                f"euler_deg=({roll:.0f},{pitch:.0f},{yaw:.0f}) color={color} "
                f"stable={stable_str} graspable={graspable_str} "
                f"dist_to_gripper={dist_str}"
            )
            lines.append(line)

        # ── Robot ──
        lines.append("Robot:")
        if snapshot.gripper_position is not None:
            gp = snapshot.gripper_position
            state = "open" if snapshot.gripper_open else "closed"
            lines.append(
                f"  gripper_pos=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) "
                f"gripper_state={state}"
            )
        else:
            lines.append("  gripper_state=unknown")

        # ── Physics ──
        if predictions:
            lines.append("Physics:")
            for name, pred in sorted(predictions.items()):
                if pred.get("stable", True):
                    disp = pred.get("displacement", 0.0)
                    if disp < 0.005:
                        phys_text = "stable, no predicted motion"
                    else:
                        phys_text = f"mostly stable (displacement={disp*100:.1f}cm over prediction horizon)"
                else:
                    disp = pred.get("displacement", 0.0)
                    phys_text = f"UNSTABLE, predicted displacement={disp*100:.1f}cm"
                lines.append(f"  {name}: {phys_text}")

        # ── Workspace ──
        if self.workspace:
            ws = self.workspace
            lines.append(
                f"Workspace: "
                f"x=[{ws.get('x_min', 0.4)},{ws.get('x_max', 1.0)}] "
                f"y=[{ws.get('y_min', -0.6)},{ws.get('y_max', 0.6)}] "
                f"table_z={ws.get('table_height', 0.7)}"
            )

        # ── Nearest Object ──
        if nearest_name is not None:
            lines.append(f"Nearest: {nearest_name} ({nearest_dist:.2f}m from gripper)")

        return "\n".join(lines)
