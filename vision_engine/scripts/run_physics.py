#!/usr/bin/env python3
"""
scripts/run_physics.py — PyBullet Physics Prediction Test

Connects to OptiTrack, mirrors the scene in PyBullet, and prints
physics predictions. Calibration tools (e.g., CS-100) are automatically
excluded from physics predictions.

If calibration.yaml exists, it is loaded and applied to transform
OptiTrack coordinates into the calibrated world frame.

Usage
-----
    cd vision_engine
    python scripts/run_physics.py
    python scripts/run_physics.py --no-optitrack   # Demo mode with fake data

Expected Output
---------------
    [Physics] Predictions at t=1712345678.42:
      cube_1:
        stable=True, displacement=0.001m
        graspable=True (fits gripper), suggested_height=0.75m
        distance_to_gripper=0.23m
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import load_config, load_calibration, is_calibration_tool
from cv.optitrack_client import OptiTrackClient, RigidBodyState
from cv.scene_state import SceneStateAggregator, SceneSnapshot
from physics.world import PhysicsWorld
from physics.predictions import PhysicsPredictor
from physics.body_registry import BodyRegistry


def create_demo_snapshot() -> SceneSnapshot:
    """Fake snapshot for testing without OptiTrack."""
    t = time.time()
    bodies = {
        "cube_1": RigidBodyState(
            name="cube_1", id=1,
            position=np.array([0.6, -0.1, 0.725]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            timestamp=t, tracking_valid=True,
        ),
        "cylinder_1": RigidBodyState(
            name="cylinder_1", id=2,
            position=np.array([0.75, 0.15, 0.74]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            timestamp=t, tracking_valid=True,
        ),
    }
    return SceneSnapshot(
        timestamp=t, rigid_bodies=bodies,
        gripper_position=np.array([0.6, 0.0, 0.85]),
        gripper_open=True,
    )


def filter_calibration_tools(snapshot: SceneSnapshot, config: dict) -> SceneSnapshot:
    """Return a new snapshot with calibration tool bodies removed."""
    filtered_bodies = {
        name: body for name, body in snapshot.rigid_bodies.items()
        if not is_calibration_tool(name, config)
    }
    return SceneSnapshot(
        timestamp=snapshot.timestamp,
        rigid_bodies=filtered_bodies,
        gripper_position=snapshot.gripper_position,
        gripper_open=snapshot.gripper_open,
        rgb_image=snapshot.rgb_image,
        depth_image=snapshot.depth_image,
    )


def main():
    parser = argparse.ArgumentParser(description="Physics prediction test")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config", "scene_config.yaml"),
    )
    parser.add_argument("--no-optitrack", action="store_true")
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()

    config = load_config(args.config)
    physics_cfg = config.get("physics", {})
    objects_cfg = config.get("objects", {})
    workspace_cfg = config.get("workspace", {})

    # Load calibration
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    calibration_transform = load_calibration(config, base_dir)

    # Create physics engine
    registry = BodyRegistry(objects_cfg)
    world = PhysicsWorld(
        gravity=physics_cfg.get("gravity", -9.81),
        table_height=workspace_cfg.get("table_height", 0.7),
    )
    predictor = PhysicsPredictor(
        world=world,
        registry=registry,
        gripper_width=physics_cfg.get("gripper_width", 0.08),
        prediction_horizon=physics_cfg.get("prediction_horizon", 2.0),
    )

    # Connect OptiTrack or use demo
    client = None
    aggregator = None
    if not args.no_optitrack:
        optitrack_cfg = config.get("optitrack", {})
        client = OptiTrackClient(
            server_ip=optitrack_cfg.get("server_ip", "192.168.0.101"),
            local_ip=optitrack_cfg.get("local_ip", "0.0.0.0"),
            multicast_ip=optitrack_cfg.get("multicast_ip", "239.255.42.99"),
            command_port=optitrack_cfg.get("command_port", 1510),
            data_port=optitrack_cfg.get("data_port", 1511),
        )
        client.start()
        aggregator = SceneStateAggregator(
            optitrack_client=client,
            calibration_transform=calibration_transform,
        )
        time.sleep(1.0)

    print(f"\n[Physics] Running predictions every 2 seconds for {args.duration}s...")
    print(f"  Prediction horizon: {physics_cfg.get('prediction_horizon', 2.0)}s")
    print(f"  Gripper width: {physics_cfg.get('gripper_width', 0.08)}m\n")

    start = time.time()
    try:
        while time.time() - start < args.duration:
            snapshot = aggregator.capture_snapshot() if aggregator else create_demo_snapshot()

            # Filter out calibration tools before physics
            physics_snapshot = filter_calibration_tools(snapshot, config)
            predictions = predictor.predict_all(physics_snapshot)

            print(f"--- Predictions at t={snapshot.timestamp:.2f} ---")
            if not predictions:
                print("  (no non-calibration objects tracked)")
            for name, pred in sorted(predictions.items()):
                stable_str = "stable" if pred["stable"] else f"UNSTABLE ({pred['displacement']*100:.1f}cm)"
                grasp_str = f"graspable ({pred['grasp_reason']})" if pred["graspable"] else f"NOT graspable ({pred['grasp_reason']})"
                dist_str = f"{pred['distance_to_gripper']:.3f}m" if pred["distance_to_gripper"] is not None else "N/A"

                print(f"  {name}:")
                print(f"    {stable_str}")
                print(f"    {grasp_str}, suggested_height={pred['suggested_grasp_height']:.3f}m")
                print(f"    distance_to_gripper={dist_str}")
            print()

            time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n[Physics] Interrupted.")

    world.disconnect()
    if client:
        client.stop()


if __name__ == "__main__":
    main()
