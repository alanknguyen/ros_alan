#!/usr/bin/env python3
"""
scripts/run_renderer.py — OpenGL 3D Scene Renderer Test

Step 3 verification script. Connects to OptiTrack V120:Trio and renders
tracked rigid bodies in a real-time 3D window.

Controls
--------
    1       Switch to birds-eye camera (top-down view)
    2       Switch to robot-view camera (behind robot)
    3       Switch to free orbit camera (mouse drag to rotate, scroll to zoom)
    S       Save snapshot to output/snapshot.png
    Q/Esc   Quit

Usage
-----
    cd vision_engine
    python scripts/run_renderer.py
    python scripts/run_renderer.py --headless     # No window, FBO only
    python scripts/run_renderer.py --no-optitrack  # Demo mode with fake data

Verification Checklist
----------------------
    1. Window opens showing table grid, workspace bounds, coordinate axes
    2. Rigid bodies appear as colored shapes at OptiTrack-reported positions
    3. Moving objects physically updates their positions in real-time
    4. Camera switching works (keys 1, 2, 3)
    5. Free camera orbits with mouse drag, zooms with scroll
    6. S key saves a screenshot
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from cv.optitrack_client import OptiTrackClient, RigidBodyState
from cv.scene_state import SceneStateAggregator, SceneSnapshot
from graphics.renderer import SceneRenderer


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_demo_snapshot() -> SceneSnapshot:
    """Create a fake snapshot for demo mode (no OptiTrack needed)."""
    t = time.time()
    bodies = {
        "cube_1": RigidBodyState(
            name="cube_1", id=1,
            position=np.array([0.6 + 0.1 * np.sin(t), -0.1, 0.725]),
            quaternion=np.array([0.0, 0.0, np.sin(t * 0.5) * 0.3, np.cos(t * 0.5)]),
            timestamp=t, tracking_valid=True,
        ),
        "cylinder_1": RigidBodyState(
            name="cylinder_1", id=2,
            position=np.array([0.75, 0.15 + 0.05 * np.cos(t * 0.7), 0.74]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            timestamp=t, tracking_valid=True,
        ),
    }
    return SceneSnapshot(
        timestamp=t,
        rigid_bodies=bodies,
        gripper_position=np.array([0.6, 0.0, 0.85]),
        gripper_open=True,
    )


def main():
    parser = argparse.ArgumentParser(description="OpenGL 3D scene renderer test")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config", "scene_config.yaml"),
    )
    parser.add_argument("--headless", action="store_true", help="No window (FBO only)")
    parser.add_argument("--no-optitrack", action="store_true", help="Demo mode with fake data")
    args = parser.parse_args()

    config = load_config(args.config)
    workspace_cfg = config.get("workspace", {})
    objects_cfg = config.get("objects", {})
    renderer_cfg = config.get("renderer", {})

    # Create renderer
    renderer = SceneRenderer(
        width=renderer_cfg.get("window_width", 1280),
        height=renderer_cfg.get("window_height", 720),
        headless=args.headless,
        background_color=tuple(renderer_cfg.get("background_color", [0.15, 0.15, 0.15])),
    )
    renderer.setup_scene(workspace_cfg)

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Connect to OptiTrack (or use demo mode)
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
        aggregator = SceneStateAggregator(optitrack_client=client)
        time.sleep(1.0)

    print("\n[Renderer] Controls: 1=birds-eye, 2=robot-view, 3=free-orbit, S=snapshot, Q=quit\n")

    # Render loop
    frame_count = 0
    start_time = time.time()

    try:
        while not renderer.should_close():
            # Get scene data
            if aggregator is not None:
                snapshot = aggregator.capture_snapshot()
            else:
                snapshot = create_demo_snapshot()

            # Update and render
            renderer.update_scene(snapshot, objects_cfg)
            renderer.render()

            frame_count += 1

            # Print FPS every 5 seconds
            elapsed = time.time() - start_time
            if elapsed >= 5.0:
                fps = frame_count / elapsed
                n_bodies = len(snapshot.rigid_bodies)
                print(f"[Renderer] {fps:.0f} fps, {n_bodies} rigid bodies tracked")
                frame_count = 0
                start_time = time.time()

    except KeyboardInterrupt:
        print("\n[Renderer] Interrupted.")

    # Cleanup
    if client:
        client.stop()
    renderer.shutdown()


if __name__ == "__main__":
    main()
