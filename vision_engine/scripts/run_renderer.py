#!/usr/bin/env python3
"""
scripts/run_renderer.py — OpenGL 3D Scene Renderer

Connects to OptiTrack V120:Trio and renders tracked rigid bodies
(including the CS-100 Calibration Square) in a real-time 3D window.

The camera auto-centers on tracked objects so they are always visible,
regardless of where they are in the OptiTrack coordinate space.

If calibration.yaml exists, it is automatically loaded and applied
to transform OptiTrack coordinates into the calibrated world frame.

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
    python scripts/run_renderer.py                  # Live OptiTrack
    python scripts/run_renderer.py --no-optitrack   # Demo mode with fake data
    python scripts/run_renderer.py --verbose         # Print object positions
    python scripts/run_renderer.py --headless        # No window, FBO only
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import load_config, load_calibration
from cv.optitrack_client import OptiTrackClient, RigidBodyState
from cv.scene_state import SceneStateAggregator, SceneSnapshot
from graphics.renderer import SceneRenderer


def create_demo_snapshot() -> SceneSnapshot:
    """Create a fake snapshot for demo mode (no OptiTrack needed)."""
    t = time.time()
    bodies = {
        "CS-100": RigidBodyState(
            name="CS-100", id=1,
            position=np.array([0.1 * np.sin(t), 0.1 * np.cos(t * 0.7), 0.005]),
            quaternion=np.array([0.0, 0.0, np.sin(t * 0.3) * 0.1, 1.0]),
            timestamp=t, tracking_valid=True,
        ),
        "cube_1": RigidBodyState(
            name="cube_1", id=2,
            position=np.array([0.3 + 0.05 * np.sin(t * 0.5), -0.1, 0.025]),
            quaternion=np.array([0.0, 0.0, np.sin(t * 0.5) * 0.3, np.cos(t * 0.5)]),
            timestamp=t, tracking_valid=True,
        ),
    }
    return SceneSnapshot(
        timestamp=t,
        rigid_bodies=bodies,
        gripper_position=np.array([0.2, 0.0, 0.15]),
        gripper_open=True,
    )


def main():
    parser = argparse.ArgumentParser(description="OpenGL 3D scene renderer")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config", "scene_config.yaml"),
    )
    parser.add_argument("--headless", action="store_true", help="No window (FBO only)")
    parser.add_argument("--no-optitrack", action="store_true", help="Demo mode with fake data")
    parser.add_argument("--verbose", action="store_true",
                        help="Print object positions every 5 seconds")
    args = parser.parse_args()

    config = load_config(args.config)
    workspace_cfg = config.get("workspace", {})
    objects_cfg = config.get("objects", {})
    renderer_cfg = config.get("renderer", {})

    # ── Load calibration if available ──
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    calibration_transform = load_calibration(config, base_dir)
    if calibration_transform is None:
        print("[Renderer] No calibration loaded — using raw OptiTrack coordinates.")
        print("  Run 'python scripts/run_calibration.py' to calibrate.")
    else:
        print("[Renderer] Calibration loaded — coordinates are in calibrated world frame.")

    # ── Create renderer ──
    renderer = SceneRenderer(
        width=renderer_cfg.get("window_width", 1280),
        height=renderer_cfg.get("window_height", 720),
        headless=args.headless,
        background_color=tuple(renderer_cfg.get("background_color", [0.15, 0.15, 0.15])),
    )
    renderer.setup_scene(workspace_cfg)

    # ── Create output directory ──
    os.makedirs("output", exist_ok=True)

    # ── Connect to OptiTrack (or use demo mode) ──
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

    print("\n[Renderer] Controls: 1=birds-eye, 2=robot-view, 3=free-orbit, S=snapshot, Q=quit\n")

    # ── Render loop ──
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

            # Print status every 5 seconds
            elapsed = time.time() - start_time
            if elapsed >= 5.0:
                fps = frame_count / elapsed
                n_bodies = sum(1 for b in snapshot.rigid_bodies.values() if b.tracking_valid)
                body_names = [n for n, b in snapshot.rigid_bodies.items() if b.tracking_valid]
                print(f"[Renderer] {fps:.0f} fps | {n_bodies} objects: {body_names}")

                if args.verbose:
                    for name, body in snapshot.rigid_bodies.items():
                        if body.tracking_valid:
                            p = body.position
                            print(f"  {name}: pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

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
