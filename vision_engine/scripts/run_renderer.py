#!/usr/bin/env python3
"""
scripts/run_renderer.py — OpenGL 3D Scene Renderer

Connects to OptiTrack V120:Trio and renders tracked rigid bodies in a
real-time 3D window. The CS-100 Calibration Square is rendered as an
L-shape frame (3 colored spheres + connecting lines).

The camera auto-centers on tracked objects so they are always visible.
If calibration.yaml exists, it is automatically loaded and applied.

Controls
--------
    1       Birds-eye camera (top-down)
    2       Robot-view camera (behind robot)
    3       Free orbit camera (mouse drag/scroll)
    S       Save snapshot to output/snapshot.png
    Q/Esc   Quit

Usage
-----
    cd vision_engine
    python scripts/run_renderer.py                  # Live OptiTrack
    python scripts/run_renderer.py --no-optitrack   # Demo mode
    python scripts/run_renderer.py --verbose        # Print positions
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
from cv.cs100_model import CS100Geometry
from graphics.renderer import SceneRenderer


def create_cs100_model(objects_cfg: dict) -> CS100Geometry:
    """Create CS100Geometry from config if any object uses cs100_lshape rendering."""
    for name, info in objects_cfg.items():
        if info.get("render_as") == "cs100_lshape":
            return CS100Geometry(
                short_arm_length=info.get("short_arm_length", 0.08),
                long_arm_length=info.get("long_arm_length", 0.10),
            )
    return None


def create_demo_snapshot() -> SceneSnapshot:
    """Animated demo snapshot with CS-100 L-shape moving on a surface."""
    t = time.time()

    # Smooth rotation + gentle translation to demo L-shape tracking
    angle = t * 0.5
    qz = np.sin(angle * 0.5)
    qw = np.cos(angle * 0.5)

    bodies = {
        "CS-100": RigidBodyState(
            name="CS-100", id=1,
            position=np.array([
                0.1 * np.sin(t * 0.3),
                0.1 * np.cos(t * 0.2),
                0.005,
            ]),
            quaternion=np.array([0.0, 0.0, qz, qw]),
            timestamp=t, tracking_valid=True,
        ),
    }
    return SceneSnapshot(
        timestamp=t,
        rigid_bodies=bodies,
        gripper_position=None,
        gripper_open=True,
    )


def main():
    parser = argparse.ArgumentParser(description="OpenGL 3D scene renderer")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config", "scene_config.yaml"),
    )
    parser.add_argument("--headless", action="store_true", help="No window (FBO only)")
    parser.add_argument("--no-optitrack", action="store_true", help="Demo mode")
    parser.add_argument("--verbose", action="store_true", help="Print object positions")
    args = parser.parse_args()

    config = load_config(args.config)
    workspace_cfg = config.get("workspace", {})
    objects_cfg = config.get("objects", {})
    renderer_cfg = config.get("renderer", {})

    # ── Load calibration ──
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    calibration_transform = load_calibration(config, base_dir)
    if calibration_transform is None:
        print("[Renderer] No calibration loaded — using raw OptiTrack coordinates.")
    else:
        print("[Renderer] Calibration loaded.")

    # ── Create CS-100 geometry model ──
    cs100_model = create_cs100_model(objects_cfg)
    if cs100_model is not None:
        print(f"[Renderer] CS-100 L-shape: short={cs100_model.short_arm_length*100:.0f}cm, "
              f"long={cs100_model.long_arm_length*100:.0f}cm")

    # ── Create renderer ──
    renderer = SceneRenderer(
        width=renderer_cfg.get("window_width", 1280),
        height=renderer_cfg.get("window_height", 720),
        headless=args.headless,
        background_color=tuple(renderer_cfg.get("background_color", [0.15, 0.15, 0.15])),
        cs100_model=cs100_model,
    )
    renderer.setup_scene(workspace_cfg)

    os.makedirs("output", exist_ok=True)

    # ── Connect to OptiTrack ──
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
            if aggregator is not None:
                snapshot = aggregator.capture_snapshot()
            else:
                snapshot = create_demo_snapshot()

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
                            q = body.quaternion
                            print(f"  {name}: pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) "
                                  f"quat=({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f})")

                            # Validate CS-100 geometry if applicable
                            if (cs100_model is not None
                                    and objects_cfg.get(name, {}).get("render_as") == "cs100_lshape"):
                                markers = cs100_model.compute_marker_positions(p, q)
                                v = cs100_model.validate_geometry(markers)
                                print(f"    L-shape: short={v['short_arm_dist_m']*100:.2f}cm "
                                      f"long={v['long_arm_dist_m']*100:.2f}cm "
                                      f"valid={'YES' if v['is_valid'] else 'NO'}")

                frame_count = 0
                start_time = time.time()

    except KeyboardInterrupt:
        print("\n[Renderer] Interrupted.")

    if client:
        client.stop()
    renderer.shutdown()


if __name__ == "__main__":
    main()
