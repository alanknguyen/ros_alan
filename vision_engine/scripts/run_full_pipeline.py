#!/usr/bin/env python3
"""
scripts/run_full_pipeline.py — Full Vision Engine Pipeline

Launches all components together:
  - OptiTrack V120:Trio capture (NatNet client)
  - Scene state aggregation (time-synchronized)
  - OpenGL 3D rendering (real-time GLFW window)
  - PyBullet physics predictions (stability, grasp feasibility)
  - OpenCV 2D annotations (projected overlays)
  - Scene-to-language conversion (structured text for LLM)

Controls
--------
    1       Birds-eye camera
    2       Robot-view camera
    3       Free orbit camera (mouse drag/scroll)
    S       Save snapshot (rendered 3D + annotated 2D + scene text → output/)
    P       Print scene description to console
    Q/Esc   Quit

Usage
-----
    cd vision_engine
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --no-optitrack   # Demo mode

Output Files (on S key)
-----------------------
    output/snapshot_3d.png        — OpenGL rendered scene
    output/snapshot_2d.png        — OpenCV annotated view
    output/snapshot_scene.txt     — Scene description text for LLM
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import cv2
from cv.optitrack_client import OptiTrackClient, RigidBodyState
from cv.scene_state import SceneStateAggregator, SceneSnapshot
from cv.annotator import SceneAnnotator
from cv.scene_to_language import SceneDescriber
from graphics.renderer import SceneRenderer
from physics.world import PhysicsWorld
from physics.predictions import PhysicsPredictor
from physics.body_registry import BodyRegistry


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_demo_snapshot() -> SceneSnapshot:
    """Animated demo snapshot for testing without OptiTrack."""
    t = time.time()
    bodies = {
        "cube_1": RigidBodyState(
            name="cube_1", id=1,
            position=np.array([
                0.6 + 0.1 * np.sin(t * 0.5),
                -0.1 + 0.05 * np.cos(t * 0.3),
                0.725,
            ]),
            quaternion=np.array([0.0, 0.0, np.sin(t * 0.2) * 0.1, 1.0]),
            timestamp=t, tracking_valid=True,
        ),
        "cylinder_1": RigidBodyState(
            name="cylinder_1", id=2,
            position=np.array([
                0.75 + 0.03 * np.sin(t * 0.7),
                0.15,
                0.74,
            ]),
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


def save_snapshot(renderer, annotated_img, scene_text, snapshot_count):
    """Save all outputs to the output/ directory."""
    os.makedirs("output", exist_ok=True)

    # 3D rendered image
    img_3d = renderer.render_snapshot()
    path_3d = f"output/snapshot_{snapshot_count:04d}_3d.png"
    cv2.imwrite(path_3d, cv2.cvtColor(img_3d, cv2.COLOR_RGB2BGR))

    # 2D annotated image
    path_2d = f"output/snapshot_{snapshot_count:04d}_2d.png"
    cv2.imwrite(path_2d, annotated_img)

    # Scene text
    path_txt = f"output/snapshot_{snapshot_count:04d}_scene.txt"
    with open(path_txt, "w") as f:
        f.write(scene_text)

    print(f"[Pipeline] Snapshot {snapshot_count} saved:")
    print(f"  3D: {path_3d}")
    print(f"  2D: {path_2d}")
    print(f"  Text: {path_txt}")


def main():
    parser = argparse.ArgumentParser(description="Full vision engine pipeline")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config", "scene_config.yaml"),
    )
    parser.add_argument("--no-optitrack", action="store_true", help="Demo mode")
    parser.add_argument("--headless", action="store_true", help="No OpenGL window")
    parser.add_argument("--print-interval", type=float, default=5.0,
                        help="Print scene description every N seconds (default 5)")
    args = parser.parse_args()

    config = load_config(args.config)
    workspace_cfg = config.get("workspace", {})
    objects_cfg = config.get("objects", {})
    renderer_cfg = config.get("renderer", {})
    physics_cfg = config.get("physics", {})

    # ── Initialize Components ──

    # 1. OptiTrack client
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

    # 2. OpenGL renderer
    renderer = SceneRenderer(
        width=renderer_cfg.get("window_width", 1280),
        height=renderer_cfg.get("window_height", 720),
        headless=args.headless,
        background_color=tuple(renderer_cfg.get("background_color", [0.15, 0.15, 0.15])),
    )
    renderer.setup_scene(workspace_cfg)

    # 3. Physics engine
    registry = BodyRegistry(objects_cfg)
    physics_world = PhysicsWorld(
        gravity=physics_cfg.get("gravity", -9.81),
        table_height=workspace_cfg.get("table_height", 0.7),
    )
    predictor = PhysicsPredictor(
        world=physics_world,
        registry=registry,
        gripper_width=physics_cfg.get("gripper_width", 0.08),
        prediction_horizon=physics_cfg.get("prediction_horizon", 2.0),
    )

    # 4. Annotator (no camera model for now — uses top-down mapping)
    annotator = SceneAnnotator(image_width=640, image_height=480)

    # 5. Scene-to-language
    describer = SceneDescriber(
        object_registry=objects_cfg,
        workspace=workspace_cfg,
    )

    # ── Output directory ──
    os.makedirs("output", exist_ok=True)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              Vision Engine — Full Pipeline                   ║
╠══════════════════════════════════════════════════════════════╣
║  1 = Birds-eye  |  2 = Robot-view  |  3 = Free orbit        ║
║  S = Save snapshot  |  P = Print scene  |  Q = Quit          ║
╚══════════════════════════════════════════════════════════════╝
""")

    # ── Main Loop ──
    frame_count = 0
    snapshot_count = 0
    fps_start = time.time()
    last_print_time = time.time()
    last_physics_time = time.time()
    latest_predictions = {}
    latest_scene_text = ""
    latest_annotated = None

    # Monkey-patch the renderer's key callback to handle S and P
    original_key_cb = renderer._key_callback
    save_requested = [False]
    print_requested = [False]

    def custom_key_callback(window, key, scancode, action, mods):
        import glfw
        if action == glfw.PRESS:
            if key == glfw.KEY_P:
                print_requested[0] = True
                return
            if key == glfw.KEY_S:
                save_requested[0] = True
                return
        original_key_cb(window, key, scancode, action, mods)

    import glfw
    glfw.set_key_callback(renderer._window, custom_key_callback)

    try:
        while not renderer.should_close():
            now = time.time()

            # ── 1. Capture scene state ──
            if aggregator is not None:
                snapshot = aggregator.capture_snapshot()
            else:
                snapshot = create_demo_snapshot()

            # ── 2. Physics predictions (every 1 second, not every frame) ──
            if now - last_physics_time >= 1.0:
                latest_predictions = predictor.predict_all(snapshot)
                last_physics_time = now

            # ── 3. Render 3D scene ──
            renderer.update_scene(snapshot, objects_cfg)
            renderer.render()

            # ── 4. Generate annotations ──
            latest_annotated = annotator.annotate(
                None, snapshot, latest_predictions, workspace_cfg
            )

            # ── 5. Generate scene description ──
            latest_scene_text = describer.describe(snapshot, latest_predictions)

            # ── Handle user requests ──
            if print_requested[0]:
                print(f"\n{'='*60}")
                print(latest_scene_text)
                print(f"{'='*60}\n")
                print_requested[0] = False

            if save_requested[0]:
                snapshot_count += 1
                save_snapshot(renderer, latest_annotated, latest_scene_text, snapshot_count)
                save_requested[0] = False

            # ── Auto-print scene description periodically ──
            if now - last_print_time >= args.print_interval:
                n_bodies = sum(1 for b in snapshot.rigid_bodies.values() if b.tracking_valid)
                elapsed = now - fps_start
                fps = frame_count / elapsed if elapsed > 0 else 0

                print(f"[Pipeline] {fps:.0f} fps | {n_bodies} objects | "
                      f"physics: {len(latest_predictions)} predictions")
                last_print_time = now

            frame_count += 1

    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted.")

    # ── Cleanup ──
    physics_world.disconnect()
    if client:
        client.stop()
    renderer.shutdown()
    print("[Pipeline] All components shut down.")


if __name__ == "__main__":
    main()
