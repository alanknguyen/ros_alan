#!/usr/bin/env python3
"""
scripts/run_cs100_tracker.py — CS-100 Center Tracker with OpenCV

Connects to OptiTrack via NatNet, tracks the CS-100 rigid body center
in real-time, and draws its motion trail on a top-down 2D view using
OpenCV. Records a 5-second trace by default.

The V120:Trio does not stream raw camera frames over NatNet — only
tracking data (positions + orientations). This script uses the rigid
body pose data as the "detection" and visualizes it with OpenCV.

The display shows:
  - Black canvas representing a top-down view of the tracking volume
  - CS-100 L-shape markers (yellow corner, orange short, green long)
  - White trail tracing the center over time
  - Coordinate readout and timing info

Controls
--------
    R       Reset trail
    S       Save current frame to output/
    Space   Pause/resume recording
    Q/Esc   Quit

Usage
-----
    cd vision_engine
    python scripts/run_cs100_tracker.py                    # Live
    python scripts/run_cs100_tracker.py --no-optitrack     # Demo mode
    python scripts/run_cs100_tracker.py --duration 10      # 10s trace
    python scripts/run_cs100_tracker.py --save             # Auto-save at end
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import load_config
from cv.optitrack_client import OptiTrackClient, RigidBodyState
from cv.scene_state import SceneStateAggregator, SceneSnapshot
from cv.cs100_model import CS100Geometry


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate Mapping: 3D world → 2D pixel
# ──────────────────────────────────────────────────────────────────────────────

class TopDownMapper:
    """
    Maps 3D world coordinates (X, Y) to 2D pixel coordinates on a canvas.

    Uses the OptiTrack coordinate frame (Z-up after conversion):
      - X → horizontal on canvas
      - Y → vertical on canvas (inverted for image coordinates)
    """

    def __init__(self, canvas_w: int, canvas_h: int, margin: float = 0.05):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.margin = margin

        # Auto-fit bounds — will be set from data
        self.x_min = -0.3
        self.x_max = 0.3
        self.y_min = -0.3
        self.y_max = 0.3

    def update_bounds(self, positions, padding: float = 0.1):
        """Expand bounds to fit all positions with padding."""
        if len(positions) == 0:
            return

        pts = np.array(positions)
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

        # Ensure minimum range
        x_range = max(x_max - x_min, 0.1)
        y_range = max(y_max - y_min, 0.1)

        # Keep aspect ratio square
        max_range = max(x_range, y_range) + padding * 2
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        self.x_min = cx - max_range / 2
        self.x_max = cx + max_range / 2
        self.y_min = cy - max_range / 2
        self.y_max = cy + max_range / 2

    def to_pixel(self, x: float, y: float) -> tuple:
        """Convert world (x, y) to pixel (px, py)."""
        usable_w = self.canvas_w * (1 - 2 * self.margin)
        usable_h = self.canvas_h * (1 - 2 * self.margin)

        px = int(self.margin * self.canvas_w +
                 (x - self.x_min) / max(self.x_max - self.x_min, 1e-6) * usable_w)
        py = int(self.margin * self.canvas_h +
                 (1.0 - (y - self.y_min) / max(self.y_max - self.y_min, 1e-6)) * usable_h)
        return (px, py)


# ──────────────────────────────────────────────────────────────────────────────
# Demo Mode
# ──────────────────────────────────────────────────────────────────────────────

def create_demo_snapshot() -> SceneSnapshot:
    """Animated demo: CS-100 moving in a figure-8 pattern."""
    t = time.time()

    # Figure-8 path
    cx = 0.15 * np.sin(t * 0.8)
    cy = 0.10 * np.sin(t * 1.6)

    angle = t * 0.5
    qz = np.sin(angle * 0.5)
    qw = np.cos(angle * 0.5)

    bodies = {
        "CS-100": RigidBodyState(
            name="CS-100", id=1,
            position=np.array([cx, cy, 0.005]),
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


# ──────────────────────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────────────────────

def draw_frame(canvas, mapper, trail, marker_positions, elapsed, duration,
               position, recording):
    """Draw the full tracker visualization."""
    canvas[:] = 20  # Near-black background

    h, w = canvas.shape[:2]

    # ── Grid lines ──
    grid_color = (40, 40, 40)
    x_range = mapper.x_max - mapper.x_min
    grid_step = 0.05 if x_range < 0.5 else 0.1

    gx = mapper.x_min
    while gx <= mapper.x_max:
        px, _ = mapper.to_pixel(gx, 0)
        cv2.line(canvas, (px, 0), (px, h), grid_color, 1)
        gx += grid_step

    gy = mapper.y_min
    while gy <= mapper.y_max:
        _, py = mapper.to_pixel(0, gy)
        cv2.line(canvas, (0, py), (w, py), grid_color, 1)
        gy += grid_step

    # ── Origin crosshair ──
    ox, oy = mapper.to_pixel(0, 0)
    cv2.line(canvas, (ox - 10, oy), (ox + 10, oy), (60, 60, 60), 1)
    cv2.line(canvas, (ox, oy - 10), (ox, oy + 10), (60, 60, 60), 1)

    # ── Trail (fading white line) ──
    if len(trail) >= 2:
        pts = [mapper.to_pixel(p[0], p[1]) for p in trail]
        n = len(pts)
        for i in range(1, n):
            # Fade from dim to bright
            alpha = int(80 + 175 * (i / n))
            color = (alpha, alpha, alpha)
            thickness = 1 if i < n * 0.5 else 2
            cv2.line(canvas, pts[i - 1], pts[i], color, thickness)

    # ── CS-100 L-shape markers ──
    if marker_positions is not None:
        colors = [
            (0, 255, 255),    # Yellow (BGR) — corner
            (0, 140, 255),    # Orange (BGR) — short arm
            (0, 255, 0),      # Green (BGR) — long arm
        ]
        labels = ["C", "8", "10"]
        marker_px = [mapper.to_pixel(m[0], m[1]) for m in marker_positions]

        # Connecting lines
        cv2.line(canvas, marker_px[0], marker_px[1], (180, 180, 180), 1)
        cv2.line(canvas, marker_px[0], marker_px[2], (180, 180, 180), 1)

        # Marker dots
        for i, (px, py) in enumerate(marker_px):
            cv2.circle(canvas, (px, py), 6, colors[i], -1)
            cv2.circle(canvas, (px, py), 6, (255, 255, 255), 1)
            cv2.putText(canvas, labels[i], (px + 8, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[i], 1)

    # ── Center dot (bright white) ──
    if position is not None:
        cpx, cpy = mapper.to_pixel(position[0], position[1])
        cv2.circle(canvas, (cpx, cpy), 4, (255, 255, 255), -1)

    # ── HUD: timing and coordinates ──
    hud_color = (200, 200, 200)
    y_text = 25

    status = "RECORDING" if recording else "PAUSED"
    status_color = (0, 200, 0) if recording else (0, 0, 200)
    cv2.putText(canvas, status, (w - 130, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    cv2.putText(canvas, f"Time: {elapsed:.1f}s / {duration:.0f}s",
                (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1)
    y_text += 22

    cv2.putText(canvas, f"Trail: {len(trail)} points",
                (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1)
    y_text += 22

    if position is not None:
        cv2.putText(canvas,
                    f"Center: ({position[0]:+.4f}, {position[1]:+.4f}, {position[2]:+.4f}) m",
                    (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, hud_color, 1)
        y_text += 22

    # ── Progress bar ──
    bar_y = h - 15
    bar_w = int((w - 20) * min(elapsed / max(duration, 0.1), 1.0))
    cv2.rectangle(canvas, (10, bar_y), (w - 10, bar_y + 8), (40, 40, 40), -1)
    if bar_w > 0:
        cv2.rectangle(canvas, (10, bar_y), (10 + bar_w, bar_y + 8), (0, 180, 0), -1)

    # ── Controls help ──
    cv2.putText(canvas, "R=Reset  S=Save  Space=Pause  Q=Quit",
                (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CS-100 center tracker with OpenCV")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config", "scene_config.yaml"),
    )
    parser.add_argument("--no-optitrack", action="store_true", help="Demo mode")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Trace duration in seconds (default: 5)")
    parser.add_argument("--width", type=int, default=800, help="Canvas width")
    parser.add_argument("--height", type=int, default=800, help="Canvas height")
    parser.add_argument("--save", action="store_true",
                        help="Auto-save trace image at the end")
    parser.add_argument("--body", default=None,
                        help="Rigid body name (default: from config)")
    args = parser.parse_args()

    config = load_config(args.config)
    objects_cfg = config.get("objects", {})

    # Determine body name
    body_name = args.body
    if body_name is None:
        cal_cfg = config.get("calibration", {})
        body_name = cal_cfg.get("tool_body", "CS-100")

    # Create CS-100 geometry model
    cs100_cfg = objects_cfg.get(body_name, objects_cfg.get("CS-100", {}))
    cs100 = CS100Geometry(
        short_arm_length=cs100_cfg.get("short_arm_length", 0.08),
        long_arm_length=cs100_cfg.get("long_arm_length", 0.10),
    )

    # Connect to OptiTrack
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
        time.sleep(1.5)

        # Wait for body
        print(f"[Tracker] Waiting for '{body_name}'...")
        found = False
        for _ in range(50):
            bodies = client.get_rigid_bodies()
            if bodies:
                available = list(bodies.keys())
                # Flexible name matching
                for name in available:
                    if (name == body_name or
                        name.lower() == body_name.lower() or
                        body_name.lower().replace("-", "") in name.lower().replace("-", "")):
                        body_name = name
                        found = True
                        break
                if found:
                    break
            time.sleep(0.1)

        if found:
            print(f"[Tracker] Found rigid body: '{body_name}'")
        else:
            bodies = client.get_rigid_bodies()
            if bodies:
                body_name = list(bodies.keys())[0]
                print(f"[Tracker] '{body_name}' not found, using first available: '{body_name}'")
            else:
                print("[Tracker] No rigid bodies detected. Running demo mode.")
                args.no_optitrack = True

    # ── Setup ──
    canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    mapper = TopDownMapper(args.width, args.height)
    trail = deque(maxlen=10000)

    os.makedirs("output", exist_ok=True)

    window_name = "CS-100 Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║              CS-100 Center Tracker                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║  R = Reset trail  |  S = Save  |  Space = Pause  |  Q = Quit       ║
║  Duration: {args.duration:.0f}s  |  Body: {body_name:<20s}                    ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    start_time = time.time()
    recording = True
    snapshot_count = 0

    try:
        while True:
            now = time.time()
            elapsed = now - start_time

            # ── Get snapshot ──
            if args.no_optitrack:
                snapshot = create_demo_snapshot()
            else:
                bodies = client.get_rigid_bodies()
                if bodies and body_name in bodies:
                    body = bodies[body_name]
                    snapshot = SceneSnapshot(
                        timestamp=now,
                        rigid_bodies={body_name: body},
                        gripper_position=None,
                        gripper_open=True,
                    )
                else:
                    snapshot = SceneSnapshot(
                        timestamp=now,
                        rigid_bodies={},
                        gripper_position=None,
                        gripper_open=True,
                    )

            # ── Extract position ──
            position = None
            marker_positions = None

            if body_name in snapshot.rigid_bodies:
                body = snapshot.rigid_bodies[body_name]
                pos = body.position
                quat = body.quaternion

                if np.all(np.isfinite(pos)) and np.linalg.norm(pos) < 100.0:
                    position = pos.copy()
                    marker_positions = cs100.compute_marker_positions(pos, quat)

                    if recording and elapsed <= args.duration:
                        trail.append(position.copy())

            # ── Update bounds every 30 frames ──
            if len(trail) > 0 and len(trail) % 30 == 0:
                mapper.update_bounds(list(trail), padding=0.05)

            # ── Draw ──
            draw_frame(canvas, mapper, trail, marker_positions,
                       elapsed if elapsed <= args.duration else args.duration,
                       args.duration, position, recording)

            cv2.imshow(window_name, canvas)

            # ── Auto-stop recording after duration ──
            if recording and elapsed > args.duration:
                recording = False
                print(f"[Tracker] Recording complete: {len(trail)} points in {args.duration:.1f}s")
                if args.save:
                    snapshot_count += 1
                    path = f"output/trace_{snapshot_count:04d}.png"
                    cv2.imwrite(path, canvas)
                    print(f"[Tracker] Auto-saved to {path}")

            # ── Handle keys ──
            key = cv2.waitKey(16) & 0xFF  # ~60fps display
            if key == ord('q') or key == 27:  # Q or Esc
                break
            elif key == ord('r'):  # Reset
                trail.clear()
                start_time = time.time()
                recording = True
                print("[Tracker] Trail reset.")
            elif key == ord('s'):  # Save
                snapshot_count += 1
                path = f"output/trace_{snapshot_count:04d}.png"
                cv2.imwrite(path, canvas)
                print(f"[Tracker] Saved to {path}")
            elif key == ord(' '):  # Pause/resume
                recording = not recording
                print(f"[Tracker] {'Recording' if recording else 'Paused'}")

    except KeyboardInterrupt:
        print("\n[Tracker] Interrupted.")

    cv2.destroyAllWindows()
    if client:
        client.stop()
    print(f"[Tracker] Done. {len(trail)} points recorded.")


if __name__ == "__main__":
    main()
