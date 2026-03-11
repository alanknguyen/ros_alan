#!/usr/bin/env python3
"""
scene_state_publisher.py — OptiTrack Scene State Publisher + CS-100 Tracker

Reads 6DOF rigid body poses from OptiTrack via NatNet and publishes them
as structured pose messages for Sauman's LLM reasoning system.

Modes
─────
    (default)   : Prints pose messages to console
    --ros       : Publishes to /llm/scene_state as ROS String messages
    --track     : Opens OpenCV window tracking CS-100 center + 2D path on desk
    --plot      : Matplotlib real-time plots of x, y, z, quat over time
    --demo      : Simulated data, no OptiTrack hardware needed

Usage
─────
    # Live OptiTrack, print to console:
    python scene_state_publisher.py --objects config/objects.yaml --ip 192.168.0.101

    # Track CS-100 with 2D path visualization:
    python scene_state_publisher.py --track --demo
    python scene_state_publisher.py --track --ip 192.168.0.101

    # Plot live pose data (diagnose coordinate values):
    python scene_state_publisher.py --plot --ip 192.168.0.101
    python scene_state_publisher.py --plot --demo --plot-window 60

    # Demo mode (no hardware):
    python scene_state_publisher.py --demo
"""

import sys
import time
import math
import argparse
import signal
import numpy as np
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml

# ── Resolve imports ──────────────────────────────────────────────────────────
_script_dir = Path(__file__).resolve().parent
_engine_dir = _script_dir.parent
if str(_engine_dir) not in sys.path:
    sys.path.insert(0, str(_engine_dir))

from cv.optitrack_client import OptiTrackClient, RigidBodyState
from cv.cs100_model import CS100Geometry
from cv.transforms import quaternion_to_euler


# ═════════════════════════════════════════════════════════════════════════════
# Object Registry
# ═════════════════════════════════════════════════════════════════════════════

def load_object_registry(yaml_path: str) -> Dict[str, str]:
    """Load objects.yaml → returns {optitrack_body_name: display_name}."""
    if not Path(yaml_path).exists():
        print(f"[SceneState] WARNING: {yaml_path} not found, using raw body names")
        return {}

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    if not raw:
        return {}

    entries = raw.get("objects", raw) if isinstance(raw, dict) else {}
    if not isinstance(entries, dict):
        return {}

    body_to_name = {}
    for display_name, props in entries.items():
        if isinstance(props, dict) and "optitrack_body" in props:
            body_to_name[props["optitrack_body"]] = display_name

    print(f"[SceneState] Loaded {len(body_to_name)} objects from {yaml_path}")
    for body, name in body_to_name.items():
        print(f"  {body} → {name}")

    return body_to_name


# ═════════════════════════════════════════════════════════════════════════════
# Format rigid body pose message
# ═════════════════════════════════════════════════════════════════════════════

def format_body_message(display_name: str, body: RigidBodyState) -> str:
    px, py, pz = body.position
    qx, qy, qz, qw = body.quaternion
    lines = [
        f"RigidBody: {display_name}",
        f"position:",
        f"  x: {px:.4f}",
        f"  y: {py:.4f}",
        f"  z: {pz:.4f}",
        f"orientation (quaternion):",
        f"  x: {qx:.4f}",
        f"  y: {qy:.4f}",
        f"  z: {qz:.4f}",
        f"  w: {qw:.4f}",
        f"timestamp: {body.timestamp:.2f}",
        f"frame: optitrack_world",
    ]
    return "\n".join(lines)


def format_scene_message(
    bodies: Dict[str, RigidBodyState],
    body_to_name: Dict[str, str],
) -> str:
    blocks = []
    for body_name, body in sorted(bodies.items()):
        if not body.tracking_valid:
            continue
        display_name = body_to_name.get(body_name, body_name)
        blocks.append(format_body_message(display_name, body))

    if not blocks:
        return "---\nNo objects tracked\n---"
    return "\n---\n".join(blocks)


# ═════════════════════════════════════════════════════════════════════════════
# Demo Data Generator
# ═════════════════════════════════════════════════════════════════════════════

class DemoOptiTrack:
    """Simulates OptiTrack data — CS-100 draws a figure-8 on the desk."""

    def __init__(self):
        self._t0 = time.time()

    def get_rigid_bodies(self) -> Dict[str, RigidBodyState]:
        t = time.time() - self._t0
        now = time.time()

        # CS-100: figure-8 (lemniscate) pattern on the desk
        scale = 0.15
        cx, cy = 0.45, 0.0  # center of desk
        denom = 1 + math.sin(t * 0.4) ** 2
        x = cx + scale * math.cos(t * 0.4) / denom
        y = cy + scale * math.sin(t * 0.4) * math.cos(t * 0.4) / denom
        z = 0.73  # table height

        # Slight rotation as it moves
        yaw = t * 0.2
        qz_val = math.sin(yaw / 2)
        qw_val = math.cos(yaw / 2)

        bodies = {
            "Rigid_3_Balls": RigidBodyState(
                name="Rigid_3_Balls", id=1,
                position=np.array([x, y, z]),
                quaternion=np.array([0.0, 0.0, qz_val, qw_val]),
                timestamp=now, tracking_valid=True,
            ),
        }

        # Also add some other objects for completeness
        bodies["rigid_body_1"] = RigidBodyState(
            name="rigid_body_1", id=2,
            position=np.array([0.30, -0.15, 0.735]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            timestamp=now, tracking_valid=True,
        )
        bodies["rigid_body_2"] = RigidBodyState(
            name="rigid_body_2", id=3,
            position=np.array([0.60, 0.10, 0.735]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            timestamp=now, tracking_valid=True,
        )

        return bodies

    def start(self):
        pass

    def stop(self):
        pass


# ═════════════════════════════════════════════════════════════════════════════
# CS-100 2D Path Tracker (OpenCV visualization)
# ═════════════════════════════════════════════════════════════════════════════

def run_tracker(args):
    """
    Track the CS-100 rigid body and visualize it in real-time on a 2D desk view.

    Shows:
    - The actual CS-100 L-shape (3 markers + arms) with live orientation
    - Center position trail (fading orange path)
    - Grid overlay for scale
    - Live readout: position, orientation (roll/pitch/yaw), height
    """
    import cv2

    BODY_NAME = args.body  # default: "Rigid_3_Balls"

    # CS-100 geometry model (8cm short arm, 10cm long arm)
    cs100 = CS100Geometry(short_arm_length=0.08, long_arm_length=0.10)

    # ── Connect ──────────────────────────────────────────────────────────
    if args.demo:
        print("[Tracker] Demo mode — simulated CS-100 figure-8 motion")
        client = DemoOptiTrack()
    else:
        print(f"[Tracker] Connecting to OptiTrack at {args.ip}...")
        client = OptiTrackClient(server_ip=args.ip)
        client.start()

    # ── Display settings ─────────────────────────────────────────────────
    WIN_W, WIN_H = 900, 750
    MARGIN = 60
    INFO_H = 80  # height of info panel at bottom

    # Desk bounds in meters (auto-adjusts)
    desk_x_range = [0.1, 0.8]
    desk_y_range = [-0.4, 0.4]

    # Colors (BGR)
    BG_COLOR        = (30, 30, 30)
    GRID_COLOR      = (55, 55, 55)
    GRID_TEXT_COLOR  = (100, 100, 100)
    DESK_COLOR      = (50, 45, 40)
    DESK_BORDER     = (80, 80, 80)
    TRAIL_COLOR     = (0, 200, 255)      # orange
    CORNER_COLOR    = (0, 255, 255)      # yellow — corner marker
    SHORT_ARM_COLOR = (0, 140, 255)      # orange — short arm marker (8cm)
    LONG_ARM_COLOR  = (0, 255, 0)        # green — long arm marker (10cm)
    ARM_LINE_COLOR  = (200, 200, 200)    # white — L-shape arms
    FILL_COLOR      = (80, 120, 60)      # dark green — L-shape fill
    CENTER_COLOR    = (255, 200, 0)      # cyan — center dot
    LOST_COLOR      = (0, 0, 200)        # red
    TEXT_COLOR      = (220, 220, 220)
    DIM_TEXT        = (140, 140, 140)

    # Trail buffer
    trail: deque = deque(maxlen=args.trail_length)

    # Auto-range
    x_min_seen, x_max_seen = 999.0, -999.0
    y_min_seen, y_max_seen = 999.0, -999.0

    def world_to_px(wx: float, wy: float) -> Tuple[int, int]:
        """Convert world XY (meters) → pixel coordinates on the canvas."""
        plot_w = WIN_W - 2 * MARGIN
        plot_h = WIN_H - 2 * MARGIN - INFO_H

        x_span = max(desk_x_range[1] - desk_x_range[0], 0.01)
        y_span = max(desk_y_range[1] - desk_y_range[0], 0.01)

        px = int(MARGIN + (wy - desk_y_range[0]) / y_span * plot_w)
        py = int(MARGIN + (1.0 - (wx - desk_x_range[0]) / x_span) * plot_h)
        return px, py

    def expand_range(x: float, y: float):
        nonlocal x_min_seen, x_max_seen, y_min_seen, y_max_seen
        pad = 0.05
        x_min_seen = min(x_min_seen, x)
        x_max_seen = max(x_max_seen, x)
        y_min_seen = min(y_min_seen, y)
        y_max_seen = max(y_max_seen, y)
        desk_x_range[0] = min(desk_x_range[0], x_min_seen - pad)
        desk_x_range[1] = max(desk_x_range[1], x_max_seen + pad)
        desk_y_range[0] = min(desk_y_range[0], y_min_seen - pad)
        desk_y_range[1] = max(desk_y_range[1], y_max_seen + pad)

    frame_count = 0
    last_valid_body = None
    tracking_lost = False
    start_time = time.time()

    print(f"[Tracker] Tracking body '{BODY_NAME}'")
    print(f"[Tracker] Keys: 'q' quit | 'c' clear trail | 'r' reset view\n")

    while True:
        t_frame = time.time()
        bodies = client.get_rigid_bodies()
        body = bodies.get(BODY_NAME)

        # ── Process body data ────────────────────────────────────────────
        current_body = None
        marker_positions = None  # (3, 3): [corner, short_arm, long_arm]
        euler_deg = None

        if body is not None and body.tracking_valid:
            pos = body.position
            quat = body.quaternion
            if (np.all(np.isfinite(pos)) and np.linalg.norm(pos) < 50.0
                    and np.linalg.norm(quat) > 0.5):
                current_body = body
                last_valid_body = body
                tracking_lost = False

                # Trail (center position)
                trail.append((float(pos[0]), float(pos[1]), t_frame))
                expand_range(float(pos[0]), float(pos[1]))

                # Compute actual marker positions from L-shape geometry
                marker_positions = cs100.compute_marker_positions(pos, quat)

                # Expand range to fit markers too
                for m in marker_positions:
                    expand_range(float(m[0]), float(m[1]))

                # Euler angles
                roll, pitch, yaw = quaternion_to_euler(
                    quat[0], quat[1], quat[2], quat[3])
                euler_deg = (np.degrees(roll), np.degrees(pitch), np.degrees(yaw))
        else:
            tracking_lost = True

        # ── Draw ─────────────────────────────────────────────────────────
        canvas = np.full((WIN_H, WIN_W, 3), BG_COLOR, dtype=np.uint8)

        # Desk rectangle
        tl = world_to_px(desk_x_range[1], desk_y_range[0])
        br = world_to_px(desk_x_range[0], desk_y_range[1])
        cv2.rectangle(canvas, tl, br, DESK_COLOR, -1)
        cv2.rectangle(canvas, tl, br, DESK_BORDER, 1)

        # Grid every 10cm
        grid_step = 0.10
        gx = math.floor(desk_x_range[0] / grid_step) * grid_step
        while gx <= desk_x_range[1]:
            p1 = world_to_px(gx, desk_y_range[0])
            p2 = world_to_px(gx, desk_y_range[1])
            cv2.line(canvas, p1, p2, GRID_COLOR, 1)
            cv2.putText(canvas, f"{gx:.1f}m", (p2[0] + 4, p2[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, GRID_TEXT_COLOR, 1)
            gx += grid_step

        gy = math.floor(desk_y_range[0] / grid_step) * grid_step
        while gy <= desk_y_range[1]:
            p1 = world_to_px(desk_x_range[0], gy)
            p2 = world_to_px(desk_x_range[1], gy)
            cv2.line(canvas, p1, p2, GRID_COLOR, 1)
            cv2.putText(canvas, f"{gy:.1f}m", (p1[0], p1[1] + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, GRID_TEXT_COLOR, 1)
            gy += grid_step

        # ── Trail (fading orange path) ───────────────────────────────────
        trail_list = list(trail)
        n = len(trail_list)
        for i in range(1, n):
            alpha = i / n
            color = (int(TRAIL_COLOR[0] * alpha),
                     int(TRAIL_COLOR[1] * alpha),
                     int(TRAIL_COLOR[2] * alpha))
            thickness = max(1, int(2 * alpha))

            p1 = world_to_px(trail_list[i - 1][0], trail_list[i - 1][1])
            p2 = world_to_px(trail_list[i][0], trail_list[i][1])

            dx = abs(trail_list[i][0] - trail_list[i - 1][0])
            dy = abs(trail_list[i][1] - trail_list[i - 1][1])
            if dx < 0.3 and dy < 0.3:
                cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)

        # ── Draw CS-100 L-shape ──────────────────────────────────────────
        if current_body is not None and marker_positions is not None:
            # Pixel positions of the 3 markers (top-down XY)
            corner_px   = world_to_px(marker_positions[0][0], marker_positions[0][1])
            short_px    = world_to_px(marker_positions[1][0], marker_positions[1][1])
            long_px     = world_to_px(marker_positions[2][0], marker_positions[2][1])
            center_px   = world_to_px(float(current_body.position[0]),
                                      float(current_body.position[1]))

            # Filled triangle (L-shape area)
            tri_pts = np.array([corner_px, short_px, long_px], dtype=np.int32)
            cv2.fillPoly(canvas, [tri_pts], FILL_COLOR, cv2.LINE_AA)

            # L-shape arms (corner → short, corner → long)
            cv2.line(canvas, corner_px, short_px, ARM_LINE_COLOR, 2, cv2.LINE_AA)
            cv2.line(canvas, corner_px, long_px, ARM_LINE_COLOR, 2, cv2.LINE_AA)
            # Hypotenuse (dashed feel — thinner)
            cv2.line(canvas, short_px, long_px, (120, 120, 120), 1, cv2.LINE_AA)

            # Marker dots
            cv2.circle(canvas, corner_px, 7, CORNER_COLOR, -1, cv2.LINE_AA)    # yellow
            cv2.circle(canvas, short_px,  6, SHORT_ARM_COLOR, -1, cv2.LINE_AA) # orange
            cv2.circle(canvas, long_px,   6, LONG_ARM_COLOR, -1, cv2.LINE_AA)  # green

            # Marker labels
            cv2.putText(canvas, "C", (corner_px[0] + 10, corner_px[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, CORNER_COLOR, 1)
            cv2.putText(canvas, "8cm", (short_px[0] + 10, short_px[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, SHORT_ARM_COLOR, 1)
            cv2.putText(canvas, "10cm", (long_px[0] + 10, long_px[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, LONG_ARM_COLOR, 1)

            # Center point (pivot / centroid)
            cv2.circle(canvas, center_px, 4, CENTER_COLOR, -1, cv2.LINE_AA)
            cv2.circle(canvas, center_px, 6, (255, 255, 255), 1, cv2.LINE_AA)

        elif last_valid_body is not None:
            # Show last known position as red outline
            lp = last_valid_body.position
            px, py_cv = world_to_px(float(lp[0]), float(lp[1]))
            cv2.circle(canvas, (px, py_cv), 12, LOST_COLOR, 2, cv2.LINE_AA)
            cv2.putText(canvas, "TRACKING LOST", (px + 16, py_cv + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, LOST_COLOR, 1)

        # ── Info panel (bottom) ──────────────────────────────────────────
        info_top = WIN_H - INFO_H
        cv2.rectangle(canvas, (0, info_top), (WIN_W, WIN_H), (40, 40, 40), -1)
        cv2.line(canvas, (0, info_top), (WIN_W, info_top), DESK_BORDER, 1)

        if current_body is not None:
            pos = current_body.position
            # Line 1: Position
            cv2.putText(canvas,
                f"Position:  X={pos[0]:.4f}  Y={pos[1]:.4f}  Z={pos[2]:.4f} m",
                (10, info_top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)

            # Line 2: Orientation
            if euler_deg is not None:
                cv2.putText(canvas,
                    f"Orientation:  Roll={euler_deg[0]:.1f}  "
                    f"Pitch={euler_deg[1]:.1f}  Yaw={euler_deg[2]:.1f} deg",
                    (10, info_top + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)

            # Line 3: Quaternion
            q = current_body.quaternion
            cv2.putText(canvas,
                f"Quaternion:  x={q[0]:.3f}  y={q[1]:.3f}  z={q[2]:.3f}  w={q[3]:.3f}",
                (10, info_top + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.38, DIM_TEXT, 1)

        elif tracking_lost:
            cv2.putText(canvas, "CS-100: TRACKING LOST", (10, info_top + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, LOST_COLOR, 1)
        else:
            cv2.putText(canvas, f"Waiting for body '{BODY_NAME}'...",
                        (10, info_top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM_TEXT, 1)

        # Right side: trail stats
        elapsed = t_frame - start_time
        cv2.putText(canvas,
            f"Trail: {len(trail_list)} pts | {elapsed:.0f}s | Frame {frame_count}",
            (WIN_W - 320, info_top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM_TEXT, 1)
        cv2.putText(canvas,
            "'q' quit | 'c' clear trail | 'r' reset view",
            (WIN_W - 320, info_top + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.30, DIM_TEXT, 1)

        # ── Title ────────────────────────────────────────────────────────
        cv2.putText(canvas, "CS-100 Rigid Body Tracker", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)

        mode_str = "DEMO" if args.demo else "LIVE"
        cv2.putText(canvas, f"[{mode_str}] Top-down XY view  |  10cm grid",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM_TEXT, 1)

        # Legend (top-right)
        lx = WIN_W - 150
        cv2.circle(canvas, (lx, 16), 5, CORNER_COLOR, -1)
        cv2.putText(canvas, "Corner", (lx + 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, CORNER_COLOR, 1)
        cv2.circle(canvas, (lx, 30), 5, SHORT_ARM_COLOR, -1)
        cv2.putText(canvas, "Short (8cm)", (lx + 10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, SHORT_ARM_COLOR, 1)
        cv2.circle(canvas, (lx, 44), 5, LONG_ARM_COLOR, -1)
        cv2.putText(canvas, "Long (10cm)", (lx + 10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, LONG_ARM_COLOR, 1)

        # ── Show ─────────────────────────────────────────────────────────
        cv2.imshow("CS-100 Rigid Body Tracker", canvas)
        key = cv2.waitKey(max(1, int(1000 / args.rate))) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("c"):
            trail.clear()
            print("[Tracker] Trail cleared")
        elif key == ord("r"):
            desk_x_range[:] = [0.1, 0.8]
            desk_y_range[:] = [-0.4, 0.4]
            x_min_seen, x_max_seen = 999.0, -999.0
            y_min_seen, y_max_seen = 999.0, -999.0
            print("[Tracker] View reset")

        frame_count += 1

    cv2.destroyAllWindows()
    if hasattr(client, "stop"):
        client.stop()
    print(f"[Tracker] Done. {frame_count} frames, {len(trail)} trail points.")


# ═════════════════════════════════════════════════════════════════════════════
# Real-Time Plot Mode (matplotlib)
# ═════════════════════════════════════════════════════════════════════════════

def run_plotter(args):
    """
    Plot x, y, z position and quaternion (qx, qy, qz, qw) over time
    using matplotlib. Useful for diagnosing coordinate ranges and tracking.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    BODY_NAME = args.body
    WINDOW_SEC = args.plot_window  # seconds of data to show

    # ── Connect ──────────────────────────────────────────────────────────
    if args.demo:
        print("[Plot] Demo mode — simulated data")
        client = DemoOptiTrack()
    else:
        print(f"[Plot] Connecting to OptiTrack at {args.ip}...")
        client = OptiTrackClient(server_ip=args.ip)
        client.start()

    # ── Data buffers ─────────────────────────────────────────────────────
    max_pts = args.rate * WINDOW_SEC
    ts = deque(maxlen=max_pts)      # relative time
    xs = deque(maxlen=max_pts)
    ys = deque(maxlen=max_pts)
    zs = deque(maxlen=max_pts)
    qxs = deque(maxlen=max_pts)
    qys = deque(maxlen=max_pts)
    qzs = deque(maxlen=max_pts)
    qws = deque(maxlen=max_pts)

    t0 = time.time()
    body_found_ever = [False]
    last_print = [0.0]       # for throttled terminal output
    frame_counter = [0]

    # ── Set up figure ────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"CS-100 Rigid Body: '{BODY_NAME}'  —  Live Pose Data",
                 fontsize=13, fontweight="bold")

    # Subplot configs: (ax, label, color, data_lists)
    configs = [
        (axes[0], "X position (m)", "#ff6b6b", [xs]),
        (axes[1], "Y position (m)", "#51cf66", [ys]),
        (axes[2], "Z position (m)", "#339af0", [zs]),
        (axes[3], "Quaternion", None, [qxs, qys, qzs, qws]),
    ]

    lines = []
    for ax, label, color, data in configs:
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        if label == "Quaternion":
            # 4 lines for qx, qy, qz, qw
            l_qx, = ax.plot([], [], color="#ff6b6b", linewidth=1, label="qx")
            l_qy, = ax.plot([], [], color="#51cf66", linewidth=1, label="qy")
            l_qz, = ax.plot([], [], color="#339af0", linewidth=1, label="qz")
            l_qw, = ax.plot([], [], color="#ffd43b", linewidth=1.5, label="qw")
            lines.extend([l_qx, l_qy, l_qz, l_qw])
            ax.legend(loc="upper left", fontsize=7, ncol=4)
        else:
            l, = ax.plot([], [], color=color, linewidth=1.2)
            lines.append(l)

    axes[-1].set_xlabel("Time (s)", fontsize=9)

    # Status text
    status_text = fig.text(0.02, 0.01, "Waiting for body...",
                           fontsize=8, color="#aaaaaa",
                           family="monospace")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ── Animation update ─────────────────────────────────────────────────
    def update(frame_num):
        t_now = time.time() - t0
        bodies = client.get_rigid_bodies()
        body = bodies.get(BODY_NAME)

        frame_counter[0] += 1
        client_frames = client.get_frame_count() if hasattr(client, 'get_frame_count') else -1

        if body is not None:
            pos = body.position
            quat = body.quaternion
            pos_ok = np.all(np.isfinite(pos)) and np.linalg.norm(pos) < 50.0
            quat_ok = np.all(np.isfinite(quat)) and np.linalg.norm(quat) > 0.1

            if pos_ok:
                body_found_ever[0] = True
                ts.append(t_now)
                xs.append(float(pos[0]))
                ys.append(float(pos[1]))
                zs.append(float(pos[2]))

                if quat_ok:
                    qxs.append(float(quat[0]))
                    qys.append(float(quat[1]))
                    qzs.append(float(quat[2]))
                    qws.append(float(quat[3]))
                else:
                    qxs.append(float("nan"))
                    qys.append(float("nan"))
                    qzs.append(float("nan"))
                    qws.append(float("nan"))

                # ── Terminal print every 0.5s ────────────────────────
                if t_now - last_print[0] >= 0.5:
                    last_print[0] = t_now
                    print(
                        f"[{t_now:6.1f}s] "
                        f"x={pos[0]:+8.4f}  y={pos[1]:+8.4f}  z={pos[2]:+8.4f}  |  "
                        f"qx={quat[0]:+6.3f} qy={quat[1]:+6.3f} "
                        f"qz={quat[2]:+6.3f} qw={quat[3]:+6.3f}  |  "
                        f"valid={body.tracking_valid}  "
                        f"nnet_frames={client_frames}"
                    )

                # Status bar on plot
                status_text.set_text(
                    f"TRACKING  |  pos=({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})  "
                    f"quat=({quat[0]:+.3f}, {quat[1]:+.3f}, {quat[2]:+.3f}, {quat[3]:+.3f})  "
                    f"valid={body.tracking_valid}  |  {t_now:.1f}s  |  NatNet frames: {client_frames}"
                )
                status_text.set_color("#51cf66")
            else:
                status_text.set_text(
                    f"INVALID DATA  |  pos={pos}  quat={quat}  |  {t_now:.1f}s")
                status_text.set_color("#ff6b6b")
        else:
            names = list(bodies.keys()) if bodies else ["(none)"]
            status_text.set_text(
                f"BODY NOT FOUND: '{BODY_NAME}'  |  "
                f"Available: {', '.join(names)}  |  {t_now:.1f}s")
            status_text.set_color("#ffd43b")

            # Print to terminal too
            if t_now - last_print[0] >= 1.0:
                last_print[0] = t_now
                print(f"[{t_now:6.1f}s] BODY '{BODY_NAME}' NOT FOUND. Available: {', '.join(names)}")

        # ── Update line data ─────────────────────────────────────────
        if len(ts) > 1:
            t_list = list(ts)
            t_min = max(0, t_list[-1] - WINDOW_SEC)
            t_max = t_list[-1] + 0.5

            # lines[0] = X, [1] = Y, [2] = Z, [3-6] = qx/qy/qz/qw
            lines[0].set_data(t_list, list(xs))
            lines[1].set_data(t_list, list(ys))
            lines[2].set_data(t_list, list(zs))
            lines[3].set_data(t_list, list(qxs))
            lines[4].set_data(t_list, list(qys))
            lines[5].set_data(t_list, list(qzs))
            lines[6].set_data(t_list, list(qws))

            for i, ax in enumerate(axes):
                ax.set_xlim(t_min, t_max)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

        return lines + [status_text]

    # ── Run ──────────────────────────────────────────────────────────────
    interval_ms = max(16, int(1000 / args.rate))
    anim = FuncAnimation(fig, update, interval=interval_ms, blit=False, cache_frame_data=False)

    print(f"[Plot] Plotting body '{BODY_NAME}' at {args.rate} Hz. Close window to stop.\n")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass

    if hasattr(client, "stop"):
        client.stop()
    print("[Plot] Done.")


# ═════════════════════════════════════════════════════════════════════════════
# Console / ROS Mode
# ═════════════════════════════════════════════════════════════════════════════

def run_publisher(args):
    """Run in console-print or ROS mode."""
    body_to_name = load_object_registry(args.objects)

    if args.demo:
        print("[SceneState] Demo mode — simulated data")
        client = DemoOptiTrack()
    else:
        print(f"[SceneState] Connecting to OptiTrack at {args.ip}...")
        client = OptiTrackClient(server_ip=args.ip)
        client.start()

    ros_pub = None
    if args.ros:
        try:
            import rospy
            from std_msgs.msg import String
            rospy.init_node("scene_state_publisher", anonymous=False)
            ros_pub = rospy.Publisher(args.topic, String, queue_size=1)
            print(f"[SceneState] ROS publisher on {args.topic}")
        except ImportError:
            print("[SceneState] ERROR: rospy not found.")
            sys.exit(1)

    running = [True]
    def on_signal(sig, frame):
        running[0] = False
        print("\n[SceneState] Stopping...")
    signal.signal(signal.SIGINT, on_signal)

    print(f"[SceneState] Publishing at {args.rate} Hz. Ctrl+C to stop.\n")
    interval = 1.0 / args.rate
    count = 0

    while running[0]:
        t0 = time.time()
        if args.ros:
            import rospy
            if rospy.is_shutdown():
                break

        bodies = client.get_rigid_bodies()
        msg_text = format_scene_message(bodies, body_to_name)

        if ros_pub is not None:
            from std_msgs.msg import String
            ros_pub.publish(String(data=msg_text))
        else:
            print(f"\033[2J\033[H", end="")
            print(f"═══ Scene State (frame {count}) ═══")
            print(msg_text)
            print(f"\n[{time.strftime('%H:%M:%S')}] Rate: {args.rate} Hz | "
                  f"Bodies: {sum(1 for b in bodies.values() if b.tracking_valid)}")

        count += 1
        elapsed = time.time() - t0
        if interval - elapsed > 0:
            time.sleep(interval - elapsed)

    if hasattr(client, "stop"):
        client.stop()
    print(f"[SceneState] Done. {count} frames published.")


# ═════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="OptiTrack Scene State Publisher + CS-100 Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scene_state_publisher.py --demo                     # console, simulated
  python scene_state_publisher.py --track --demo             # visual tracker, simulated
  python scene_state_publisher.py --track --ip 192.168.0.101 # visual tracker, live
  python scene_state_publisher.py --plot --ip 192.168.0.101  # plot xyz + quat live
  python scene_state_publisher.py --plot --demo              # plot with simulated data
  python scene_state_publisher.py --ros                      # ROS publisher, live
        """,
    )
    parser.add_argument("--objects", default="config/objects.yaml",
                        help="Path to objects.yaml (default: config/objects.yaml)")
    parser.add_argument("--ip", default="192.168.0.101",
                        help="OptiTrack server IP (default: 192.168.0.101)")
    parser.add_argument("--rate", type=int, default=30,
                        help="Update rate in Hz (default: 30)")
    parser.add_argument("--ros", action="store_true",
                        help="Publish as ROS String messages")
    parser.add_argument("--topic", default="/llm/scene_state",
                        help="ROS topic name (default: /llm/scene_state)")
    parser.add_argument("--demo", action="store_true",
                        help="Simulated data, no OptiTrack needed")
    parser.add_argument("--track", action="store_true",
                        help="Open CS-100 2D path tracker visualization")
    parser.add_argument("--plot", action="store_true",
                        help="Plot x, y, z, quat values over time (matplotlib)")
    parser.add_argument("--plot-window", type=int, default=30,
                        help="Seconds of data visible in plot (default: 30)")
    parser.add_argument("--body", default="Rigid_3_Balls",
                        help="Rigid body name to track (default: Rigid_3_Balls)")
    parser.add_argument("--trail-length", type=int, default=2000,
                        help="Max trail points to keep (default: 2000)")

    args = parser.parse_args()

    if args.plot:
        run_plotter(args)
    elif args.track:
        run_tracker(args)
    else:
        run_publisher(args)


if __name__ == "__main__":
    main()
