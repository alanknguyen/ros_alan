#!/usr/bin/env python3
"""
scripts/run_cs100_tracker.py — CS-100 Center Tracker with Timeline Scrubber

Records the CS-100 rigid body motion for a configurable duration, then
switches to playback mode where you can scrub through the timeline like
a video player to inspect the L-shape position at any moment.

Two phases:
  RECORDING — captures pose data from OptiTrack at ~120Hz
  PLAYBACK  — scrub through the recorded timeline with mouse/keys

Controls (Recording)
--------------------
    Space       Pause/resume recording
    Q/Esc       Stop recording early → enter playback

Controls (Playback)
-------------------
    Click/drag  Scrub on the timeline bar
    Left/Right  Step forward/backward one frame
    Home/End    Jump to start/end
    Space       Play/pause auto-playback
    R           Re-record (restart)
    S           Save current frame to output/
    Q/Esc       Quit

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import load_config
from cv.optitrack_client import OptiTrackClient, RigidBodyState
from cv.scene_state import SceneSnapshot
from cv.cs100_model import CS100Geometry


# ──────────────────────────────────────────────────────────────────────────────
# Recorded Frame
# ──────────────────────────────────────────────────────────────────────────────

class RecordedFrame:
    """A single recorded frame of CS-100 pose data."""
    __slots__ = ("timestamp", "position", "quaternion", "marker_positions")

    def __init__(self, timestamp, position, quaternion, marker_positions):
        self.timestamp = timestamp
        self.position = position
        self.quaternion = quaternion
        self.marker_positions = marker_positions


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate Mapping: 3D world → 2D pixel
# ──────────────────────────────────────────────────────────────────────────────

class TopDownMapper:
    """Maps 3D world (X, Y) to 2D pixel coordinates on canvas."""

    def __init__(self, canvas_w: int, canvas_h: int, margin: float = 0.06):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.margin = margin
        self.x_min = -0.3
        self.x_max = 0.3
        self.y_min = -0.3
        self.y_max = 0.3

    def fit_to_data(self, frames, padding: float = 0.05):
        """Set bounds to fit all recorded positions."""
        if not frames:
            return
        pts = np.array([f.position[:2] for f in frames])
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        max_range = max(x_max - x_min, y_max - y_min, 0.1) + padding * 2
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        self.x_min = cx - max_range / 2
        self.x_max = cx + max_range / 2
        self.y_min = cy - max_range / 2
        self.y_max = cy + max_range / 2

    def to_pixel(self, x: float, y: float) -> tuple:
        """Convert world (x, y) to pixel (px, py)."""
        uw = self.canvas_w * (1 - 2 * self.margin)
        uh = self.canvas_h * (1 - 2 * self.margin)
        px = int(self.margin * self.canvas_w +
                 (x - self.x_min) / max(self.x_max - self.x_min, 1e-6) * uw)
        py = int(self.margin * self.canvas_h +
                 (1.0 - (y - self.y_min) / max(self.y_max - self.y_min, 1e-6)) * uh)
        return (px, py)


# ──────────────────────────────────────────────────────────────────────────────
# Demo Mode
# ──────────────────────────────────────────────────────────────────────────────

def generate_demo_frames(cs100: CS100Geometry, duration: float,
                         fps: float = 120.0) -> list:
    """Generate demo frames: CS-100 moving in a figure-8 pattern."""
    frames = []
    n = int(duration * fps)
    t0 = time.time()
    for i in range(n):
        t = i / fps
        cx = 0.15 * np.sin(t * 0.8)
        cy = 0.10 * np.sin(t * 1.6)
        angle = t * 0.5
        qz = np.sin(angle * 0.5)
        qw = np.cos(angle * 0.5)

        pos = np.array([cx, cy, 0.005])
        quat = np.array([0.0, 0.0, qz, qw])
        markers = cs100.compute_marker_positions(pos, quat)
        frames.append(RecordedFrame(t0 + t, pos, quat, markers))
    return frames


# ──────────────────────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────────────────────

TIMELINE_H = 50  # Height of timeline bar area at bottom

# Colors (BGR)
COL_CORNER = (0, 255, 255)      # Yellow
COL_SHORT  = (0, 140, 255)      # Orange
COL_LONG   = (0, 255, 0)        # Green
COL_LINE   = (180, 180, 180)    # Light gray
COL_HUD    = (200, 200, 200)    # Text
COL_GRID   = (40, 40, 40)       # Grid
COL_BAR_BG = (50, 50, 50)       # Timeline background
COL_BAR_FG = (0, 180, 0)        # Timeline filled
COL_CURSOR = (0, 220, 255)      # Timeline cursor


def draw_grid(canvas, mapper):
    """Draw background grid lines."""
    h, w = canvas.shape[:2]
    x_range = mapper.x_max - mapper.x_min
    step = 0.05 if x_range < 0.5 else 0.1

    gx = mapper.x_min
    while gx <= mapper.x_max:
        px, _ = mapper.to_pixel(gx, 0)
        cv2.line(canvas, (px, 0), (px, h - TIMELINE_H), COL_GRID, 1)
        gx += step

    gy = mapper.y_min
    while gy <= mapper.y_max:
        _, py = mapper.to_pixel(0, gy)
        if py < h - TIMELINE_H:
            cv2.line(canvas, (0, py), (w, py), COL_GRID, 1)
        gy += step

    # Origin crosshair
    ox, oy = mapper.to_pixel(0, 0)
    if oy < h - TIMELINE_H:
        cv2.line(canvas, (ox - 12, oy), (ox + 12, oy), (60, 60, 60), 1)
        cv2.line(canvas, (ox, max(oy - 12, 0)), (ox, min(oy + 12, h - TIMELINE_H)), (60, 60, 60), 1)


def draw_trail(canvas, mapper, frames, end_idx, h_limit):
    """Draw fading trail from frame 0 to end_idx."""
    if end_idx < 1:
        return
    n = min(end_idx + 1, len(frames))
    pts = [mapper.to_pixel(frames[i].position[0], frames[i].position[1])
           for i in range(n)]

    for i in range(1, len(pts)):
        if pts[i][1] >= h_limit or pts[i - 1][1] >= h_limit:
            continue
        alpha = int(60 + 195 * (i / len(pts)))
        color = (alpha, alpha, alpha)
        thickness = 1 if i < len(pts) * 0.7 else 2
        cv2.line(canvas, pts[i - 1], pts[i], color, thickness)


def draw_lshape(canvas, mapper, frame, h_limit):
    """Draw the CS-100 L-shape markers at a given frame."""
    if frame is None or frame.marker_positions is None:
        return

    colors = [COL_CORNER, COL_SHORT, COL_LONG]
    labels = ["C", "8cm", "10cm"]
    mpx = [mapper.to_pixel(m[0], m[1]) for m in frame.marker_positions]

    # Only draw if within canvas area (above timeline)
    for px, py in mpx:
        if py >= h_limit:
            return

    # Lines
    cv2.line(canvas, mpx[0], mpx[1], COL_LINE, 1, cv2.LINE_AA)
    cv2.line(canvas, mpx[0], mpx[2], COL_LINE, 1, cv2.LINE_AA)

    # Marker dots
    for i, (px, py) in enumerate(mpx):
        cv2.circle(canvas, (px, py), 7, colors[i], -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, labels[i], (px + 10, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[i], 1, cv2.LINE_AA)

    # Center dot
    cpx, cpy = mapper.to_pixel(frame.position[0], frame.position[1])
    cv2.circle(canvas, (cpx, cpy), 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_timeline(canvas, frames, cursor_idx, playing, mode_label):
    """Draw the timeline scrubber bar at the bottom."""
    h, w = canvas.shape[:2]
    bar_top = h - TIMELINE_H
    n = max(len(frames), 1)

    # Background
    cv2.rectangle(canvas, (0, bar_top), (w, h), (30, 30, 30), -1)
    cv2.line(canvas, (0, bar_top), (w, bar_top), (60, 60, 60), 1)

    # Bar area
    bar_x0 = 15
    bar_x1 = w - 15
    bar_y = bar_top + 20
    bar_h = 10

    # Background bar
    cv2.rectangle(canvas, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), COL_BAR_BG, -1)

    # Filled portion
    if n > 1:
        fill_w = int((bar_x1 - bar_x0) * cursor_idx / (n - 1))
        cv2.rectangle(canvas, (bar_x0, bar_y), (bar_x0 + fill_w, bar_y + bar_h),
                      COL_BAR_FG, -1)

    # Cursor handle
    if n > 1:
        cx = bar_x0 + int((bar_x1 - bar_x0) * cursor_idx / (n - 1))
    else:
        cx = bar_x0
    cv2.rectangle(canvas, (cx - 3, bar_y - 3), (cx + 3, bar_y + bar_h + 3),
                  COL_CURSOR, -1)

    # Time labels
    if frames:
        t0 = frames[0].timestamp
        t_cur = frames[min(cursor_idx, len(frames) - 1)].timestamp - t0
        t_end = frames[-1].timestamp - t0

        cv2.putText(canvas, f"{t_cur:.2f}s",
                    (bar_x0, bar_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_HUD, 1)
        cv2.putText(canvas, f"{t_end:.2f}s",
                    (bar_x1 - 40, bar_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_HUD, 1)

    # Frame counter
    cv2.putText(canvas, f"Frame {cursor_idx}/{n - 1}",
                (bar_x0, bar_top + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    # Mode + play state
    play_icon = "||" if playing else ">"
    cv2.putText(canvas, f"{mode_label}  [{play_icon}]",
                (bar_x1 - 160, bar_top + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_HUD, 1)


def draw_hud(canvas, frame, mode):
    """Draw coordinate readout at top of canvas."""
    h, w = canvas.shape[:2]
    y = 22

    if mode == "recording":
        cv2.putText(canvas, "RECORDING", (w - 130, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)
    elif mode == "playback":
        cv2.putText(canvas, "PLAYBACK", (w - 120, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 180, 0), 1)

    if frame is not None:
        p = frame.position
        cv2.putText(canvas,
                    f"Center: ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}) m",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_HUD, 1)

    # Help
    if mode == "playback":
        cv2.putText(canvas, "Click timeline | Arrows=step | Space=play | R=re-record | S=save | Q=quit",
                    (10, h - TIMELINE_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 90), 1)
    else:
        cv2.putText(canvas, "Space=pause | Q=stop+playback",
                    (10, h - TIMELINE_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 90), 1)


def render_frame(canvas, mapper, frames, cursor_idx, mode, playing):
    """Render the full visualization for a given cursor position."""
    canvas[:] = 20

    h = canvas.shape[0]
    scene_h = h - TIMELINE_H

    draw_grid(canvas, mapper)

    if frames:
        frame = frames[min(cursor_idx, len(frames) - 1)]
        draw_trail(canvas, mapper, frames, cursor_idx, scene_h)
        draw_lshape(canvas, mapper, frame, scene_h)
        draw_hud(canvas, frame, mode)
    else:
        draw_hud(canvas, None, mode)

    draw_timeline(canvas, frames, cursor_idx, playing,
                  "REC" if mode == "recording" else "PLAY")


# ──────────────────────────────────────────────────────────────────────────────
# Timeline mouse interaction
# ──────────────────────────────────────────────────────────────────────────────

class TimelineDragger:
    """Handles mouse click/drag on the timeline bar."""

    def __init__(self, canvas_w, canvas_h):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.dragging = False
        self.bar_x0 = 15
        self.bar_x1 = canvas_w - 15
        self.bar_top = canvas_h - TIMELINE_H

    def update_size(self, w, h):
        self.canvas_w = w
        self.canvas_h = h
        self.bar_x0 = 15
        self.bar_x1 = w - 15
        self.bar_top = h - TIMELINE_H

    def hit_test(self, x, y):
        """Check if click is in timeline area."""
        return (y >= self.bar_top and
                self.bar_x0 <= x <= self.bar_x1)

    def x_to_index(self, x, n_frames):
        """Convert pixel x to frame index."""
        if n_frames <= 1:
            return 0
        frac = (x - self.bar_x0) / max(self.bar_x1 - self.bar_x0, 1)
        frac = np.clip(frac, 0.0, 1.0)
        return int(frac * (n_frames - 1))


# ──────────────────────────────────────────────────────────────────────────────
# Recording Phase
# ──────────────────────────────────────────────────────────────────────────────

def record_live(client, body_name, cs100, duration, canvas, mapper,
                window_name, dragger):
    """Record frames from OptiTrack, showing live preview."""
    frames = []
    start_time = time.time()
    recording = True
    paused = False

    print(f"[Tracker] Recording for {duration:.1f}s...")

    while True:
        now = time.time()
        elapsed = now - start_time

        # Get data
        bodies = client.get_rigid_bodies()
        if bodies:
            # Flexible name match
            matched = None
            for name in bodies:
                if (name == body_name or
                    name.lower() == body_name.lower() or
                    body_name.lower().replace("-", "").replace("_", "") in
                    name.lower().replace("-", "").replace("_", "")):
                    matched = name
                    break

            if matched:
                body = bodies[matched]
                pos = body.position
                quat = body.quaternion

                if (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))
                        and np.linalg.norm(pos) < 100.0
                        and np.linalg.norm(quat) > 0.5):

                    markers = cs100.compute_marker_positions(pos, quat)

                    if recording and not paused:
                        frames.append(RecordedFrame(now, pos.copy(), quat.copy(), markers))

                        # Update bounds periodically
                        if len(frames) % 60 == 0:
                            mapper.fit_to_data(frames, padding=0.05)

        # Render live preview
        idx = len(frames) - 1 if frames else 0
        render_frame(canvas, mapper, frames, idx, "recording", not paused)

        # Recording progress overlay
        h, w = canvas.shape[:2]
        pct = min(elapsed / max(duration, 0.1), 1.0)
        cv2.putText(canvas, f"{elapsed:.1f}s / {duration:.0f}s  ({len(frames)} frames)",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_HUD, 1)

        cv2.imshow(window_name, canvas)

        # Auto-stop
        if recording and elapsed >= duration:
            print(f"[Tracker] Recording complete: {len(frames)} frames in {duration:.1f}s")
            break

        key = cv2.waitKey(8) & 0xFF
        if key == ord('q') or key == 27:
            print(f"[Tracker] Stopped early: {len(frames)} frames in {elapsed:.1f}s")
            break
        elif key == ord(' '):
            paused = not paused
            print(f"[Tracker] {'Paused' if paused else 'Recording'}")

    if frames:
        mapper.fit_to_data(frames, padding=0.05)

    return frames


# ──────────────────────────────────────────────────────────────────────────────
# Playback Phase
# ──────────────────────────────────────────────────────────────────────────────

def playback(frames, canvas, mapper, window_name, dragger, auto_save):
    """Interactive timeline scrubber playback."""
    if not frames:
        print("[Tracker] No frames to play back.")
        return

    n = len(frames)
    cursor = 0
    playing = False
    play_speed = 1.0
    play_t0 = None
    play_cursor0 = 0
    snapshot_count = 0

    duration = frames[-1].timestamp - frames[0].timestamp
    print(f"[Tracker] Playback: {n} frames, {duration:.2f}s. "
          f"Click timeline to scrub, arrows to step.")

    if auto_save:
        snapshot_count += 1
        # Render final frame and save
        render_frame(canvas, mapper, frames, n - 1, "playback", False)
        path = f"output/trace_{snapshot_count:04d}.png"
        cv2.imwrite(path, canvas)
        print(f"[Tracker] Auto-saved to {path}")

    mouse_state = {"down": False}

    def on_mouse(event, x, y, flags, param):
        nonlocal cursor, playing
        if event == cv2.EVENT_LBUTTONDOWN:
            if dragger.hit_test(x, y):
                mouse_state["down"] = True
                cursor = dragger.x_to_index(x, n)
                playing = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if mouse_state["down"] and dragger.hit_test(x, y):
                cursor = dragger.x_to_index(x, n)
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_state["down"] = False

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        # Auto-play
        if playing and not mouse_state["down"]:
            if play_t0 is None:
                play_t0 = time.time()
                play_cursor0 = cursor
            elapsed_play = (time.time() - play_t0) * play_speed
            # Map elapsed time to frame index
            if duration > 0:
                target_t = frames[play_cursor0].timestamp - frames[0].timestamp + elapsed_play
                # Find closest frame
                for i in range(play_cursor0, n):
                    if frames[i].timestamp - frames[0].timestamp >= target_t:
                        cursor = i
                        break
                else:
                    cursor = n - 1
                    playing = False
                    play_t0 = None
            else:
                cursor = min(cursor + 1, n - 1)
                if cursor >= n - 1:
                    playing = False

        cursor = max(0, min(cursor, n - 1))

        render_frame(canvas, mapper, frames, cursor, "playback", playing)
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(16) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            playing = not playing
            if playing:
                play_t0 = None  # Reset play timer
            print(f"[Tracker] {'Playing' if playing else 'Paused'}")
        elif key == 81 or key == 2:  # Left arrow
            cursor = max(0, cursor - 1)
            playing = False
        elif key == 83 or key == 3:  # Right arrow
            cursor = min(n - 1, cursor + 1)
            playing = False
        elif key == 80 or key == 0:  # Up arrow — skip 10 frames forward
            cursor = min(n - 1, cursor + 10)
            playing = False
        elif key == 82 or key == 1:  # Down arrow — skip 10 frames back
            cursor = max(0, cursor - 10)
            playing = False
        elif key == ord('a'):  # Home — jump to start
            cursor = 0
            playing = False
        elif key == ord('e'):  # End — jump to end
            cursor = n - 1
            playing = False
        elif key == ord('s'):
            snapshot_count += 1
            path = f"output/trace_{snapshot_count:04d}.png"
            cv2.imwrite(path, canvas)
            print(f"[Tracker] Saved to {path}")
        elif key == ord('r'):
            return "rerecord"

    return "quit"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CS-100 tracker with timeline scrubber")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config", "scene_config.yaml"),
    )
    parser.add_argument("--no-optitrack", action="store_true", help="Demo mode")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Recording duration in seconds (default: 5)")
    parser.add_argument("--width", type=int, default=900, help="Window width")
    parser.add_argument("--height", type=int, default=750, help="Window height")
    parser.add_argument("--save", action="store_true",
                        help="Auto-save trace image after recording")
    parser.add_argument("--body", default=None,
                        help="Rigid body name (default: from config)")
    args = parser.parse_args()

    config = load_config(args.config)
    objects_cfg = config.get("objects", {})

    # Determine body name
    body_name = args.body
    if body_name is None:
        cal_cfg = config.get("calibration", {})
        body_name = cal_cfg.get("tool_body", "Rigid_3_Balls")

    # Create CS-100 geometry model — search by body name then fall back
    cs100_cfg = objects_cfg.get(body_name, {})
    if not cs100_cfg:
        # Search for any cs100_lshape entry
        for name, cfg in objects_cfg.items():
            if cfg.get("render_as") == "cs100_lshape":
                cs100_cfg = cfg
                break

    cs100 = CS100Geometry(
        short_arm_length=cs100_cfg.get("short_arm_length", 0.08),
        long_arm_length=cs100_cfg.get("long_arm_length", 0.10),
    )

    # Canvas and mapper
    canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    mapper = TopDownMapper(args.width, args.height - TIMELINE_H)
    dragger = TimelineDragger(args.width, args.height)

    os.makedirs("output", exist_ok=True)
    window_name = "CS-100 Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║              CS-100 Center Tracker + Timeline                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Phase 1: RECORD for {args.duration:.0f}s                                       ║
║  Phase 2: PLAYBACK — scrub timeline to inspect any moment           ║
║                                                                      ║
║  Body: {body_name:<30s}                             ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # Connect to OptiTrack
    client = None
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

        print(f"[Tracker] NatNet: {client.natnet_version}, "
              f"Server: {client.server_app_name or 'no response'}, "
              f"Bodies: {dict(client._id_to_name)}")

        # Wait for body
        found = False
        for _ in range(50):
            bodies = client.get_rigid_bodies()
            if bodies:
                for name in bodies:
                    if (name == body_name or
                        name.lower() == body_name.lower() or
                        body_name.lower().replace("-", "").replace("_", "") in
                        name.lower().replace("-", "").replace("_", "")):
                        body_name = name
                        found = True
                        break
                if found:
                    break
                # If preferred not found, take first available
                if not found:
                    body_name = list(bodies.keys())[0]
                    found = True
                    break
            time.sleep(0.1)

        if found:
            print(f"[Tracker] Using: '{body_name}'")
        else:
            print("[Tracker] No bodies detected, switching to demo mode.")
            args.no_optitrack = True

    # ── Main loop (supports re-recording) ──
    while True:
        if args.no_optitrack:
            print("[Tracker] Generating demo data...")
            frames = generate_demo_frames(cs100, args.duration)
            mapper.fit_to_data(frames, padding=0.05)
        else:
            frames = record_live(client, body_name, cs100, args.duration,
                                 canvas, mapper, window_name, dragger)

        if not frames:
            print("[Tracker] No frames recorded.")
            break

        result = playback(frames, canvas, mapper, window_name, dragger, args.save)
        if result == "rerecord":
            print("[Tracker] Re-recording...")
            continue
        break

    cv2.destroyAllWindows()
    if client:
        client.stop()
    print(f"[Tracker] Done.")


if __name__ == "__main__":
    main()
