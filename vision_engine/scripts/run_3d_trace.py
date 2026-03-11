#!/usr/bin/env python3
"""
run_3d_trace.py — Real-Time 3D Trace Renderer for OptiTrack Rigid Bodies

Writers: Nguyen Nguyen (Alan), Sauman Raaj

Plots the live 3D trajectory of a tracked rigid body (default: rigid_body_12)
using matplotlib's 3D axes. The trace fades from dim to bright as it progresses,
showing the object's current position as a highlighted sphere.

Coordinate System
-----------------
Uses raw Y-up values from Motive by default (convert_to_zup=False)
so you see exactly what OptiTrack reports:
    X = right,  Y = up (height),  Z = toward cameras

Use --zup to convert to robotics Z-up convention.

Usage
-----
    # Live, track rigid_body_12:
    python run_3d_trace.py --body rigid_body_12 --ip 192.168.0.101

    # Demo mode (simulated data, no hardware):
    python run_3d_trace.py --demo

    # With Z-up conversion:
    python run_3d_trace.py --body rigid_body_12 --ip 192.168.0.101 --zup

    # Longer trail, slower update:
    python run_3d_trace.py --demo --trail 5000 --rate 15
"""

import sys
import time
import math
import argparse
import numpy as np
from collections import deque
from pathlib import Path

# Resolve imports from parent directory
_script_dir = Path(__file__).resolve().parent
_engine_dir = _script_dir.parent
if str(_engine_dir) not in sys.path:
    sys.path.insert(0, str(_engine_dir))

from cv.optitrack_client import OptiTrackClient, RigidBodyState


# --- Demo Data Generator ---

class DemoOptiTrack3D:
    """Simulates a rigid body tracing a 3D helix for demo/testing."""

    def __init__(self, body_name: str = "rigid_body_12"):
        self._t0 = time.time()
        self._body_name = body_name

    def get_rigid_bodies(self):
        t = time.time() - self._t0
        now = time.time()

        # 3D helix: circular XZ motion + slow Y drift
        radius = 0.20
        speed = 0.5
        x = radius * math.cos(t * speed)
        z = radius * math.sin(t * speed)
        y = 0.8 + 0.05 * math.sin(t * 0.15)  # gentle vertical bob

        # Add some noise to make it realistic
        x += 0.002 * math.sin(t * 7.3)
        z += 0.002 * math.cos(t * 5.1)

        qw = math.cos(t * speed / 2)
        qy = math.sin(t * speed / 2)

        return {
            self._body_name: RigidBodyState(
                name=self._body_name, id=12,
                position=np.array([x, y, z]),
                quaternion=np.array([0.0, qy, 0.0, qw]),
                timestamp=now, tracking_valid=True,
            ),
        }

    def get_frame_count(self):
        return int((time.time() - self._t0) * 120)

    def start(self):
        pass

    def stop(self):
        pass


# --- Main 3D Trace Renderer ---

def run_3d_trace(args):
    """Real-time 3D trace of a rigid body using matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    BODY_NAME = args.body
    TRAIL_MAX = args.trail

    # Connect
    if args.demo:
        print(f"[3D Trace] Demo mode — simulated 3D helix for '{BODY_NAME}'")
        client = DemoOptiTrack3D(body_name=BODY_NAME)
    else:
        convert = args.zup
        print(f"[3D Trace] Connecting to OptiTrack at {args.ip} "
              f"({'Z-up' if convert else 'raw Y-up'})...")
        client = OptiTrackClient(server_ip=args.ip, convert_to_zup=convert)
        client.start()

    # Data buffers
    xs = deque(maxlen=TRAIL_MAX)
    ys = deque(maxlen=TRAIL_MAX)
    zs = deque(maxlen=TRAIL_MAX)
    ts_buf = deque(maxlen=TRAIL_MAX)

    t0 = time.time()
    last_print = [0.0]

    # Set up figure
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    coord_label = "Z-up" if args.zup else "Y-up (raw)"
    ax.set_title(f"3D Trace — '{BODY_NAME}'  [{coord_label}]",
                 fontsize=13, fontweight="bold", pad=15)

    if args.zup:
        ax.set_xlabel("X (m)", fontsize=9)
        ax.set_ylabel("Y (m)", fontsize=9)
        ax.set_zlabel("Z (m) ↑ UP", fontsize=9)
    else:
        ax.set_xlabel("X (m) ← right →", fontsize=9)
        ax.set_ylabel("Y (m) ↑ UP", fontsize=9)
        ax.set_zlabel("Z (m) ← fwd →", fontsize=9)

    ax.tick_params(labelsize=7)

    # Initialize plot elements
    trail_line, = ax.plot([], [], [], color="#ff6b6b", linewidth=1.0, alpha=0.6)
    current_dot, = ax.plot([], [], [], 'o', color="#ffd43b", markersize=10,
                           markeredgecolor="white", markeredgewidth=1.5)
    ghost_dot, = ax.plot([], [], [], 'o', color="#ff6b6b", markersize=5, alpha=0.3)

    # Status text
    status_text = fig.text(0.02, 0.02, "Waiting for body...",
                           fontsize=8, color="#aaaaaa", family="monospace")
    pos_text = fig.text(0.02, 0.05, "",
                        fontsize=9, color="#51cf66", family="monospace")

    # For auto-scaling
    range_pad = 0.05
    seen_data = [False]

    def update(frame_num):
        t_now = time.time() - t0
        bodies = client.get_rigid_bodies()
        body = bodies.get(BODY_NAME)

        if body is not None:
            pos = body.position
            if np.all(np.isfinite(pos)) and np.linalg.norm(pos) < 50.0:
                px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
                xs.append(px)
                ys.append(py)
                zs.append(pz)
                ts_buf.append(t_now)
                seen_data[0] = True

                # Update trail line
                x_arr = list(xs)
                y_arr = list(ys)
                z_arr = list(zs)
                trail_line.set_data(x_arr, y_arr)
                trail_line.set_3d_properties(z_arr)

                # Current position (bright dot)
                current_dot.set_data([px], [py])
                current_dot.set_3d_properties([pz])

                # Ghost dot at first point
                if len(xs) > 1:
                    ghost_dot.set_data([x_arr[0]], [y_arr[0]])
                    ghost_dot.set_3d_properties([z_arr[0]])

                # Auto-scale axes
                if len(xs) > 2:
                    x_min, x_max = min(x_arr), max(x_arr)
                    y_min, y_max = min(y_arr), max(y_arr)
                    z_min, z_max = min(z_arr), max(z_arr)

                    # Ensure minimum range so axes don't collapse
                    for arr_min, arr_max in [(x_min, x_max), (y_min, y_max), (z_min, z_max)]:
                        if arr_max - arr_min < 0.02:
                            mid = (arr_max + arr_min) / 2
                            arr_min = mid - 0.01
                            arr_max = mid + 0.01

                    x_span = x_max - x_min
                    y_span = y_max - y_min
                    z_span = z_max - z_min

                    ax.set_xlim(x_min - range_pad * x_span, x_max + range_pad * x_span)
                    ax.set_ylim(y_min - range_pad * y_span, y_max + range_pad * y_span)
                    ax.set_zlim(z_min - range_pad * z_span, z_max + range_pad * z_span)

                # Status text
                q = body.quaternion
                pos_text.set_text(
                    f"pos=({px:+.4f}, {py:+.4f}, {pz:+.4f})  "
                    f"quat=({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}, {q[3]:+.3f})"
                )
                status_text.set_text(
                    f"TRACKING  |  {len(xs)} pts  |  {t_now:.1f}s"
                )
                status_text.set_color("#51cf66")

                # Terminal print every 1s
                if t_now - last_print[0] >= 1.0:
                    last_print[0] = t_now
                    print(f"[{t_now:6.1f}s] "
                          f"pos=({px:+.5f}, {py:+.5f}, {pz:+.5f})  "
                          f"|  {len(xs)} trail pts")
            else:
                status_text.set_text(f"INVALID DATA  |  {t_now:.1f}s")
                status_text.set_color("#ff6b6b")
        else:
            names = list(bodies.keys()) if bodies else ["(none)"]
            status_text.set_text(
                f"BODY NOT FOUND: '{BODY_NAME}'  |  "
                f"Available: {', '.join(names)}  |  {t_now:.1f}s"
            )
            status_text.set_color("#ffd43b")

            if t_now - last_print[0] >= 2.0:
                last_print[0] = t_now
                print(f"[{t_now:6.1f}s] BODY '{BODY_NAME}' NOT FOUND. "
                      f"Available: {', '.join(names)}")

        return trail_line, current_dot, ghost_dot, status_text, pos_text

    # Run animation
    interval_ms = max(16, int(1000 / args.rate))
    anim = FuncAnimation(fig, update, interval=interval_ms,
                         blit=False, cache_frame_data=False)

    print(f"[3D Trace] Tracking '{BODY_NAME}' at {args.rate} Hz")
    print(f"[3D Trace] Trail buffer: {TRAIL_MAX} points")
    print(f"[3D Trace] Close matplotlib window to stop.\n")

    try:
        plt.show()
    except KeyboardInterrupt:
        pass

    if hasattr(client, "stop"):
        client.stop()
    print(f"[3D Trace] Done. {len(xs)} trail points recorded.")


# --- Entry Point ---

def main():
    parser = argparse.ArgumentParser(
        description="Real-time 3D trace of OptiTrack rigid body",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_3d_trace.py --demo                                   # simulated helix
  python run_3d_trace.py --body rigid_body_12 --ip 192.168.0.101  # live tracking
  python run_3d_trace.py --body rigid_body_12 --ip 192.168.0.101 --zup  # Z-up
  python run_3d_trace.py --demo --trail 5000 --rate 15            # longer trail
        """,
    )
    parser.add_argument("--body", default="rigid_body_12",
                        help="Rigid body name to track (default: rigid_body_12)")
    parser.add_argument("--ip", default="192.168.0.101",
                        help="OptiTrack server IP (default: 192.168.0.101)")
    parser.add_argument("--rate", type=int, default=30,
                        help="Update rate in Hz (default: 30)")
    parser.add_argument("--trail", type=int, default=3000,
                        help="Max trail points to keep (default: 3000)")
    parser.add_argument("--demo", action="store_true",
                        help="Simulated data, no OptiTrack needed")
    parser.add_argument("--zup", action="store_true",
                        help="Convert to Z-up (robotics convention). "
                             "Default: raw Y-up from Motive.")

    args = parser.parse_args()
    run_3d_trace(args)


if __name__ == "__main__":
    main()
