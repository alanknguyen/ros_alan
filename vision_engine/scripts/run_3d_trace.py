#!/usr/bin/env python3
"""
run_3d_trace.py — Real-Time 3D Trace Renderer for OptiTrack

Writers: Nguyen Nguyen (Alan), Sauman Raaj

Plots the live 3D trajectory of a single tracked rigid body using
matplotlib's 3D axes. Shows the object's current position as a bright
dot with a fading trail behind it.

Coordinate System
-----------------
Raw Y-up from Motive by default (convert_to_zup=False):
    X = right,  Y = up (height),  Z = toward cameras

Use --zup to convert to robotics Z-up convention.

Usage
-----
    python run_3d_trace.py --body rigid_body_12 --ip 192.168.0.101
    python run_3d_trace.py --demo
    python run_3d_trace.py --body rigid_body_12 --ip 192.168.0.101 --zup
    python run_3d_trace.py --demo --trail 5000 --rate 15
"""

from __future__ import annotations

import sys
import time
import math
import argparse
import numpy as np
from collections import deque
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_engine_dir = _script_dir.parent
if str(_engine_dir) not in sys.path:
    sys.path.insert(0, str(_engine_dir))

from cv.optitrack_client import OptiTrackClient, RigidBodyState


# --- Demo ---

class DemoOptiTrack3D:
    """Simulates a rigid body tracing a 3D helix."""

    def __init__(self, body_name="rigid_body_12"):
        self._t0 = time.time()
        self._body_name = body_name

    def get_rigid_bodies(self):
        t = time.time() - self._t0
        now = time.time()
        r, s = 0.20, 0.5
        x = r * math.cos(t * s) + 0.002 * math.sin(t * 7.3)
        z = r * math.sin(t * s) + 0.002 * math.cos(t * 5.1)
        y = 0.8 + 0.05 * math.sin(t * 0.15)
        qw = math.cos(t * s / 2)
        qy = math.sin(t * s / 2)
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


# --- Main ---

def run_3d_trace(args):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    BODY_NAME = args.body
    TRAIL_MAX = args.trail

    if args.demo:
        print(f"[3D Trace] Demo mode — simulated helix for '{BODY_NAME}'")
        client = DemoOptiTrack3D(body_name=BODY_NAME)
    else:
        print(f"[3D Trace] Connecting to OptiTrack at {args.ip} "
              f"({'Z-up' if args.zup else 'raw Y-up'})...")
        client = OptiTrackClient(server_ip=args.ip, convert_to_zup=args.zup)
        client.start()

    xs = deque(maxlen=TRAIL_MAX)
    ys = deque(maxlen=TRAIL_MAX)
    zs = deque(maxlen=TRAIL_MAX)

    t0 = time.time()
    last_print = [0.0]

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    coord = "Z-up" if args.zup else "Y-up (raw)"
    ax.set_title(f"3D Trace — '{BODY_NAME}'  [{coord}]",
                 fontsize=13, fontweight="bold", pad=15)
    if args.zup:
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m) ↑")
    else:
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m) ↑ UP")
        ax.set_zlabel("Z (m)")
    ax.tick_params(labelsize=7)

    trail_line, = ax.plot([], [], [], color="#ff6b6b", linewidth=1.0, alpha=0.6)
    current_dot, = ax.plot([], [], [], 'o', color="#ffd43b", markersize=10,
                           markeredgecolor="white", markeredgewidth=1.5)
    status_text = fig.text(0.02, 0.02, "Waiting...",
                           fontsize=8, color="#aaaaaa", family="monospace")
    pos_text = fig.text(0.02, 0.05, "",
                        fontsize=9, color="#51cf66", family="monospace")

    pad = 0.05

    def update(_):
        t_now = time.time() - t0
        bodies = client.get_rigid_bodies()
        body = bodies.get(BODY_NAME)

        if body is not None:
            pos = body.position
            if np.all(np.isfinite(pos)) and np.linalg.norm(pos) < 50.0:
                px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
                xs.append(px); ys.append(py); zs.append(pz)

                xa, ya, za = list(xs), list(ys), list(zs)
                trail_line.set_data(xa, ya)
                trail_line.set_3d_properties(za)
                current_dot.set_data([px], [py])
                current_dot.set_3d_properties([pz])

                if len(xs) > 2:
                    for arr, setter in [
                        (xa, ax.set_xlim), (ya, ax.set_ylim), (za, ax.set_zlim)
                    ]:
                        lo, hi = min(arr), max(arr)
                        span = max(hi - lo, 0.02)
                        setter(lo - pad * span, hi + pad * span)

                q = body.quaternion
                pos_text.set_text(
                    f"pos=({px:+.4f}, {py:+.4f}, {pz:+.4f})  "
                    f"quat=({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}, {q[3]:+.3f})")
                status_text.set_text(f"TRACKING  |  {len(xs)} pts  |  {t_now:.1f}s")
                status_text.set_color("#51cf66")

                if t_now - last_print[0] >= 1.0:
                    last_print[0] = t_now
                    print(f"[{t_now:6.1f}s] pos=({px:+.5f}, {py:+.5f}, {pz:+.5f})")
            else:
                status_text.set_text(f"INVALID DATA  |  {t_now:.1f}s")
                status_text.set_color("#ff6b6b")
        else:
            names = list(bodies.keys()) if bodies else ["(none)"]
            status_text.set_text(
                f"NOT FOUND: '{BODY_NAME}'  |  available: {', '.join(names)}")
            status_text.set_color("#ffd43b")

        return trail_line, current_dot, status_text, pos_text

    anim = FuncAnimation(fig, update, interval=max(16, int(1000 / args.rate)),
                         blit=False, cache_frame_data=False)

    print(f"[3D Trace] Tracking '{BODY_NAME}' at {args.rate} Hz, trail={TRAIL_MAX}")
    print(f"[3D Trace] Close window to stop.\n")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass

    if hasattr(client, "stop"):
        client.stop()
    print(f"[3D Trace] Done. {len(xs)} trail points.")


def main():
    p = argparse.ArgumentParser(description="Real-time 3D trace of OptiTrack rigid body")
    p.add_argument("--body", default="rigid_body_12")
    p.add_argument("--ip", default="192.168.0.101")
    p.add_argument("--rate", type=int, default=30)
    p.add_argument("--trail", type=int, default=3000)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--zup", action="store_true")
    main_args = p.parse_args()
    run_3d_trace(main_args)


if __name__ == "__main__":
    main()
