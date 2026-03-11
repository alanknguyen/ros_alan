#!/usr/bin/env python3
"""
run_3d_trace.py — Real-Time 3D Trace Renderer for OptiTrack

Writers: Nguyen Nguyen (Alan), Sauman Raaj

Plots live 3D trajectories of tracked rigid bodies using matplotlib.
Automatically picks up any new body that appears in the OptiTrack stream
and gives it a unique color. Each body gets its own trail.

Coordinate System
-----------------
Raw Y-up from Motive by default (convert_to_zup=False):
    X = right,  Y = up (height),  Z = toward cameras

Use --zup to convert to robotics Z-up convention.

Usage
-----
    python run_3d_trace.py --ip 192.168.0.101
    python run_3d_trace.py --demo
    python run_3d_trace.py --ip 192.168.0.101 --zup
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

# Colors assigned to bodies in discovery order
COLORS = ["#ff6b6b", "#4ecdc4", "#ffd43b", "#cc5de8", "#339af0",
           "#51cf66", "#ff922b", "#f06595", "#20c997", "#a9e34b"]


# --- Demo ---

class DemoOptiTrack3D:
    """Simulates two rigid bodies. Second one appears after 6 seconds."""

    def __init__(self):
        self._t0 = time.time()

    def get_rigid_bodies(self):
        t = time.time() - self._t0
        now = time.time()
        bodies = {}

        # Body 1: helix (always present)
        x1 = 0.20 * math.cos(t * 0.5)
        z1 = 0.20 * math.sin(t * 0.5)
        y1 = 0.80 + 0.05 * math.sin(t * 0.15)
        bodies["rigid_body_12"] = RigidBodyState(
            name="rigid_body_12", id=12,
            position=np.array([x1, y1, z1]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            timestamp=now, tracking_valid=True,
        )

        # Body 2: figure-8 (appears at t=6s)
        if t >= 6.0:
            t2 = t - 6.0
            d = 1.0 + math.sin(t2 * 0.7) ** 2
            x2 = 0.25 + 0.12 * math.cos(t2 * 0.7) / d
            z2 = 0.12 * math.sin(t2 * 0.7) * math.cos(t2 * 0.7) / d
            y2 = 0.90 + 0.03 * math.sin(t2 * 0.3)
            bodies["rigid_body_13"] = RigidBodyState(
                name="rigid_body_13", id=13,
                position=np.array([x2, y2, z2]),
                quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
                timestamp=now, tracking_valid=True,
            )

        return bodies

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

    TRAIL_MAX = args.trail

    if args.demo:
        print("[3D Trace] Demo mode — body 1 now, body 2 at t=6s")
        client = DemoOptiTrack3D()
    else:
        print(f"[3D Trace] Connecting to OptiTrack at {args.ip} "
              f"({'Z-up' if args.zup else 'raw Y-up'})...")
        client = OptiTrackClient(server_ip=args.ip, convert_to_zup=args.zup)
        client.start()

    # Per-body storage: name -> {xs, ys, zs, trail_line, dot, color}
    tracked = {}

    t0 = time.time()
    last_print = [0.0]
    color_idx = [0]

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    coord = "Z-up" if args.zup else "Y-up (raw)"
    ax.set_title(f"3D Trace  [{coord}]", fontsize=13, fontweight="bold", pad=15)
    if args.zup:
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m) ↑")
    else:
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m) ↑ UP"); ax.set_zlabel("Z (m)")
    ax.tick_params(labelsize=7)

    status_text = fig.text(0.02, 0.02, "Scanning...",
                           fontsize=8, color="#aaaaaa", family="monospace")

    pad = 0.05

    def add_body(name):
        """Register a new body with its own trail and color."""
        c = COLORS[color_idx[0] % len(COLORS)]
        color_idx[0] += 1
        line, = ax.plot([], [], [], color=c, linewidth=1.2, alpha=0.6, label=name)
        dot, = ax.plot([], [], [], 'o', color=c, markersize=10,
                       markeredgecolor="white", markeredgewidth=1.5)
        tracked[name] = {
            "xs": deque(maxlen=TRAIL_MAX),
            "ys": deque(maxlen=TRAIL_MAX),
            "zs": deque(maxlen=TRAIL_MAX),
            "line": line, "dot": dot, "color": c,
        }
        ax.legend(loc="upper left", fontsize=7, framealpha=0.3)
        print(f"  [NEW] {name} — color {c}")

    def update(_):
        t_now = time.time() - t0
        bodies = client.get_rigid_bodies()

        for name, body in bodies.items():
            pos = body.position
            if not (np.all(np.isfinite(pos)) and np.linalg.norm(pos) < 50.0):
                continue

            # First time seeing this body? Register it.
            if name not in tracked:
                add_body(name)

            b = tracked[name]
            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            b["xs"].append(px); b["ys"].append(py); b["zs"].append(pz)

            xa, ya, za = list(b["xs"]), list(b["ys"]), list(b["zs"])
            b["line"].set_data(xa, ya); b["line"].set_3d_properties(za)
            b["dot"].set_data([px], [py]); b["dot"].set_3d_properties([pz])

        # Auto-scale axes across ALL bodies
        all_x, all_y, all_z = [], [], []
        for b in tracked.values():
            all_x.extend(b["xs"]); all_y.extend(b["ys"]); all_z.extend(b["zs"])

        if len(all_x) > 2:
            for arr, setter in [
                (all_x, ax.set_xlim), (all_y, ax.set_ylim), (all_z, ax.set_zlim)
            ]:
                lo, hi = min(arr), max(arr)
                span = max(hi - lo, 0.02)
                setter(lo - pad * span, hi + pad * span)

        # Status
        total_pts = sum(len(b["xs"]) for b in tracked.values())
        status_text.set_text(
            f"{len(tracked)} bodies  |  {total_pts} pts  |  {t_now:.1f}s")
        status_text.set_color("#51cf66" if tracked else "#ffd43b")

        # Terminal log
        if t_now - last_print[0] >= 2.0 and tracked:
            last_print[0] = t_now
            parts = [f"{n}: ({list(b['xs'])[-1]:+.4f}, {list(b['ys'])[-1]:+.4f}, "
                     f"{list(b['zs'])[-1]:+.4f})"
                     for n, b in tracked.items() if b["xs"]]
            print(f"  [{t_now:6.1f}s] {' | '.join(parts)}")

    anim = FuncAnimation(fig, update, interval=max(16, int(1000 / args.rate)),
                         blit=False, cache_frame_data=False)

    print(f"[3D Trace] {args.rate} Hz, trail={TRAIL_MAX} per body")
    print(f"[3D Trace] Close window to stop.\n")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass

    if hasattr(client, "stop"):
        client.stop()
    print(f"[3D Trace] Done. {len(tracked)} bodies tracked.")


def main():
    p = argparse.ArgumentParser(description="Real-time 3D trace of OptiTrack rigid bodies")
    p.add_argument("--ip", default="192.168.0.101")
    p.add_argument("--rate", type=int, default=30)
    p.add_argument("--trail", type=int, default=3000)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--zup", action="store_true")
    run_3d_trace(p.parse_args())


if __name__ == "__main__":
    main()
