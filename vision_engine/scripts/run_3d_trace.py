#!/usr/bin/env python3
"""
run_3d_trace.py — Multi-Body 3D Trace Renderer for OptiTrack

Writers: Nguyen Nguyen (Alan), Sauman Raaj

Automatically discovers and traces ALL rigid bodies visible to OptiTrack
in real-time 3D. When a new body enters the tracking volume, it is
detected, assigned a unique color, and traced independently.

Identity Model
--------------
Each rigid body in Motive has a unique integer ID (assigned when you
create the rigid body). Our NatNet client maps these to names like
"rigid_body_12". This ID is the GROUND TRUTH for identity — it never
changes, even if the physical ball is moved, occluded, or leaves and
re-enters the tracking volume.

The tracker uses a three-tier lifecycle:

    ACTIVE   — body is in the current OptiTrack frame, position is valid.
               Trail is drawn solid, bright dot at current position.

    STALE    — body was in a recent frame but dropped out (occlusion,
               edge of FOV). Trail stays visible, dot dims. If the body
               returns within STALE_TIMEOUT seconds, it seamlessly
               resumes. This prevents flicker on brief tracking drops.

    LOST     — body has been missing for > STALE_TIMEOUT seconds.
               Trail fades to dim. If it returns, a console message
               announces the recovery.

Anti-Confusion Logic
--------------------
- Identity is NEVER based on position. Two balls can cross paths and
  their trails will never swap, because Motive's rigid body IDs are
  stable.
- Proximity warning: when two active bodies are within 5cm of each
  other, a warning is printed. This alerts you that the physical balls
  are close and Motive might struggle with marker assignment — but the
  software will still track them by their Motive IDs.
- Each body's trail buffer is completely independent (separate deques).
- The only way a body can be "confused" is if Motive itself swaps IDs,
  which requires re-creating rigid bodies.

Coordinate System
-----------------
Raw Y-up from Motive by default (no conversion):
    X = right,  Y = up (height),  Z = toward cameras

Use --zup to convert to robotics Z-up convention.

Usage
-----
    # Auto-discover all bodies, live:
    python run_3d_trace.py --ip 192.168.0.101

    # Demo mode (body 1 at start, body 2 appears after 6s):
    python run_3d_trace.py --demo

    # Only trace bodies matching a prefix:
    python run_3d_trace.py --ip 192.168.0.101 --filter rigid_body_

    # With Z-up conversion:
    python run_3d_trace.py --ip 192.168.0.101 --zup
"""

from __future__ import annotations

import sys
import time
import math
import argparse
import numpy as np
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Resolve imports from parent directory
_script_dir = Path(__file__).resolve().parent
_engine_dir = _script_dir.parent
if str(_engine_dir) not in sys.path:
    sys.path.insert(0, str(_engine_dir))

from cv.optitrack_client import OptiTrackClient, RigidBodyState


# --- Constants ---

# Lifecycle thresholds
STALE_TIMEOUT_SEC = 2.0     # seconds before ACTIVE->STALE transitions to LOST
PROXIMITY_WARN_M = 0.05     # 5cm — warn when two bodies are this close

# Color palette — perceptually distinct, ordered by visibility on dark bg
COLOR_PALETTE = [
    ("#ff6b6b", "Coral"),       # body 1
    ("#4ecdc4", "Cyan"),        # body 2
    ("#ffd43b", "Gold"),        # body 3
    ("#cc5de8", "Violet"),      # body 4
    ("#339af0", "Sky Blue"),    # body 5
    ("#51cf66", "Green"),       # body 6
    ("#ff922b", "Orange"),      # body 7
    ("#f06595", "Pink"),        # body 8
    ("#20c997", "Teal"),        # body 9
    ("#a9e34b", "Lime"),        # body 10
]


# --- Per-Body Tracking State ---

class TrackedBody:
    """
    Complete tracking state for a single rigid body.

    Each instance owns its own trail buffers and matplotlib artists.
    Identity is locked to the Motive rigid body name (which encodes
    the Motive-assigned integer ID).
    """

    def __init__(self, name: str, motive_id: int, color_hex: str,
                 color_label: str, trail_max: int, discovery_time: float):
        # Identity (immutable after creation)
        self.name = name
        self.motive_id = motive_id
        self.color_hex = color_hex
        self.color_label = color_label

        # Lifecycle
        self.state = "active"           # "active", "stale", or "lost"
        self.first_seen_t = discovery_time
        self.last_seen_t = discovery_time
        self.last_active_t = discovery_time
        self.total_samples = 0
        self.stale_notified = False     # avoid spamming stale messages
        self.lost_notified = False      # avoid spamming lost messages

        # Trail data (independent per body)
        self.trail_x = deque(maxlen=trail_max)
        self.trail_y = deque(maxlen=trail_max)
        self.trail_z = deque(maxlen=trail_max)
        self.trail_t = deque(maxlen=trail_max)

        # Latest pose
        self.last_position = None       # np.ndarray (3,)
        self.last_quaternion = None     # np.ndarray (4,)

        # Matplotlib artists (created lazily when first drawn)
        self.trail_line = None
        self.current_dot = None
        self.label_text = None

    def record_sample(self, position: np.ndarray, quaternion: np.ndarray,
                      t: float) -> None:
        """Record a valid position sample and update lifecycle."""
        px, py, pz = float(position[0]), float(position[1]), float(position[2])
        self.trail_x.append(px)
        self.trail_y.append(py)
        self.trail_z.append(pz)
        self.trail_t.append(t)
        self.last_position = position.copy()
        self.last_quaternion = quaternion.copy()
        self.last_seen_t = t
        self.last_active_t = t
        self.total_samples += 1

        # Reset lifecycle flags on re-acquisition
        was_lost = self.state == "lost"
        self.state = "active"
        self.stale_notified = False
        self.lost_notified = False

        return was_lost  # caller can print recovery message

    def update_lifecycle(self, t_now: float) -> Optional[str]:
        """
        Update lifecycle state based on time since last seen.
        Returns a transition event string, or None.

        Transitions:
            active -> stale   (after 0.5s of no data)
            stale  -> lost    (after STALE_TIMEOUT_SEC total)
        """
        if self.state == "active":
            if t_now - self.last_active_t > 0.5:
                self.state = "stale"
                if not self.stale_notified:
                    self.stale_notified = True
                    return "stale"

        if self.state == "stale":
            if t_now - self.last_active_t > STALE_TIMEOUT_SEC:
                self.state = "lost"
                if not self.lost_notified:
                    self.lost_notified = True
                    return "lost"

        return None

    @property
    def age(self) -> float:
        """Seconds since first discovery."""
        return self.last_seen_t - self.first_seen_t

    @property
    def trail_length(self) -> int:
        return len(self.trail_x)


# --- Body Registry ---

class BodyRegistry:
    """
    Manages discovery, tracking, and retirement of multiple rigid bodies.

    This is the core anti-confusion layer. Bodies are keyed by their
    Motive name (e.g., "rigid_body_12") which is derived from the
    integer ID assigned in Motive. This ID is stable across tracking
    drops, occlusion, and re-entry.
    """

    def __init__(self, trail_max: int, name_filter: Optional[str] = None):
        self.trail_max = trail_max
        self.name_filter = name_filter      # only track names containing this
        self._bodies: OrderedDict[str, TrackedBody] = OrderedDict()
        self._color_index = 0
        self._discovery_order: List[str] = []

    @property
    def bodies(self) -> OrderedDict[str, TrackedBody]:
        return self._bodies

    @property
    def active_bodies(self) -> List[TrackedBody]:
        return [b for b in self._bodies.values() if b.state == "active"]

    @property
    def visible_bodies(self) -> List[TrackedBody]:
        """Bodies with trails to draw (active or stale or recently lost)."""
        return [b for b in self._bodies.values() if b.trail_length > 0]

    def update(self, optitrack_bodies: Dict[str, RigidBodyState],
               t_now: float) -> List[str]:
        """
        Process a new frame of OptiTrack data.

        Returns a list of event strings for console output:
            "[NEW] rigid_body_13 (Cyan) — 2 bodies now tracked"
            "[STALE] rigid_body_12 — no data for 0.5s"
            "[LOST] rigid_body_12 — no data for 2.0s"
            "[RECOVERED] rigid_body_12 — back after 3.1s"
            "[PROXIMITY] rigid_body_12 <-> rigid_body_13: 3.2cm apart"
        """
        events = []

        # Phase 1: Process each body in the current frame
        for name, rb_state in optitrack_bodies.items():
            # Apply name filter if set
            if self.name_filter and self.name_filter not in name:
                continue

            # Validate position data
            pos = rb_state.position
            quat = rb_state.quaternion
            if not (np.all(np.isfinite(pos)) and np.linalg.norm(pos) < 50.0):
                continue
            if not (np.all(np.isfinite(quat)) and np.linalg.norm(quat) > 0.1):
                continue

            if name not in self._bodies:
                # NEW BODY DISCOVERED
                color_hex, color_label = self._next_color()
                body = TrackedBody(
                    name=name,
                    motive_id=rb_state.id,
                    color_hex=color_hex,
                    color_label=color_label,
                    trail_max=self.trail_max,
                    discovery_time=t_now,
                )
                self._bodies[name] = body
                self._discovery_order.append(name)
                events.append(
                    f"[NEW] {name} (id={rb_state.id}, {color_label}) "
                    f"— {len(self._bodies)} body(ies) now tracked"
                )

            # Record sample
            tracked = self._bodies[name]
            was_lost = tracked.record_sample(pos, quat, t_now)
            if was_lost:
                gap = t_now - tracked.last_active_t + (t_now - tracked.last_seen_t)
                events.append(
                    f"[RECOVERED] {name} — back after "
                    f"{t_now - tracked.first_seen_t:.1f}s"
                )

        # Phase 2: Update lifecycle for bodies NOT in current frame
        for name, tracked in self._bodies.items():
            if name not in optitrack_bodies or (
                self.name_filter and self.name_filter not in name
            ):
                event = tracked.update_lifecycle(t_now)
                if event == "stale":
                    events.append(
                        f"[STALE] {name} — no data for "
                        f"{t_now - tracked.last_active_t:.1f}s"
                    )
                elif event == "lost":
                    events.append(
                        f"[LOST] {name} — no data for "
                        f"{t_now - tracked.last_active_t:.1f}s"
                    )

        # Phase 3: Proximity warnings between active bodies
        active = self.active_bodies
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                if a.last_position is not None and b.last_position is not None:
                    dist = float(np.linalg.norm(a.last_position - b.last_position))
                    if dist < PROXIMITY_WARN_M:
                        events.append(
                            f"[PROXIMITY] {a.name} <-> {b.name}: "
                            f"{dist*100:.1f}cm apart — watch for Motive ID swap!"
                        )

        return events

    def _next_color(self) -> Tuple[str, str]:
        """Assign the next color from the palette (wraps around)."""
        idx = self._color_index % len(COLOR_PALETTE)
        self._color_index += 1
        return COLOR_PALETTE[idx]

    def get_global_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the bounding box enclosing ALL body trails.
        Returns (min_xyz, max_xyz) or None if no data.
        """
        all_x, all_y, all_z = [], [], []
        for body in self._bodies.values():
            if body.trail_length > 0:
                all_x.extend(body.trail_x)
                all_y.extend(body.trail_y)
                all_z.extend(body.trail_z)

        if not all_x:
            return None

        mins = np.array([min(all_x), min(all_y), min(all_z)])
        maxs = np.array([max(all_x), max(all_y), max(all_z)])

        # Enforce minimum span so axes don't collapse
        for i in range(3):
            if maxs[i] - mins[i] < 0.02:
                mid = (maxs[i] + mins[i]) / 2
                mins[i] = mid - 0.01
                maxs[i] = mid + 0.01

        return mins, maxs


# --- Demo Data Generator (Multi-Body) ---

class DemoMultiBody:
    """
    Simulates multiple rigid bodies appearing over time.

    Body 1 ("rigid_body_12"): appears at t=0, traces a 3D helix.
    Body 2 ("rigid_body_13"): appears at t=6s, traces a tilted figure-8.
    Body 2 briefly disappears at t=14-15s to test stale/recovery logic.
    """

    def __init__(self):
        self._t0 = time.time()

    def get_rigid_bodies(self) -> Dict[str, RigidBodyState]:
        t = time.time() - self._t0
        now = time.time()
        bodies = {}

        # Body 1: 3D helix — always present from t=0
        r1 = 0.18
        s1 = 0.5
        x1 = r1 * math.cos(t * s1)
        z1 = r1 * math.sin(t * s1)
        y1 = 0.80 + 0.04 * math.sin(t * 0.2)
        x1 += 0.001 * math.sin(t * 7.3)
        z1 += 0.001 * math.cos(t * 5.1)

        qw1 = math.cos(t * s1 / 2)
        qy1 = math.sin(t * s1 / 2)

        bodies["rigid_body_12"] = RigidBodyState(
            name="rigid_body_12", id=12,
            position=np.array([x1, y1, z1]),
            quaternion=np.array([0.0, qy1, 0.0, qw1]),
            timestamp=now, tracking_valid=True,
        )

        # Body 2: tilted figure-8 — appears at t=6s
        # Briefly disappears at t=14-15s to test stale logic
        if t >= 6.0:
            in_dropout = 14.0 <= t <= 15.0
            if not in_dropout:
                t2 = t - 6.0
                s2 = 0.7
                r2 = 0.12
                denom = 1.0 + math.sin(t2 * s2) ** 2
                x2 = 0.25 + r2 * math.cos(t2 * s2) / denom
                z2 = r2 * math.sin(t2 * s2) * math.cos(t2 * s2) / denom
                y2 = 0.90 + 0.03 * math.sin(t2 * 0.3)
                x2 += 0.001 * math.sin(t2 * 6.1)
                z2 += 0.001 * math.cos(t2 * 4.7)

                qw2 = math.cos(t2 * s2 * 0.3)
                qz2 = math.sin(t2 * s2 * 0.3)

                bodies["rigid_body_13"] = RigidBodyState(
                    name="rigid_body_13", id=13,
                    position=np.array([x2, y2, z2]),
                    quaternion=np.array([0.0, 0.0, qz2, qw2]),
                    timestamp=now, tracking_valid=True,
                )

        return bodies

    def get_frame_count(self):
        return int((time.time() - self._t0) * 120)

    def start(self):
        pass

    def stop(self):
        pass


# --- Matplotlib 3D Renderer ---

def _ensure_artists(body: TrackedBody, ax) -> None:
    """Lazily create matplotlib artists for a body on first draw."""
    if body.trail_line is None:
        # Trail line
        body.trail_line, = ax.plot(
            [], [], [],
            color=body.color_hex,
            linewidth=1.4,
            alpha=0.7,
            label=f"{body.name} ({body.color_label})",
        )
        # Current position dot
        body.current_dot, = ax.plot(
            [], [], [], 'o',
            color=body.color_hex,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=10,
        )
        # Floating label near current position
        body.label_text = ax.text(
            0, 0, 0, "",
            fontsize=7,
            color=body.color_hex,
            fontweight="bold",
            zorder=11,
        )


def _update_body_artists(body: TrackedBody) -> None:
    """Update a body's matplotlib artists from its trail data."""
    if body.trail_line is None:
        return

    if body.trail_length > 0:
        x_arr = list(body.trail_x)
        y_arr = list(body.trail_y)
        z_arr = list(body.trail_z)

        # Trail line
        body.trail_line.set_data(x_arr, y_arr)
        body.trail_line.set_3d_properties(z_arr)

        # Style based on lifecycle state
        if body.state == "active":
            body.trail_line.set_alpha(0.7)
            body.trail_line.set_linewidth(1.4)
            body.current_dot.set_alpha(1.0)
            body.current_dot.set_markersize(10)
        elif body.state == "stale":
            body.trail_line.set_alpha(0.4)
            body.trail_line.set_linewidth(1.0)
            body.current_dot.set_alpha(0.5)
            body.current_dot.set_markersize(8)
        else:  # lost
            body.trail_line.set_alpha(0.15)
            body.trail_line.set_linewidth(0.7)
            body.current_dot.set_alpha(0.15)
            body.current_dot.set_markersize(6)

        # Current dot at last known position
        if body.last_position is not None:
            lp = body.last_position
            body.current_dot.set_data([float(lp[0])], [float(lp[1])])
            body.current_dot.set_3d_properties([float(lp[2])])

            # Label slightly offset
            body.label_text.set_position_3d(
                (float(lp[0]) + 0.015,
                 float(lp[1]) + 0.015,
                 float(lp[2]) + 0.015)
            )
            state_tag = ""
            if body.state == "stale":
                state_tag = " [STALE]"
            elif body.state == "lost":
                state_tag = " [LOST]"
            body.label_text.set_text(f"{body.name}{state_tag}")
            body.label_text.set_alpha(
                1.0 if body.state == "active" else
                0.5 if body.state == "stale" else 0.2
            )
    else:
        # No data yet — hide artists
        body.trail_line.set_data([], [])
        body.trail_line.set_3d_properties([])
        body.current_dot.set_data([], [])
        body.current_dot.set_3d_properties([])
        body.label_text.set_text("")


# --- Main 3D Trace Renderer ---

def run_3d_trace(args):
    """Multi-body real-time 3D trace using matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    TRAIL_MAX = args.trail

    # Connect
    if args.demo:
        print("[3D Trace] Demo mode — body 1 from start, body 2 appears at t=6s")
        print("[3D Trace] Body 2 briefly drops out at t=14-15s (stale test)")
        client = DemoMultiBody()
    else:
        convert = args.zup
        print(f"[3D Trace] Connecting to OptiTrack at {args.ip} "
              f"({'Z-up' if convert else 'raw Y-up'})...")
        client = OptiTrackClient(server_ip=args.ip, convert_to_zup=convert)
        client.start()

    # Body registry — the core tracking layer
    name_filter = args.filter if args.filter else None
    registry = BodyRegistry(trail_max=TRAIL_MAX, name_filter=name_filter)

    t0 = time.time()
    last_print = [0.0]
    last_proximity_warn = [0.0]

    # Set up figure
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    coord_label = "Z-up" if args.zup else "Y-up (raw)"
    ax.set_title(f"Multi-Body 3D Trace  [{coord_label}]",
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

    # Status text
    status_text = fig.text(0.02, 0.02, "Scanning for bodies...",
                           fontsize=8, color="#aaaaaa", family="monospace")
    body_list_text = fig.text(0.02, 0.05, "",
                              fontsize=8, color="#cccccc", family="monospace")
    event_text = fig.text(0.98, 0.02, "",
                          fontsize=8, color="#ffd43b", family="monospace",
                          ha="right")

    range_pad = 0.08
    last_event_time = [0.0]
    last_event_msg = [""]

    def update(frame_num):
        t_now = time.time() - t0

        # Get current frame from OptiTrack
        raw_bodies = client.get_rigid_bodies()

        # Update registry — this handles discovery, tracking, lifecycle
        events = registry.update(raw_bodies, t_now)

        # Print events to terminal
        for ev in events:
            # Rate-limit proximity warnings (max once per 3 seconds)
            if "[PROXIMITY]" in ev:
                if t_now - last_proximity_warn[0] < 3.0:
                    continue
                last_proximity_warn[0] = t_now
            print(f"  [{t_now:7.1f}s] {ev}")
            last_event_time[0] = t_now
            last_event_msg[0] = ev

        # Ensure matplotlib artists exist for all tracked bodies
        for body in registry.visible_bodies:
            _ensure_artists(body, ax)

        # Update all body artists
        for body in registry.bodies.values():
            _update_body_artists(body)

        # Auto-scale axes to encompass all trails
        bounds = registry.get_global_bounds()
        if bounds is not None:
            mins, maxs = bounds
            span = maxs - mins
            ax.set_xlim(mins[0] - range_pad * span[0],
                        maxs[0] + range_pad * span[0])
            ax.set_ylim(mins[1] - range_pad * span[1],
                        maxs[1] + range_pad * span[1])
            ax.set_zlim(mins[2] - range_pad * span[2],
                        maxs[2] + range_pad * span[2])

        # Update legend if new bodies appeared
        if events and any("[NEW]" in e for e in events):
            handles = []
            labels = []
            for body in registry.bodies.values():
                if body.trail_line is not None:
                    handles.append(body.trail_line)
                    labels.append(f"{body.name} ({body.color_label})")
            if handles:
                ax.legend(handles, labels, loc="upper left", fontsize=7,
                          framealpha=0.3)

        # Build status strings
        n_active = len(registry.active_bodies)
        n_total = len(registry.bodies)
        total_pts = sum(b.trail_length for b in registry.bodies.values())

        if n_total == 0:
            available = list(raw_bodies.keys()) if raw_bodies else ["(none)"]
            status_text.set_text(
                f"No bodies tracked  |  Available: {', '.join(available)}  "
                f"|  {t_now:.1f}s"
            )
            status_text.set_color("#ffd43b")
        else:
            status_text.set_text(
                f"{n_active} active / {n_total} tracked  |  "
                f"{total_pts} trail pts  |  {t_now:.1f}s"
            )
            status_text.set_color("#51cf66" if n_active > 0 else "#ff6b6b")

        # Body list with positions
        body_lines = []
        for body in registry.bodies.values():
            if body.last_position is not None:
                p = body.last_position
                state_icon = "●" if body.state == "active" else (
                    "◐" if body.state == "stale" else "○")
                body_lines.append(
                    f"{state_icon} {body.name}: "
                    f"({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})  "
                    f"[{body.trail_length} pts]"
                )
        body_list_text.set_text("  |  ".join(body_lines) if body_lines else "")

        # Show recent event (fades after 5s)
        if t_now - last_event_time[0] < 5.0:
            event_text.set_text(last_event_msg[0])
            fade = max(0.2, 1.0 - (t_now - last_event_time[0]) / 5.0)
            event_text.set_alpha(fade)
        else:
            event_text.set_text("")

        # Terminal print every 2s
        if t_now - last_print[0] >= 2.0 and n_total > 0:
            last_print[0] = t_now
            parts = []
            for body in registry.bodies.values():
                if body.last_position is not None:
                    p = body.last_position
                    parts.append(
                        f"{body.name}[{body.state[0].upper()}]: "
                        f"({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})"
                    )
            print(f"  [{t_now:7.1f}s] {' | '.join(parts)}")

        # Collect all artists for return
        artists = [status_text, body_list_text, event_text]
        for body in registry.bodies.values():
            if body.trail_line is not None:
                artists.extend([body.trail_line, body.current_dot,
                                body.label_text])
        return artists

    # Run animation
    interval_ms = max(16, int(1000 / args.rate))
    anim = FuncAnimation(fig, update, interval=interval_ms,
                         blit=False, cache_frame_data=False)

    filter_msg = f" (filter: '{name_filter}')" if name_filter else ""
    print(f"\n[3D Trace] Auto-discovery mode at {args.rate} Hz{filter_msg}")
    print(f"[3D Trace] Trail buffer: {TRAIL_MAX} points per body")
    print(f"[3D Trace] Stale timeout: {STALE_TIMEOUT_SEC}s")
    print(f"[3D Trace] Proximity warning: <{PROXIMITY_WARN_M*100:.0f}cm")
    print(f"[3D Trace] Close matplotlib window to stop.\n")

    try:
        plt.show()
    except KeyboardInterrupt:
        pass

    # Summary
    print(f"\n[3D Trace] Session summary:")
    for body in registry.bodies.values():
        print(f"  {body.name} ({body.color_label}): "
              f"{body.total_samples} samples, "
              f"{body.trail_length} trail pts, "
              f"first seen at {body.first_seen_t - t0:.1f}s")

    if hasattr(client, "stop"):
        client.stop()
    print(f"[3D Trace] Done.")


# --- Entry Point ---

def main():
    parser = argparse.ArgumentParser(
        description="Multi-body 3D trace — auto-discovers and tracks all OptiTrack rigid bodies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_3d_trace.py --demo                              # multi-body demo
  python run_3d_trace.py --ip 192.168.0.101                  # auto-discover all
  python run_3d_trace.py --ip 192.168.0.101 --filter rigid_  # only rigid_ bodies
  python run_3d_trace.py --ip 192.168.0.101 --zup            # Z-up conversion
  python run_3d_trace.py --demo --trail 5000 --rate 15       # longer trail
        """,
    )
    parser.add_argument("--ip", default="192.168.0.101",
                        help="OptiTrack server IP (default: 192.168.0.101)")
    parser.add_argument("--rate", type=int, default=30,
                        help="Update rate in Hz (default: 30)")
    parser.add_argument("--trail", type=int, default=3000,
                        help="Max trail points PER BODY (default: 3000)")
    parser.add_argument("--demo", action="store_true",
                        help="Simulated multi-body data, no hardware needed")
    parser.add_argument("--zup", action="store_true",
                        help="Convert to Z-up (robotics). Default: raw Y-up.")
    parser.add_argument("--filter", default=None,
                        help="Only track bodies whose name contains this string "
                             "(e.g., 'rigid_body_')")

    args = parser.parse_args()
    run_3d_trace(args)


if __name__ == "__main__":
    main()
