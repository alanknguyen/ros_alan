#!/usr/bin/env python3
"""
scene_state_publisher.py — OptiTrack Scene State Publisher

Reads 6DOF rigid body poses from OptiTrack via NatNet and publishes them
as structured pose messages for Sauman's LLM reasoning system.

Output format (per rigid body, per frame):
─────────────────────────────────────────
    RigidBody: red_cube
    position:
      x: 0.412
      y: -0.185
      z: 0.732
    orientation (quaternion):
      x: 0.02
      y: 0.71
      z: -0.01
      w: 0.70
    timestamp: 1712345678.42
    frame: optitrack_world

Modes
─────
    --ros       : Publishes to /llm/scene_state as ROS String messages (YAML text)
    --standalone: Prints to console (default, no ROS needed)
    --demo      : Simulated data, no OptiTrack hardware needed

Usage
─────
    # Live OptiTrack, print to console:
    python scene_state_publisher.py --objects config/objects.yaml --ip 192.168.0.101

    # Demo mode (no hardware):
    python scene_state_publisher.py --demo

    # ROS mode:
    rosrun spatial_reasoning_benchmark scene_state_publisher.py --ros
"""

import sys
import time
import math
import argparse
import signal
import numpy as np
from pathlib import Path
from typing import Dict, Optional

import yaml

# ── Resolve imports ──────────────────────────────────────────────────────────
_script_dir = Path(__file__).resolve().parent
_engine_dir = _script_dir.parent
if str(_engine_dir) not in sys.path:
    sys.path.insert(0, str(_engine_dir))

from cv.optitrack_client import OptiTrackClient, RigidBodyState


# ═════════════════════════════════════════════════════════════════════════════
# Object Registry
# ═════════════════════════════════════════════════════════════════════════════

def load_object_registry(yaml_path: str) -> Dict[str, str]:
    """
    Load objects.yaml → returns {optitrack_body_name: display_name}.

    objects.yaml format:
        objects:
          red_cube:
            optitrack_body: "rigid_body_1"
            class: cube
            color: red
            dimensions_m: [0.05, 0.05, 0.05]

    Returns: {"rigid_body_1": "red_cube", "rigid_body_2": "blue_cylinder", ...}
    """
    if not Path(yaml_path).exists():
        print(f"[SceneState] WARNING: {yaml_path} not found, using raw body names")
        return {}

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    if not raw:
        return {}

    # Support both top-level dict and nested "objects:" key
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
# Format one rigid body as a pose message
# ═════════════════════════════════════════════════════════════════════════════

def format_body_message(display_name: str, body: RigidBodyState) -> str:
    """
    Format a single rigid body state into the output text format.

    Returns:
        RigidBody: red_cube
        position:
          x: 0.4120
          y: -0.1850
          z: 0.7320
        orientation (quaternion):
          x: 0.0200
          y: 0.7100
          z: -0.0100
          w: 0.7000
        timestamp: 1712345678.42
        frame: optitrack_world
    """
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
    """
    Format all tracked rigid bodies into a single scene state message.
    Skips bodies with tracking_valid=False.
    """
    blocks = []
    for body_name, body in sorted(bodies.items()):
        if not body.tracking_valid:
            continue

        display_name = body_to_name.get(body_name, body_name)
        blocks.append(format_body_message(display_name, body))

    if not blocks:
        return "---\nNo objects tracked\n---"

    separator = "\n---\n"
    return separator.join(blocks)


# ═════════════════════════════════════════════════════════════════════════════
# Demo Data Generator
# ═════════════════════════════════════════════════════════════════════════════

class DemoOptiTrack:
    """Simulates OptiTrack data for testing without hardware."""

    def __init__(self):
        self._t0 = time.time()

    def get_rigid_bodies(self) -> Dict[str, RigidBodyState]:
        t = time.time() - self._t0
        now = time.time()
        return {
            "rigid_body_1": RigidBodyState(
                name="rigid_body_1", id=1,
                position=np.array([0.412 + 0.05 * math.sin(t * 0.3), -0.185, 0.732]),
                quaternion=np.array([0.02, 0.71, -0.01, 0.70]),
                timestamp=now, tracking_valid=True,
            ),
            "rigid_body_2": RigidBodyState(
                name="rigid_body_2", id=2,
                position=np.array([0.30, 0.10, 0.735]),
                quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
                timestamp=now, tracking_valid=True,
            ),
            "rigid_body_3": RigidBodyState(
                name="rigid_body_3", id=3,
                position=np.array([0.55, 0.20 + 0.03 * math.sin(t * 0.5), 0.725]),
                quaternion=np.array([0.0, 0.0, math.sin(t * 0.1), math.cos(t * 0.1)]),
                timestamp=now, tracking_valid=True,
            ),
        }

    def start(self):
        pass

    def stop(self):
        pass


# ═════════════════════════════════════════════════════════════════════════════
# Main Loop
# ═════════════════════════════════════════════════════════════════════════════

def run(args):
    # Load object names
    body_to_name = load_object_registry(args.objects)

    # Connect to OptiTrack (or demo)
    if args.demo:
        print("[SceneState] Demo mode — simulated data")
        client = DemoOptiTrack()
    else:
        print(f"[SceneState] Connecting to OptiTrack at {args.ip}...")
        client = OptiTrackClient(server_ip=args.ip)
        client.start()

    # Optional ROS publisher
    ros_pub = None
    if args.ros:
        try:
            import rospy
            from std_msgs.msg import String
            rospy.init_node("scene_state_publisher", anonymous=False)
            ros_pub = rospy.Publisher(args.topic, String, queue_size=1)
            print(f"[SceneState] ROS publisher on {args.topic}")
        except ImportError:
            print("[SceneState] ERROR: rospy not found. Use --standalone or install ROS.")
            sys.exit(1)

    # Graceful shutdown
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

        # Check ROS shutdown
        if args.ros:
            import rospy
            if rospy.is_shutdown():
                break

        bodies = client.get_rigid_bodies()
        msg_text = format_scene_message(bodies, body_to_name)

        # Publish / print
        if ros_pub is not None:
            from std_msgs.msg import String
            ros_pub.publish(String(data=msg_text))
        else:
            # Console output
            print(f"\033[2J\033[H", end="")  # clear screen
            print(f"═══ Scene State (frame {count}) ═══")
            print(msg_text)
            print(f"\n[{time.strftime('%H:%M:%S')}] Rate: {args.rate} Hz | "
                  f"Bodies: {sum(1 for b in bodies.values() if b.tracking_valid)}")

        count += 1
        elapsed = time.time() - t0
        if interval - elapsed > 0:
            time.sleep(interval - elapsed)

    # Cleanup
    if hasattr(client, "stop"):
        client.stop()
    print(f"[SceneState] Done. {count} frames published.")


# ═════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="OptiTrack Scene State Publisher")
    parser.add_argument("--objects", default="config/objects.yaml",
                        help="Path to objects.yaml (default: config/objects.yaml)")
    parser.add_argument("--ip", default="192.168.0.101",
                        help="OptiTrack server IP (default: 192.168.0.101)")
    parser.add_argument("--rate", type=int, default=10,
                        help="Publish rate in Hz (default: 10)")
    parser.add_argument("--ros", action="store_true",
                        help="Publish as ROS String messages")
    parser.add_argument("--topic", default="/llm/scene_state",
                        help="ROS topic name (default: /llm/scene_state)")
    parser.add_argument("--demo", action="store_true",
                        help="Simulated data, no OptiTrack needed")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
