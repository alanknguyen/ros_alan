#!/usr/bin/env python3
"""
scripts/run_capture.py — OptiTrack V120:Trio Capture Test

Step 1 verification script. Connects to the OptiTrack V120:Trio system via NatNet
and prints live rigid body tracking data to the console.

Usage
-----
    cd vision_engine
    python scripts/run_capture.py                          # Use defaults from config
    python scripts/run_capture.py --server-ip 192.168.0.110  # Override server IP
    python scripts/run_capture.py --duration 60              # Run for 60 seconds
    python scripts/run_capture.py --raw                      # Show Y-up (no conversion)

Expected Output
---------------
    [OptiTrack] Connecting to 192.168.0.110...
    [OptiTrack] Server: Motive v3.0.0, NatNet v4.0.0
    [OptiTrack] Rigid bodies defined: cube_1, cylinder_1
    [OptiTrack] Receiving rigid body data...

    --- Frame 1 (t=1712345678.42) ---
      cube_1:     pos=( 0.412, -0.185,  0.732) quat=(0.020, 0.710, -0.010, 0.700) valid=True
      cylinder_1: pos=( 0.550,  0.100,  0.710) quat=(0.000, 0.000,  0.000, 1.000) valid=True

Verification Checklist
----------------------
    1. Server connects and prints app name + version
    2. Rigid body names match what's defined in Motive
    3. Position values are in meters, change when you move objects
    4. With default settings (Z-up), Z should be height above floor
       (objects on a table at ~0.7m should show z ≈ 0.7)
    5. tracking_valid=True when object is visible, False when occluded
    6. Frame rate: you should see ~120 frames per second (print rate is throttled to 10 Hz)
"""

import sys
import os
import time
import argparse

# Add vision_engine root to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from cv.optitrack_client import OptiTrackClient
from cv.transforms import euler_degrees_from_quaternion


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="OptiTrack V120:Trio capture test — prints live rigid body data"
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config", "scene_config.yaml"),
        help="Path to scene_config.yaml (default: config/scene_config.yaml)",
    )
    parser.add_argument(
        "--server-ip",
        default=None,
        help="Override OptiTrack server IP (default: from config file)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="How long to run in seconds (default: 30)",
    )
    parser.add_argument(
        "--print-rate",
        type=float,
        default=10.0,
        help="Console print rate in Hz (default: 10, actual capture is ~120 Hz)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Show raw Y-up coordinates (skip Z-up conversion)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    optitrack_cfg = config.get("optitrack", {})

    # Apply overrides
    server_ip = args.server_ip or optitrack_cfg.get("server_ip", "192.168.0.101")
    local_ip = optitrack_cfg.get("local_ip", "0.0.0.0")
    multicast_ip = optitrack_cfg.get("multicast_ip", "239.255.42.99")
    command_port = optitrack_cfg.get("command_port", 1510)
    data_port = optitrack_cfg.get("data_port", 1511)
    convert_to_zup = not args.raw

    # Create client
    client = OptiTrackClient(
        server_ip=server_ip,
        local_ip=local_ip,
        multicast_ip=multicast_ip,
        command_port=command_port,
        data_port=data_port,
        convert_to_zup=convert_to_zup,
    )

    # Start capture
    client.start()

    coord_label = "Z-up" if convert_to_zup else "Y-up (raw)"
    print(f"\nCoordinate frame: {coord_label}")
    print(f"Printing at {args.print_rate} Hz for {args.duration} seconds...")
    print(f"(Actual capture rate: ~120 Hz from V120:Trio)")
    print(f"Press Ctrl+C to stop early.\n")

    # Print loop (throttled to print_rate)
    print_interval = 1.0 / args.print_rate
    start_time = time.time()
    last_print_time = 0
    last_frame_count = 0

    try:
        while time.time() - start_time < args.duration:
            now = time.time()
            if now - last_print_time < print_interval:
                time.sleep(0.001)
                continue

            last_print_time = now
            bodies = client.get_rigid_bodies()
            current_frame = client.get_frame_count()

            # Compute frame rate
            elapsed = now - start_time
            fps = current_frame / elapsed if elapsed > 0 else 0

            if not bodies:
                print(f"[t={now:.2f}] No rigid bodies received yet... "
                      f"(frames={current_frame}, {fps:.0f} fps)")
                continue

            # Print header
            print(f"--- Frame {current_frame} (t={now:.2f}, {fps:.0f} fps) ---")

            # Print each rigid body
            for name, state in sorted(bodies.items()):
                p = state.position
                q = state.quaternion
                roll, pitch, yaw = euler_degrees_from_quaternion(q[0], q[1], q[2], q[3])

                print(
                    f"  {name:20s}: "
                    f"pos=({p[0]:7.3f}, {p[1]:7.3f}, {p[2]:7.3f}) "
                    f"quat=({q[0]:6.3f}, {q[1]:6.3f}, {q[2]:6.3f}, {q[3]:6.3f}) "
                    f"euler=({roll:6.1f}, {pitch:6.1f}, {yaw:6.1f})° "
                    f"valid={state.tracking_valid}"
                )

            print()  # Blank line between frames

    except KeyboardInterrupt:
        print("\n[Capture] Interrupted by user.")

    # Summary
    total_frames = client.get_frame_count()
    total_time = time.time() - start_time
    avg_fps = total_frames / total_time if total_time > 0 else 0
    print(f"\n[Summary] Received {total_frames} frames in {total_time:.1f}s "
          f"(avg {avg_fps:.0f} fps)")

    # Cleanup
    client.stop()


if __name__ == "__main__":
    main()
