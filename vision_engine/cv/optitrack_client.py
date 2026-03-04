"""
vision_engine/cv/optitrack_client.py — NatNet Protocol Client for OptiTrack V120:Trio

Implements a pure-Python NatNet 3.x/4.x client that connects to the OptiTrack Motive
software and receives rigid body tracking data over UDP.

Architecture
------------
The OptiTrack V120:Trio is a self-contained 3-camera tracking bar. It connects via
USB to a Windows PC running Motive, which processes marker data and streams it via
the NatNet protocol over the local network.

NatNet uses two UDP channels:
  - **Command socket** (unicast, default port 1510): Request/response for server info,
    model definitions, and configuration. Client sends requests to the server IP.
  - **Data socket** (multicast, default port 1511): Server broadcasts frame data
    (rigid body poses, markers, etc.) to a multicast group. Client joins the group
    to receive frames.

Protocol Details
----------------
NatNet messages have a simple header:
  - uint16 message_type
  - uint16 payload_size_bytes
  - [payload_size_bytes of data]

Key message types:
  - 0x0001 (NAT_CONNECT): Client → Server connection request
  - 0x0002 (NAT_SERVERINFO): Server → Client info (app name, version, NatNet version)
  - 0x0005 (NAT_MODELDEF): Server → Client model definitions (rigid body names)
  - 0x0007 (NAT_FRAMEOFDATA): Server → Client frame data (poses, markers)
  - 0x0009 (NAT_REQUEST_MODELDEF): Client → Server request for model definitions

Rigid body data within a frame:
  - int32 id
  - float32 x, y, z (position in meters)
  - float32 qx, qy, qz, qw (orientation as quaternion)
  - [NatNet 3.0+] int16 params — bit 0 = tracking valid

The V120:Trio typically streams at 120 Hz in Motive's default configuration.

Coordinate System
-----------------
OptiTrack/Motive defaults to a **Y-up** right-handed system:
  - X = right
  - Y = up
  - Z = towards the cameras (into the tracking volume)

This client optionally converts to **Z-up** (robotics convention) using
transforms.position_yup_to_zup() and transforms.quaternion_yup_to_zup().

Usage
-----
    from cv.optitrack_client import OptiTrackClient

    client = OptiTrackClient(server_ip="192.168.0.110")
    client.start()

    # In your main loop:
    bodies = client.get_rigid_bodies()
    for name, state in bodies.items():
        print(f"{name}: pos={state.position}, quat={state.quaternion}")

    client.stop()
"""

import socket
import struct
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable

from cv.transforms import position_yup_to_zup, quaternion_yup_to_zup


# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RigidBodyState:
    """
    State of a single tracked rigid body at a moment in time.

    Attributes
    ----------
    name : str
        Rigid body name as defined in Motive (e.g., "cube_1").
    id : int
        Numeric ID assigned by Motive.
    position : np.ndarray, shape (3,)
        Position (x, y, z) in meters, in Z-up frame (after conversion).
    quaternion : np.ndarray, shape (4,)
        Orientation as quaternion (x, y, z, w) in Z-up frame.
    timestamp : float
        Unix timestamp (seconds since epoch) when this frame was received.
    tracking_valid : bool
        True if Motive is actively tracking this rigid body (not occluded).
    """
    name: str
    id: int
    position: np.ndarray
    quaternion: np.ndarray
    timestamp: float
    tracking_valid: bool


# ──────────────────────────────────────────────────────────────────────────────
# NatNet Message Types
# ──────────────────────────────────────────────────────────────────────────────

NAT_CONNECT            = 0  # Client → Server: connect request
NAT_SERVERINFO         = 1  # Server → Client: server info response
NAT_REQUEST            = 2  # Client → Server: generic request
NAT_RESPONSE           = 3  # Server → Client: generic response
NAT_REQUEST_MODELDEF   = 4  # Client → Server: request model definitions
NAT_MODELDEF           = 5  # Server → Client: model definitions
NAT_REQUEST_FRAMEOFDATA = 6 # Client → Server: request single frame
NAT_FRAMEOFDATA        = 7  # Server → Client: frame data
NAT_UNRECOGNIZED       = 100


# ──────────────────────────────────────────────────────────────────────────────
# NatNet Client
# ──────────────────────────────────────────────────────────────────────────────

class OptiTrackClient:
    """
    Pure-Python NatNet client for receiving rigid body data from OptiTrack Motive.

    Designed for the V120:Trio system but compatible with any OptiTrack setup
    running Motive with NatNet streaming enabled.

    Parameters
    ----------
    server_ip : str
        IP address of the Motive PC (e.g., "192.168.0.110").
    local_ip : str
        IP address of this machine's network interface. Use "0.0.0.0" to bind
        to all interfaces (default). Set to a specific IP if you have multiple
        network interfaces and need to control which one joins multicast.
    multicast_ip : str
        NatNet multicast group address (default "239.255.42.99").
    command_port : int
        NatNet command port (default 1510).
    data_port : int
        NatNet data port (default 1511).
    convert_to_zup : bool
        If True (default), convert OptiTrack Y-up coordinates to Z-up.
    """

    def __init__(
        self,
        server_ip: str,
        local_ip: str = "0.0.0.0",
        multicast_ip: str = "239.255.42.99",
        command_port: int = 1510,
        data_port: int = 1511,
        convert_to_zup: bool = True,
    ):
        self.server_ip = server_ip
        self.local_ip = local_ip
        self.multicast_ip = multicast_ip
        self.command_port = command_port
        self.data_port = data_port
        self.convert_to_zup = convert_to_zup

        # Server info (populated after connection)
        self.server_app_name: str = ""
        self.server_app_version: tuple = (0, 0, 0, 0)
        self.natnet_version: tuple = (0, 0, 0, 0)

        # Rigid body name lookup: id → name (populated from model definitions)
        self._id_to_name: Dict[int, str] = {}

        # Latest rigid body states (thread-safe access via lock)
        self._rigid_bodies: Dict[str, RigidBodyState] = {}
        self._lock = threading.Lock()

        # Frame counter and callback
        self._frame_count: int = 0
        self._callback: Optional[Callable] = None

        # Sockets and threads
        self._command_socket: Optional[socket.socket] = None
        self._data_socket: Optional[socket.socket] = None
        self._command_thread: Optional[threading.Thread] = None
        self._data_thread: Optional[threading.Thread] = None
        self._running = False

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Connect to Motive and begin receiving rigid body data.

        This method:
          1. Creates command (unicast) and data (multicast) UDP sockets
          2. Sends a NAT_CONNECT message to Motive
          3. Requests model definitions (rigid body names)
          4. Starts background listener threads for both sockets

        Blocks briefly to wait for server response and model definitions.
        """
        print(f"[OptiTrack] Connecting to {self.server_ip}...")

        # Create command socket (unicast)
        self._command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._command_socket.bind((self.local_ip, 0))  # Ephemeral port
        self._command_socket.settimeout(3.0)

        # Create data socket (multicast)
        self._data_socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        self._data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Bind to the data port
        # On Linux, bind to multicast group; on macOS, bind to INADDR_ANY
        try:
            self._data_socket.bind((self.multicast_ip, self.data_port))
        except OSError:
            self._data_socket.bind(("", self.data_port))

        # Join multicast group
        mreq = struct.pack(
            "4s4s",
            socket.inet_aton(self.multicast_ip),
            socket.inet_aton(self.local_ip),
        )
        self._data_socket.setsockopt(
            socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq
        )
        self._data_socket.settimeout(1.0)

        # Start listener threads
        self._running = True
        self._data_thread = threading.Thread(
            target=self._data_listener, daemon=True, name="NatNet-Data"
        )
        self._command_thread = threading.Thread(
            target=self._command_listener, daemon=True, name="NatNet-Command"
        )
        self._data_thread.start()
        self._command_thread.start()

        # Send connection request
        self._send_command(NAT_CONNECT, b"")

        # Wait for server info
        time.sleep(0.5)
        if self.server_app_name:
            version_str = ".".join(str(v) for v in self.server_app_version)
            natnet_str = ".".join(str(v) for v in self.natnet_version)
            print(f"[OptiTrack] Server: {self.server_app_name} v{version_str}, "
                  f"NatNet v{natnet_str}")
        else:
            print("[OptiTrack] Warning: No server response received. "
                  "Check IP and network connectivity.")

        # Request model definitions (to get rigid body names)
        self._send_command(NAT_REQUEST_MODELDEF, b"")
        time.sleep(0.5)

        if self._id_to_name:
            names = ", ".join(self._id_to_name.values())
            print(f"[OptiTrack] Rigid bodies defined: {names}")
        else:
            print("[OptiTrack] Warning: No rigid body definitions received. "
                  "Check Motive has rigid bodies defined and streaming is enabled.")

        print("[OptiTrack] Receiving rigid body data...")

    def stop(self) -> None:
        """
        Stop receiving data and close sockets.
        """
        self._running = False

        if self._data_thread and self._data_thread.is_alive():
            self._data_thread.join(timeout=2.0)
        if self._command_thread and self._command_thread.is_alive():
            self._command_thread.join(timeout=2.0)

        if self._data_socket:
            try:
                self._data_socket.close()
            except Exception:
                pass
        if self._command_socket:
            try:
                self._command_socket.close()
            except Exception:
                pass

        print("[OptiTrack] Disconnected.")

    def get_rigid_bodies(self) -> Dict[str, RigidBodyState]:
        """
        Get the latest rigid body states.

        Returns a snapshot (copy) of the current rigid body data. Thread-safe.

        Returns
        -------
        bodies : dict[str, RigidBodyState]
            Map of rigid body name → state. Empty if no data received yet.
        """
        with self._lock:
            return dict(self._rigid_bodies)

    def get_frame_count(self) -> int:
        """Return the number of frames received since start()."""
        return self._frame_count

    def set_callback(self, fn: Callable[[Dict[str, RigidBodyState]], None]) -> None:
        """
        Set a callback function that is called on every new frame.

        The callback receives a dict of rigid body states (same as get_rigid_bodies()).
        It runs in the data listener thread — keep it fast to avoid dropped frames.

        Parameters
        ----------
        fn : callable
            Function(dict[str, RigidBodyState]) → None
        """
        self._callback = fn

    # ──────────────────────────────────────────────────────────────────────
    # Socket Communication
    # ──────────────────────────────────────────────────────────────────────

    def _send_command(self, msg_type: int, payload: bytes) -> None:
        """Send a NatNet command message to the server."""
        # NatNet header: uint16 msg_type + uint16 payload_size
        header = struct.pack("<HH", msg_type, len(payload))
        self._command_socket.sendto(
            header + payload,
            (self.server_ip, self.command_port),
        )

    def _data_listener(self) -> None:
        """Background thread: receive and parse multicast data frames."""
        while self._running:
            try:
                data, addr = self._data_socket.recvfrom(65536)
                if len(data) < 4:
                    continue
                msg_type, payload_size = struct.unpack_from("<HH", data, 0)
                if msg_type == NAT_FRAMEOFDATA:
                    self._parse_frame_data(data, 4)
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    continue
                break

    def _command_listener(self) -> None:
        """Background thread: receive command socket responses."""
        while self._running:
            try:
                data, addr = self._command_socket.recvfrom(65536)
                if len(data) < 4:
                    continue
                msg_type, payload_size = struct.unpack_from("<HH", data, 0)

                if msg_type == NAT_SERVERINFO:
                    self._parse_server_info(data, 4)
                elif msg_type == NAT_MODELDEF:
                    self._parse_model_def(data, 4)
                elif msg_type == NAT_RESPONSE:
                    pass  # Generic response, ignore
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    continue
                break

    # ──────────────────────────────────────────────────────────────────────
    # Packet Parsers
    # ──────────────────────────────────────────────────────────────────────

    def _parse_server_info(self, data: bytes, offset: int) -> None:
        """
        Parse NAT_SERVERINFO response.

        Layout:
          - char[256] app_name (null-terminated)
          - uint8[4]  app_version (major, minor, build, revision)
          - uint8[4]  natnet_version (major, minor, build, revision)
        """
        try:
            # App name: 256-byte null-terminated string
            app_name_raw = data[offset:offset + 256]
            null_idx = app_name_raw.find(b'\x00')
            if null_idx >= 0:
                self.server_app_name = app_name_raw[:null_idx].decode("utf-8", errors="replace")
            else:
                self.server_app_name = app_name_raw.decode("utf-8", errors="replace").strip()
            offset += 256

            # App version: 4 bytes
            self.server_app_version = struct.unpack_from("4B", data, offset)
            offset += 4

            # NatNet version: 4 bytes
            self.natnet_version = struct.unpack_from("4B", data, offset)
        except (struct.error, IndexError):
            pass

    def _parse_model_def(self, data: bytes, offset: int) -> None:
        """
        Parse NAT_MODELDEF response to extract rigid body names and IDs.

        Layout:
          - int32 num_datasets
          - For each dataset:
            - int32 dataset_type (0=markerset, 1=rigid body, 2=skeleton)
            - [type-specific data]

        For rigid body definitions (type 1):
          - char[256] name (null-terminated) [NatNet 3.0+: variable-length]
          - int32 id
          - int32 parent_id
          - float32[3] offset (x, y, z)
        """
        try:
            num_datasets = struct.unpack_from("<i", data, offset)[0]
            offset += 4

            for _ in range(num_datasets):
                if offset + 4 > len(data):
                    break

                dataset_type = struct.unpack_from("<i", data, offset)[0]
                offset += 4

                if dataset_type == 0:
                    # Marker set definition — skip
                    offset = self._skip_markerset_def(data, offset)
                elif dataset_type == 1:
                    # Rigid body definition — extract name and ID
                    offset = self._parse_rigid_body_def(data, offset)
                elif dataset_type == 2:
                    # Skeleton definition — skip
                    offset = self._skip_skeleton_def(data, offset)
                else:
                    # Unknown type, try to skip gracefully
                    break

        except (struct.error, IndexError) as e:
            print(f"[OptiTrack] Warning: Error parsing model definitions: {e}")

    def _parse_rigid_body_def(self, data: bytes, offset: int) -> int:
        """Parse a single rigid body definition. Returns new offset."""
        # Name: null-terminated string
        # NatNet 3.0+ uses variable-length strings, but for compatibility
        # we read until null byte
        name, offset = self._read_cstring(data, offset)

        # ID, parent ID, offset position
        rb_id, parent_id = struct.unpack_from("<ii", data, offset)
        offset += 8

        # Offset (x, y, z) — relative to parent
        offset += 12  # 3 * float32

        self._id_to_name[rb_id] = name
        return offset

    def _skip_markerset_def(self, data: bytes, offset: int) -> int:
        """Skip a marker set definition. Returns new offset."""
        # Name
        _, offset = self._read_cstring(data, offset)
        # Number of markers
        num_markers = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        # Marker names
        for _ in range(num_markers):
            _, offset = self._read_cstring(data, offset)
        return offset

    def _skip_skeleton_def(self, data: bytes, offset: int) -> int:
        """Skip a skeleton definition. Returns new offset."""
        # Name
        _, offset = self._read_cstring(data, offset)
        # Skeleton ID
        offset += 4
        # Number of rigid bodies in skeleton
        num_bodies = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        for _ in range(num_bodies):
            offset = self._parse_rigid_body_def(data, offset)
        return offset

    def _parse_frame_data(self, data: bytes, offset: int) -> None:
        """
        Parse NAT_FRAMEOFDATA message containing rigid body poses.

        This is the main data path — called at frame rate (~120 Hz).

        Handles both NatNet 2.x and 3.0+ formats:
          - NatNet 2.x: rigid bodies include per-body marker data
          - NatNet 3.0+: rigid bodies have only pose + meanError + params

        Layout:
          - int32 frame_number
          - int32 num_marker_sets → [skip marker set data]
          - int32 num_other_markers → [skip]
          - int32 num_rigid_bodies → [parse each rigid body]
          - ... (skeletons, labeled markers, etc.)
        """
        try:
            now = time.time()
            natnet_major = self.natnet_version[0] if self.natnet_version[0] > 0 else 3

            # Frame number
            frame_number = struct.unpack_from("<i", data, offset)[0]
            offset += 4

            # ── Marker Sets (skip) ──
            num_marker_sets = struct.unpack_from("<i", data, offset)[0]
            offset += 4
            for _ in range(num_marker_sets):
                # Name
                _, offset = self._read_cstring(data, offset)
                # Number of markers
                num_markers = struct.unpack_from("<i", data, offset)[0]
                offset += 4
                # Skip marker positions (3 floats each)
                offset += num_markers * 12

            # ── Other Markers (unidentified, skip) ──
            num_other_markers = struct.unpack_from("<i", data, offset)[0]
            offset += 4
            offset += num_other_markers * 12  # 3 floats each

            # ── Rigid Bodies (PARSE) ──
            num_rigid_bodies = struct.unpack_from("<i", data, offset)[0]
            offset += 4

            new_bodies: Dict[str, RigidBodyState] = {}

            for _ in range(num_rigid_bodies):
                if offset + 32 > len(data):
                    break

                # ID
                rb_id = struct.unpack_from("<i", data, offset)[0]
                offset += 4

                # Position (x, y, z) — in Motive's coordinate system
                px, py, pz = struct.unpack_from("<fff", data, offset)
                offset += 12

                # Quaternion (qx, qy, qz, qw) — in Motive's coordinate system
                qx, qy, qz, qw = struct.unpack_from("<ffff", data, offset)
                offset += 16

                # ── NatNet 2.x: per-body marker data (removed in 3.0) ──
                if natnet_major < 3:
                    n_body_markers = struct.unpack_from("<i", data, offset)[0]
                    offset += 4
                    # Marker positions: float32[nMarkers * 3]
                    offset += n_body_markers * 12
                    # Marker IDs: int32[nMarkers] (NatNet >= 2.0)
                    offset += n_body_markers * 4
                    # Marker sizes: float32[nMarkers] (NatNet >= 2.0)
                    offset += n_body_markers * 4

                # ── Mean marker error (float32) — NatNet >= 2.0 ──
                tracking_valid = True
                if offset + 4 <= len(data):
                    offset += 4  # Skip mean marker error

                # ── Params (int16) — bit 0 = tracking valid ──
                if offset + 2 <= len(data):
                    params = struct.unpack_from("<H", data, offset)[0]
                    offset += 2
                    tracking_valid = bool(params & 0x0001)

                # Convert coordinates
                pos = np.array([px, py, pz], dtype=np.float64)
                quat = np.array([qx, qy, qz, qw], dtype=np.float64)

                if self.convert_to_zup:
                    pos = position_yup_to_zup(pos)
                    quat = quaternion_yup_to_zup(quat)

                # Look up name
                name = self._id_to_name.get(rb_id, f"rigid_body_{rb_id}")

                new_bodies[name] = RigidBodyState(
                    name=name,
                    id=rb_id,
                    position=pos,
                    quaternion=quat,
                    timestamp=now,
                    tracking_valid=tracking_valid,
                )

            # Update shared state
            if new_bodies:
                with self._lock:
                    self._rigid_bodies = new_bodies
                self._frame_count += 1

                # Invoke callback if set
                if self._callback is not None:
                    try:
                        self._callback(new_bodies)
                    except Exception as e:
                        print(f"[OptiTrack] Callback error: {e}")

        except (struct.error, IndexError):
            # Malformed packet — skip silently (common during version negotiation)
            pass

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _read_cstring(data: bytes, offset: int) -> tuple:
        """
        Read a null-terminated C string from binary data.

        Returns (string, new_offset).
        """
        end = data.index(b'\x00', offset)
        s = data[offset:end].decode("utf-8", errors="replace")
        return s, end + 1
