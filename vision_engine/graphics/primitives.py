"""
vision_engine/graphics/primitives.py — 3D Mesh Generators

Generates vertex data (position + normal + color) for primitive shapes.
All meshes are returned as numpy arrays ready for upload to OpenGL vertex
buffers via moderngl.

Vertex Format
-------------
Each vertex has 9 floats: [x, y, z, nx, ny, nz, r, g, b]
  - (x, y, z): position
  - (nx, ny, nz): surface normal (for lighting)
  - (r, g, b): vertex color in [0, 1]

All primitives are generated centered at the origin with unit dimensions.
Scale and position are applied at render time via the model matrix.
"""

import numpy as np
from typing import Tuple


def make_cube(color: Tuple[float, float, float] = (0.8, 0.2, 0.2)) -> np.ndarray:
    """
    Generate a unit cube centered at origin (side length = 1.0).

    Each face has its own vertices with correct face normals for Phong shading.
    6 faces × 2 triangles × 3 vertices = 36 vertices.

    Parameters
    ----------
    color : tuple of 3 floats
        RGB color in [0, 1].

    Returns
    -------
    vertices : np.ndarray, shape (36, 9)
        Vertex data [x, y, z, nx, ny, nz, r, g, b].
    """
    r, g, b = color
    s = 0.5  # half-size

    # Define 6 faces with positions and normals
    # Each face: 4 corners → 2 triangles (6 vertices)
    faces = [
        # Front face (+Z)
        (( s,  s,  s), (-s,  s,  s), (-s, -s,  s), ( s, -s,  s), ( 0,  0,  1)),
        # Back face (-Z)
        ((-s,  s, -s), ( s,  s, -s), ( s, -s, -s), (-s, -s, -s), ( 0,  0, -1)),
        # Top face (+Y... but in Z-up this is +Z, handled by model matrix)
        (( s,  s, -s), (-s,  s, -s), (-s,  s,  s), ( s,  s,  s), ( 0,  1,  0)),
        # Bottom face (-Y)
        (( s, -s,  s), (-s, -s,  s), (-s, -s, -s), ( s, -s, -s), ( 0, -1,  0)),
        # Right face (+X)
        (( s,  s, -s), ( s,  s,  s), ( s, -s,  s), ( s, -s, -s), ( 1,  0,  0)),
        # Left face (-X)
        ((-s,  s,  s), (-s,  s, -s), (-s, -s, -s), (-s, -s,  s), (-1,  0,  0)),
    ]

    verts = []
    for p0, p1, p2, p3, normal in faces:
        nx, ny, nz = normal
        # Triangle 1: p0, p1, p2
        verts.append([*p0, nx, ny, nz, r, g, b])
        verts.append([*p1, nx, ny, nz, r, g, b])
        verts.append([*p2, nx, ny, nz, r, g, b])
        # Triangle 2: p0, p2, p3
        verts.append([*p0, nx, ny, nz, r, g, b])
        verts.append([*p2, nx, ny, nz, r, g, b])
        verts.append([*p3, nx, ny, nz, r, g, b])

    return np.array(verts, dtype=np.float32)


def make_cylinder(
    segments: int = 24,
    color: Tuple[float, float, float] = (0.2, 0.2, 0.8),
) -> np.ndarray:
    """
    Generate a unit cylinder (radius=0.5, height=1.0) centered at origin,
    aligned along the Z axis.

    Parameters
    ----------
    segments : int
        Number of segments around the circumference (default 24).
    color : tuple of 3 floats
        RGB color in [0, 1].

    Returns
    -------
    vertices : np.ndarray, shape (N, 9)
        Vertex data [x, y, z, nx, ny, nz, r, g, b].
    """
    r, g, b = color
    radius = 0.5
    half_h = 0.5
    verts = []

    for i in range(segments):
        a0 = 2.0 * np.pi * i / segments
        a1 = 2.0 * np.pi * (i + 1) / segments

        c0, s0 = np.cos(a0), np.sin(a0)
        c1, s1 = np.cos(a1), np.sin(a1)

        x0, y0 = radius * c0, radius * s0
        x1, y1 = radius * c1, radius * s1

        # Side wall: 2 triangles
        # Normal points radially outward
        nx0, ny0 = c0, s0
        nx1, ny1 = c1, s1

        verts.append([x0, y0,  half_h, nx0, ny0, 0, r, g, b])
        verts.append([x1, y1,  half_h, nx1, ny1, 0, r, g, b])
        verts.append([x1, y1, -half_h, nx1, ny1, 0, r, g, b])

        verts.append([x0, y0,  half_h, nx0, ny0, 0, r, g, b])
        verts.append([x1, y1, -half_h, nx1, ny1, 0, r, g, b])
        verts.append([x0, y0, -half_h, nx0, ny0, 0, r, g, b])

        # Top cap
        verts.append([0, 0, half_h, 0, 0, 1, r, g, b])
        verts.append([x0, y0, half_h, 0, 0, 1, r, g, b])
        verts.append([x1, y1, half_h, 0, 0, 1, r, g, b])

        # Bottom cap
        verts.append([0, 0, -half_h, 0, 0, -1, r, g, b])
        verts.append([x1, y1, -half_h, 0, 0, -1, r, g, b])
        verts.append([x0, y0, -half_h, 0, 0, -1, r, g, b])

    return np.array(verts, dtype=np.float32)


def make_sphere(
    rings: int = 16,
    sectors: int = 24,
    color: Tuple[float, float, float] = (0.2, 0.8, 0.2),
) -> np.ndarray:
    """
    Generate a unit sphere (radius=0.5) centered at origin.

    Parameters
    ----------
    rings : int
        Number of horizontal rings (default 16).
    sectors : int
        Number of vertical sectors (default 24).
    color : tuple of 3 floats
        RGB color in [0, 1].

    Returns
    -------
    vertices : np.ndarray, shape (N, 9)
        Vertex data [x, y, z, nx, ny, nz, r, g, b].
    """
    r_c, g_c, b_c = color
    radius = 0.5
    verts = []

    for i in range(rings):
        theta0 = np.pi * i / rings
        theta1 = np.pi * (i + 1) / rings

        for j in range(sectors):
            phi0 = 2.0 * np.pi * j / sectors
            phi1 = 2.0 * np.pi * (j + 1) / sectors

            # 4 corners of this quad
            def sphere_point(theta, phi):
                x = radius * np.sin(theta) * np.cos(phi)
                y = radius * np.sin(theta) * np.sin(phi)
                z = radius * np.cos(theta)
                nx = np.sin(theta) * np.cos(phi)
                ny = np.sin(theta) * np.sin(phi)
                nz = np.cos(theta)
                return [x, y, z, nx, ny, nz, r_c, g_c, b_c]

            p00 = sphere_point(theta0, phi0)
            p10 = sphere_point(theta1, phi0)
            p11 = sphere_point(theta1, phi1)
            p01 = sphere_point(theta0, phi1)

            verts.extend([p00, p10, p11])
            verts.extend([p00, p11, p01])

    return np.array(verts, dtype=np.float32)


def make_grid(
    size: float = 2.0,
    spacing: float = 0.1,
    color: Tuple[float, float, float] = (0.4, 0.4, 0.4),
) -> np.ndarray:
    """
    Generate a flat grid of lines on the XY plane (at Z=0).

    Used for the table surface visualization. Lines run parallel to X and Y.

    Parameters
    ----------
    size : float
        Total grid size in meters (centered at origin).
    spacing : float
        Distance between grid lines in meters.
    color : tuple of 3 floats
        RGB color in [0, 1].

    Returns
    -------
    vertices : np.ndarray, shape (N, 6)
        Vertex data [x, y, z, r, g, b] — no normals (flat shader).
    """
    r, g, b = color
    half = size / 2.0
    lines = []

    # Lines parallel to X
    y = -half
    while y <= half + 1e-6:
        lines.append([-half, y, 0, r, g, b])
        lines.append([ half, y, 0, r, g, b])
        y += spacing

    # Lines parallel to Y
    x = -half
    while x <= half + 1e-6:
        lines.append([x, -half, 0, r, g, b])
        lines.append([x,  half, 0, r, g, b])
        x += spacing

    return np.array(lines, dtype=np.float32)


def make_axis_triad(length: float = 0.1) -> np.ndarray:
    """
    Generate a coordinate axis triad (RGB = XYZ).

    Three colored lines from origin along each axis:
      - X axis: Red
      - Y axis: Green
      - Z axis: Blue

    Parameters
    ----------
    length : float
        Length of each axis line in meters.

    Returns
    -------
    vertices : np.ndarray, shape (6, 6)
        Vertex data [x, y, z, r, g, b] — no normals (flat shader).
    """
    return np.array([
        # X axis (red)
        [0, 0, 0, 1, 0, 0],
        [length, 0, 0, 1, 0, 0],
        # Y axis (green)
        [0, 0, 0, 0, 1, 0],
        [0, length, 0, 0, 1, 0],
        # Z axis (blue)
        [0, 0, 0, 0, 0, 1],
        [0, 0, length, 0, 0, 1],
    ], dtype=np.float32)


def make_wireframe_box(
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    z_min: float, z_max: float,
    color: Tuple[float, float, float] = (0.5, 0.5, 0.0),
) -> np.ndarray:
    """
    Generate wireframe edges of an axis-aligned box.

    12 edges × 2 vertices = 24 vertices for GL_LINES.

    Parameters
    ----------
    x_min, x_max, y_min, y_max, z_min, z_max : float
        Box bounds.
    color : tuple of 3 floats
        RGB color in [0, 1].

    Returns
    -------
    vertices : np.ndarray, shape (24, 6)
        Vertex data [x, y, z, r, g, b].
    """
    r, g, b = color
    corners = [
        (x_min, y_min, z_min), (x_max, y_min, z_min),
        (x_max, y_max, z_min), (x_min, y_max, z_min),
        (x_min, y_min, z_max), (x_max, y_min, z_max),
        (x_max, y_max, z_max), (x_min, y_max, z_max),
    ]

    edges = [
        (0,1),(1,2),(2,3),(3,0),  # Bottom face
        (4,5),(5,6),(6,7),(7,4),  # Top face
        (0,4),(1,5),(2,6),(3,7),  # Vertical edges
    ]

    verts = []
    for i, j in edges:
        verts.append([*corners[i], r, g, b])
        verts.append([*corners[j], r, g, b])

    return np.array(verts, dtype=np.float32)
