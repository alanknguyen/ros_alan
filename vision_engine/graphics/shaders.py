"""
vision_engine/graphics/shaders.py — GLSL Shader Programs

Contains vertex and fragment shader source code as Python strings.
These are compiled at runtime by moderngl.

Shader Programs
---------------
1. PHONG — Standard Phong lighting for solid objects (cubes, cylinders, etc.)
   - Per-vertex color
   - Single directional light
   - Ambient + diffuse + specular components

2. FLAT — Flat unlit color for lines (axes, grid, workspace bounds)
   - Per-vertex color, no lighting
   - Used for wireframe overlays

3. POINT_CLOUD — Colored points for RGB-D point cloud rendering
   - Per-vertex color from RGB image
   - Configurable point size
"""

# ──────────────────────────────────────────────────────────────────────────────
# 1. Phong-Lit Solid Shader
# ──────────────────────────────────────────────────────────────────────────────

PHONG_VERTEX = """
#version 330 core

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;

out vec3 v_position;   // World-space position (for specular)
out vec3 v_normal;     // World-space normal
out vec3 v_color;      // Vertex color

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_position = world_pos.xyz;
    v_normal = mat3(transpose(inverse(u_model))) * in_normal;
    v_color = in_color;
    gl_Position = u_projection * u_view * world_pos;
}
"""

PHONG_FRAGMENT = """
#version 330 core

uniform vec3 u_light_dir;       // Directional light direction (normalized)
uniform vec3 u_light_color;     // Light color (usually white)
uniform vec3 u_ambient_color;   // Ambient light color
uniform vec3 u_camera_pos;      // Camera position (for specular)

in vec3 v_position;
in vec3 v_normal;
in vec3 v_color;

out vec4 frag_color;

void main() {
    vec3 normal = normalize(v_normal);
    vec3 light_dir = normalize(-u_light_dir);  // Point towards light

    // Ambient
    vec3 ambient = u_ambient_color * v_color;

    // Diffuse
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = diff * u_light_color * v_color;

    // Specular (Blinn-Phong)
    vec3 view_dir = normalize(u_camera_pos - v_position);
    vec3 half_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, half_dir), 0.0), 32.0);
    vec3 specular = spec * u_light_color * 0.3;

    vec3 result = ambient + diffuse + specular;
    frag_color = vec4(result, 1.0);
}
"""

# ──────────────────────────────────────────────────────────────────────────────
# 2. Flat (Unlit) Line/Wire Shader
# ──────────────────────────────────────────────────────────────────────────────

FLAT_VERTEX = """
#version 330 core

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

in vec3 in_position;
in vec3 in_color;

out vec3 v_color;

void main() {
    v_color = in_color;
    gl_Position = u_projection * u_view * u_model * vec4(in_position, 1.0);
}
"""

FLAT_FRAGMENT = """
#version 330 core

in vec3 v_color;
out vec4 frag_color;

void main() {
    frag_color = vec4(v_color, 1.0);
}
"""

# ──────────────────────────────────────────────────────────────────────────────
# 3. Point Cloud Shader
# ──────────────────────────────────────────────────────────────────────────────

POINT_CLOUD_VERTEX = """
#version 330 core

uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_point_size;

in vec3 in_position;
in vec3 in_color;

out vec3 v_color;

void main() {
    v_color = in_color;
    gl_Position = u_projection * u_view * vec4(in_position, 1.0);
    gl_PointSize = u_point_size;
}
"""

POINT_CLOUD_FRAGMENT = """
#version 330 core

in vec3 v_color;
out vec4 frag_color;

void main() {
    // Circular point shape (discard corners of the point quad)
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (dot(coord, coord) > 0.25) discard;

    frag_color = vec4(v_color, 1.0);
}
"""
