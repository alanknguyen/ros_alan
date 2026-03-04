# vision_engine/graphics — OpenGL 3D Rendering module
#
# This module handles:
#   - Real-time 3D scene visualization for human operators (GLFW window)
#   - Offscreen framebuffer rendering for VLM snapshot capture
#   - Scene graph management (objects, table, axes, workspace bounds)
#   - Multiple camera viewpoints (birds-eye, robot-view, free orbit)
#   - GLSL Phong shading with directional lighting
#   - Mesh generation for primitive shapes (cube, cylinder, sphere)
#   - RGB-D point cloud rendering as colored GL_POINTS
