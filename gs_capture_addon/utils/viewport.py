"""
Viewport visualization utilities for camera preview and coverage display.
Uses GPU module for efficient drawing.
"""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix
import math


class CameraVisualizer:
    """GPU-accelerated camera visualization in viewport."""

    _instance = None
    _handle = None

    def __init__(self):
        self.cameras = []
        self.show_frustums = True
        self.show_coverage = False
        self.frustum_color = (0.2, 0.6, 1.0, 0.5)
        self.frustum_length = 1.0
        self._shader = None
        self._batch = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = CameraVisualizer()
        return cls._instance

    def set_cameras(self, cameras):
        """Set cameras to visualize.

        Args:
            cameras: List of camera objects
        """
        self.cameras = cameras
        self._rebuild_batch()

    def _rebuild_batch(self):
        """Rebuild GPU batch for drawing."""
        if not self.cameras:
            self._batch = None
            return

        vertices = []
        colors = []

        for cam in self.cameras:
            frustum_verts = self._get_frustum_vertices(cam)
            vertices.extend(frustum_verts)
            # Color for each vertex
            for _ in frustum_verts:
                colors.append(self.frustum_color)

        if vertices:
            self._shader = gpu.shader.from_builtin('FLAT_COLOR')
            self._batch = batch_for_shader(
                self._shader, 'LINES',
                {"pos": vertices, "color": colors}
            )

    def _get_frustum_vertices(self, camera):
        """Calculate frustum line vertices for a camera.

        Args:
            camera: Blender camera object

        Returns:
            List of Vector vertices for line drawing
        """
        cam_data = camera.data
        mat = camera.matrix_world

        # Camera position
        pos = mat.translation

        # Get camera properties
        if cam_data.type == 'PERSP':
            angle = cam_data.angle
        else:
            angle = math.radians(50)  # Default for ortho

        # Calculate frustum corners at frustum_length distance
        aspect = 16 / 9  # Assume 16:9
        half_height = math.tan(angle / 2) * self.frustum_length
        half_width = half_height * aspect

        # Local frustum corners (camera looks down -Z)
        corners_local = [
            Vector((-half_width, -half_height, -self.frustum_length)),
            Vector((half_width, -half_height, -self.frustum_length)),
            Vector((half_width, half_height, -self.frustum_length)),
            Vector((-half_width, half_height, -self.frustum_length)),
        ]

        # Transform to world space
        corners_world = [mat @ c for c in corners_local]

        # Create line pairs
        lines = []

        # Lines from camera to corners
        for corner in corners_world:
            lines.append(pos)
            lines.append(corner)

        # Lines connecting corners (rectangle)
        for i in range(4):
            lines.append(corners_world[i])
            lines.append(corners_world[(i + 1) % 4])

        return lines

    def draw(self):
        """Draw callback for viewport."""
        if not self._batch or not self.show_frustums:
            return

        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(1.5)

        self._shader.bind()
        self._batch.draw(self._shader)

        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)

    def register(self):
        """Register draw handler."""
        if CameraVisualizer._handle is None:
            CameraVisualizer._handle = bpy.types.SpaceView3D.draw_handler_add(
                self.draw, (), 'WINDOW', 'POST_VIEW'
            )

    def unregister(self):
        """Unregister draw handler."""
        if CameraVisualizer._handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(
                CameraVisualizer._handle, 'WINDOW'
            )
            CameraVisualizer._handle = None

    def clear(self):
        """Clear all cameras and batch."""
        self.cameras = []
        self._batch = None


class CoverageVisualizer:
    """Visualize camera coverage on mesh using vertex colors."""

    def __init__(self):
        self.coverage_attribute_name = "gs_coverage"

    def create_coverage_colors(self, mesh_obj, coverage_data):
        """Create vertex color layer showing coverage.

        Args:
            mesh_obj: Blender mesh object
            coverage_data: Dict mapping vertex index to coverage count
        """
        mesh = mesh_obj.data

        # Remove existing coverage attribute
        if self.coverage_attribute_name in mesh.color_attributes:
            mesh.color_attributes.remove(
                mesh.color_attributes[self.coverage_attribute_name]
            )

        # Create new color attribute
        color_attr = mesh.color_attributes.new(
            name=self.coverage_attribute_name,
            type='FLOAT_COLOR',
            domain='POINT'
        )

        # Calculate min/max for normalization
        if coverage_data:
            max_coverage = max(coverage_data.values())
            min_coverage = min(coverage_data.values())
        else:
            max_coverage = 1
            min_coverage = 0

        # Set colors based on coverage
        for vert in mesh.vertices:
            count = coverage_data.get(vert.index, 0)

            # Normalize to 0-1
            if max_coverage > min_coverage:
                normalized = (count - min_coverage) / (max_coverage - min_coverage)
            else:
                normalized = 1.0

            # Color gradient: red (poor) -> yellow -> green (good)
            if normalized < 0.5:
                r = 1.0
                g = normalized * 2
                b = 0.0
            else:
                r = 1.0 - (normalized - 0.5) * 2
                g = 1.0
                b = 0.0

            # Set color for this vertex
            color_attr.data[vert.index].color = (r, g, b, 1.0)

    def remove_coverage_colors(self, mesh_obj):
        """Remove coverage color layer.

        Args:
            mesh_obj: Blender mesh object
        """
        mesh = mesh_obj.data
        if self.coverage_attribute_name in mesh.color_attributes:
            mesh.color_attributes.remove(
                mesh.color_attributes[self.coverage_attribute_name]
            )

    def set_viewport_shading_for_coverage(self, context):
        """Set viewport to show vertex colors.

        Args:
            context: Blender context
        """
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'SOLID'
                        space.shading.color_type = 'VERTEX'


def get_camera_visualizer():
    """Get the global camera visualizer instance."""
    return CameraVisualizer.get_instance()


def show_camera_frustums(cameras, frustum_length=1.0):
    """Show camera frustums in viewport.

    Args:
        cameras: List of camera objects
        frustum_length: Length of frustum visualization
    """
    viz = get_camera_visualizer()
    viz.frustum_length = frustum_length
    viz.set_cameras(cameras)
    viz.show_frustums = True
    viz.register()

    _tag_view3d_redraw()


def hide_camera_frustums():
    """Hide camera frustums."""
    viz = get_camera_visualizer()
    viz.show_frustums = False
    viz.clear()
    viz.unregister()

    _tag_view3d_redraw()


def cleanup_visualizers():
    """Cleanup all visualizers on addon unload."""
    viz = get_camera_visualizer()
    viz.unregister()
    viz.clear()


def _tag_view3d_redraw():
    """Best-effort redraw request for all 3D viewports."""
    context = bpy.context
    screen = getattr(context, "screen", None)
    if screen is None:
        return
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()
