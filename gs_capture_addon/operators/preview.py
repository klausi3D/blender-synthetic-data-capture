"""
Preview and utility operators.
Camera preview visualization and file browser helpers.
"""

import bpy
import os
import subprocess
import sys
from bpy.types import Operator

from ..core.camera import (
    generate_camera_positions,
    get_objects_combined_bounds,
    create_camera_at_position,
    delete_gs_cameras,
)


class GSCAPTURE_OT_preview_cameras(Operator):
    """Create preview cameras to visualize capture setup."""
    bl_idname = "gs_capture.preview_cameras"
    bl_label = "Preview Cameras"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.gs_capture_settings

        # Get selected objects
        selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not selected:
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}

        # Remove existing preview cameras
        delete_gs_cameras(context)

        # Calculate bounds
        center, radius = get_objects_combined_bounds(selected)

        # Calculate distance
        if settings.camera_distance_mode == 'AUTO':
            distance = radius * settings.camera_distance_multiplier
        else:
            distance = settings.camera_distance

        # Generate camera positions
        points = generate_camera_positions(
            settings.camera_distribution,
            settings.camera_count,
            min_elevation=settings.min_elevation,
            max_elevation=settings.max_elevation,
            ring_count=settings.ring_count
        )

        # Create cameras
        for i, point in enumerate(points):
            cam_pos = center + point * distance
            cam = create_camera_at_position(context, cam_pos, center, f"GS_Cam_{i:04d}")
            cam.data.lens = settings.focal_length

            # Make cameras smaller for visualization
            cam.data.display_size = 0.2

        self.report({'INFO'}, f"Created {len(points)} preview cameras")
        return {'FINISHED'}


class GSCAPTURE_OT_clear_preview(Operator):
    """Remove preview cameras."""
    bl_idname = "gs_capture.clear_preview"
    bl_label = "Clear Preview"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        delete_gs_cameras(context)
        try:
            from ..utils.viewport import hide_camera_frustums
            hide_camera_frustums()
        except Exception:
            # Keep preview clear robust even if viewport utils are unavailable.
            pass
        self.report({'INFO'}, "Cleared preview cameras")
        return {'FINISHED'}


class GSCAPTURE_OT_open_output_folder(Operator):
    """Open the output folder in file browser."""
    bl_idname = "gs_capture.open_output_folder"
    bl_label = "Open Output Folder"

    def execute(self, context):
        settings = context.scene.gs_capture_settings
        output_path = bpy.path.abspath(settings.output_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        # Open folder in system file browser
        if sys.platform == 'win32':
            os.startfile(output_path)
        elif sys.platform == 'darwin':
            subprocess.run(['open', output_path])
        else:
            subprocess.run(['xdg-open', output_path])

        return {'FINISHED'}
