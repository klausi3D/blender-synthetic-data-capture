"""
Camera settings panel.
"""

import bpy
from bpy.types import Panel


class GSCAPTURE_PT_camera_panel(Panel):
    """Camera distribution and settings panel."""
    bl_label = "Camera Settings"
    bl_idname = "GSCAPTURE_PT_camera_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_parent_id = "GSCAPTURE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.gs_capture_settings

        layout.prop(settings, "camera_count")
        layout.prop(settings, "camera_distribution")

        if settings.camera_distribution == 'MULTI_RING':
            layout.prop(settings, "ring_count")

        row = layout.row(align=True)
        row.prop(settings, "min_elevation")
        row.prop(settings, "max_elevation")

        layout.separator()
        layout.prop(settings, "camera_distance_mode")

        if settings.camera_distance_mode == 'AUTO':
            layout.prop(settings, "camera_distance_multiplier")
        else:
            layout.prop(settings, "camera_distance")

        layout.prop(settings, "focal_length")
