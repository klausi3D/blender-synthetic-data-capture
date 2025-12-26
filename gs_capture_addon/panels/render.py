"""
Render settings panel.
Uses Blender's native render settings directly for proper sync.
"""

import bpy
from bpy.types import Panel


class GSCAPTURE_PT_render_panel(Panel):
    """Render settings panel."""
    bl_label = "Render Settings"
    bl_idname = "GSCAPTURE_PT_render_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_parent_id = "GSCAPTURE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.gs_capture_settings
        rd = scene.render

        # Render speed preset
        layout.prop(settings, "render_speed_preset")

        # Use Blender's native render settings directly
        layout.prop(rd, "engine", text="Engine")

        row = layout.row(align=True)
        row.prop(rd, "resolution_x", text="W")
        row.prop(rd, "resolution_y", text="H")

        # Show appropriate samples setting based on engine
        if rd.engine == 'CYCLES':
            layout.prop(scene.cycles, "samples", text="Samples")
        else:
            layout.prop(scene.eevee, "taa_render_samples", text="Samples")

        # File format from image settings
        layout.prop(rd.image_settings, "file_format")

        # Show transparent background option (only for formats that support alpha)
        file_format = rd.image_settings.file_format
        if file_format in ('PNG', 'OPEN_EXR', 'OPEN_EXR_MULTILAYER', 'TARGA', 'TARGA_RAW'):
            layout.prop(settings, "transparent_background")

        # Info box
        box = layout.box()
        box.label(text="These are Blender's render settings", icon='INFO')
        box.label(text="Changes sync with Render Properties")
