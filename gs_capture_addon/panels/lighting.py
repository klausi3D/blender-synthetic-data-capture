"""
Lighting and materials panel.
"""

import bpy
from bpy.types import Panel


class GSCAPTURE_PT_lighting_panel(Panel):
    """Lighting and materials settings panel."""
    bl_label = "Lighting & Materials"
    bl_idname = "GSCAPTURE_PT_lighting_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_parent_id = "GSCAPTURE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.gs_capture_settings

        layout.prop(settings, "lighting_mode")

        if settings.lighting_mode in ('WHITE', 'GRAY'):
            layout.prop(settings, "background_strength")

        if settings.lighting_mode == 'GRAY':
            layout.prop(settings, "gray_value")

        if settings.lighting_mode == 'HDR':
            layout.prop(settings, "hdr_path")
            layout.prop(settings, "hdr_strength")

        if settings.lighting_mode != 'KEEP':
            layout.prop(settings, "disable_scene_lights")

        layout.separator()
        layout.prop(settings, "material_mode")

        # Show info about material mode
        if settings.material_mode == 'VERTEX_COLOR':
            box = layout.box()
            box.label(text="Uses vertex colors if present", icon='INFO')
            box.label(text="Falls back to neutral gray otherwise")
        elif settings.material_mode == 'MATCAP':
            box = layout.box()
            box.label(text="Normal-based shading for form", icon='INFO')
