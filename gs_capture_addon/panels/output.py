"""
Output settings panel.
"""

import bpy
from bpy.types import Panel


class GSCAPTURE_PT_output_panel(Panel):
    """Output settings panel."""
    bl_label = "Output"
    bl_idname = "GSCAPTURE_PT_output_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_parent_id = "GSCAPTURE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.gs_capture_settings

        layout.prop(settings, "output_path")

        row = layout.row()
        row.operator("gs_capture.open_output_folder", text="Open Folder", icon='FILE_FOLDER')

        layout.separator()
        layout.label(text="Export Formats:", icon='EXPORT')

        col = layout.column(align=True)
        col.prop(settings, "export_colmap", text="COLMAP (sparse/0/)")
        col.prop(settings, "export_transforms_json", text="transforms.json")

        layout.separator()
        layout.label(text="Additional Outputs:", icon='IMAGE_DATA')

        col = layout.column(align=True)
        col.prop(settings, "export_depth", text="Depth Maps")
        col.prop(settings, "export_normals", text="Normal Maps")
        col.prop(settings, "export_masks", text="Object Masks")

        # Checkpoint settings
        layout.separator()
        layout.label(text="Resume/Checkpoint:", icon='FILE_REFRESH')

        col = layout.column(align=True)
        col.prop(settings, "enable_checkpoints", text="Enable Checkpoints")
        if settings.enable_checkpoints:
            col.prop(settings, "checkpoint_interval", text="Save Every N Images")
            col.prop(settings, "auto_resume", text="Auto Resume")
