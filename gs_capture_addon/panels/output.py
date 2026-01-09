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

        # Mask options when enabled
        if settings.export_masks:
            box = layout.box()
            box.label(text="Mask Settings:", icon='MOD_MASK')
            box.prop(settings, "mask_source")
            box.prop(settings, "mask_format")

            # Info about mask format
            if settings.mask_format == 'GSL':
                box.label(text="For GS-Lightning training", icon='INFO')

            if settings.mask_source == 'ALPHA':
                box.label(text="Requires transparent background", icon='INFO')

        # Checkpoint settings
        layout.separator()
        layout.label(text="Resume/Checkpoint:", icon='FILE_REFRESH')

        col = layout.column(align=True)
        col.prop(settings, "enable_checkpoints", text="Enable Checkpoints")
        if settings.enable_checkpoints:
            col.prop(settings, "checkpoint_interval", text="Save Every N Images")
            col.prop(settings, "auto_resume", text="Auto Resume")

        # Size Estimation
        layout.separator()
        box = layout.box()
        box.label(text="Estimated Size:", icon='DISK_DRIVE')
        from ..core.validation import estimate_capture_size
        estimate = estimate_capture_size(settings, context)
        row = box.row()
        row.label(text=f"Images: {estimate['images_mb']:.0f} MB")
        if estimate['depth_mb'] > 0:
            row.label(text=f"Depth: {estimate['depth_mb']:.0f} MB")
        if estimate['normals_mb'] > 0:
            row = box.row()
            row.label(text=f"Normals: {estimate['normals_mb']:.0f} MB")
        box.label(text=f"Total: {estimate['total_gb']:.2f} GB", icon='INFO')
        if estimate['warning']:
            box.label(text=estimate['warning'], icon='ERROR')
