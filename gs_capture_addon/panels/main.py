"""
Main panel with quick capture buttons and progress display.
"""

import bpy
from bpy.types import Panel


class GSCAPTURE_PT_main_panel(Panel):
    """Main GS Capture panel."""
    bl_label = "GS Capture"
    bl_idname = "GSCAPTURE_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.gs_capture_settings

        # Main capture buttons
        box = layout.box()
        box.label(text="Quick Capture", icon='RENDER_STILL')

        if settings.is_rendering:
            # Show progress
            box.label(text=settings.current_render_info)
            box.prop(settings, "render_progress", text="Progress")
            box.operator("gs_capture.cancel_capture", text="Cancel", icon='CANCEL')
        else:
            row = box.row(align=True)
            row.scale_y = 1.5
            row.operator("gs_capture.capture_selected", text="Capture Selected", icon='RENDER_STILL')

            row = box.row(align=True)
            row.operator("gs_capture.preview_cameras", text="Preview", icon='CAMERA_DATA')
            row.operator("gs_capture.clear_preview", text="Clear", icon='X')

        # Checkpoint status
        if settings.enable_checkpoints:
            from ..utils.checkpoint import load_checkpoint
            output_path = bpy.path.abspath(settings.output_path)
            checkpoint = load_checkpoint(output_path)
            if checkpoint:
                box = layout.box()
                box.label(text="Checkpoint Found", icon='FILE_REFRESH')
                completed = len(checkpoint.get('completed_images', []))
                total = checkpoint.get('total_cameras', 0)
                box.label(text=f"Progress: {completed}/{total} images")
                if settings.auto_resume:
                    box.label(text="Will auto-resume", icon='INFO')
