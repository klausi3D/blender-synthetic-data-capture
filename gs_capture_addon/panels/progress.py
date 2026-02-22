"""
Progress panel showing capture status and controls.
Only visible during active capture or when last capture stats exist.
"""

import bpy
from bpy.types import Panel


def format_time(seconds):
    """Format seconds into MM:SS or HH:MM:SS string."""
    if seconds <= 0:
        return "0:00"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


class GSCAPTURE_PT_progress_panel(Panel):
    """Progress panel showing capture status."""
    bl_label = "Capture Progress"
    bl_idname = "GSCAPTURE_PT_progress_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_order = 0  # Show at top

    @classmethod
    def poll(cls, context):
        """Only show when capturing or when last capture stats exist."""
        settings = context.scene.gs_capture_settings
        return settings.is_rendering or settings.last_capture_images > 0

    def draw(self, context):
        layout = self.layout
        settings = context.scene.gs_capture_settings

        if settings.is_rendering:
            self.draw_active_progress(layout, settings)
        else:
            self.draw_last_capture(layout, settings)

    def draw_active_progress(self, layout, settings):
        """Draw progress for active capture."""
        box = layout.box()

        # Header with spinning icon
        row = box.row()
        row.label(text="Capturing...", icon='RENDER_ANIMATION')

        # Progress bar
        row = box.row()
        row.enabled = False
        row.prop(settings, "render_progress", text="")

        # Current/Total
        row = box.row()
        row.label(text=f"{settings.capture_current} / {settings.capture_total} images")
        percentage = (settings.capture_current / settings.capture_total * 100) if settings.capture_total > 0 else 0
        row.label(text=f"({percentage:.1f}%)")

        box.separator()

        # Current camera
        if settings.capture_current_camera:
            row = box.row()
            row.label(text="Camera:", icon='CAMERA_DATA')
            row.label(text=settings.capture_current_camera)

        # Current object(s)
        if settings.capture_current_object:
            row = box.row()
            row.label(text="Object:", icon='OBJECT_DATA')
            row.label(text=settings.capture_current_object)

        box.separator()

        # Time stats
        col = box.column(align=True)

        # Elapsed time
        row = col.row()
        row.label(text="Elapsed:", icon='TIME')
        row.label(text=format_time(settings.capture_elapsed_seconds))

        # ETA
        row = col.row()
        row.label(text="ETA:", icon='PREVIEW_RANGE')
        row.label(text=f"~{format_time(settings.capture_eta_seconds)}")

        # Rate
        row = col.row()
        row.label(text="Rate:", icon='SORTTIME')
        row.label(text=f"{settings.capture_rate:.2f} img/sec")

        box.separator()

        # Cancel button
        row = box.row()
        row.scale_y = 1.5
        row.alert = True
        row.operator("gs_capture.cancel_capture", text="Cancel Capture", icon='CANCEL')

    def draw_last_capture(self, layout, settings):
        """Draw last capture statistics."""
        box = layout.box()

        # Header
        row = box.row()
        if settings.last_capture_success:
            row.label(text="Last Capture", icon='CHECKMARK')
        else:
            row.label(text="Last Capture (Cancelled)", icon='X')

        # Stats
        col = box.column(align=True)

        row = col.row()
        row.label(text="Images:", icon='IMAGE_DATA')
        row.label(text=str(settings.last_capture_images))

        row = col.row()
        row.label(text="Duration:", icon='TIME')
        row.label(text=format_time(settings.last_capture_duration))

        if settings.last_capture_duration > 0 and settings.last_capture_images > 0:
            rate = settings.last_capture_images / settings.last_capture_duration
            row = col.row()
            row.label(text="Avg Rate:", icon='SORTTIME')
            row.label(text=f"{rate:.2f} img/sec")

        # Path (truncated)
        if settings.last_capture_path:
            box.separator()
            row = box.row()
            row.label(text="Output:", icon='FOLDER_REDIRECT')
            # Truncate path if too long
            path = settings.last_capture_path
            if len(path) > 35:
                path = "..." + path[-32:]
            row.label(text=path)

        # Clear button
        box.separator()
        row = box.row()
        row.operator("gs_capture.clear_last_capture", text="Dismiss", icon='X')


class GSCAPTURE_OT_clear_last_capture(bpy.types.Operator):
    """Clear last capture statistics."""
    bl_idname = "gs_capture.clear_last_capture"
    bl_label = "Clear Last Capture Stats"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.gs_capture_settings
        settings.last_capture_images = 0
        settings.last_capture_duration = 0.0
        settings.last_capture_path = ""
        settings.last_capture_success = False
        return {'FINISHED'}


# Registration
classes = [
    GSCAPTURE_PT_progress_panel,
    GSCAPTURE_OT_clear_last_capture,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
