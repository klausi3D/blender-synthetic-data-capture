"""
Framework Presets Panel - UI for selecting and applying framework presets.

This panel provides quick configuration for different Gaussian Splatting
frameworks like 3DGS, Nerfstudio, Postshot, Polycam, and more.
"""

import bpy
from bpy.types import Panel

from ..core.presets import PRESETS, get_preset_enum_items


class GSCAPTURE_PT_PresetsPanel(Panel):
    """Framework preset selection panel.

    Allows users to quickly configure capture settings for their
    target training framework with a single click.
    """

    bl_label = "Framework Preset"
    bl_idname = "GSCAPTURE_PT_presets"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_order = 0  # Show at top

    def draw_header(self, context):
        """Draw panel header with icon."""
        self.layout.label(text="", icon='PRESET')

    def draw(self, context):
        """Draw the preset selection UI."""
        layout = self.layout
        settings = context.scene.gs_capture_settings

        # Preset selector
        row = layout.row(align=True)
        row.prop(settings, "framework_preset", text="")
        row.operator("gs_capture.apply_preset", text="", icon='CHECKMARK')

        # Show preset info
        preset_id = settings.framework_preset
        if preset_id in PRESETS:
            preset = PRESETS[preset_id]

            box = layout.box()

            # Preset description
            col = box.column(align=True)
            col.scale_y = 0.8

            # Wrap description text
            words = preset.description.split()
            line = ""
            for word in words:
                if len(line + word) > 35:
                    col.label(text=line)
                    line = word + " "
                else:
                    line += word + " "
            if line:
                col.label(text=line.strip())

            # Recommended settings
            box.separator()

            col = box.column(align=True)
            col.label(text="Recommended Settings:", icon='INFO')

            min_cam, max_cam = preset.recommended_cameras
            col.label(text=f"  Cameras: {min_cam} - {max_cam}")

            res_w, res_h = preset.recommended_resolution
            col.label(text=f"  Resolution: {res_w} x {res_h}")

            col.label(text=f"  Format: {preset.file_format}")

            # Export formats
            exports = []
            if preset.export_colmap:
                exports.append("COLMAP")
            if preset.export_transforms_json:
                exports.append("transforms.json")
            col.label(text=f"  Exports: {', '.join(exports)}")

            # Website link
            if preset.website:
                box.separator()
                row = box.row()
                row.operator(
                    "wm.url_open",
                    text="Documentation",
                    icon='URL'
                ).url = preset.website

            # Notes
            if preset.notes:
                box.separator()
                notes_box = box.box()
                notes_box.scale_y = 0.7

                # Wrap notes text
                words = preset.notes.split()
                line = ""
                for word in words:
                    if len(line + word) > 40:
                        notes_box.label(text=line, icon='BLANK1')
                        line = word + " "
                    else:
                        line += word + " "
                if line:
                    notes_box.label(text=line.strip(), icon='BLANK1')


class GSCAPTURE_PT_PresetsQuickSettings(Panel):
    """Quick settings subpanel showing current preset values."""

    bl_label = "Current Settings"
    bl_idname = "GSCAPTURE_PT_presets_quick"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_parent_id = "GSCAPTURE_PT_presets"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        """Draw current settings comparison."""
        layout = self.layout
        settings = context.scene.gs_capture_settings
        scene = context.scene

        preset_id = settings.framework_preset
        if preset_id not in PRESETS:
            layout.label(text="Select a preset")
            return

        preset = PRESETS[preset_id]

        # Compare current vs recommended
        col = layout.column(align=True)

        # Camera count
        current_cameras = settings.camera_count
        min_cam, max_cam = preset.recommended_cameras

        row = col.row()
        row.label(text="Cameras:")
        if min_cam <= current_cameras <= max_cam:
            row.label(text=f"{current_cameras}", icon='CHECKMARK')
        else:
            row.label(text=f"{current_cameras}", icon='ERROR')
            row.label(text=f"({min_cam}-{max_cam})")

        # Resolution
        current_x = scene.render.resolution_x
        current_y = scene.render.resolution_y
        rec_x, rec_y = preset.recommended_resolution

        row = col.row()
        row.label(text="Resolution:")
        if current_x == rec_x and current_y == rec_y:
            row.label(text=f"{current_x}x{current_y}", icon='CHECKMARK')
        else:
            row.label(text=f"{current_x}x{current_y}", icon='DOT')

        # File format
        current_format = scene.render.image_settings.file_format
        row = col.row()
        row.label(text="Format:")
        if current_format == preset.file_format:
            row.label(text=current_format, icon='CHECKMARK')
        else:
            row.label(text=current_format, icon='DOT')

        # Background
        row = col.row()
        row.label(text="White BG:")
        if settings.transparent_background != preset.white_background:
            row.label(text="Yes" if not settings.transparent_background else "No", icon='CHECKMARK')
        else:
            row.label(text="Yes" if not settings.transparent_background else "No", icon='DOT')


# Registration
classes = [
    GSCAPTURE_PT_PresetsPanel,
    GSCAPTURE_PT_PresetsQuickSettings,
]


def register():
    """Register preset panels."""
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    """Unregister preset panels."""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
