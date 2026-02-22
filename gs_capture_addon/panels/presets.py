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
            col = box.column(align=True)
            col.scale_y = 0.85

            # Compact description preview (max 2 lines).
            words = preset.description.split()
            desc_lines = []
            line = ""
            truncated = False
            for word in words:
                if len(line + word) > 44:
                    desc_lines.append(line.strip())
                    line = word + " "
                    if len(desc_lines) >= 2:
                        truncated = True
                        break
                else:
                    line += word + " "
            if len(desc_lines) < 2 and line.strip():
                desc_lines.append(line.strip())
            for entry in desc_lines:
                col.label(text=entry)
            if truncated:
                col.label(text="...", icon='DOT')

            min_cam, max_cam = preset.recommended_cameras
            res_w, res_h = preset.recommended_resolution
            exports = []
            if preset.export_colmap:
                exports.append("COLMAP")
            if preset.export_transforms_json:
                exports.append("transforms.json")
            export_text = ", ".join(exports) if exports else "None"
            col.separator()
            col.label(
                text=f"Rec: {min_cam}-{max_cam} cams | {res_w}x{res_h} | {preset.file_format} | {export_text}",
                icon='SETTINGS',
            )

            # Show first note as a compact hint.
            first_note = ""
            if isinstance(preset.notes, list) and preset.notes:
                first_note = str(preset.notes[0]).strip()
            elif isinstance(preset.notes, str):
                first_note = preset.notes.strip().splitlines()[0] if preset.notes.strip() else ""
            if first_note:
                clipped_note = first_note[:72] + ("..." if len(first_note) > 72 else "")
                col.label(text=clipped_note, icon='INFO')

            box.separator()
            actions = box.row(align=True)
            info_op = actions.operator("gs_capture.preset_info", text="Details", icon='INFO')
            info_op.preset_id = preset_id
            actions.operator("gs_capture.compare_presets", text="", icon='PRESET')
            if preset.website:
                actions.operator("wm.url_open", text="", icon='URL').url = preset.website


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

        # Compare current vs recommended and show only drift to reduce noise.
        mismatches = []

        # Camera count drift
        current_cameras = settings.camera_count
        min_cam, max_cam = preset.recommended_cameras
        if not (min_cam <= current_cameras <= max_cam):
            mismatches.append(("Cameras", str(current_cameras), f"{min_cam}-{max_cam}"))

        # Resolution drift
        current_x = scene.render.resolution_x
        current_y = scene.render.resolution_y
        rec_x, rec_y = preset.recommended_resolution
        if current_x != rec_x or current_y != rec_y:
            mismatches.append(("Resolution", f"{current_x}x{current_y}", f"{rec_x}x{rec_y}"))

        # File format drift
        current_format = scene.render.image_settings.file_format
        if current_format != preset.file_format:
            mismatches.append(("Format", current_format, preset.file_format))

        # Background drift (UI stores transparency; presets store white BG).
        current_white_bg = not settings.transparent_background
        preset_white_bg = preset.white_background
        if current_white_bg != preset_white_bg:
            mismatches.append(
                (
                    "White BG",
                    "Yes" if current_white_bg else "No",
                    "Yes" if preset_white_bg else "No",
                )
            )

        # Export drift.
        current_exports = []
        if settings.export_colmap:
            current_exports.append("COLMAP")
        if settings.export_transforms_json:
            current_exports.append("transforms.json")
        preset_exports = []
        if preset.export_colmap:
            preset_exports.append("COLMAP")
        if preset.export_transforms_json:
            preset_exports.append("transforms.json")
        if set(current_exports) != set(preset_exports):
            mismatches.append(
                (
                    "Exports",
                    ", ".join(current_exports) if current_exports else "None",
                    ", ".join(preset_exports) if preset_exports else "None",
                )
            )

        if not mismatches:
            layout.label(text="Current capture settings match preset.", icon='CHECKMARK')
            return

        layout.label(text=f"{len(mismatches)} setting(s) differ from preset", icon='ERROR')
        col = layout.column(align=True)
        for label, current, recommended in mismatches:
            row = col.row(align=True)
            row.label(text=f"{label}:")
            row.label(text=current, icon='DOT')
            row.label(text=f"-> {recommended}")

        layout.separator()
        layout.operator("gs_capture.apply_preset", text="Re-apply Preset", icon='FILE_REFRESH')


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
