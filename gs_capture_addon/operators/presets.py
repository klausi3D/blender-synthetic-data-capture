"""
Preset Operators - Apply framework presets to capture settings.

Provides operators for applying and managing framework-specific
configurations for optimal Gaussian Splatting training data.
"""

import bpy
from bpy.types import Operator
from bpy.props import StringProperty, EnumProperty

from ..core.presets import PRESETS, apply_preset_to_settings, get_preset_enum_items


class GSCAPTURE_OT_ApplyPreset(Operator):
    """Apply the selected framework preset to capture settings.

    This operator configures camera count, resolution, file format,
    and export options based on the selected framework's requirements.
    """

    bl_idname = "gs_capture.apply_preset"
    bl_label = "Apply Preset"
    bl_description = "Apply framework preset to configure optimal settings"
    bl_options = {'REGISTER', 'UNDO'}

    preset_id: StringProperty(
        name="Preset ID",
        description="ID of preset to apply (uses current selection if empty)",
        default=""
    )

    def execute(self, context):
        """Apply the preset settings."""
        settings = context.scene.gs_capture_settings

        # Use provided preset_id or current selection
        preset_id = self.preset_id or settings.framework_preset

        if preset_id not in PRESETS:
            self.report({'ERROR'}, f"Unknown preset: {preset_id}")
            return {'CANCELLED'}

        preset = PRESETS[preset_id]

        # Apply preset settings
        apply_preset_to_settings(preset, settings, context.scene)

        self.report({'INFO'}, f"Applied preset: {preset.name}")
        return {'FINISHED'}


class GSCAPTURE_OT_PresetInfo(Operator):
    """Show detailed information about a framework preset.

    Opens a popup with comprehensive information about the selected
    preset including requirements, recommended settings, and tips.
    """

    bl_idname = "gs_capture.preset_info"
    bl_label = "Preset Information"
    bl_description = "Show detailed information about the selected preset"

    preset_id: EnumProperty(
        name="Preset",
        items=get_preset_enum_items
    )

    def execute(self, context):
        """Show info popup."""
        return context.window_manager.invoke_popup(self, width=400)

    def draw(self, context):
        """Draw preset information."""
        layout = self.layout

        if self.preset_id not in PRESETS:
            layout.label(text="Unknown preset")
            return

        preset = PRESETS[self.preset_id]

        # Header
        row = layout.row()
        row.label(text=preset.name, icon='PRESET')

        # Description
        box = layout.box()
        col = box.column(align=True)
        col.scale_y = 0.8
        for line in preset.description.split('\n'):
            col.label(text=line)

        # Recommended settings
        layout.separator()
        layout.label(text="Recommended Settings:", icon='SETTINGS')

        col = layout.column(align=True)

        min_cam, max_cam = preset.recommended_cameras
        col.label(text=f"Camera Count: {min_cam} - {max_cam}")

        res_w, res_h = preset.recommended_resolution
        col.label(text=f"Resolution: {res_w} x {res_h}")

        col.label(text=f"File Format: {preset.file_format}")
        col.label(text=f"White Background: {'Yes' if preset.white_background else 'No'}")

        # Export formats
        layout.separator()
        layout.label(text="Export Formats:", icon='EXPORT')

        col = layout.column(align=True)
        if preset.export_colmap:
            col.label(text="✓ COLMAP (cameras.txt, images.txt, points3D.txt)")
        if preset.export_transforms_json:
            col.label(text="✓ transforms.json (NeRF/Instant-NGP format)")

        # Notes
        if preset.notes:
            layout.separator()
            layout.label(text="Notes:", icon='INFO')
            box = layout.box()
            col = box.column(align=True)
            col.scale_y = 0.8
            for line in preset.notes.split('\n'):
                col.label(text=line)

        # Website
        if preset.website:
            layout.separator()
            layout.operator("wm.url_open", text="Visit Website", icon='URL').url = preset.website

    def invoke(self, context, event):
        """Invoke the popup."""
        settings = context.scene.gs_capture_settings
        self.preset_id = settings.framework_preset
        return context.window_manager.invoke_popup(self, width=400)


class GSCAPTURE_OT_ComparePresets(Operator):
    """Compare settings across different framework presets.

    Shows a comparison table of recommended settings for all
    available framework presets.
    """

    bl_idname = "gs_capture.compare_presets"
    bl_label = "Compare Presets"
    bl_description = "Compare recommended settings across all presets"

    def execute(self, context):
        """Show comparison popup."""
        return context.window_manager.invoke_popup(self, width=600)

    def draw(self, context):
        """Draw preset comparison table."""
        layout = self.layout

        layout.label(text="Framework Preset Comparison", icon='PRESET')
        layout.separator()

        # Header row
        header = layout.row()
        header.label(text="Framework")
        header.label(text="Cameras")
        header.label(text="Resolution")
        header.label(text="Format")
        header.label(text="COLMAP")
        header.label(text="JSON")

        layout.separator()

        # Preset rows
        for preset_id, preset in PRESETS.items():
            row = layout.row()
            row.label(text=preset.name)

            min_cam, max_cam = preset.recommended_cameras
            row.label(text=f"{min_cam}-{max_cam}")

            res_w, res_h = preset.recommended_resolution
            row.label(text=f"{res_w}x{res_h}")

            row.label(text=preset.file_format)

            row.label(text="✓" if preset.export_colmap else "")
            row.label(text="✓" if preset.export_transforms_json else "")

    def invoke(self, context, event):
        """Invoke the popup."""
        return context.window_manager.invoke_popup(self, width=600)


# Registration
classes = [
    GSCAPTURE_OT_ApplyPreset,
    GSCAPTURE_OT_PresetInfo,
    GSCAPTURE_OT_ComparePresets,
]


def register():
    """Register preset operators."""
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    """Unregister preset operators."""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
