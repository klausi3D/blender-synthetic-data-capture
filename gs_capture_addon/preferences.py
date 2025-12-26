"""
Addon Preferences for GS Capture.

Provides user preferences for configuring training backend paths,
default settings, and addon behavior.
"""

import bpy
from bpy.types import AddonPreferences
from bpy.props import (
    StringProperty,
    BoolProperty,
    EnumProperty,
    IntProperty,
)


class GSCapturePreferences(AddonPreferences):
    """Addon preferences for GS Capture.

    Configure paths to training backends, default settings,
    and addon behavior.
    """

    bl_idname = __package__

    # ==========================================================================
    # BACKEND PATHS
    # ==========================================================================

    gaussian_splatting_path: StringProperty(
        name="3DGS Path",
        description="Path to 3D Gaussian Splatting repository (contains train.py)",
        default="",
        subtype='DIR_PATH'
    )

    nerfstudio_env: StringProperty(
        name="Nerfstudio Environment",
        description="Name of conda environment with Nerfstudio installed",
        default="nerfstudio"
    )

    gsplat_examples_path: StringProperty(
        name="gsplat Examples Path",
        description="Path to gsplat examples directory",
        default="",
        subtype='DIR_PATH'
    )

    # ==========================================================================
    # DEFAULT SETTINGS
    # ==========================================================================

    default_framework: EnumProperty(
        name="Default Framework",
        description="Default framework preset for new captures",
        items=[
            ('GAUSSIAN_SPLATTING', "3D Gaussian Splatting", ""),
            ('INSTANT_NGP', "Instant-NGP", ""),
            ('NERFSTUDIO', "Nerfstudio", ""),
            ('POSTSHOT', "Postshot", ""),
            ('POLYCAM', "Polycam", ""),
            ('LUMA_AI', "Luma AI", ""),
            ('GSPLAT', "gsplat", ""),
        ],
        default='GAUSSIAN_SPLATTING'
    )

    default_camera_count: IntProperty(
        name="Default Camera Count",
        description="Default number of cameras to generate",
        default=100,
        min=8,
        max=500
    )

    default_iterations: IntProperty(
        name="Default Iterations",
        description="Default training iterations",
        default=30000,
        min=1000,
        max=100000
    )

    # ==========================================================================
    # BEHAVIOR SETTINGS
    # ==========================================================================

    auto_validate: BoolProperty(
        name="Auto Validate",
        description="Automatically validate settings before capture",
        default=True
    )

    show_advanced_options: BoolProperty(
        name="Show Advanced Options",
        description="Show advanced options in panels by default",
        default=False
    )

    enable_viewport_preview: BoolProperty(
        name="Enable Viewport Preview",
        description="Enable GPU-based camera preview in viewport",
        default=True
    )

    preview_camera_color: EnumProperty(
        name="Preview Camera Color",
        description="Color scheme for camera preview",
        items=[
            ('BLUE', "Blue", "Blue camera indicators"),
            ('GREEN', "Green", "Green camera indicators"),
            ('ORANGE', "Orange", "Orange camera indicators"),
            ('WHITE', "White", "White camera indicators"),
        ],
        default='BLUE'
    )

    # ==========================================================================
    # PERFORMANCE SETTINGS
    # ==========================================================================

    max_concurrent_renders: IntProperty(
        name="Max Concurrent Renders",
        description="Maximum concurrent render operations (batch mode)",
        default=1,
        min=1,
        max=4
    )

    checkpoint_interval: IntProperty(
        name="Checkpoint Interval",
        description="Save checkpoint every N images",
        default=10,
        min=1,
        max=100
    )

    def draw(self, context):
        """Draw preferences UI."""
        layout = self.layout

        # Training Backend Paths
        box = layout.box()
        box.label(text="Training Backend Paths", icon='TOOL_SETTINGS')

        col = box.column(align=True)

        # 3DGS Path
        row = col.row()
        row.prop(self, "gaussian_splatting_path")
        self._draw_path_status(row, self.gaussian_splatting_path, "train.py")

        # Nerfstudio env
        row = col.row()
        row.prop(self, "nerfstudio_env")

        # gsplat path
        row = col.row()
        row.prop(self, "gsplat_examples_path")
        self._draw_path_status(row, self.gsplat_examples_path, "simple_trainer.py")

        # Default Settings
        layout.separator()
        box = layout.box()
        box.label(text="Default Settings", icon='PREFERENCES')

        col = box.column()
        col.prop(self, "default_framework")
        col.prop(self, "default_camera_count")
        col.prop(self, "default_iterations")

        # Behavior
        layout.separator()
        box = layout.box()
        box.label(text="Behavior", icon='MODIFIER')

        col = box.column()
        col.prop(self, "auto_validate")
        col.prop(self, "show_advanced_options")
        col.prop(self, "enable_viewport_preview")

        if self.enable_viewport_preview:
            col.prop(self, "preview_camera_color")

        # Performance
        layout.separator()
        box = layout.box()
        box.label(text="Performance", icon='SORTTIME')

        col = box.column()
        col.prop(self, "max_concurrent_renders")
        col.prop(self, "checkpoint_interval")

        # Documentation links
        layout.separator()
        box = layout.box()
        box.label(text="Documentation & Support", icon='HELP')

        row = box.row(align=True)
        row.operator(
            "wm.url_open",
            text="Documentation",
            icon='URL'
        ).url = "https://github.com/your-repo/gs-capture-blender"

        row.operator(
            "wm.url_open",
            text="Report Issue",
            icon='ERROR'
        ).url = "https://github.com/your-repo/gs-capture-blender/issues"

    def _draw_path_status(self, row, path, check_file):
        """Draw path validation status."""
        import os

        if not path:
            row.label(text="", icon='BLANK1')
            return

        full_path = bpy.path.abspath(path)
        file_path = os.path.join(full_path, check_file)

        if os.path.exists(file_path):
            row.label(text="", icon='CHECKMARK')
        else:
            row.label(text="", icon='ERROR')


def get_preferences():
    """Get addon preferences.

    Returns:
        GSCapturePreferences or None
    """
    addon = bpy.context.preferences.addons.get(__package__)
    if addon:
        return addon.preferences
    return None


# Registration
def register():
    """Register preferences."""
    bpy.utils.register_class(GSCapturePreferences)


def unregister():
    """Unregister preferences."""
    bpy.utils.unregister_class(GSCapturePreferences)
