# SPDX-License-Identifier: MIT
# Gaussian Splatting Capture Addon for Blender 4.x/5.0
# Generates training images for 3D Gaussian Splatting from Blender scenes

"""
GS Capture - Gaussian Splatting Training Data Generator

A professional-grade Blender addon for generating training data
for Gaussian Splatting and NeRF models.

Features:
- Multi-distribution camera placement (Fibonacci, hemisphere, ring)
- Framework presets (3DGS, Nerfstudio, Postshot, Polycam, Luma AI)
- Material problem detection and scene analysis
- Depth map, normal map, and object mask export
- Checkpoint/resume system for long captures
- Coverage analysis with viewport visualization
- COLMAP and transforms.json export

"""

bl_info = {
    "name": "GS Capture - Gaussian Splatting Training Data Generator",
    "author": "Alexander Klaus",
    "version": (2, 2, 1),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > GS Capture",
    "description": "Professional training data generator for 3D Gaussian Splatting with scene analysis, presets, and validation",
    "category": "Render",
    "doc_url": "https://github.com/klausi3D/blender-synthetic-data-capture",
    "tracker_url": "https://github.com/klausi3D/blender-synthetic-data-capture/issues",
}

import bpy
from bpy.props import PointerProperty

# Import all modules
from . import properties
from . import preferences

# Operators
from .operators import capture, analysis, batch, preview
from .operators import presets as preset_operators
from .operators import training as training_operators
from .operators import coverage as coverage_operators

# Panels
from .panels import main, output, camera, adaptive, render, lighting
from .panels import batch as batch_panel
from .panels import presets as preset_panels
from .panels import training as training_panels
from .panels import warnings as warnings_panel
from .panels import progress as progress_panel


# All classes to register (order matters for UI)
classes = [
    # Property Groups (must be first)
    properties.GSCaptureObjectItem,
    properties.GSCaptureObjectGroup,
    properties.GSCaptureCheckpoint,
    properties.GSCaptureSettings,

    # Operators - Analysis
    analysis.GSCAPTURE_OT_analyze_selected,
    analysis.GSCAPTURE_OT_analyze_scene,
    analysis.GSCAPTURE_OT_export_analysis_report,
    analysis.GSCAPTURE_OT_apply_recommendations,

    # Operators - Capture
    capture.GSCAPTURE_OT_capture_selected,
    capture.GSCAPTURE_OT_capture_collection,
    capture.GSCAPTURE_OT_cancel_capture,

    # Operators - Batch
    batch.GSCAPTURE_OT_batch_capture,
    batch.GSCAPTURE_OT_add_object_group,
    batch.GSCAPTURE_OT_remove_object_group,
    batch.GSCAPTURE_OT_add_to_group,
    batch.GSCAPTURE_OT_remove_from_group,

    # Operators - Preview
    preview.GSCAPTURE_OT_preview_cameras,
    preview.GSCAPTURE_OT_clear_preview,
    preview.GSCAPTURE_OT_open_output_folder,

    # Operators - Presets
    preset_operators.GSCAPTURE_OT_ApplyPreset,
    preset_operators.GSCAPTURE_OT_PresetInfo,
    preset_operators.GSCAPTURE_OT_ComparePresets,

    # Operators - Training
    training_operators.GSCAPTURE_OT_StartTraining,
    training_operators.GSCAPTURE_OT_StopTraining,
    training_operators.GSCAPTURE_OT_ClearTraining,
    training_operators.GSCAPTURE_OT_BrowseTrainingData,
    training_operators.GSCAPTURE_OT_BrowseTrainingOutput,
    training_operators.GSCAPTURE_OT_OpenTrainingOutput,
    training_operators.GSCAPTURE_OT_ShowInstallInstructions,
    training_operators.GSCAPTURE_OT_UseLastCapture,
    training_operators.GSCAPTURE_OT_ReloadCustomBackends,
    training_operators.GSCAPTURE_OT_OpenCustomBackendsFolder,

    # Operators - Coverage
    coverage_operators.GSCAPTURE_OT_show_coverage_heatmap,
    coverage_operators.GSCAPTURE_OT_clear_coverage_heatmap,
    coverage_operators.GSCAPTURE_OT_analyze_coverage,

    # Operators - Warnings/Analysis Panel
    warnings_panel.GSCAPTURE_OT_analyze_scene_mvp,
    warnings_panel.GSCAPTURE_OT_show_material_problems,
    warnings_panel.GSCAPTURE_OT_fix_material_problems,

    # Panels (order determines UI layout)
    progress_panel.GSCAPTURE_PT_progress_panel,  # Progress panel at very top
    progress_panel.GSCAPTURE_OT_clear_last_capture,  # Clear stats operator
    warnings_panel.GSCAPTURE_PT_warnings,  # Scene Analysis
    preset_panels.GSCAPTURE_PT_PresetsPanel,
    preset_panels.GSCAPTURE_PT_PresetsQuickSettings,
    main.GSCAPTURE_PT_main_panel,
    output.GSCAPTURE_PT_output_panel,
    camera.GSCAPTURE_PT_camera_panel,
    adaptive.GSCAPTURE_PT_adaptive_panel,
    render.GSCAPTURE_PT_render_panel,
    lighting.GSCAPTURE_PT_lighting_panel,
    batch_panel.GSCAPTURE_PT_batch_panel,
    training_panels.GSCAPTURE_OT_ApplyRecommendedExportSettings,
    training_panels.GSCAPTURE_PT_TrainingPanel,
    training_panels.GSCAPTURE_PT_TrainingAdvanced,
    training_panels.GSCAPTURE_PT_TrainingCustomBackends,
    training_panels.GSCAPTURE_PT_TrainingLog,
]


def register():
    """Register all addon classes and properties."""
    # Register preferences first
    preferences.register()

    # Register all classes
    for cls in classes:
        bpy.utils.register_class(cls)

    # Register main settings property
    bpy.types.Scene.gs_capture_settings = PointerProperty(
        type=properties.GSCaptureSettings,
        name="GS Capture Settings",
        description="Settings for Gaussian Splatting training data capture"
    )


def unregister():
    """Unregister all addon classes and properties."""
    # Unregister in reverse order
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    # Remove settings property
    del bpy.types.Scene.gs_capture_settings

    # Unregister preferences last
    preferences.unregister()


if __name__ == "__main__":
    register()
