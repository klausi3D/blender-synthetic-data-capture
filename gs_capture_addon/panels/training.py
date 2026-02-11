"""
Training Panel - UI for integrated Gaussian Splatting training.

This panel provides controls for training Gaussian Splatting models
directly from Blender, with real-time progress monitoring.
"""

import os
import bpy
from bpy.types import Panel, Operator

from ..core.training import get_available_backends, get_all_backends, get_running_process
from ..core.training.base import TrainingStatus
from ..utils.folder_structure import get_export_settings, validate_structure, count_images


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_export_warnings(settings):
    """Get list of export setting warnings for the selected backend.

    Args:
        settings: GSCaptureSettings property group

    Returns:
        list: Warning messages about mismatched export settings
    """
    backend_id = settings.training_backend
    recommended = get_export_settings(backend_id)
    warnings = []

    if recommended.get('export_colmap') and not settings.export_colmap:
        warnings.append("Enable 'Export COLMAP Format' for this backend")

    if recommended.get('export_transforms_json') and not settings.export_transforms_json:
        warnings.append("Enable 'Export transforms.json' for this backend")

    if recommended.get('export_masks') and not settings.export_masks:
        warnings.append("Enable 'Export Object Masks' for GS-Lightning")

    if backend_id == 'gs_lightning' and settings.export_masks and settings.mask_format != 'GSL':
        warnings.append("Set mask format to 'GS-Lightning' for proper naming")

    return warnings


def _get_backend_status(backend_id, data_path):
    """Get backend availability and data compatibility status.

    Args:
        backend_id: Backend identifier string
        data_path: Path to training data directory

    Returns:
        tuple: (status_icon, status_text, status_alert)
            status_icon: Blender icon name
            status_text: Status description
            status_alert: Whether to show alert styling
    """
    backends = get_available_backends()
    all_backends = get_all_backends()

    # Check if backend is installed
    if backend_id not in backends:
        if backend_id in all_backends:
            return 'ERROR', "Backend not installed", True
        return 'ERROR', "Unknown backend", True

    backend = backends[backend_id]

    # Check if data path is valid
    if not data_path:
        return 'CHECKMARK', "Backend ready", False

    data_path_normalized = os.path.normpath(bpy.path.abspath(data_path))

    if not os.path.exists(data_path_normalized):
        return 'QUESTION', "Data path not found", True

    # Validate data structure
    is_valid, message = backend.validate_data(data_path_normalized)

    if is_valid:
        return 'CHECKMARK', "Ready to train", False
    else:
        return 'ERROR', "Data format mismatch", True


def _get_data_path_status(settings, backend_id):
    """Get status indicator for the training data path.

    Args:
        settings: GSCaptureSettings property group
        backend_id: Backend identifier string

    Returns:
        tuple: (icon, tooltip, is_valid)
    """
    data_path = settings.training_data_path

    if not data_path:
        return 'BLANK1', "No path specified", False

    data_path_normalized = os.path.normpath(bpy.path.abspath(data_path))

    if not os.path.exists(data_path_normalized):
        return 'ERROR', "Path does not exist", False

    # Check structure
    is_valid, missing_required, missing_optional = validate_structure(data_path_normalized, backend_id)

    if not is_valid:
        return 'ERROR', f"Missing: {', '.join(missing_required[:2])}", False

    # Count images
    image_count = count_images(data_path_normalized)

    if image_count == 0:
        return 'ERROR', "No images found", False

    if image_count < 10:
        return 'SORTTIME', f"{image_count} images (low count)", True

    if missing_optional:
        return 'SORTTIME', f"{image_count} images (missing optional data)", True

    return 'CHECKMARK', f"{image_count} images ready", True


# =============================================================================
# OPERATORS
# =============================================================================

class GSCAPTURE_OT_ApplyRecommendedExportSettings(Operator):
    """Apply recommended export settings for the selected training backend"""
    bl_idname = "gs_capture.apply_recommended_export"
    bl_label = "Apply Recommended Settings"
    bl_description = "Automatically configure export settings to match the selected training backend requirements"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.gs_capture_settings
        backend_id = settings.training_backend

        recommended = get_export_settings(backend_id)

        settings.export_colmap = recommended.get('export_colmap', True)
        settings.export_transforms_json = recommended.get('export_transforms_json', False)
        settings.export_masks = recommended.get('export_masks', False)

        if 'mask_format' in recommended:
            settings.mask_format = recommended['mask_format']

        self.report({'INFO'}, f"Applied recommended settings for {backend_id}")
        return {'FINISHED'}


# =============================================================================
# PANELS
# =============================================================================

class GSCAPTURE_PT_TrainingPanel(Panel):
    """Main training control panel.

    Provides interface for:
    - Backend selection (3DGS, Nerfstudio, gsplat)
    - Training configuration
    - Start/stop controls
    - Progress monitoring
    """

    bl_label = "Training"
    bl_idname = "GSCAPTURE_PT_training"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_order = 50  # Show after capture panels

    def draw_header(self, context):
        """Draw panel header with icon."""
        self.layout.label(text="", icon='PLAY')

    def draw(self, context):
        """Draw the training control UI."""
        layout = self.layout
        settings = context.scene.gs_capture_settings

        # Check for running process
        process = get_running_process()

        if process and process.is_running:
            self._draw_running(layout, process)
        else:
            self._draw_setup(layout, context, settings)

    def _draw_setup(self, layout, context, settings):
        """Draw training setup UI when not running."""
        backends = get_available_backends()
        current_backend = settings.training_backend

        # Backend selection with status indicator
        box = layout.box()
        header_row = box.row()
        header_row.label(text="Training Backend:", icon='TOOL_SETTINGS')

        row = box.row(align=True)
        row.prop(settings, "training_backend", text="")

        # Backend status indicator
        status_icon, status_text, is_alert = _get_backend_status(
            current_backend, settings.training_data_path
        )

        status_row = box.row()
        status_row.alert = is_alert
        status_row.label(text=status_text, icon=status_icon)

        if current_backend not in backends:
            box.operator(
                "gs_capture.show_install_instructions",
                text="Installation Guide",
                icon='HELP'
            )

        # Export Settings Warnings
        self._draw_export_warnings(context, layout, settings)

        # Data path with validation indicator
        layout.separator()
        box = layout.box()
        box.label(text="Training Data:", icon='FILE_FOLDER')

        # Path input with status icon
        row = box.row(align=True)
        row.prop(settings, "training_data_path", text="")
        row.operator("gs_capture.browse_training_data", text="", icon='FILEBROWSER')

        # Data path validation indicator
        path_icon, path_tooltip, path_valid = _get_data_path_status(settings, current_backend)
        if settings.training_data_path:
            status_row = box.row()
            status_row.alert = not path_valid
            status_row.label(text=path_tooltip, icon=path_icon)

        # Use Last Capture button - show if last capture path exists
        if settings.last_capture_path:
            if os.path.exists(settings.last_capture_path):
                row = box.row(align=True)
                row.operator("gs_capture.use_last_capture", text="Use Last Capture", icon='FILE_REFRESH')

        # Output path
        box = layout.box()
        box.label(text="Output:", icon='EXPORT')

        row = box.row(align=True)
        row.prop(settings, "training_output_path", text="")
        row.operator("gs_capture.browse_training_output", text="", icon='FILEBROWSER')

        # Import options
        import_box = box.box()
        import_box.label(text="Import Trained Splat:", icon='IMPORT')

        col = import_box.column(align=True)
        col.prop(settings, "training_import_location", text="Location")
        col.prop(settings, "training_import_uniform_scale", text="Scale")
        col.prop(settings, "training_import_replace_selection")

        row = import_box.row(align=True)
        row.enabled = bool(settings.training_output_path)
        op = row.operator("gs_capture.open_training_output", text="Import Trained Splat", icon='IMPORT')
        op.action = 'IMPORT_SPLAT'

        # Training parameters
        layout.separator()
        box = layout.box()
        box.label(text="Parameters:", icon='PREFERENCES')

        col = box.column(align=True)
        col.prop(settings, "training_iterations", text="Iterations")
        col.prop(settings, "training_save_every", text="Save Every")

        row = box.row()
        row.prop(settings, "training_white_background", text="White Background")

        # Start Training button - always visible
        layout.separator()

        # Big start button in highlighted box
        start_box = layout.box()
        start_box.alert = False  # Make it stand out

        can_start = bool(
            current_backend in backends and
            settings.training_data_path and
            settings.training_output_path
        )

        if not can_start:
            start_box.label(text="Set paths above to enable training", icon='INFO')

        row = start_box.row(align=True)
        row.scale_y = 2.0
        row.enabled = can_start
        row.operator("gs_capture.start_training", text="START TRAINING", icon='PLAY')

    def _draw_export_warnings(self, context, layout, settings):
        """Draw warnings if export settings don't match backend requirements."""
        warnings = _get_export_warnings(settings)

        if warnings:
            box = layout.box()
            header_row = box.row()
            header_row.label(text="Export Settings Warning:", icon='ERROR')

            col = box.column(align=True)
            col.scale_y = 0.8
            for warning in warnings:
                row = col.row()
                row.label(text=f"  {warning}")

            # Apply Recommended Settings button
            row = box.row()
            row.operator(
                "gs_capture.apply_recommended_export",
                text="Apply Recommended Settings",
                icon='SETTINGS'
            )

    def _draw_error_display(self, layout, error_message):
        """Draw an improved error message display with suggestions.

        Args:
            layout: Blender UI layout
            error_message: The error message string
        """
        from ..utils.errors import get_error_suggestion

        box = layout.box()
        box.alert = True

        # Header
        header_row = box.row()
        header_row.label(text="Error Details:", icon='ERROR')

        # Error message (truncated for display)
        col = box.column(align=True)
        col.scale_y = 0.75

        error_lines = error_message.strip().split('\n')[:8]
        for line in error_lines:
            # Truncate long lines
            display_line = line.strip()
            if len(display_line) > 55:
                display_line = display_line[:52] + "..."
            if display_line:
                col.label(text=display_line)

        # Try to get suggestion based on error
        suggestion = get_error_suggestion(error_message)
        if suggestion:
            layout.separator()
            suggestion_box = layout.box()
            row = suggestion_box.row()
            row.label(text="Suggestion:", icon='LIGHT')
            col = suggestion_box.column(align=True)
            col.scale_y = 0.75
            for line in suggestion.split('\n')[:3]:
                col.label(text=line)

    def _draw_running(self, layout, process):
        """Draw training progress UI when running."""
        progress = process.progress

        # Status header
        box = layout.box()

        if progress.status == TrainingStatus.RUNNING:
            row = box.row()
            row.label(text="Training in Progress", icon='TIME')
        elif progress.status == TrainingStatus.COMPLETED:
            row = box.row()
            row.label(text="Training Complete!", icon='CHECKMARK')
        elif progress.status == TrainingStatus.FAILED:
            row = box.row()
            row.alert = True
            row.label(text="Training Failed", icon='ERROR')

        # Progress bar
        layout.separator()

        col = layout.column(align=True)
        col.label(text=f"Iteration: {progress.iteration} / {progress.total_iterations}")

        # Progress percentage
        pct = progress.progress_percent
        col.progress(
            factor=pct / 100.0,
            type='BAR',
            text=f"{pct:.1f}%"
        )

        # Metrics
        if progress.loss > 0 or progress.psnr > 0:
            row = layout.row(align=True)
            if progress.loss > 0:
                row.label(text=f"Loss: {progress.loss:.4f}")
            if progress.psnr > 0:
                row.label(text=f"PSNR: {progress.psnr:.2f}")

        # Time info
        if progress.elapsed_seconds > 0:
            layout.separator()
            col = layout.column(align=True)

            # Elapsed
            elapsed_min = int(progress.elapsed_seconds // 60)
            elapsed_sec = int(progress.elapsed_seconds % 60)
            col.label(text=f"Elapsed: {elapsed_min}m {elapsed_sec}s")

            # ETA
            if progress.eta_seconds > 0:
                col.label(text=f"ETA: {progress.eta_formatted}")

        # Error message - improved display
        if progress.error:
            layout.separator()
            self._draw_error_display(layout, progress.error)

        # Stop button
        layout.separator()

        row = layout.row(align=True)
        row.scale_y = 1.5

        if progress.status == TrainingStatus.RUNNING:
            row.operator("gs_capture.stop_training", text="Stop Training", icon='PAUSE')
        else:
            row.operator("gs_capture.clear_training", text="Close", icon='X')

            # Show result path
            if progress.status == TrainingStatus.COMPLETED:
                result_path = process.get_result_path()
                if result_path:
                    layout.separator()
                    box = layout.box()
                    box.label(text="Output:", icon='CHECKMARK')
                    col = box.column(align=True)
                    col.scale_y = 0.8
                    col.label(text=result_path)

                    row = box.row(align=True)
                    import_op = row.operator("gs_capture.open_training_output", text="Import Trained Splat", icon='IMPORT')
                    import_op.action = 'IMPORT_SPLAT'

                    folder_op = row.operator("gs_capture.open_training_output", text="Open Folder", icon='FILE_FOLDER')
                    folder_op.action = 'OPEN_FOLDER'


class GSCAPTURE_PT_TrainingAdvanced(Panel):
    """Advanced training settings panel."""

    bl_label = "Advanced"
    bl_idname = "GSCAPTURE_PT_training_advanced"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_parent_id = "GSCAPTURE_PT_training"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        """Only show when not training."""
        process = get_running_process()
        return not (process and process.is_running)

    def draw(self, context):
        """Draw advanced settings."""
        layout = self.layout
        settings = context.scene.gs_capture_settings

        # GPU selection
        col = layout.column(align=True)
        col.prop(settings, "training_gpu_id", text="GPU ID")

        # Densification (3DGS specific)
        layout.separator()
        layout.label(text="Densification (3DGS):", icon='PARTICLE_DATA')

        col = layout.column(align=True)
        col.prop(settings, "densify_from_iter", text="Start")
        col.prop(settings, "densify_until_iter", text="End")
        col.prop(settings, "densification_interval", text="Interval")

        # Extra arguments
        layout.separator()
        layout.label(text="Extra Arguments:", icon='CONSOLE')
        layout.prop(settings, "training_extra_args", text="")


class GSCAPTURE_PT_TrainingLog(Panel):
    """Training output log panel."""

    bl_label = "Log"
    bl_idname = "GSCAPTURE_PT_training_log"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_parent_id = "GSCAPTURE_PT_training"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        """Only show when training is running or completed."""
        process = get_running_process()
        return process is not None

    def draw(self, context):
        """Draw training log output."""
        layout = self.layout

        process = get_running_process()
        if not process:
            layout.label(text="No training process")
            return

        # Get recent output lines
        lines = process.get_output_lines(max_lines=20)

        if not lines:
            layout.label(text="No output yet...")
            return

        box = layout.box()
        col = box.column(align=True)
        col.scale_y = 0.6

        for line in lines[-15:]:  # Show last 15 lines
            # Truncate long lines
            text = line.strip()
            if len(text) > 60:
                text = text[:57] + "..."
            col.label(text=text)


class GSCAPTURE_PT_TrainingCustomBackends(Panel):
    """Custom backends management panel."""

    bl_label = "Custom Backends"
    bl_idname = "GSCAPTURE_PT_training_custom_backends"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_parent_id = "GSCAPTURE_PT_training"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        """Only show when not training."""
        process = get_running_process()
        return not (process and process.is_running)

    def draw(self, context):
        """Draw custom backends management UI."""
        layout = self.layout

        from ..core.training import load_custom_backends, get_custom_backends_dir

        # Show custom backends directory
        custom_dir = get_custom_backends_dir()

        col = layout.column(align=True)
        col.label(text="Custom backends location:", icon='FILE_FOLDER')
        col.label(text=os.path.basename(custom_dir))

        # Management buttons
        layout.separator()
        col = layout.column(align=True)
        col.operator(
            "gs_capture.reload_custom_backends",
            text="Reload Custom Backends",
            icon='FILE_REFRESH'
        )
        col.operator(
            "gs_capture.open_custom_backends_folder",
            text="Open Backends Folder",
            icon='FILEBROWSER'
        )

        # List loaded custom backends
        custom_backends = load_custom_backends()
        if custom_backends:
            layout.separator()
            box = layout.box()
            box.label(text=f"Loaded: {len(custom_backends)}", icon='CHECKMARK')

            col = box.column(align=True)
            col.scale_y = 0.8
            for backend_id, backend in custom_backends.items():
                status = "Ready" if backend.is_available() else "Not Available"
                icon = 'CHECKMARK' if backend.is_available() else 'ERROR'
                col.label(text=f"  {backend.name}: {status}", icon=icon)
        else:
            layout.separator()
            box = layout.box()
            box.label(text="No custom backends loaded", icon='INFO')
            col = box.column(align=True)
            col.scale_y = 0.75
            col.label(text="Add YAML/JSON config files to")
            col.label(text="the custom_backends folder.")


# Registration
classes = [
    GSCAPTURE_OT_ApplyRecommendedExportSettings,
    GSCAPTURE_PT_TrainingPanel,
    GSCAPTURE_PT_TrainingAdvanced,
    GSCAPTURE_PT_TrainingCustomBackends,
    GSCAPTURE_PT_TrainingLog,
]


def register():
    """Register training panels."""
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    """Unregister training panels."""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
