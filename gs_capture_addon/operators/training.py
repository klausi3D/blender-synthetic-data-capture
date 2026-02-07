"""
Training Operators - Control Gaussian Splatting training from Blender.

Provides operators for starting, stopping, and managing training
processes with real-time progress updates.
"""

import os
import bpy
from bpy.types import Operator
from bpy.props import StringProperty

from ..core.training import (
    get_available_backends,
    get_all_backends,
    get_running_process,
    start_training,
    stop_training
)
from ..core.training.base import TrainingConfig, TrainingStatus
from ..utils.paths import normalize_path, validate_directory, check_disk_space
from ..utils.folder_structure import (
    validate_structure,
    check_mask_naming,
    count_images,
    get_export_settings,
    EXPORT_SETTINGS
)
from ..utils.errors import get_error_message


class GSCAPTURE_OT_StartTraining(Operator):
    """Start Gaussian Splatting training.

    Launches the training subprocess using the selected backend
    and monitors progress through a modal timer.
    """

    bl_idname = "gs_capture.start_training"
    bl_label = "Start Training"
    bl_description = "Start training Gaussian Splatting model"

    _timer = None
    _process = None

    @classmethod
    def poll(cls, context):
        """Check if training can start."""
        process = get_running_process()
        if process and process.is_running:
            return False

        settings = context.scene.gs_capture_settings
        return (
            settings.training_data_path and
            settings.training_output_path
        )

    def execute(self, context):
        """Start the training process."""
        settings = context.scene.gs_capture_settings

        # Run pre-flight check
        preflight_ok, errors, warnings = self._preflight_check(context)

        # Report warnings (but continue)
        for warning in warnings:
            # Truncate long warnings for Blender's report
            short_warning = warning.split('\n')[0][:200]
            self.report({'WARNING'}, short_warning)

        # If errors, report and cancel
        if not preflight_ok:
            for error in errors:
                # Report first line of error (Blender has limited report space)
                short_error = error.split('\n')[0][:200]
                self.report({'ERROR'}, short_error)
            return {'CANCELLED'}

        # Get backend
        backends = get_available_backends()
        backend_id = settings.training_backend

        if backend_id not in backends:
            self.report({'ERROR'}, f"Backend '{backend_id}' not available")
            return {'CANCELLED'}

        backend = backends[backend_id]

        # Validate data with backend-specific validation
        is_valid, message = backend.validate_data(settings.training_data_path)
        if not is_valid:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

        # Build config
        config = TrainingConfig(
            data_path=settings.training_data_path,
            output_path=settings.training_output_path,
            iterations=settings.training_iterations,
            save_iterations=self._get_save_iterations(settings),
            white_background=settings.training_white_background,
            gpu_id=settings.training_gpu_id,
            densify_from_iter=settings.densify_from_iter,
            densify_until_iter=settings.densify_until_iter,
            densification_interval=settings.densification_interval,
            extra_args=settings.training_extra_args.split() if settings.training_extra_args else [],
        )

        # Start training
        self._redraw_pending = False
        self._process = start_training(
            backend=backend,
            config=config,
            progress_callback=self._on_progress
        )

        if not self._process.is_running:
            error = self._process.progress.error or "Failed to start"
            self.report({'ERROR'}, error)
            return {'CANCELLED'}

        # Start timer for UI updates
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)

        self.report({'INFO'}, f"Training started with {backend.name}")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        """Handle modal events for progress updates."""
        if event.type == 'TIMER':
            # Force UI redraw
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

            # Check if process finished
            if self._process and not self._process.is_running:
                self._finish(context)
                return {'FINISHED'}

        return {'PASS_THROUGH'}

    def cancel(self, context):
        """Cancel the modal operation."""
        self._finish(context)

    def _finish(self, context):
        """Clean up timer."""
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def _get_save_iterations(self, settings):
        """Generate save iteration list."""
        save_every = settings.training_save_every
        total = settings.training_iterations

        if save_every <= 0:
            return [total]

        saves = list(range(save_every, total, save_every))
        if total not in saves:
            saves.append(total)

        return saves

    def _on_progress(self, progress):
        """Handle progress updates from training."""
        # This is called from the training thread; do not call Blender API here.
        # We only mark that a redraw is desired and let the modal timer handle UI updates.
        self._latest_progress = progress
        self._redraw_pending = True

    def _preflight_check(self, context):
        """Perform pre-flight validation before starting training.

        Returns:
            tuple: (success, errors_list, warnings_list)
        """
        settings = context.scene.gs_capture_settings
        errors = []
        warnings = []

        backend_id = settings.training_backend

        # Normalize paths
        data_path = normalize_path(settings.training_data_path)
        output_path = normalize_path(settings.training_output_path)

        # Validate data path exists
        is_valid, normalized_data, error_msg = validate_directory(data_path, must_exist=True)
        if not is_valid:
            errors.append(get_error_message('invalid_data_path',
                                            data_path=data_path,
                                            error=error_msg))
            return False, errors, warnings

        # Validate output path can be created
        is_valid, normalized_output, error_msg = validate_directory(output_path, create=True)
        if not is_valid:
            errors.append(get_error_message('invalid_output_path',
                                            output_path=output_path,
                                            error=error_msg))
            return False, errors, warnings

        # Validate folder structure matches backend requirements
        structure_valid, missing_required, missing_optional = validate_structure(
            normalized_data, backend_id
        )

        if not structure_valid:
            # Determine specific error type
            if any('sparse/0' in m for m in missing_required):
                errors.append(get_error_message('missing_colmap', data_path=normalized_data))
            elif any('transforms.json' in m for m in missing_required):
                errors.append(get_error_message('missing_transforms', data_path=normalized_data))
            else:
                errors.append(f"Missing required items:\n" + "\n".join(f"  - {m}" for m in missing_required))

        if missing_optional:
            warnings.append(f"Missing optional items:\n" + "\n".join(f"  - {m}" for m in missing_optional))

        # Check image count
        image_count = count_images(normalized_data)
        if image_count == 0:
            errors.append(get_error_message('missing_images'))
        elif image_count < 10:
            warnings.append(f"Low image count ({image_count}). Training may produce poor results with few images.")

        # Check mask naming for GS-Lightning
        if backend_id == 'gs_lightning':
            mask_valid, mask_msg = check_mask_naming(normalized_data, backend_id)
            if not mask_valid:
                errors.append(get_error_message('mask_naming_mismatch'))
            elif "Partial" in mask_msg:
                warnings.append(mask_msg)

        # Check disk space (estimate 10GB needed)
        has_space, free_gb, space_error = check_disk_space(normalized_output, required_gb=10.0)
        if not has_space:
            errors.append(get_error_message('insufficient_disk_space',
                                            free_gb=free_gb,
                                            required_gb=10.0))

        # Check export settings suggestions
        settings_warnings = self._get_settings_suggestions(context)
        warnings.extend(settings_warnings)

        return len(errors) == 0, errors, warnings

    def _get_settings_suggestions(self, context):
        """Compare current export settings with recommended settings for the backend.

        Returns:
            list: List of warning messages for mismatched settings
        """
        settings = context.scene.gs_capture_settings
        backend_id = settings.training_backend
        warnings = []

        recommended = get_export_settings(backend_id)

        # Check COLMAP export setting
        if recommended.get('export_colmap', False) and not settings.export_colmap:
            warnings.append(
                f"Recommended: Enable 'Export COLMAP Format' for {backend_id}. "
                "Future captures should have this enabled."
            )

        # Check transforms.json setting
        if recommended.get('export_transforms_json', False) and not settings.export_transforms_json:
            warnings.append(
                f"Recommended: Enable 'Export transforms.json' for {backend_id}. "
                "Future captures should have this enabled."
            )

        # Check mask settings for GS-Lightning
        if backend_id == 'gs_lightning':
            if recommended.get('export_masks', False) and not settings.export_masks:
                warnings.append(
                    "Recommended: Enable 'Export Object Masks' for GS-Lightning. "
                    "Masks improve background removal quality."
                )
            if settings.export_masks and settings.mask_format != 'GSL':
                warnings.append(
                    "Recommended: Set mask format to 'GS-Lightning' for proper mask naming."
                )

        return warnings


class GSCAPTURE_OT_StopTraining(Operator):
    """Stop the running training process.

    Terminates the training subprocess and cleans up resources.
    """

    bl_idname = "gs_capture.stop_training"
    bl_label = "Stop Training"
    bl_description = "Stop the current training process"

    @classmethod
    def poll(cls, context):
        """Check if training is running."""
        process = get_running_process()
        return process and process.is_running

    def execute(self, context):
        """Stop the training."""
        stop_training()
        self.report({'INFO'}, "Training stopped")
        return {'FINISHED'}


class GSCAPTURE_OT_ClearTraining(Operator):
    """Clear the finished training state.

    Clears the training process reference to allow starting new training.
    """

    bl_idname = "gs_capture.clear_training"
    bl_label = "Clear Training"
    bl_description = "Clear training state"

    def execute(self, context):
        """Clear training state."""
        from ..core.training.process import TrainingProcess
        TrainingProcess.clear_active_process()
        return {'FINISHED'}


class GSCAPTURE_OT_BrowseTrainingData(Operator):
    """Browse for training data directory.

    Opens file browser to select the captured data directory.
    """

    bl_idname = "gs_capture.browse_training_data"
    bl_label = "Browse Training Data"
    bl_description = "Select training data directory"

    directory: StringProperty(
        subtype='DIR_PATH'
    )

    def execute(self, context):
        """Set the training data path."""
        settings = context.scene.gs_capture_settings
        settings.training_data_path = self.directory
        return {'FINISHED'}

    def invoke(self, context, event):
        """Open file browser."""
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class GSCAPTURE_OT_BrowseTrainingOutput(Operator):
    """Browse for training output directory.

    Opens file browser to select the output directory.
    """

    bl_idname = "gs_capture.browse_training_output"
    bl_label = "Browse Training Output"
    bl_description = "Select training output directory"

    directory: StringProperty(
        subtype='DIR_PATH'
    )

    def execute(self, context):
        """Set the training output path."""
        settings = context.scene.gs_capture_settings
        settings.training_output_path = self.directory
        return {'FINISHED'}

    def invoke(self, context, event):
        """Open file browser."""
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class GSCAPTURE_OT_OpenTrainingOutput(Operator):
    """Open the training output folder.

    Opens the file explorer at the training output directory.
    """

    bl_idname = "gs_capture.open_training_output"
    bl_label = "Open Output Folder"
    bl_description = "Open training output directory in file explorer"

    def execute(self, context):
        """Open the folder."""
        process = get_running_process()
        if process:
            output_path = process.config.output_path
        else:
            settings = context.scene.gs_capture_settings
            output_path = settings.training_output_path

        if not output_path or not os.path.exists(output_path):
            self.report({'ERROR'}, "Output directory not found")
            return {'CANCELLED'}

        import subprocess
        import platform

        if platform.system() == "Windows":
            subprocess.Popen(['explorer', output_path])
        elif platform.system() == "Darwin":
            subprocess.Popen(['open', output_path])
        else:
            subprocess.Popen(['xdg-open', output_path])

        return {'FINISHED'}


class GSCAPTURE_OT_ShowInstallInstructions(Operator):
    """Show installation instructions for the selected backend.

    Opens a popup with step-by-step installation instructions.
    """

    bl_idname = "gs_capture.show_install_instructions"
    bl_label = "Installation Guide"
    bl_description = "Show installation instructions for the selected training backend"

    def execute(self, context):
        """Show install popup."""
        return context.window_manager.invoke_popup(self, width=500)

    def draw(self, context):
        """Draw installation instructions."""
        layout = self.layout
        settings = context.scene.gs_capture_settings

        backend_id = settings.training_backend

        # Get backend (handles both built-in and custom)
        from ..core.training import get_all_backends

        all_backends = get_all_backends()

        if backend_id not in all_backends:
            layout.label(text="Unknown backend")
            return

        backend = all_backends[backend_id]

        # Header
        layout.label(text=f"Installing {backend.name}", icon='IMPORT')
        layout.separator()

        # Instructions
        box = layout.box()
        col = box.column(align=True)
        col.scale_y = 0.8

        for line in backend.install_instructions.strip().split('\n'):
            col.label(text=line)

        # Website link
        if backend.website:
            layout.separator()
            layout.operator("wm.url_open", text="Visit Website", icon='URL').url = backend.website

    def invoke(self, context, event):
        """Invoke the popup."""
        return context.window_manager.invoke_popup(self, width=500)


class GSCAPTURE_OT_UseLastCapture(Operator):
    """Set training data path to last capture output.

    Automatically sets the training data path to the output
    directory from the most recent capture session.
    """

    bl_idname = "gs_capture.use_last_capture"
    bl_label = "Use Last Capture"
    bl_description = "Set training data path to the output from the last capture"

    @classmethod
    def poll(cls, context):
        """Check if last capture path is available."""
        settings = context.scene.gs_capture_settings
        return (
            settings.last_capture_path and
            os.path.exists(settings.last_capture_path)
        )

    def execute(self, context):
        """Set the training data path to last capture."""
        settings = context.scene.gs_capture_settings
        settings.training_data_path = settings.last_capture_path
        self.report({'INFO'}, f"Set training data to: {settings.last_capture_path}")
        return {'FINISHED'}


class GSCAPTURE_OT_ReloadCustomBackends(Operator):
    """Reload custom backend configurations.

    Rescans the custom_backends directory for YAML/JSON configuration files.
    """

    bl_idname = "gs_capture.reload_custom_backends"
    bl_label = "Reload Custom Backends"
    bl_description = "Reload custom backend configurations from disk"

    def execute(self, context):
        """Reload custom backends."""
        from ..core.training import reload_custom_backends

        backends = reload_custom_backends()

        if backends:
            self.report({'INFO'}, f"Reloaded {len(backends)} custom backend(s)")
        else:
            self.report({'INFO'}, "No custom backends found")

        # Force UI update
        for area in context.screen.areas:
            area.tag_redraw()

        return {'FINISHED'}


class GSCAPTURE_OT_OpenCustomBackendsFolder(Operator):
    """Open the custom backends folder.

    Opens the file explorer at the custom_backends directory.
    """

    bl_idname = "gs_capture.open_custom_backends_folder"
    bl_label = "Open Custom Backends Folder"
    bl_description = "Open the custom backends configuration folder"

    def execute(self, context):
        """Open the folder."""
        from ..core.training import get_custom_backends_dir

        custom_dir = get_custom_backends_dir()

        # Create if it doesn't exist
        if not os.path.exists(custom_dir):
            try:
                os.makedirs(custom_dir, exist_ok=True)
                self.report({'INFO'}, f"Created custom backends folder: {custom_dir}")
            except OSError as e:
                self.report({'ERROR'}, f"Failed to create folder: {e}")
                return {'CANCELLED'}

        import subprocess
        import platform

        if platform.system() == "Windows":
            subprocess.Popen(['explorer', custom_dir])
        elif platform.system() == "Darwin":
            subprocess.Popen(['open', custom_dir])
        else:
            subprocess.Popen(['xdg-open', custom_dir])

        return {'FINISHED'}


# Registration
classes = [
    GSCAPTURE_OT_StartTraining,
    GSCAPTURE_OT_StopTraining,
    GSCAPTURE_OT_ClearTraining,
    GSCAPTURE_OT_BrowseTrainingData,
    GSCAPTURE_OT_BrowseTrainingOutput,
    GSCAPTURE_OT_OpenTrainingOutput,
    GSCAPTURE_OT_ShowInstallInstructions,
    GSCAPTURE_OT_UseLastCapture,
    GSCAPTURE_OT_ReloadCustomBackends,
    GSCAPTURE_OT_OpenCustomBackendsFolder,
]


def register():
    """Register training operators."""
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    """Unregister training operators."""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
