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
    get_running_process,
    start_training,
    stop_training
)
from ..core.training.base import TrainingConfig, TrainingStatus


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

        # Get backend
        backends = get_available_backends()
        backend_id = settings.training_backend

        if backend_id not in backends:
            self.report({'ERROR'}, f"Backend '{backend_id}' not available")
            return {'CANCELLED'}

        backend = backends[backend_id]

        # Validate data
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
        # This is called from the training thread
        # We just need to trigger a redraw
        pass


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
        TrainingProcess._active_process = None
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

        # Get backend class
        from ..core.training.gaussian_splatting import GaussianSplattingBackend
        from ..core.training.nerfstudio import NerfstudioBackend
        from ..core.training.gsplat import GsplatBackend

        backends = {
            'gaussian_splatting': GaussianSplattingBackend,
            'nerfstudio': NerfstudioBackend,
            'gsplat': GsplatBackend,
        }

        if backend_id not in backends:
            layout.label(text="Unknown backend")
            return

        backend_cls = backends[backend_id]

        # Header
        layout.label(text=f"Installing {backend_cls.name}", icon='IMPORT')
        layout.separator()

        # Instructions
        box = layout.box()
        col = box.column(align=True)
        col.scale_y = 0.8

        for line in backend_cls.install_instructions.strip().split('\n'):
            col.label(text=line)

        # Website link
        if backend_cls.website:
            layout.separator()
            layout.operator("wm.url_open", text="Visit Website", icon='URL').url = backend_cls.website

    def invoke(self, context, event):
        """Invoke the popup."""
        return context.window_manager.invoke_popup(self, width=500)


# Registration
classes = [
    GSCAPTURE_OT_StartTraining,
    GSCAPTURE_OT_StopTraining,
    GSCAPTURE_OT_ClearTraining,
    GSCAPTURE_OT_BrowseTrainingData,
    GSCAPTURE_OT_BrowseTrainingOutput,
    GSCAPTURE_OT_OpenTrainingOutput,
    GSCAPTURE_OT_ShowInstallInstructions,
]


def register():
    """Register training operators."""
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    """Unregister training operators."""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
