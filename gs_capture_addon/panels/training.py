"""
Training Panel - UI for integrated Gaussian Splatting training.

This panel provides controls for training Gaussian Splatting models
directly from Blender, with real-time progress monitoring.
"""

import bpy
from bpy.types import Panel

from ..core.training import get_available_backends, get_running_process
from ..core.training.base import TrainingStatus


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
        # Backend selection
        box = layout.box()
        box.label(text="Training Backend:", icon='TOOL_SETTINGS')

        row = box.row(align=True)
        row.prop(settings, "training_backend", text="")

        # Check availability
        backends = get_available_backends()
        current_backend = settings.training_backend

        if current_backend not in backends:
            row = box.row()
            row.alert = True
            row.label(text="Backend not found", icon='ERROR')

            box.operator(
                "gs_capture.show_install_instructions",
                text="Installation Guide",
                icon='HELP'
            )
        else:
            row = box.row()
            row.label(text="Ready", icon='CHECKMARK')

        # Data path
        layout.separator()
        box = layout.box()
        box.label(text="Training Data:", icon='FILE_FOLDER')

        row = box.row(align=True)
        row.prop(settings, "training_data_path", text="")
        row.operator("gs_capture.browse_training_data", text="", icon='FILEBROWSER')

        # Validate data
        if settings.training_data_path:
            if current_backend in backends:
                backend = backends[current_backend]
                is_valid, message = backend.validate_data(settings.training_data_path)

                row = box.row()
                if is_valid:
                    row.label(text=message, icon='CHECKMARK')
                else:
                    row.alert = True
                    row.label(text=message, icon='ERROR')

        # Output path
        box = layout.box()
        box.label(text="Output:", icon='EXPORT')

        row = box.row(align=True)
        row.prop(settings, "training_output_path", text="")
        row.operator("gs_capture.browse_training_output", text="", icon='FILEBROWSER')

        # Training parameters
        layout.separator()
        box = layout.box()
        box.label(text="Parameters:", icon='PREFERENCES')

        col = box.column(align=True)
        col.prop(settings, "training_iterations", text="Iterations")
        col.prop(settings, "training_save_every", text="Save Every")

        row = box.row()
        row.prop(settings, "training_white_background", text="White Background")

        # Start button
        layout.separator()

        can_start = (
            current_backend in backends and
            settings.training_data_path and
            settings.training_output_path
        )

        row = layout.row(align=True)
        row.scale_y = 2.0
        row.enabled = can_start
        row.operator("gs_capture.start_training", text="Start Training", icon='PLAY')

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

        # Error message
        if progress.error:
            layout.separator()
            box = layout.box()
            box.alert = True
            col = box.column(align=True)
            col.scale_y = 0.8
            for line in progress.error.split('\n')[:5]:
                col.label(text=line)

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
                    box.operator("gs_capture.open_training_output", text="Open Folder", icon='FILE_FOLDER')


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


# Registration
classes = [
    GSCAPTURE_PT_TrainingPanel,
    GSCAPTURE_PT_TrainingAdvanced,
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
