"""
Main capture operators for rendering multi-view images.
Includes checkpoint support, depth/normal/mask export.
"""

import bpy
import copy
import glob
import json
import os
import threading
import time
from datetime import datetime, timezone
from bpy.props import BoolProperty
from bpy.types import Operator
from mathutils import Vector

from ..core.analysis import calculate_adaptive_settings
from ..core.camera import (
    generate_camera_positions,
    get_objects_combined_bounds,
    create_camera_at_position,
    delete_gs_cameras,
)
from ..core.export import (
    export_colmap_cameras,
    export_transforms_json,
    cleanup_compositor_nodes,
    extract_alpha_mask,
    get_image_extension,
    get_compositor_tree,
    get_or_create_render_layers_node,
    find_socket_by_names,
    configure_output_file_node,
    set_output_file_basename,
)
from ..core.validation import (
    ValidationLevel,
    ValidationResult,
    validate_all,
    validation_result_to_dict,
)
from ..preferences import get_preferences
from ..utils.lighting import (
    setup_neutral_lighting,
    store_lighting_state,
    restore_lighting,
    get_eevee_engine_name,
)
from ..utils.materials import override_materials, restore_materials
from ..utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    validate_checkpoint,
    clear_checkpoint,
    create_checkpoint,
    get_checkpoint_path,
    get_missing_images,
    calculate_settings_hash,
    settings_hash_matches,
)
from ..utils.paths import validate_path_length


class _AsyncCheckpointWriter:
    """Background checkpoint writer to avoid blocking the render loop."""

    def __init__(self, output_path):
        self._output_path = output_path
        self._lock = threading.Lock()
        self._pending = None
        self._pending_version = 0
        self._written_version = 0
        self._errors = []
        self._last_error = None
        self._event = threading.Event()
        self._stop_requested = False
        self._thread = threading.Thread(
            target=self._run,
            name="GSCheckpointWriter",
            daemon=True,
        )
        self._thread.start()

    def request_save(self, checkpoint_data):
        snapshot = copy.deepcopy(checkpoint_data)
        with self._lock:
            self._pending = snapshot
            self._pending_version += 1
            self._event.set()
            return self._pending_version

    def flush(self, timeout=None):
        start = time.time()
        while True:
            with self._lock:
                pending_version = self._pending_version
                written_version = self._written_version
                pending_exists = self._pending is not None
            if pending_version == 0 or (written_version >= pending_version and not pending_exists):
                return True
            if timeout is not None and (time.time() - start) >= timeout:
                return False
            time.sleep(0.01)

    def stop(self, timeout=None):
        self.flush(timeout=timeout)
        with self._lock:
            self._stop_requested = True
            self._event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self):
        while True:
            self._event.wait()
            with self._lock:
                if self._stop_requested and self._pending is None:
                    break
                snapshot = self._pending
                version = self._pending_version
                self._pending = None
                self._event.clear()
            if snapshot is None:
                continue
            success, error = save_checkpoint(self._output_path, snapshot)
            with self._lock:
                if success:
                    self._last_error = None
                elif error and error != self._last_error:
                    self._errors.append(error)
                    self._last_error = error
            with self._lock:
                if version > self._written_version:
                    self._written_version = version

    def pop_errors(self):
        with self._lock:
            errors = list(self._errors)
            self._errors.clear()
        return errors


class GSCAPTURE_OT_capture_selected(Operator):
    """Capture images of selected objects for Gaussian Splatting training."""
    bl_idname = "gs_capture.capture_selected"
    bl_label = "Capture Selected"
    bl_options = {'REGISTER', 'UNDO'}

    # Note: Instance variables are initialized in execute() to avoid
    # class-level mutable defaults being shared across instances.
    # These type hints document the expected instance attributes.
    _timer: object
    _cameras: list
    _current_camera_index: int
    _output_path: str
    _original_camera: object
    _original_light_states: dict
    _original_materials: dict
    _original_file_format: object
    _original_hide_render: dict
    _original_film_transparent: bool
    _target_objects: list
    _adaptive_result: object
    _target_format: str
    _image_ext: str
    _save_manually: bool
    _checkpoint_data: dict
    _images_to_render: list
    _depth_output_node: object
    _normal_output_node: object
    _mask_output_node: object
    _images_path: str
    _depth_path: str
    _normals_path: str
    _masks_path: str
    _image_path_prefix: str
    _image_path_suffix: str
    _mask_path_prefix: str
    _mask_path_suffix: str
    _mask_path_prefix_gsl: str
    _mask_path_suffix_gsl: str
    _original_engine: object
    _original_eevee_samples: object
    _original_cycles_samples: object
    _export_errors: bool
    _checkpoint_writer: object
    _checkpoints_enabled: bool
    _pre_capture_validation_result: object
    _capture_view_layer_name: str
    _original_view_layer_pass_flags: dict
    _original_object_pass_indices: dict
    _render_in_progress: bool
    _active_camera_actual_index: int
    _active_image_path: str
    _active_needs_alpha_mask: bool

    preflight_only: BoolProperty(
        name="Validation Summary Only",
        description="Show validation summary without starting capture",
        default=False,
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    def _get_auto_validate_enabled(self):
        prefs = get_preferences()
        return prefs.auto_validate if prefs else True

    def _get_validation_result(self, context, settings, force=False):
        if not force and not self._get_auto_validate_enabled():
            return None
        cached = getattr(self, "_pre_capture_validation_result", None)
        if cached is not None:
            return cached
        result = validate_all(context, settings)
        self._pre_capture_validation_result = result
        return result

    def _report_validation_issues(self, validation_result):
        if validation_result is None:
            return
        for issue in validation_result.issues:
            if issue.level == ValidationLevel.INFO:
                continue
            report_level = {'ERROR'} if issue.level == ValidationLevel.ERROR else {'WARNING'}
            message = issue.message
            if issue.suggestion and issue.level == ValidationLevel.ERROR:
                message = f"{issue.message} ({issue.suggestion})"
            self.report(report_level, message)

    def _should_show_pre_capture_dialog(self, validation_result):
        if validation_result is None:
            return False
        if self.preflight_only:
            return True
        return any(
            issue.level in (ValidationLevel.ERROR, ValidationLevel.WARNING)
            for issue in validation_result.issues
        )

    def _dialog_icon_for_issue(self, issue):
        if issue.level == ValidationLevel.ERROR:
            return 'ERROR'
        if issue.level == ValidationLevel.WARNING:
            return 'INFO'
        return 'DOT'

    def _draw_validation_dialog(self, layout, validation_result):
        if validation_result is None:
            layout.label(text="Validation disabled in preferences", icon='INFO')
            return

        if validation_result.can_proceed:
            header_icon = 'CHECKMARK' if validation_result.warning_count == 0 else 'INFO'
        else:
            header_icon = 'ERROR'

        title = "Pre-Capture Validation Summary" if self.preflight_only else "Pre-Capture Validation"
        layout.label(text=title, icon=header_icon)
        layout.label(
            text=(
                f"Errors: {validation_result.error_count} | "
                f"Warnings: {validation_result.warning_count} | "
                f"Info: {validation_result.info_count}"
            )
        )

        if not validation_result.issues:
            layout.label(text="No validation issues found.", icon='CHECKMARK')
            return

        box = layout.box()
        col = box.column(align=True)
        max_items = 12
        for issue in validation_result.issues[:max_items]:
            col.label(
                text=f"[{issue.category}] {issue.message}",
                icon=self._dialog_icon_for_issue(issue),
            )
            if issue.suggestion:
                col.label(text=f"Fix: {issue.suggestion}", icon='RIGHTARROW')

        hidden_count = len(validation_result.issues) - max_items
        if hidden_count > 0:
            box.label(text=f"... and {hidden_count} more issue(s)", icon='INFO')

        if not validation_result.can_proceed and not self.preflight_only:
            layout.label(text="Capture is blocked until errors are fixed.", icon='CANCEL')

    def _count_non_empty_files(self, directory, pattern):
        """Count files that match pattern and have non-zero size."""
        matches = glob.glob(os.path.join(directory, pattern))
        count = 0
        for path in matches:
            try:
                if os.path.getsize(path) > 0:
                    count += 1
            except OSError:
                continue
        return count

    def _build_post_export_validation(self, settings, image_ext):
        """Validate exported artifacts and return issues plus check details."""
        expected = len(self._cameras)
        result = ValidationResult()
        checks = []

        def add_check(label, folder, pattern, required):
            found = self._count_non_empty_files(folder, pattern)
            check = {
                "name": label,
                "folder": folder,
                "pattern": pattern,
                "expected": expected if required else 0,
                "found": found,
                "required": required,
            }
            checks.append(check)

            if required and found < expected:
                result.add_error(
                    "export",
                    f"{label}: expected {expected}, found {found}",
                    f"Re-run capture to generate missing {label.lower()} files",
                )

        add_check("images", self._images_path, f"image_*.{image_ext}", required=True)

        if settings.export_depth:
            add_check(
                "depth",
                self._depth_path,
                f"depth_*.{self._depth_output_extension()}",
                required=True,
            )

        if settings.export_normals:
            add_check("normals", self._normals_path, "normal_*.exr", required=True)

        if settings.export_masks:
            if settings.mask_source == 'ALPHA':
                pattern = f"image_*.{image_ext}.png" if settings.mask_format == 'GSL' else "mask_*.png"
            else:
                pattern = f"mask_*.{self._object_index_mask_extension()}"
            add_check("masks", self._masks_path, pattern, required=True)

        return result, checks

    def _write_validation_report(self, settings, image_ext, export_failures, post_result, checks):
        """Write a JSON validation artifact to the output directory."""
        output_file = os.path.join(self._output_path, "validation_report.json")
        payload = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "blender_version": ".".join(str(part) for part in bpy.app.version),
            "output_path": self._output_path,
            "capture": {
                "expected_cameras": len(self._cameras),
                "rendered_this_run": len(self._images_to_render),
                "completed_images": len(self._checkpoint_data.get("completed_images", [])) if self._checkpoint_data else 0,
                "export_errors": self._export_errors,
                "mask_source": settings.mask_source,
                "mask_format": settings.mask_format,
                "image_extension": image_ext,
            },
            "pre_capture_validation": validation_result_to_dict(
                getattr(self, "_pre_capture_validation_result", None)
            ),
            "post_export_validation": {
                "result": validation_result_to_dict(post_result),
                "artifact_checks": checks,
            },
            "export_failures": list(export_failures),
        }

        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return output_file

    def invoke(self, context, event):
        settings = context.scene.gs_capture_settings
        force_validation = self.preflight_only
        self._pre_capture_validation_result = None

        validation_result = self._get_validation_result(context, settings, force=force_validation)
        if not self._should_show_pre_capture_dialog(validation_result):
            return self.execute(context)

        return context.window_manager.invoke_props_dialog(self, width=620)

    def draw(self, context):
        layout = self.layout
        self._draw_validation_dialog(layout, getattr(self, "_pre_capture_validation_result", None))

    def _get_all_children(self, obj):
        """Recursively get all children of an object."""
        children = []
        for child in obj.children:
            children.append(child)
            children.extend(self._get_all_children(child))
        return children

    def _hide_non_target_objects(self, context, target_objects):
        """Hide all objects except target objects and their children from render."""
        visible_objects = set(target_objects)

        settings = context.scene.gs_capture_settings
        if settings.include_children:
            for obj in target_objects:
                visible_objects.update(self._get_all_children(obj))

        self._original_hide_render = {}
        for obj in context.scene.objects:
            self._original_hide_render[obj.name] = obj.hide_render
            if obj not in visible_objects:
                obj.hide_render = True

    def _restore_object_visibility(self, context):
        """Restore original hide_render state for all objects."""
        for obj_name, hide_render in self._original_hide_render.items():
            obj = context.scene.objects.get(obj_name)
            if obj:
                obj.hide_render = hide_render

    def _snapshot_view_layer_pass_flags(self, view_layer, settings):
        """Remember view-layer pass flags that this capture run may modify."""
        if self._original_view_layer_pass_flags:
            return

        flags = {}
        if settings.export_depth and hasattr(view_layer, "use_pass_z"):
            flags["use_pass_z"] = view_layer.use_pass_z
        if settings.export_normals and hasattr(view_layer, "use_pass_normal"):
            flags["use_pass_normal"] = view_layer.use_pass_normal
        if settings.export_masks and settings.mask_source == 'OBJECT_INDEX':
            if hasattr(view_layer, "use_pass_object_index"):
                flags["use_pass_object_index"] = view_layer.use_pass_object_index

        self._original_view_layer_pass_flags = flags
        self._capture_view_layer_name = view_layer.name if flags else ""

    def _restore_view_layer_pass_flags(self, context):
        """Restore any view-layer pass flags changed for capture exports."""
        if not self._original_view_layer_pass_flags:
            return

        view_layer = context.scene.view_layers.get(self._capture_view_layer_name)
        if view_layer is None:
            view_layer = context.view_layer

        for attr, value in self._original_view_layer_pass_flags.items():
            if hasattr(view_layer, attr):
                setattr(view_layer, attr, value)

        self._original_view_layer_pass_flags = {}
        self._capture_view_layer_name = ""

    def _snapshot_target_object_pass_indices(self):
        """Remember original pass indices before assigning mask IDs."""
        if self._original_object_pass_indices:
            return

        self._original_object_pass_indices = {
            obj.name: obj.pass_index
            for obj in self._target_objects
            if obj and hasattr(obj, "pass_index")
        }

    def _restore_target_object_pass_indices(self, context):
        """Restore target object pass indices after capture."""
        if not self._original_object_pass_indices:
            return

        for obj_name, pass_index in self._original_object_pass_indices.items():
            obj = context.scene.objects.get(obj_name)
            if obj and hasattr(obj, "pass_index"):
                obj.pass_index = pass_index

        self._original_object_pass_indices = {}

    def _validate_path_length(self, path, label, report_level):
        is_valid, _, error = validate_path_length(path)
        if not is_valid:
            self.report(report_level, f"{label} path is too long for Windows. {error}")
            return False
        return True

    def _validate_output_paths(self, settings, image_ext):
        depth_ext = self._depth_output_extension()
        object_mask_ext = self._object_index_mask_extension()
        paths = [
            ("Output directory", self._output_path),
            ("Images directory", self._images_path),
        ]

        if settings.export_depth:
            paths.append(("Depth directory", self._depth_path))
        if settings.export_normals:
            paths.append(("Normals directory", self._normals_path))
        if settings.export_masks:
            paths.append(("Masks directory", self._masks_path))

        # Sample output files
        paths.append(("Image file", os.path.join(self._images_path, f"image_0000.{image_ext}")))

        if settings.export_depth:
            paths.append(("Depth file", os.path.join(self._depth_path, f"depth_0000.{depth_ext}")))
        if settings.export_normals:
            paths.append(("Normal file", os.path.join(self._normals_path, "normal_0000.exr")))
        if settings.export_masks:
            if settings.mask_source == 'ALPHA':
                if settings.mask_format == 'GSL':
                    mask_name = f"image_0000.{image_ext}.png"
                else:
                    mask_name = "mask_0000.png"
            else:
                mask_name = f"mask_0000.{object_mask_ext}"
            paths.append(("Mask file", os.path.join(self._masks_path, mask_name)))

        for label, path in paths:
            if not self._validate_path_length(path, label, {'ERROR'}):
                return False

        if self._checkpoints_enabled:
            checkpoint_path = get_checkpoint_path(self._output_path)
            if not self._validate_path_length(checkpoint_path, "Checkpoint file", {'WARNING'}):
                self._checkpoints_enabled = False

        return True

    def _depth_output_extension(self):
        # Blender 5.x compositor file outputs currently emit EXR for value outputs.
        return "exr" if bpy.app.version >= (5, 0, 0) else "png"

    def _object_index_mask_extension(self):
        # Blender 5.x compositor file outputs currently emit EXR for value outputs.
        return "exr" if bpy.app.version >= (5, 0, 0) else "png"

    def _setup_compositor_outputs(self, context, settings):
        """Setup compositor nodes for depth, normal, and mask outputs."""
        scene = context.scene
        tree = get_compositor_tree(scene, create=True)
        if tree is None:
            self.report({'WARNING'}, "Could not create compositor node tree")
            return

        # Get or create render layers node
        render_layers = get_or_create_render_layers_node(tree)

        # Enable required passes
        view_layer = context.view_layer
        self._snapshot_view_layer_pass_flags(view_layer, settings)

        if settings.export_depth:
            view_layer.use_pass_z = True
            self._setup_depth_output(tree, render_layers)

        if settings.export_normals:
            view_layer.use_pass_normal = True
            self._setup_normal_output(tree, render_layers)

        if settings.export_masks and settings.mask_source == 'OBJECT_INDEX':
            view_layer.use_pass_object_index = True
            # Set object indices
            self._snapshot_target_object_pass_indices()
            for i, obj in enumerate(self._target_objects):
                obj.pass_index = i + 1
            self._setup_mask_output(tree, render_layers)

    def _setup_depth_output(self, tree, render_layers):
        """Setup depth map output node."""
        depth_socket = find_socket_by_names(render_layers.outputs, {"depth", "z"})
        if depth_socket is None:
            self.report({'WARNING'}, "Depth pass output not found; skipping depth export")
            return

        # Normalize node
        normalize = tree.nodes.new('CompositorNodeNormalize')
        normalize.name = "GS_Depth_Normalize"
        normalize.location = (300, -100)

        # File output
        file_output = tree.nodes.new('CompositorNodeOutputFile')
        file_output.name = "GS_Depth_Output"
        file_output.location = (500, -100)
        depth_ext = self._depth_output_extension()
        depth_format = 'OPEN_EXR' if depth_ext == 'exr' else 'PNG'
        depth_color_mode = None if depth_ext == 'exr' else 'BW'
        depth_color_depth = None if depth_ext == 'exr' else '16'
        output_socket = configure_output_file_node(
            file_output,
            os.path.join(self._output_path, "depth"),
            "depth_0000",
            file_format=depth_format,
            color_mode=depth_color_mode,
            color_depth=depth_color_depth,
            slot_name='Depth',
            socket_type='FLOAT',
        )
        if output_socket is None:
            self.report({'WARNING'}, "Could not configure depth file output node")
            return

        # Connect
        tree.links.new(depth_socket, normalize.inputs[0])
        tree.links.new(normalize.outputs[0], output_socket)

        self._depth_output_node = file_output

    def _setup_normal_output(self, tree, render_layers):
        """Setup normal map output node."""
        normal_socket = find_socket_by_names(render_layers.outputs, {"normal"})
        if normal_socket is None:
            self.report({'WARNING'}, "Normal pass output not found; skipping normal export")
            return

        # File output for normals
        file_output = tree.nodes.new('CompositorNodeOutputFile')
        file_output.name = "GS_Normal_Output"
        file_output.location = (500, -250)
        output_socket = configure_output_file_node(
            file_output,
            os.path.join(self._output_path, "normals"),
            "normal_0000",
            file_format='OPEN_EXR',
            color_mode='RGB',
            slot_name='Normal',
            socket_type='VECTOR',
        )
        if output_socket is None:
            self.report({'WARNING'}, "Could not configure normal file output node")
            return

        # Connect normal output
        tree.links.new(normal_socket, output_socket)

        self._normal_output_node = file_output

    def _setup_mask_output(self, tree, render_layers):
        """Setup object mask output node for all target objects.

        Creates ID mask nodes for each target object index and combines
        them to create a single mask covering all target objects.
        """
        num_objects = len(self._target_objects)

        if num_objects == 0:
            return

        # Create ID Mask nodes for each object index
        index_output = None
        wanted = {'indexob', 'object index'}
        for socket in render_layers.outputs:
            socket_name = str(getattr(socket, 'name', '')).lower()
            socket_id = str(getattr(socket, 'identifier', '')).lower()
            if socket_name in wanted or socket_id in wanted:
                index_output = socket
                break
        if index_output is None:
            self.report({'WARNING'}, "Object Index pass output not found; skipping object-index masks")
            return

        id_masks = []
        for i in range(num_objects):
            id_mask = tree.nodes.new('CompositorNodeIDMask')
            id_mask.name = f"GS_ID_Mask_{i}"
            id_mask.location = (300, -400 - i * 100)
            # Blender 4.x uses node properties; Blender 5.x uses input sockets.
            if hasattr(id_mask, 'index'):
                id_mask.index = i + 1  # Object indices start at 1
                if hasattr(id_mask, 'use_antialiasing'):
                    id_mask.use_antialiasing = True
            else:
                index_input = find_socket_by_names(id_mask.inputs, {"index"})
                if index_input is not None and hasattr(index_input, 'default_value'):
                    index_input.default_value = i + 1
                aa_input = find_socket_by_names(id_mask.inputs, {"anti-alias", "anti alias", "anti_alias"})
                if aa_input is not None and hasattr(aa_input, 'default_value'):
                    aa_input.default_value = True
            tree.links.new(index_output, id_mask.inputs[0])
            id_masks.append(id_mask)

        # Combine masks using Maximum math nodes
        if len(id_masks) == 1:
            # Only one object, use its mask directly
            combined_output = id_masks[0].outputs[0]
        else:
            # Combine multiple masks with Maximum nodes
            combined_output = id_masks[0].outputs[0]
            for i in range(1, len(id_masks)):
                math_node = tree.nodes.new('CompositorNodeMath')
                math_node.name = f"GS_Mask_Combine_{i}"
                math_node.operation = 'MAXIMUM'
                math_node.location = (500, -400 - (i - 1) * 50)
                tree.links.new(combined_output, math_node.inputs[0])
                tree.links.new(id_masks[i].outputs[0], math_node.inputs[1])
                combined_output = math_node.outputs[0]

        # File output for mask
        file_output = tree.nodes.new('CompositorNodeOutputFile')
        file_output.name = "GS_Mask_Output"
        file_output.location = (700, -400)
        mask_ext = self._object_index_mask_extension()
        mask_format = 'OPEN_EXR' if mask_ext == 'exr' else 'PNG'
        mask_color_mode = None if mask_ext == 'exr' else 'BW'
        output_socket = configure_output_file_node(
            file_output,
            os.path.join(self._output_path, "masks"),
            "mask_0000",
            file_format=mask_format,
            color_mode=mask_color_mode,
            slot_name='Mask',
            socket_type='FLOAT',
        )
        if output_socket is None:
            self.report({'WARNING'}, "Could not configure mask file output node")
            return

        # Connect combined mask to output
        tree.links.new(combined_output, output_socket)

        self._mask_output_node = file_output

    def _update_output_filenames(self, index):
        """Update compositor output node filenames for current frame."""
        if self._depth_output_node:
            set_output_file_basename(self._depth_output_node, f"depth_{index:04d}")

        if self._normal_output_node:
            set_output_file_basename(self._normal_output_node, f"normal_{index:04d}")

        if self._mask_output_node:
            set_output_file_basename(self._mask_output_node, f"mask_{index:04d}")

    def _verify_output_file(self, directory, prefix, ext):
        """Check for a non-empty output file written by compositor or export."""
        pattern = os.path.join(directory, f"{prefix}*.{ext}")
        matches = glob.glob(pattern)
        for path in matches:
            try:
                if os.path.getsize(path) > 0:
                    return True
            except OSError:
                continue
        return False

    def _reconcile_checkpoint_camera_count(self, checkpoint, current_total):
        """Ensure checkpoint camera count aligns with current cameras."""
        if not checkpoint:
            return None

        checkpoint_total = checkpoint.get('total_cameras')
        if not isinstance(checkpoint_total, int) or checkpoint_total <= 0:
            self.report({'WARNING'}, "Checkpoint has invalid camera count. Starting fresh.")
            return None

        if checkpoint_total != current_total:
            self.report(
                {'WARNING'},
                f"Checkpoint camera count ({checkpoint_total}) does not match current camera count ({current_total}). "
                "Resuming with reconciled checkpoint."
            )

        completed = checkpoint.get('completed_images')
        if not isinstance(completed, list):
            completed = []

        valid_completed = [
            idx for idx in completed
            if isinstance(idx, int) and 0 <= idx < current_total
        ]
        if len(valid_completed) != len(completed):
            removed = len(completed) - len(valid_completed)
            if removed > 0:
                self.report(
                    {'INFO'},
                    f"Discarded {removed} checkpoint indices outside current camera range."
                )

        if len(valid_completed) > 1:
            valid_completed = sorted(set(valid_completed))

        checkpoint['total_cameras'] = current_total
        checkpoint['completed_images'] = valid_completed

        current_index = checkpoint.get('current_index')
        if not isinstance(current_index, int) or current_index < 0:
            current_index = 0
        if current_total > 0 and current_index >= current_total:
            current_index = current_total - 1
        checkpoint['current_index'] = current_index

        return checkpoint

    def _get_expected_output_dirs(self, settings):
        """Return expected output subdirectories based on current settings."""
        expected = [("images", self._images_path)]
        if settings.export_depth:
            expected.append(("depth", self._depth_path))
        if settings.export_normals:
            expected.append(("normals", self._normals_path))
        if settings.export_masks:
            expected.append(("masks", self._masks_path))
        return expected

    def _find_missing_output_dirs(self, settings):
        """Return list of expected output dir labels that are missing."""
        missing = []
        for label, path in self._get_expected_output_dirs(settings):
            if not os.path.isdir(path):
                missing.append(label)
        return missing

    def _build_checkpoint_file_specs(self, settings, image_ext):
        """Build expected output filenames used to validate checkpoint resume."""
        specs = [
            {
                "label": "Image",
                "directory": self._images_path,
                "template": f"image_{{idx:04d}}.{image_ext}",
            }
        ]

        if settings.export_depth:
            specs.append(
                {
                    "label": "Depth",
                    "directory": self._depth_path,
                    "template": f"depth_{{idx:04d}}.{self._depth_output_extension()}",
                }
            )

        if settings.export_normals:
            specs.append(
                {
                    "label": "Normal",
                    "directory": self._normals_path,
                    "template": "normal_{idx:04d}.exr",
                }
            )

        if settings.export_masks:
            if settings.mask_source == 'ALPHA':
                if settings.mask_format == 'GSL':
                    mask_template = f"image_{{idx:04d}}.{image_ext}.png"
                else:
                    mask_template = "mask_{idx:04d}.png"
            else:
                mask_template = f"mask_{{idx:04d}}.{self._object_index_mask_extension()}"

            specs.append(
                {
                    "label": "Mask",
                    "directory": self._masks_path,
                    "template": mask_template,
                }
            )

        return specs

    def _ensure_directory(self, path, label):
        """Create directory with user-facing error handling."""
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            self.report({'ERROR'}, f"Failed to create {label}: {exc}")
            return False

        if not os.path.isdir(path):
            self.report({'ERROR'}, f"{label} exists but is not a directory: {path}")
            return False

        return True

    def _shutdown_checkpoint_writer(self, timeout=None):
        if self._checkpoint_writer:
            self._checkpoint_writer.stop(timeout=timeout)
            for error in self._checkpoint_writer.pop_errors():
                self.report({'WARNING'}, error)
            self._checkpoint_writer = None

    def _drain_checkpoint_writer_errors(self):
        if not self._checkpoint_writer:
            return
        for error in self._checkpoint_writer.pop_errors():
            self.report({'WARNING'}, error)

    def execute(self, context):
        cached_validation = getattr(self, "_pre_capture_validation_result", None)

        # Initialize instance variables (avoid class-level mutable defaults)
        self._timer = None
        self._cameras = []
        self._current_camera_index = 0
        self._output_path = ""
        self._original_camera = None
        self._original_light_states = {}
        self._original_materials = {}
        self._original_file_format = None
        self._original_hide_render = {}
        self._original_film_transparent = False
        self._target_objects = []
        self._adaptive_result = None
        self._target_format = 'PNG'
        self._image_ext = 'png'
        self._save_manually = False
        self._checkpoint_data = None
        self._images_to_render = []
        self._depth_output_node = None
        self._normal_output_node = None
        self._mask_output_node = None
        self._images_path = ""
        self._depth_path = ""
        self._normals_path = ""
        self._masks_path = ""
        self._image_path_prefix = ""
        self._image_path_suffix = ""
        self._mask_path_prefix = ""
        self._mask_path_suffix = ""
        self._mask_path_prefix_gsl = ""
        self._mask_path_suffix_gsl = ""
        self._original_engine = None
        self._original_eevee_samples = None
        self._original_cycles_samples = None
        self._export_errors = False
        self._checkpoint_writer = None
        self._checkpoints_enabled = False
        self._pre_capture_validation_result = cached_validation
        self._capture_view_layer_name = ""
        self._original_view_layer_pass_flags = {}
        self._original_object_pass_indices = {}

        settings = context.scene.gs_capture_settings
        rd = context.scene.render

        settings.cancel_requested = False

        # Apply render speed preset
        self._original_engine = rd.engine
        self._original_eevee_samples = context.scene.eevee.taa_render_samples if hasattr(context.scene.eevee, 'taa_render_samples') else 64
        self._original_cycles_samples = context.scene.cycles.samples if hasattr(context.scene, 'cycles') else 128

        if settings.render_speed_preset == 'FAST':
            engine_name, warning = get_eevee_engine_name()
            if warning:
                self.report({'WARNING'}, warning)
            rd.engine = engine_name
            if hasattr(context.scene.eevee, 'taa_render_samples'):
                context.scene.eevee.taa_render_samples = 16
        elif settings.render_speed_preset == 'BALANCED':
            engine_name, warning = get_eevee_engine_name()
            if warning:
                self.report({'WARNING'}, warning)
            rd.engine = engine_name
            if hasattr(context.scene.eevee, 'taa_render_samples'):
                context.scene.eevee.taa_render_samples = 64
        elif settings.render_speed_preset == 'QUALITY':
            rd.engine = 'CYCLES'
            context.scene.cycles.samples = 128

        def restore_render_settings():
            if self._original_engine:
                rd.engine = self._original_engine
            if self._original_eevee_samples and hasattr(context.scene.eevee, 'taa_render_samples'):
                context.scene.eevee.taa_render_samples = self._original_eevee_samples
            if self._original_cycles_samples and hasattr(context.scene, 'cycles'):
                context.scene.cycles.samples = self._original_cycles_samples

        validation_result = self._get_validation_result(
            context,
            settings,
            force=self.preflight_only,
        )
        self._report_validation_issues(validation_result)

        if self.preflight_only:
            restore_render_settings()
            if validation_result is None:
                self.report({'INFO'}, "Validation is disabled in preferences")
            elif validation_result.can_proceed:
                self.report(
                    {'INFO'},
                    (
                        f"Validation passed with "
                        f"{validation_result.warning_count} warning(s) and {validation_result.info_count} info item(s)"
                    ),
                )
            else:
                self.report({'ERROR'}, "Validation failed. Resolve errors before capture.")
                return {'CANCELLED'}
            return {'FINISHED'}

        if validation_result is not None and not validation_result.can_proceed:
            restore_render_settings()
            return {'CANCELLED'}

        # Get selected objects
        selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not selected:
            self.report({'ERROR'}, "No mesh objects selected")
            restore_render_settings()
            return {'CANCELLED'}

        self._target_objects = selected

        self._checkpoints_enabled = settings.enable_checkpoints

        # Store original file format for potential restoration
        self._original_file_format = rd.image_settings.file_format
        self._target_format = rd.image_settings.file_format
        self._image_ext = get_image_extension(self._target_format)
        image_ext = self._image_ext or 'png'

        # Setup output directory
        self._output_path = bpy.path.abspath(settings.output_path)
        self._images_path = os.path.join(self._output_path, "images")
        self._depth_path = os.path.join(self._output_path, "depth")
        self._normals_path = os.path.join(self._output_path, "normals")
        self._masks_path = os.path.join(self._output_path, "masks")

        if not self._validate_output_paths(settings, image_ext):
            restore_render_settings()
            return {'CANCELLED'}
        checkpoint_file_specs = self._build_checkpoint_file_specs(settings, image_ext)

        # Check for checkpoint resume
        settings_hash = calculate_settings_hash(settings, context.scene)
        legacy_settings_hash = calculate_settings_hash(settings, context.scene, legacy=True)
        checkpoint = None

        if self._checkpoints_enabled and settings.auto_resume:
            checkpoint, checkpoint_error = load_checkpoint(self._output_path)
            if checkpoint_error:
                self.report({'WARNING'}, checkpoint_error)
            if checkpoint:
                # Validate checkpoint
                if settings_hash_matches(checkpoint, settings_hash, legacy_settings_hash):
                    if checkpoint.get('settings_hash') != settings_hash or checkpoint.get('settings_hash_legacy') is None:
                        checkpoint['settings_hash'] = settings_hash
                        checkpoint['settings_hash_legacy'] = legacy_settings_hash
                    missing_dirs = self._find_missing_output_dirs(settings)
                    if missing_dirs:
                        missing_list = ", ".join(missing_dirs)
                        self.report(
                            {'WARNING'},
                            f"Checkpoint found but output folders are missing ({missing_list}). Starting fresh."
                        )
                        success, error = clear_checkpoint(self._output_path)
                        if not success and error:
                            self.report({'WARNING'}, error)
                        checkpoint = None
                    else:
                        checkpoint_valid, checkpoint, checkpoint_error = validate_checkpoint(
                            self._output_path,
                            settings_hash,
                            legacy_settings_hash=legacy_settings_hash,
                            checkpoint_data=checkpoint,
                            file_specs=checkpoint_file_specs,
                        )
                        if not checkpoint_valid:
                            reason = checkpoint_error or "checkpoint outputs are missing or corrupted"
                            self.report({'WARNING'}, f"Checkpoint invalid ({reason}). Starting fresh.")
                            success, error = clear_checkpoint(self._output_path)
                            if not success and error:
                                self.report({'WARNING'}, error)
                            checkpoint = None
                        else:
                            self.report({'INFO'}, f"Resuming from image {checkpoint['current_index']}")
                else:
                    self.report({'WARNING'}, "Settings changed, starting fresh")
                    checkpoint = None

        if not self._ensure_directory(self._output_path, "output directory"):
            restore_render_settings()
            return {'CANCELLED'}
        if not self._ensure_directory(self._images_path, "images directory"):
            restore_render_settings()
            return {'CANCELLED'}

        # Create additional directories for exports
        if settings.export_depth:
            if not self._ensure_directory(self._depth_path, "depth directory"):
                restore_render_settings()
                return {'CANCELLED'}
        if settings.export_normals:
            if not self._ensure_directory(self._normals_path, "normals directory"):
                restore_render_settings()
                return {'CANCELLED'}
        if settings.export_masks:
            if not self._ensure_directory(self._masks_path, "masks directory"):
                restore_render_settings()
                return {'CANCELLED'}

        # Hide non-target objects
        self._hide_non_target_objects(context, selected)

        # Run adaptive analysis if enabled
        scene = context.scene
        rd = scene.render

        if settings.use_adaptive_capture:
            self._adaptive_result = calculate_adaptive_settings(
                selected, settings.adaptive_quality_preset
            )
            actual_camera_count = self._adaptive_result.recommended_camera_count
            # Adaptive can suggest resolution, but we apply it to Blender's settings
            suggested_res = self._adaptive_result.recommended_resolution
            rd.resolution_x = suggested_res[0]
            rd.resolution_y = suggested_res[1]
            actual_distance_mult = self._adaptive_result.recommended_distance_multiplier

            settings.analysis_recommended_cameras = actual_camera_count
            settings.analysis_quality_preset = self._adaptive_result.quality_preset
        else:
            actual_camera_count = settings.camera_count
            actual_distance_mult = settings.camera_distance_multiplier

        # Store original state
        self._original_camera = context.scene.camera
        self._original_light_states = store_lighting_state(context)

        # Calculate bounds
        center, radius = get_objects_combined_bounds(selected)

        # Calculate camera distance
        if settings.camera_distance_mode == 'AUTO':
            distance = radius * actual_distance_mult
        else:
            distance = settings.camera_distance

        # Generate camera positions
        hotspots = []
        if settings.use_adaptive_capture and settings.adaptive_use_hotspots and self._adaptive_result:
            hotspots = self._adaptive_result.detail_hotspots

        points = generate_camera_positions(
            settings.camera_distribution,
            actual_camera_count,
            min_elevation=settings.min_elevation,
            max_elevation=settings.max_elevation,
            ring_count=settings.ring_count,
            hotspots=hotspots,
            hotspot_bias=settings.adaptive_hotspot_bias if settings.use_adaptive_capture else 0.0
        )

        # Create cameras
        self._cameras = []
        for i, point in enumerate(points):
            cam_pos = center + point * distance
            cam = create_camera_at_position(context, cam_pos, center, f"GS_Cam_{i:04d}")
            cam.data.lens = settings.focal_length
            self._cameras.append(cam)

        # Reconcile checkpoint camera count against current cameras
        if checkpoint:
            checkpoint = self._reconcile_checkpoint_camera_count(checkpoint, len(self._cameras))

        # Determine which images need rendering
        if checkpoint and checkpoint.get('completed_images'):
            completed = set(checkpoint['completed_images'])
            self._images_to_render = [i for i in range(len(self._cameras)) if i not in completed]
            self._checkpoint_data = checkpoint
        else:
            self._images_to_render = list(range(len(self._cameras)))
            self._checkpoint_data = create_checkpoint(
                0, len(self._cameras),
                settings_hash=settings_hash,
                settings_hash_legacy=legacy_settings_hash,
                completed_images=[]
            )

        if self._checkpoints_enabled:
            self._checkpoint_writer = _AsyncCheckpointWriter(self._output_path)

        # Setup lighting
        if settings.lighting_mode != 'KEEP':
            warning = setup_neutral_lighting(context, settings)
            if warning:
                self.report({'WARNING'}, warning)

        # Override materials if requested
        if settings.material_mode != 'ORIGINAL':
            self._original_materials = override_materials(selected, settings.material_mode)

        # Use Blender's native render settings directly (no duplication)
        # User sets resolution, samples, engine, file format in Blender's Render Properties
        scene = context.scene
        rd = scene.render

        self._image_path_prefix = os.path.join(self._images_path, "image_")
        self._image_path_suffix = f".{image_ext}"
        self._mask_path_prefix = os.path.join(self._masks_path, "mask_")
        self._mask_path_suffix = ".png"
        self._mask_path_prefix_gsl = os.path.join(self._masks_path, "image_")
        self._mask_path_suffix_gsl = f".{image_ext}.png"
        self._save_manually = False

        # Ensure RGBA mode for PNG
        try:
            if self._target_format == 'PNG':
                rd.image_settings.color_mode = 'RGBA'

            # Handle transparent background
            self._original_film_transparent = rd.film_transparent
            if settings.transparent_background and self._target_format in ('PNG', 'OPEN_EXR', 'OPEN_EXR_MULTILAYER'):
                rd.film_transparent = True
        except TypeError:
            self._save_manually = True
            self.report({'INFO'}, "Scene in video mode - will save images manually")

        # Setup compositor for depth/normal/mask exports (not needed for alpha masks)
        needs_compositor = (
            settings.export_depth or
            settings.export_normals or
            (settings.export_masks and settings.mask_source == 'OBJECT_INDEX')
        )
        if needs_compositor:
            self._setup_compositor_outputs(context, settings)

        # Start rendering
        self._current_camera_index = 0
        self._render_in_progress = False
        self._active_camera_actual_index = -1
        self._active_image_path = ""
        self._active_needs_alpha_mask = False
        settings.is_rendering = True
        settings.render_progress = 0.0

        # Initialize extended progress tracking
        total_to_render = len(self._images_to_render)
        settings.capture_current = 0
        settings.capture_total = total_to_render
        settings.capture_start_time = time.time()
        settings.capture_elapsed_seconds = 0.0
        settings.capture_eta_seconds = 0.0
        settings.capture_rate = 0.0
        settings.capture_current_camera = ""
        settings.capture_current_object = ", ".join([obj.name for obj in self._target_objects[:3]])
        if len(self._target_objects) > 3:
            settings.capture_current_object += f" (+{len(self._target_objects) - 3} more)"

        # Add timer for modal
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)

        # Start built-in progress bar
        wm.progress_begin(0, total_to_render)

        return {'RUNNING_MODAL'}

    def _is_render_job_running(self):
        """Best-effort check whether Blender currently runs a render job."""
        is_job_running = getattr(bpy.app, "is_job_running", None)
        if callable(is_job_running):
            try:
                return bool(is_job_running("RENDER"))
            except Exception:
                return False
        return False

    def _request_render_cancel(self):
        """Request cancellation of the active render job."""
        try:
            bpy.ops.render.cancel()
        except Exception:
            pass

    def _start_async_render(self, context, settings):
        """Start rendering the current camera asynchronously."""
        actual_index = self._images_to_render[self._current_camera_index]
        cam = self._cameras[actual_index]
        context.scene.camera = cam

        # Update output filenames for compositor nodes.
        self._update_output_filenames(actual_index)

        image_path = f"{self._image_path_prefix}{actual_index:04d}{self._image_path_suffix}"
        need_alpha_mask = settings.export_masks and settings.mask_source == 'ALPHA'

        self._active_camera_actual_index = actual_index
        self._active_image_path = image_path
        self._active_needs_alpha_mask = need_alpha_mask

        try:
            if need_alpha_mask or self._save_manually:
                result = bpy.ops.render.render('INVOKE_DEFAULT')
            else:
                context.scene.render.filepath = image_path
                result = bpy.ops.render.render('INVOKE_DEFAULT', write_still=True)
        except Exception as exc:
            self._export_errors = True
            self.report({'ERROR'}, f"Failed to start render {actual_index}: {exc}")
            return False

        if 'CANCELLED' in set(result):
            self._export_errors = True
            self.report({'ERROR'}, f"Render {actual_index} was cancelled before start")
            return False

        self._render_in_progress = True
        settings.capture_current_camera = cam.name if cam else ""
        return True

    def _finalize_completed_render(self, context, settings):
        """Finalize outputs for the render that just completed."""
        actual_index = self._active_camera_actual_index
        image_path = self._active_image_path
        need_alpha_mask = self._active_needs_alpha_mask

        save_success = False
        mask_ok = True

        try:
            # For alpha/manual modes we save the render result explicitly after render completes.
            if need_alpha_mask or self._save_manually:
                render_result = bpy.data.images.get('Render Result')
                if not render_result:
                    self._export_errors = True
                    self.report({'ERROR'}, "Render Result not available after render")
                    return False

                render_result.save_render(filepath=image_path)

                if need_alpha_mask:
                    if settings.mask_format == 'GSL':
                        mask_path = f"{self._mask_path_prefix_gsl}{actual_index:04d}{self._mask_path_suffix_gsl}"
                    else:
                        mask_path = f"{self._mask_path_prefix}{actual_index:04d}{self._mask_path_suffix}"
                    try:
                        mask_success, mask_error = extract_alpha_mask(image_path, mask_path)
                        if not mask_success:
                            mask_ok = False
                            if mask_error:
                                self.report({'WARNING'}, mask_error)
                    except Exception as exc:
                        mask_ok = False
                        self._export_errors = True
                        self.report({'ERROR'}, f"Failed to save mask {actual_index}: {exc}")

            # Verify file was written successfully.
            image_ok = os.path.exists(image_path) and os.path.getsize(image_path) > 0
            if not image_ok:
                self._export_errors = True
                self.report({'WARNING'}, f"Image {actual_index} may not have saved correctly")
            save_success = image_ok and mask_ok

            # Verify additional outputs written by compositor (depth/normals/masks).
            if save_success and settings.export_depth:
                depth_ext = self._depth_output_extension()
                depth_ok = self._verify_output_file(
                    self._depth_path,
                    f"depth_{actual_index:04d}",
                    depth_ext
                )
                if not depth_ok:
                    save_success = False
                    self._export_errors = True
                    self.report({'ERROR'}, f"Depth map {actual_index} may not have saved correctly")

            if save_success and settings.export_normals:
                normal_ok = self._verify_output_file(
                    self._normals_path,
                    f"normal_{actual_index:04d}",
                    "exr"
                )
                if not normal_ok:
                    save_success = False
                    self._export_errors = True
                    self.report({'ERROR'}, f"Normal map {actual_index} may not have saved correctly")

            if save_success and settings.export_masks and settings.mask_source == 'OBJECT_INDEX':
                mask_ext = self._object_index_mask_extension()
                mask_ok = self._verify_output_file(
                    self._masks_path,
                    f"mask_{actual_index:04d}",
                    mask_ext
                )
                if not mask_ok:
                    save_success = False
                    self._export_errors = True
                    self.report({'ERROR'}, f"Mask {actual_index} may not have saved correctly")

        except Exception as exc:
            self._export_errors = True
            self.report({'ERROR'}, f"Failed to finalize image {actual_index}: {exc}")
            save_success = False

        finally:
            self._render_in_progress = False

        return save_success

    def modal(self, context, event):
        settings = context.scene.gs_capture_settings

        if settings.cancel_requested:
            # If a render job is active, cancel it first and wait for Blender
            # to finish unwinding before cleaning up operator state.
            if self._render_in_progress and self._is_render_job_running():
                settings.current_render_info = "Cancelling current render..."
                self._request_render_cancel()
                return {'RUNNING_MODAL'}
            settings.cancel_requested = False
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'ESC':
            settings.cancel_requested = True
            if self._render_in_progress and self._is_render_job_running():
                settings.current_render_info = "Cancelling current render..."
                self._request_render_cancel()
                return {'RUNNING_MODAL'}
            settings.cancel_requested = False
            self.cancel(context)
            return {'CANCELLED'}

        # Do not swallow non-timer UI events while capture is running.
        # This keeps the sidebar responsive (e.g. cancel button clicks).
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        self._drain_checkpoint_writer_errors()

        if self._current_camera_index >= len(self._images_to_render):
            self.finish(context)
            return {'FINISHED'}

        # If a render is active, wait for it to finish (or be canceled).
        if self._render_in_progress:
            if self._is_render_job_running():
                return {'RUNNING_MODAL'}

            # Render completed. If user requested cancel while render was active,
            # honor it before finalizing this frame as completed.
            if settings.cancel_requested:
                settings.cancel_requested = False
                self.cancel(context)
                return {'CANCELLED'}

            actual_index = self._active_camera_actual_index
            cam = self._cameras[actual_index] if 0 <= actual_index < len(self._cameras) else None
            save_success = self._finalize_completed_render(context, settings)

            # Only update checkpoint if save was successful.
            if save_success:
                self._checkpoint_data['completed_images'].append(actual_index)
                self._checkpoint_data['current_index'] = self._current_camera_index

            if self._checkpoints_enabled:
                if (self._current_camera_index + 1) % settings.checkpoint_interval == 0:
                    if self._checkpoint_writer:
                        self._checkpoint_writer.request_save(self._checkpoint_data)
                    else:
                        success, error = save_checkpoint(self._output_path, self._checkpoint_data)
                        if not success and error:
                            self.report({'WARNING'}, error)

            # Update progress.
            self._current_camera_index += 1
            total_to_render = len(self._images_to_render)
            settings.render_progress = (self._current_camera_index / total_to_render) * 100
            settings.current_render_info = f"Rendering {self._current_camera_index}/{total_to_render}"

            # Update extended progress tracking.
            settings.capture_current = self._current_camera_index
            elapsed = time.time() - settings.capture_start_time
            settings.capture_elapsed_seconds = elapsed

            if self._current_camera_index > 0:
                settings.capture_rate = self._current_camera_index / elapsed
                remaining = total_to_render - self._current_camera_index
                settings.capture_eta_seconds = remaining / settings.capture_rate if settings.capture_rate > 0 else 0

            settings.capture_current_camera = cam.name if cam else ""

            # Update built-in progress bar.
            context.window_manager.progress_update(self._current_camera_index)

            # Force UI redraw.
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Start next camera render asynchronously.
        total_to_render = len(self._images_to_render)
        settings.current_render_info = f"Rendering {self._current_camera_index + 1}/{total_to_render}"
        if not self._start_async_render(context, settings):
            settings.cancel_requested = False
            self.cancel(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def finish(self, context):
        settings = context.scene.gs_capture_settings
        rd = context.scene.render
        export_failures = []

        # Export camera data using Blender's actual resolution
        image_ext = self._image_ext or 'png'
        if settings.export_colmap:
            colmap_path = os.path.join(self._output_path, "sparse", "0")
            is_valid, _, error = validate_path_length(colmap_path)
            if not is_valid:
                export_failures.append(f"COLMAP export failed: path is too long for Windows. {error}")
            else:
                if not self._ensure_directory(colmap_path, "COLMAP output directory"):
                    export_failures.append("COLMAP export failed: unable to create output directory.")
                else:
                    try:
                        _, warning = export_colmap_cameras(
                            self._cameras, colmap_path,
                            rd.resolution_x, rd.resolution_y,
                            image_ext=image_ext,
                            export_binary=settings.export_colmap_binary,
                            initial_point_count=settings.colmap_initial_point_count,
                            point_sampling_mode=settings.colmap_point_sampling,
                            target_objects=self._target_objects,
                        )
                        if warning:
                            self.report({'WARNING'}, warning)
                    except Exception as e:
                        export_failures.append(f"COLMAP export failed: {e}")

        if settings.export_transforms_json:
            depth_ext = self._depth_output_extension()
            if settings.export_masks and settings.mask_source == 'OBJECT_INDEX':
                mask_ext = self._object_index_mask_extension()
            else:
                mask_ext = 'png'
            try:
                export_transforms_json(
                    self._cameras, self._output_path,
                    rd.resolution_x, rd.resolution_y,
                    include_depth=settings.export_depth,
                    include_masks=settings.export_masks,
                    image_ext=image_ext,
                    depth_ext=depth_ext,
                    mask_ext=mask_ext,
                    mask_format=settings.mask_format
                )
            except Exception as e:
                export_failures.append(f"transforms.json export failed: {e}")

        if export_failures:
            self._export_errors = True
            for message in export_failures:
                self.report({'ERROR'}, message)

        post_result, artifact_checks = self._build_post_export_validation(settings, image_ext)
        if post_result.issues:
            self._report_validation_issues(post_result)
            if not post_result.can_proceed:
                self._export_errors = True

        try:
            report_path = self._write_validation_report(
                settings,
                image_ext,
                export_failures,
                post_result,
                artifact_checks,
            )
            self.report({'INFO'}, f"Validation report written: {report_path}")
        except Exception as exc:
            self.report({'WARNING'}, f"Failed to write validation report: {exc}")

        if self._checkpoints_enabled:
            self._shutdown_checkpoint_writer()

        # Clear checkpoint on success
        if self._checkpoints_enabled and not self._export_errors:
            success, error = clear_checkpoint(self._output_path)
            if not success and error:
                self.report({'WARNING'}, error)

        # Store last capture stats before cleanup
        elapsed = time.time() - settings.capture_start_time
        settings.last_capture_images = len(self._cameras)
        settings.last_capture_duration = elapsed
        settings.last_capture_path = self._output_path
        settings.last_capture_success = not self._export_errors

        # End built-in progress bar
        context.window_manager.progress_end()

        # Cleanup
        self.cleanup(context)

        self._render_in_progress = False
        settings.is_rendering = False
        settings.cancel_requested = False
        settings.render_progress = 100.0
        settings.current_render_info = "Complete!"

        if self._export_errors:
            self.report({'WARNING'}, f"Capture completed with errors. Output in {self._output_path}")
        else:
            self.report({'INFO'}, f"Captured {len(self._cameras)} images to {self._output_path}")

    def cleanup(self, context):
        """Restore original state and cleanup."""
        settings = context.scene.gs_capture_settings

        self._shutdown_checkpoint_writer()

        # Remove timer
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        # Restore camera
        if self._original_camera:
            context.scene.camera = self._original_camera

        # Restore lighting
        restore_lighting(context, self._original_light_states)

        # Restore materials
        if self._original_materials:
            restore_materials(self._original_materials)

        # Restore object visibility
        self._restore_object_visibility(context)

        # Restore file format
        if self._original_file_format:
            try:
                context.scene.render.image_settings.file_format = self._original_file_format
            except Exception as e:
                print(f"Warning: Could not restore file format: {e}")

        # Restore film transparent
        context.scene.render.film_transparent = self._original_film_transparent

        # Restore render engine and samples
        if self._original_engine:
            context.scene.render.engine = self._original_engine
        if self._original_eevee_samples and hasattr(context.scene.eevee, 'taa_render_samples'):
            context.scene.eevee.taa_render_samples = self._original_eevee_samples
        if self._original_cycles_samples and hasattr(context.scene, 'cycles'):
            context.scene.cycles.samples = self._original_cycles_samples

        # Shared restore path for both finish() and cancel().
        self._restore_target_object_pass_indices(context)
        self._restore_view_layer_pass_flags(context)

        # Delete GS cameras
        delete_gs_cameras(context)

        # Cleanup compositor nodes
        cleanup_compositor_nodes(context)

    def cancel(self, context):
        """Handle cancellation."""
        settings = context.scene.gs_capture_settings

        # Save checkpoint before cancelling
        if self._checkpoints_enabled and self._checkpoint_data:
            if self._checkpoint_writer:
                self._checkpoint_writer.request_save(self._checkpoint_data)
                self._shutdown_checkpoint_writer()
            else:
                success, error = save_checkpoint(self._output_path, self._checkpoint_data)
                if not success and error:
                    self.report({'WARNING'}, error)
            self.report({'WARNING'},
                        f"Cancelled. Progress saved ({len(self._checkpoint_data.get('completed_images', []))} images)")

        # Store partial capture stats
        elapsed = time.time() - settings.capture_start_time if settings.capture_start_time > 0 else 0
        settings.last_capture_images = settings.capture_current
        settings.last_capture_duration = elapsed
        settings.last_capture_path = self._output_path
        settings.last_capture_success = False

        # End built-in progress bar
        context.window_manager.progress_end()

        self.cleanup(context)
        self._render_in_progress = False
        settings.is_rendering = False
        settings.cancel_requested = False
        settings.current_render_info = "Cancelled"
        
        # Reset progress tracking
        settings.capture_current = 0
        settings.capture_eta_seconds = 0
        settings.capture_rate = 0


class GSCAPTURE_OT_cancel_capture(Operator):
    """Cancel the current capture session."""
    bl_idname = "gs_capture.cancel_capture"
    bl_label = "Cancel Capture"
    bl_description = "Cancel the active capture session"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return context.scene.gs_capture_settings.is_rendering

    def execute(self, context):
        settings = context.scene.gs_capture_settings
        if settings.cancel_requested:
            self.report({'INFO'}, "Cancel already requested")
            return {'CANCELLED'}

        settings.cancel_requested = True
        settings.current_render_info = "Cancelling..."

        # If Blender is currently rendering, request immediate render cancel too.
        try:
            is_job_running = getattr(bpy.app, "is_job_running", None)
            if callable(is_job_running) and is_job_running("RENDER"):
                bpy.ops.render.cancel()
        except Exception:
            pass

        self.report({'INFO'}, "Cancel requested")
        return {'FINISHED'}


class GSCAPTURE_OT_capture_collection(Operator):
    """Capture all objects in a collection as one scene."""
    bl_idname = "gs_capture.capture_collection"
    bl_label = "Capture Collection"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.gs_capture_settings

        # Get target collection
        if settings.target_collection:
            collection = bpy.data.collections.get(settings.target_collection)
        else:
            collection = context.view_layer.active_layer_collection.collection

        if not collection:
            self.report({'ERROR'}, "No collection specified")
            return {'CANCELLED'}

        # Get all mesh objects in collection
        objects = []
        if settings.include_nested_collections:
            objects = self._get_all_meshes_recursive(collection)
        else:
            objects = [obj for obj in collection.objects if obj.type == 'MESH']

        if not objects:
            self.report({'ERROR'}, "No mesh objects in collection")
            return {'CANCELLED'}

        # Select the objects and run main capture
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects:
            obj.select_set(True)

        return bpy.ops.gs_capture.capture_selected()

    def _get_all_meshes_recursive(self, collection):
        """Get all mesh objects from collection and children."""
        meshes = [obj for obj in collection.objects if obj.type == 'MESH']
        for child in collection.children:
            meshes.extend(self._get_all_meshes_recursive(child))
        return meshes
