"""
Main capture operators for rendering multi-view images.
Includes checkpoint support, depth/normal/mask export.
"""

import bpy
import os
import numpy as np
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
)
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
    clear_checkpoint,
    create_checkpoint,
    get_missing_images,
    calculate_settings_hash,
)


class GSCAPTURE_OT_capture_selected(Operator):
    """Capture images of selected objects for Gaussian Splatting training."""
    bl_idname = "gs_capture.capture_selected"
    bl_label = "Capture Selected"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _cameras = []
    _current_camera_index = 0
    _output_path = ""
    _original_camera = None
    _original_light_states = {}
    _original_materials = {}
    _original_file_format = None
    _original_hide_render = {}
    _original_film_transparent = False
    _target_objects = []
    _adaptive_result = None
    _target_format = 'PNG'
    _save_manually = False
    _checkpoint_data = None
    _images_to_render = []  # Indices of images that need rendering
    _depth_output_node = None
    _normal_output_node = None
    _mask_output_node = None

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

    def _setup_compositor_outputs(self, context, settings):
        """Setup compositor nodes for depth, normal, and mask outputs."""
        scene = context.scene
        scene.use_nodes = True
        tree = scene.node_tree

        # Get or create render layers node
        render_layers = None
        for node in tree.nodes:
            if node.type == 'R_LAYERS':
                render_layers = node
                break
        if not render_layers:
            render_layers = tree.nodes.new('CompositorNodeRLayers')
            render_layers.location = (0, 0)

        # Enable required passes
        view_layer = scene.view_layers["ViewLayer"]

        if settings.export_depth:
            view_layer.use_pass_z = True
            self._setup_depth_output(tree, render_layers)

        if settings.export_normals:
            view_layer.use_pass_normal = True
            self._setup_normal_output(tree, render_layers)

        if settings.export_masks:
            view_layer.use_pass_object_index = True
            # Set object indices
            for i, obj in enumerate(self._target_objects):
                obj.pass_index = i + 1
            self._setup_mask_output(tree, render_layers)

    def _setup_depth_output(self, tree, render_layers):
        """Setup depth map output node."""
        # Normalize node
        normalize = tree.nodes.new('CompositorNodeNormalize')
        normalize.name = "GS_Depth_Normalize"
        normalize.location = (300, -100)

        # File output
        file_output = tree.nodes.new('CompositorNodeOutputFile')
        file_output.name = "GS_Depth_Output"
        file_output.location = (500, -100)
        file_output.base_path = os.path.join(self._output_path, "depth")
        file_output.format.file_format = 'PNG'
        file_output.format.color_mode = 'BW'
        file_output.format.color_depth = '16'

        # Connect
        tree.links.new(render_layers.outputs['Depth'], normalize.inputs[0])
        tree.links.new(normalize.outputs[0], file_output.inputs[0])

        self._depth_output_node = file_output

    def _setup_normal_output(self, tree, render_layers):
        """Setup normal map output node."""
        # File output for normals
        file_output = tree.nodes.new('CompositorNodeOutputFile')
        file_output.name = "GS_Normal_Output"
        file_output.location = (500, -250)
        file_output.base_path = os.path.join(self._output_path, "normals")
        file_output.format.file_format = 'OPEN_EXR'
        file_output.format.color_mode = 'RGB'

        # Connect normal output
        tree.links.new(render_layers.outputs['Normal'], file_output.inputs[0])

        self._normal_output_node = file_output

    def _setup_mask_output(self, tree, render_layers):
        """Setup object mask output node."""
        # ID Mask node
        id_mask = tree.nodes.new('CompositorNodeIDMask')
        id_mask.name = "GS_ID_Mask"
        id_mask.location = (300, -400)
        id_mask.index = 1
        id_mask.use_antialiasing = True

        # File output for mask
        file_output = tree.nodes.new('CompositorNodeOutputFile')
        file_output.name = "GS_Mask_Output"
        file_output.location = (500, -400)
        file_output.base_path = os.path.join(self._output_path, "masks")
        file_output.format.file_format = 'PNG'
        file_output.format.color_mode = 'BW'

        # Connect
        tree.links.new(render_layers.outputs['IndexOB'], id_mask.inputs[0])
        tree.links.new(id_mask.outputs[0], file_output.inputs[0])

        self._mask_output_node = file_output

    def _update_output_filenames(self, index):
        """Update compositor output node filenames for current frame."""
        if self._depth_output_node:
            self._depth_output_node.file_slots[0].path = f"depth_{index:04d}"

        if self._normal_output_node:
            self._normal_output_node.file_slots[0].path = f"normal_{index:04d}"

        if self._mask_output_node:
            self._mask_output_node.file_slots[0].path = f"mask_{index:04d}"

    def execute(self, context):
        settings = context.scene.gs_capture_settings

        # Get selected objects
        selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not selected:
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}

        self._target_objects = selected

        # Setup output directory
        self._output_path = bpy.path.abspath(settings.output_path)
        images_path = os.path.join(self._output_path, "images")
        os.makedirs(images_path, exist_ok=True)

        # Create additional directories for exports
        if settings.export_depth:
            os.makedirs(os.path.join(self._output_path, "depth"), exist_ok=True)
        if settings.export_normals:
            os.makedirs(os.path.join(self._output_path, "normals"), exist_ok=True)
        if settings.export_masks:
            os.makedirs(os.path.join(self._output_path, "masks"), exist_ok=True)

        # Check for checkpoint resume
        settings_hash = calculate_settings_hash(settings, context.scene)
        checkpoint = None

        if settings.enable_checkpoints and settings.auto_resume:
            checkpoint = load_checkpoint(self._output_path)
            if checkpoint:
                # Validate checkpoint
                if checkpoint.get('settings_hash') == settings_hash:
                    self.report({'INFO'}, f"Resuming from image {checkpoint['current_index']}")
                else:
                    self.report({'WARNING'}, "Settings changed, starting fresh")
                    checkpoint = None

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
                completed_images=[]
            )

        # Setup lighting
        if settings.lighting_mode != 'KEEP':
            setup_neutral_lighting(context, settings)

        # Override materials if requested
        if settings.material_mode != 'ORIGINAL':
            self._original_materials = override_materials(selected, settings.material_mode)

        # Use Blender's native render settings directly (no duplication)
        # User sets resolution, samples, engine, file format in Blender's Render Properties
        scene = context.scene
        rd = scene.render

        # Store original file format for potential restoration
        self._original_file_format = rd.image_settings.file_format
        self._target_format = rd.image_settings.file_format
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

        # Setup compositor for depth/normal/mask exports
        if settings.export_depth or settings.export_normals or settings.export_masks:
            self._setup_compositor_outputs(context, settings)

        # Start rendering
        self._current_camera_index = 0
        settings.is_rendering = True
        settings.render_progress = 0.0

        # Add timer for modal
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        settings = context.scene.gs_capture_settings

        if event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            if self._current_camera_index >= len(self._images_to_render):
                self.finish(context)
                return {'FINISHED'}

            # Get actual camera index from render list
            actual_index = self._images_to_render[self._current_camera_index]
            cam = self._cameras[actual_index]
            context.scene.camera = cam

            # Update output filenames for compositor nodes
            self._update_output_filenames(actual_index)

            # Set output path with correct extension for file format
            format_to_ext = {
                'PNG': 'png',
                'JPEG': 'jpg',
                'OPEN_EXR': 'exr',
                'OPEN_EXR_MULTILAYER': 'exr',
                'TARGA': 'tga',
                'TARGA_RAW': 'tga',
                'BMP': 'bmp',
                'TIFF': 'tiff',
            }
            ext = format_to_ext.get(self._target_format, 'png')
            image_path = os.path.join(
                self._output_path, "images",
                f"image_{actual_index:04d}.{ext}"
            )

            if self._save_manually:
                bpy.ops.render.render()
                render_result = bpy.data.images.get('Render Result')
                if render_result:
                    render_result.save_render(filepath=image_path)
            else:
                context.scene.render.filepath = image_path
                bpy.ops.render.render(write_still=True)

            # Update checkpoint
            self._checkpoint_data['completed_images'].append(actual_index)
            self._checkpoint_data['current_index'] = self._current_camera_index

            if settings.enable_checkpoints:
                if (self._current_camera_index + 1) % settings.checkpoint_interval == 0:
                    save_checkpoint(self._output_path, self._checkpoint_data)

            # Update progress
            self._current_camera_index += 1
            total_to_render = len(self._images_to_render)
            settings.render_progress = (self._current_camera_index / total_to_render) * 100
            settings.current_render_info = f"Rendering {self._current_camera_index}/{total_to_render}"

        return {'RUNNING_MODAL'}

    def finish(self, context):
        settings = context.scene.gs_capture_settings
        rd = context.scene.render

        # Export camera data using Blender's actual resolution
        if settings.export_colmap:
            colmap_path = os.path.join(self._output_path, "sparse", "0")
            os.makedirs(colmap_path, exist_ok=True)
            export_colmap_cameras(
                self._cameras, colmap_path,
                rd.resolution_x, rd.resolution_y
            )

        if settings.export_transforms_json:
            export_transforms_json(
                self._cameras, self._output_path,
                rd.resolution_x, rd.resolution_y,
                include_depth=settings.export_depth,
                include_masks=settings.export_masks
            )

        # Clear checkpoint on success
        if settings.enable_checkpoints:
            clear_checkpoint(self._output_path)

        # Cleanup
        self.cleanup(context)

        settings.is_rendering = False
        settings.render_progress = 100.0
        settings.current_render_info = "Complete!"

        self.report({'INFO'}, f"Captured {len(self._cameras)} images to {self._output_path}")

    def cleanup(self, context):
        """Restore original state and cleanup."""
        settings = context.scene.gs_capture_settings

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
            except:
                pass

        # Restore film transparent
        context.scene.render.film_transparent = self._original_film_transparent

        # Delete GS cameras
        delete_gs_cameras(context)

        # Cleanup compositor nodes
        cleanup_compositor_nodes(context)

    def cancel(self, context):
        """Handle cancellation."""
        settings = context.scene.gs_capture_settings

        # Save checkpoint before cancelling
        if settings.enable_checkpoints and self._checkpoint_data:
            save_checkpoint(self._output_path, self._checkpoint_data)
            self.report({'WARNING'},
                        f"Cancelled. Progress saved ({len(self._checkpoint_data.get('completed_images', []))} images)")

        self.cleanup(context)
        settings.is_rendering = False
        settings.current_render_info = "Cancelled"


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
