"""
Batch processing operators for capturing multiple objects/collections.
"""

import bpy
import os
from bpy.types import Operator


class GSCAPTURE_OT_batch_capture(Operator):
    """Run batch capture based on batch mode settings."""
    bl_idname = "gs_capture.batch_capture"
    bl_label = "Batch Capture"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.gs_capture_settings
        original_output = settings.output_path

        items_to_capture = []

        if settings.batch_mode == 'SELECTED':
            # Capture all selected as one
            selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
            if selected:
                items_to_capture.append(('selected', selected))

        elif settings.batch_mode == 'EACH_SELECTED':
            # Capture each selected object individually
            for obj in context.selected_objects:
                if obj.type == 'MESH':
                    items_to_capture.append((obj.name, [obj]))

        elif settings.batch_mode == 'COLLECTION':
            # Capture specified collection
            if settings.target_collection:
                collection = bpy.data.collections.get(settings.target_collection)
                if collection:
                    meshes = self._get_collection_meshes(collection, settings.include_nested_collections)
                    if meshes:
                        items_to_capture.append((collection.name, meshes))

        elif settings.batch_mode == 'COLLECTIONS':
            # Capture each collection separately
            for collection in bpy.data.collections:
                meshes = self._get_collection_meshes(collection, settings.include_nested_collections)
                if meshes:
                    items_to_capture.append((collection.name, meshes))

        elif settings.batch_mode == 'GROUPS':
            # Use custom object groups
            for group in settings.object_groups:
                meshes = [item.obj for item in group.objects if item.obj and item.obj.type == 'MESH']
                if meshes:
                    items_to_capture.append((group.name, meshes))

        if not items_to_capture:
            self.report({'ERROR'}, "No items to capture")
            return {'CANCELLED'}

        # Process each item
        completed = 0
        failed = 0

        for name, objects in items_to_capture:
            # Set output path for this item
            settings.output_path = os.path.join(bpy.path.abspath(original_output), name)

            # Select objects
            bpy.ops.object.select_all(action='DESELECT')
            for obj in objects:
                obj.select_set(True)

            # Run capture
            try:
                result = bpy.ops.gs_capture.capture_selected()
                if result == {'FINISHED'}:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                self.report({'WARNING'}, f"Failed to capture {name}: {e}")
                failed += 1

        # Restore original output path
        settings.output_path = original_output

        self.report({'INFO'}, f"Batch complete: {completed} succeeded, {failed} failed")
        return {'FINISHED'}

    def _get_collection_meshes(self, collection, include_nested):
        """Get mesh objects from collection."""
        meshes = [obj for obj in collection.objects if obj.type == 'MESH']
        if include_nested:
            for child in collection.children:
                meshes.extend(self._get_collection_meshes(child, True))
        return meshes


class GSCAPTURE_OT_add_object_group(Operator):
    """Add a new object group for batch capture."""
    bl_idname = "gs_capture.add_object_group"
    bl_label = "Add Group"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.gs_capture_settings
        group = settings.object_groups.add()
        group.name = f"Group {len(settings.object_groups)}"
        settings.active_group_index = len(settings.object_groups) - 1
        return {'FINISHED'}


class GSCAPTURE_OT_remove_object_group(Operator):
    """Remove the active object group."""
    bl_idname = "gs_capture.remove_object_group"
    bl_label = "Remove Group"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.gs_capture_settings
        if settings.object_groups:
            settings.object_groups.remove(settings.active_group_index)
            settings.active_group_index = max(0, settings.active_group_index - 1)
        return {'FINISHED'}


class GSCAPTURE_OT_add_to_group(Operator):
    """Add selected objects to the active group."""
    bl_idname = "gs_capture.add_to_group"
    bl_label = "Add to Group"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.gs_capture_settings

        if not settings.object_groups:
            self.report({'ERROR'}, "No groups exist")
            return {'CANCELLED'}

        group = settings.object_groups[settings.active_group_index]

        for obj in context.selected_objects:
            if obj.type == 'MESH':
                # Check if already in group
                if not any(item.obj == obj for item in group.objects):
                    item = group.objects.add()
                    item.obj = obj

        return {'FINISHED'}


class GSCAPTURE_OT_remove_from_group(Operator):
    """Remove object from group."""
    bl_idname = "gs_capture.remove_from_group"
    bl_label = "Remove from Group"
    bl_options = {'REGISTER', 'UNDO'}

    index: bpy.props.IntProperty()

    def execute(self, context):
        settings = context.scene.gs_capture_settings

        if settings.object_groups:
            group = settings.object_groups[settings.active_group_index]
            if 0 <= self.index < len(group.objects):
                group.objects.remove(self.index)

        return {'FINISHED'}
