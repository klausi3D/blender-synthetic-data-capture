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

    _timer = None
    _items_to_capture = None
    _current_index = 0
    _original_output = ""
    _original_selection = None
    _original_active = None
    _waiting_for_capture = False
    _completed = 0
    _failed = 0
    _cancel_batch = False

    @classmethod
    def poll(cls, context):
        return not context.scene.gs_capture_settings.is_rendering

    def execute(self, context):
        settings = context.scene.gs_capture_settings

        self._items_to_capture = self._build_items_to_capture(context, settings)
        if not self._items_to_capture:
            self.report({'ERROR'}, "No items to capture")
            return {'CANCELLED'}

        self._original_output = settings.output_path
        self._original_selection = list(context.selected_objects)
        self._original_active = context.view_layer.objects.active
        self._current_index = -1
        self._waiting_for_capture = False
        self._completed = 0
        self._failed = 0
        self._cancel_batch = False

        if not self._start_next_capture(context):
            self._restore_state(context)
            return {'CANCELLED'}

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        settings = context.scene.gs_capture_settings

        if event.type == 'ESC':
            self._cancel_batch = True
            if settings.is_rendering:
                settings.cancel_requested = True
            return {'RUNNING_MODAL'}

        if event.type == 'TIMER':
            if self._cancel_batch:
                if settings.is_rendering:
                    return {'RUNNING_MODAL'}
                self._finish(context, cancelled=True)
                return {'CANCELLED'}

            if self._waiting_for_capture and not settings.is_rendering:
                if settings.last_capture_success:
                    self._completed += 1
                else:
                    self._failed += 1

                self._waiting_for_capture = False

                if not self._start_next_capture(context):
                    self._finish(context)
                    self.report({'INFO'}, f"Batch complete: {self._completed} succeeded, {self._failed} failed")
                    return {'FINISHED'}

        return {'PASS_THROUGH'}

    def _build_items_to_capture(self, context, settings):
        items_to_capture = []

        if settings.batch_mode == 'SCENE':
            all_meshes = [obj for obj in context.scene.objects
                          if obj.type == 'MESH' and obj.visible_get()]
            if all_meshes:
                items_to_capture.append(('scene', all_meshes))

        elif settings.batch_mode == 'SELECTED':
            selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
            if selected:
                items_to_capture.append(('selected', selected))

        elif settings.batch_mode == 'EACH_SELECTED':
            for obj in context.selected_objects:
                if obj.type == 'MESH':
                    items_to_capture.append((obj.name, [obj]))

        elif settings.batch_mode == 'COLLECTION':
            if settings.target_collection:
                collection = bpy.data.collections.get(settings.target_collection)
                if collection:
                    meshes = self._get_collection_meshes(collection, settings.include_nested_collections)
                    if meshes:
                        items_to_capture.append((collection.name, meshes))

        elif settings.batch_mode == 'COLLECTIONS':
            for collection in bpy.data.collections:
                meshes = self._get_collection_meshes(collection, settings.include_nested_collections)
                if meshes:
                    items_to_capture.append((collection.name, meshes))

        elif settings.batch_mode == 'GROUPS':
            for group in settings.object_groups:
                meshes = [item.obj for item in group.objects if item.obj and item.obj.type == 'MESH']
                if meshes:
                    items_to_capture.append((group.name, meshes))

        return items_to_capture

    def _start_next_capture(self, context):
        settings = context.scene.gs_capture_settings
        self._current_index += 1

        if self._current_index >= len(self._items_to_capture):
            return False

        name, objects = self._items_to_capture[self._current_index]

        settings.output_path = os.path.join(bpy.path.abspath(self._original_output), name)

        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects:
            obj.select_set(True)
        if objects:
            context.view_layer.objects.active = objects[0]

        try:
            result = bpy.ops.gs_capture.capture_selected()
        except Exception as e:
            self.report({'WARNING'}, f"Failed to capture {name}: {e}")
            self._failed += 1
            return self._start_next_capture(context)

        if result == {'CANCELLED'}:
            self._failed += 1
            return self._start_next_capture(context)

        self._waiting_for_capture = True
        return True

    def _finish(self, context, cancelled=False):
        self._restore_state(context)
        if cancelled:
            self.report({'WARNING'}, "Batch capture cancelled")

    def _restore_state(self, context):
        settings = context.scene.gs_capture_settings

        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        settings.output_path = self._original_output

        bpy.ops.object.select_all(action='DESELECT')
        if self._original_selection:
            for obj in self._original_selection:
                if obj and obj.name in bpy.data.objects:
                    obj.select_set(True)

        if self._original_active and self._original_active.name in bpy.data.objects:
            context.view_layer.objects.active = self._original_active

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

        if not (0 <= settings.active_group_index < len(settings.object_groups)):
            self.report({'ERROR'}, "Invalid group index")
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

        if settings.object_groups and 0 <= settings.active_group_index < len(settings.object_groups):
            group = settings.object_groups[settings.active_group_index]
            if 0 <= self.index < len(group.objects):
                group.objects.remove(self.index)

        return {'FINISHED'}
