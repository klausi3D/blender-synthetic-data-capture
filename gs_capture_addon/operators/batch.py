"""
Batch processing operators for capturing multiple objects/collections.
"""

import bpy
import json
import os
from datetime import datetime, timezone
from bpy.types import Operator

from ..utils.asset_library import (
    discover_asset_library_blend_files,
    make_output_subfolder_name,
    relative_blend_path,
    resolve_asset_library_path,
)


ASSET_LIBRARY_MANIFEST = "asset_library_manifest.json"


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
    _current_item = None
    _current_asset_import = None
    _output_root_abs = ""
    _asset_mode = False
    _asset_manifest_entries = None
    _manifest_written = False

    @classmethod
    def poll(cls, context):
        return not context.scene.gs_capture_settings.is_rendering

    def execute(self, context):
        settings = context.scene.gs_capture_settings
        if settings.batch_running:
            self.report({'WARNING'}, "Batch capture is already running")
            return {'CANCELLED'}

        self._original_output = settings.output_path
        self._output_root_abs = bpy.path.abspath(self._original_output)
        self._asset_mode = settings.batch_mode == 'ASSET_LIBRARY'
        self._asset_manifest_entries = {}
        self._manifest_written = False

        self._items_to_capture = self._build_items_to_capture(context, settings)
        if not self._items_to_capture:
            self.report({'ERROR'}, "No items to capture")
            return {'CANCELLED'}
        if self._is_render_job_running():
            self.report({'WARNING'}, "Cannot start batch while another render job is active")
            return {'CANCELLED'}

        self._original_selection = list(context.selected_objects)
        self._original_active = context.view_layer.objects.active
        self._current_index = -1
        self._current_item = None
        self._current_asset_import = None
        self._waiting_for_capture = False
        self._completed = 0
        self._failed = 0
        self._cancel_batch = False
        settings.batch_running = True
        settings.batch_cancel_requested = False
        settings.cancel_requested = False

        if not self._start_next_capture(context):
            self._finish(context)
            self.report({'INFO'}, f"Batch complete: {self._completed} succeeded, {self._failed} failed")
            return {'FINISHED'}

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        settings = context.scene.gs_capture_settings
        try:
            if settings.batch_cancel_requested:
                self._cancel_batch = True

            if event.type == 'ESC':
                self._cancel_batch = True
                settings.batch_cancel_requested = True
                if settings.is_rendering:
                    settings.cancel_requested = True
                return {'RUNNING_MODAL'}

            if event.type == 'TIMER':
                if self._waiting_for_capture:
                    if settings.is_rendering or self._is_render_job_running():
                        if self._cancel_batch:
                            settings.cancel_requested = True
                        return {'RUNNING_MODAL'}

                    self._finalize_completed_item(context, cancelled=self._cancel_batch)
                    self._waiting_for_capture = False
                    if self._cancel_batch:
                        self._finish(context, cancelled=True)
                        return {'CANCELLED'}

                    if not self._start_next_capture(context):
                        self._finish(context)
                        self.report({'INFO'}, f"Batch complete: {self._completed} succeeded, {self._failed} failed")
                        return {'FINISHED'}

                if self._cancel_batch:
                    if settings.is_rendering:
                        settings.cancel_requested = True
                        return {'RUNNING_MODAL'}
                    if self._is_render_job_running():
                        return {'RUNNING_MODAL'}
                    self._finish(context, cancelled=True)
                    return {'CANCELLED'}

            return {'PASS_THROUGH'}
        except Exception as exc:
            self._finish(context, cancelled=True)
            self.report({'ERROR'}, f"Batch capture stopped due to unexpected error: {exc}")
            return {'CANCELLED'}

    def _is_render_job_running(self):
        is_job_running = getattr(bpy.app, "is_job_running", None)
        if callable(is_job_running):
            try:
                return bool(is_job_running("RENDER"))
            except Exception:
                return False
        return False

    def _build_items_to_capture(self, context, settings):
        items_to_capture = []

        if settings.batch_mode == 'SCENE':
            all_meshes = [obj for obj in context.scene.objects
                          if obj.type == 'MESH' and obj.visible_get()]
            if all_meshes:
                items_to_capture.append({
                    "type": "OBJECTS",
                    "name": "scene",
                    "objects": all_meshes,
                })

        elif settings.batch_mode == 'SELECTED':
            selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
            if selected:
                items_to_capture.append({
                    "type": "OBJECTS",
                    "name": "selected",
                    "objects": selected,
                })

        elif settings.batch_mode == 'EACH_SELECTED':
            for obj in context.selected_objects:
                if obj.type == 'MESH':
                    items_to_capture.append({
                        "type": "OBJECTS",
                        "name": obj.name,
                        "objects": [obj],
                    })

        elif settings.batch_mode == 'COLLECTION':
            if settings.target_collection:
                collection = bpy.data.collections.get(settings.target_collection)
                if collection:
                    meshes = self._get_collection_meshes(collection, settings.include_nested_collections)
                    if meshes:
                        items_to_capture.append({
                            "type": "OBJECTS",
                            "name": collection.name,
                            "objects": meshes,
                        })

        elif settings.batch_mode == 'COLLECTIONS':
            for collection in bpy.data.collections:
                meshes = self._get_collection_meshes(collection, settings.include_nested_collections)
                if meshes:
                    items_to_capture.append({
                        "type": "OBJECTS",
                        "name": collection.name,
                        "objects": meshes,
                    })

        elif settings.batch_mode == 'GROUPS':
            for group in settings.object_groups:
                meshes = [item.obj for item in group.objects if item.obj and item.obj.type == 'MESH']
                if meshes:
                    items_to_capture.append({
                        "type": "OBJECTS",
                        "name": group.name,
                        "objects": meshes,
                    })

        elif settings.batch_mode == 'ASSET_LIBRARY':
            library_root, error = resolve_asset_library_path(
                settings.asset_library_name,
                context=context,
            )
            if error:
                self.report({'ERROR'}, error)
                return []

            blend_files = discover_asset_library_blend_files(
                library_root,
                recursive=settings.asset_library_recursive,
                filename_filter=settings.asset_library_filename_filter,
            )
            if not blend_files:
                self.report(
                    {'ERROR'},
                    f"No .blend files found in asset library '{settings.asset_library_name}'",
                )
                return []

            used_names = set()
            for index, blend_file in enumerate(blend_files):
                output_subfolder = make_output_subfolder_name(
                    blend_file,
                    library_root,
                    used_names=used_names,
                    fallback_index=index,
                )
                items_to_capture.append(
                    {
                        "type": "ASSET_FILE",
                        "asset_index": index,
                        "name": output_subfolder,
                        "output_subfolder": output_subfolder,
                        "blend_path": blend_file,
                        "relative_path": relative_blend_path(blend_file, library_root),
                    }
                )

        return items_to_capture

    def _start_next_capture(self, context):
        while True:
            if self._cancel_batch:
                return False

            self._current_index += 1
            if self._current_index >= len(self._items_to_capture):
                return False

            item = self._items_to_capture[self._current_index]
            self._current_item = item

            if self._start_item_capture(context, item):
                self._waiting_for_capture = True
                return True

            self._current_item = None
            self._waiting_for_capture = False

    def _start_item_capture(self, context, item):
        item_type = item.get("type")
        if item_type == "OBJECTS":
            return self._start_object_item_capture(context, item)
        if item_type == "ASSET_FILE":
            return self._start_asset_item_capture(context, item)

        self.report({'WARNING'}, f"Unsupported batch item type: {item_type}")
        self._failed += 1
        return False

    def _start_object_item_capture(self, context, item):
        settings = context.scene.gs_capture_settings
        name = item.get("name", "capture")
        objects = [obj for obj in item.get("objects", []) if obj and obj.name in bpy.data.objects]
        if not objects:
            self.report({'WARNING'}, f"Skipping '{name}': no valid mesh objects remain")
            self._failed += 1
            return False

        settings.output_path = os.path.join(self._output_root_abs, name)
        self._select_objects(context, objects)
        return self._invoke_capture_operator(name=name, on_failure=None)

    def _start_asset_item_capture(self, context, item):
        settings = context.scene.gs_capture_settings
        name = item.get("name", "asset_capture")
        blend_path = item.get("blend_path", "")

        import_state, error = self._import_asset_file(context, blend_path)
        if not import_state:
            self._record_asset_result(item, "failed", error or "failed to import asset file")
            self._failed += 1
            return False

        mesh_objects = [obj for obj in import_state.get("mesh_objects", []) if obj and obj.name in bpy.data.objects]
        if not mesh_objects:
            self._record_asset_result(item, "failed", "asset file has no mesh objects")
            self._cleanup_asset_import(context, import_state)
            self._failed += 1
            return False

        self._current_asset_import = import_state
        settings.output_path = os.path.join(self._output_root_abs, item.get("output_subfolder", name))
        self._select_objects(context, mesh_objects)

        def _on_failure(failure_message):
            self._record_asset_result(item, "failed", failure_message)
            self._cleanup_current_asset_import(context)

        return self._invoke_capture_operator(name=name, on_failure=_on_failure)

    def _invoke_capture_operator(self, name, on_failure):
        try:
            result = bpy.ops.gs_capture.capture_selected()
        except Exception as exc:
            message = f"Failed to capture {name}: {exc}"
            self.report({'WARNING'}, message)
            self._failed += 1
            if on_failure:
                on_failure(str(exc))
            return False

        if 'CANCELLED' in set(result):
            message = f"Capture cancelled for {name}"
            self.report({'WARNING'}, message)
            self._failed += 1
            if on_failure:
                on_failure("capture operator returned CANCELLED")
            return False

        return True

    def _select_objects(self, context, objects):
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects:
            obj.select_set(True)
        context.view_layer.objects.active = objects[0] if objects else None

    def _import_asset_file(self, context, blend_path):
        imported_objects = []
        imported_mesh_data = set()
        imported_materials = set()
        temp_collection = None

        try:
            with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
                data_to.objects = list(data_from.objects)
        except Exception as exc:
            return None, f"Could not load {blend_path}: {exc}"

        imported_objects = [obj for obj in (data_to.objects or []) if obj is not None]
        mesh_objects = [obj for obj in imported_objects if obj.type == 'MESH']
        if not mesh_objects:
            self._remove_objects(imported_objects)
            return None, f"No mesh objects found in {blend_path}"

        try:
            temp_collection = bpy.data.collections.new(
                f"GS_AssetBatch_{self._current_index:04d}"
            )
            context.scene.collection.children.link(temp_collection)

            for obj in mesh_objects:
                if obj.name not in temp_collection.objects:
                    temp_collection.objects.link(obj)
                if getattr(obj, "data", None) is not None:
                    imported_mesh_data.add(obj.data)
                    for material in obj.data.materials:
                        if material:
                            imported_materials.add(material)
        except Exception as exc:
            self._remove_objects(imported_objects)
            if temp_collection and temp_collection.name in bpy.data.collections:
                bpy.data.collections.remove(temp_collection)
            return None, f"Failed to link mesh objects from {blend_path}: {exc}"

        non_mesh_objects = [obj for obj in imported_objects if obj.type != 'MESH']
        self._remove_objects(non_mesh_objects)

        return {
            "blend_path": blend_path,
            "collection": temp_collection,
            "mesh_objects": mesh_objects,
            "mesh_data": list(imported_mesh_data),
            "materials": list(imported_materials),
        }, ""

    def _remove_objects(self, objects):
        for obj in objects:
            if obj and obj.name in bpy.data.objects:
                try:
                    bpy.data.objects.remove(obj, do_unlink=True)
                except Exception:
                    pass

    def _cleanup_current_asset_import(self, context):
        if self._current_asset_import:
            self._cleanup_asset_import(context, self._current_asset_import)
            self._current_asset_import = None

    def _cleanup_asset_import(self, context, import_state):
        mesh_objects = import_state.get("mesh_objects", [])
        self._remove_objects(mesh_objects)

        collection = import_state.get("collection")
        if collection and collection.name in bpy.data.collections:
            for scene in bpy.data.scenes:
                try:
                    if collection in scene.collection.children:
                        scene.collection.children.unlink(collection)
                except Exception:
                    pass
            try:
                bpy.data.collections.remove(collection)
            except Exception:
                pass

        for mesh in import_state.get("mesh_data", []):
            if mesh and mesh.name in bpy.data.meshes and mesh.users == 0:
                try:
                    bpy.data.meshes.remove(mesh)
                except Exception:
                    pass

        for material in import_state.get("materials", []):
            if material and material.name in bpy.data.materials and material.users == 0:
                try:
                    bpy.data.materials.remove(material)
                except Exception:
                    pass

    def _finalize_completed_item(self, context, cancelled=False):
        settings = context.scene.gs_capture_settings
        success = bool(settings.last_capture_success)
        item = self._current_item

        if success:
            self._completed += 1
        else:
            self._failed += 1

        if item and item.get("type") == "ASSET_FILE":
            if success:
                status = "succeeded"
                error = ""
            elif cancelled:
                status = "cancelled"
                error = "capture cancelled"
            else:
                status = "failed"
                error = "capture failed"
            self._record_asset_result(item, status, error)

        self._cleanup_current_asset_import(context)
        self._current_item = None

    def _record_asset_result(self, item, status, error=""):
        if not self._asset_mode:
            return

        asset_index = item.get("asset_index")
        output_subfolder = item.get("output_subfolder", item.get("name", ""))
        output_path = os.path.join(self._output_root_abs, output_subfolder)
        entry = {
            "index": asset_index,
            "blend_file": item.get("blend_path", ""),
            "relative_blend_file": item.get("relative_path", ""),
            "output_subfolder": output_subfolder,
            "output_path": output_path,
            "status": status,
            "error": error or "",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._asset_manifest_entries[asset_index] = entry

    def _finish(self, context, cancelled=False):
        settings = context.scene.gs_capture_settings
        self._cleanup_current_asset_import(context)
        self._write_asset_manifest(cancelled=cancelled)
        settings.batch_running = False
        settings.batch_cancel_requested = False
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

    def _write_asset_manifest(self, cancelled=False):
        if not self._asset_mode or self._manifest_written:
            return

        os.makedirs(self._output_root_abs, exist_ok=True)
        entries = []
        for item in self._items_to_capture:
            if item.get("type") != "ASSET_FILE":
                continue

            asset_index = item.get("asset_index")
            existing = self._asset_manifest_entries.get(asset_index)
            if existing is None:
                status = "cancelled" if cancelled else "not_started"
                existing = {
                    "index": asset_index,
                    "blend_file": item.get("blend_path", ""),
                    "relative_blend_file": item.get("relative_path", ""),
                    "output_subfolder": item.get("output_subfolder", item.get("name", "")),
                    "output_path": os.path.join(
                        self._output_root_abs,
                        item.get("output_subfolder", item.get("name", "")),
                    ),
                    "status": status,
                    "error": "batch cancelled before processing" if cancelled else "",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            entries.append(existing)

        counts = {
            "succeeded": 0,
            "failed": 0,
            "cancelled": 0,
            "not_started": 0,
        }
        for entry in entries:
            status = entry.get("status")
            if status in counts:
                counts[status] += 1

        payload = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "batch_cancelled": bool(cancelled),
            "output_root": self._output_root_abs,
            "summary": {
                "total_files": len(entries),
                "succeeded": counts["succeeded"],
                "failed": counts["failed"],
                "cancelled": counts["cancelled"],
                "not_started": counts["not_started"],
            },
            "files": entries,
        }

        manifest_path = os.path.join(self._output_root_abs, ASSET_LIBRARY_MANIFEST)
        try:
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            self.report({'INFO'}, f"Asset library manifest written: {manifest_path}")
        except Exception as exc:
            self.report({'WARNING'}, f"Failed to write asset library manifest: {exc}")

        self._manifest_written = True

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
