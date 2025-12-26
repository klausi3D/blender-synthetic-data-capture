"""
Batch processing panel.
"""

import bpy
from bpy.types import Panel


class GSCAPTURE_PT_batch_panel(Panel):
    """Batch processing panel."""
    bl_label = "Batch Processing"
    bl_idname = "GSCAPTURE_PT_batch_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_parent_id = "GSCAPTURE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.gs_capture_settings

        layout.prop(settings, "batch_mode")

        # Show collection selector when in COLLECTION mode
        if settings.batch_mode == 'COLLECTION':
            box = layout.box()
            box.label(text="Collection Settings:", icon='OUTLINER_COLLECTION')

            # Collection search/selector
            box.prop_search(settings, "target_collection", bpy.data, "collections", text="Collection")
            box.prop(settings, "include_nested_collections")

            # Show active collection info if no collection specified
            if not settings.target_collection:
                active_coll = context.view_layer.active_layer_collection.collection
                box.label(text=f"Active: {active_coll.name}", icon='INFO')

            # Quick capture button for collection
            row = box.row()
            row.scale_y = 1.3
            row.operator("gs_capture.capture_collection", text="Capture Collection", icon='RENDER_STILL')

        layout.prop(settings, "include_children")

        # Scene analysis for batch planning
        box = layout.box()
        box.label(text="Scene Analysis:", icon='SCENE_DATA')
        box.operator("gs_capture.analyze_scene", text="Analyze Entire Scene", icon='VIEWZOOM')
        box.label(text="Creates report of all collections", icon='INFO')

        if settings.batch_mode == 'GROUPS':
            box = layout.box()
            box.label(text="Object Groups:")

            row = box.row()
            row.operator("gs_capture.add_object_group", text="Add Group", icon='ADD')
            row.operator("gs_capture.remove_object_group", text="Remove", icon='REMOVE')

            for i, group in enumerate(settings.object_groups):
                group_box = box.box()
                row = group_box.row()
                row.prop(group, "expanded", text="",
                         icon='DISCLOSURE_TRI_DOWN' if group.expanded else 'DISCLOSURE_TRI_RIGHT',
                         emboss=False)
                row.prop(group, "name", text="")

                if i == settings.active_group_index:
                    row.label(text="", icon='RADIOBUT_ON')
                else:
                    row.label(text="", icon='RADIOBUT_OFF')

                if group.expanded:
                    for j, item in enumerate(group.objects):
                        item_row = group_box.row()
                        item_row.prop(item, "obj", text="")
                        op = item_row.operator("gs_capture.remove_from_group", text="", icon='X')
                        op.index = j

                    if i == settings.active_group_index:
                        group_box.operator("gs_capture.add_to_group", text="Add Selected", icon='ADD')

        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        row.operator("gs_capture.batch_capture", text="Run Batch Capture", icon='RENDER_ANIMATION')
