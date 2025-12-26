"""
Adaptive analysis panel.
"""

import bpy
from bpy.types import Panel


class GSCAPTURE_PT_adaptive_panel(Panel):
    """Adaptive capture analysis panel."""
    bl_label = "Adaptive Analysis"
    bl_idname = "GSCAPTURE_PT_adaptive_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_parent_id = "GSCAPTURE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.gs_capture_settings

        # Enable/disable adaptive
        layout.prop(settings, "use_adaptive_capture")

        if not settings.use_adaptive_capture:
            layout.label(text="Enable for automatic optimization", icon='INFO')
            return

        layout.prop(settings, "adaptive_quality_preset")

        box = layout.box()
        box.label(text="Hotspot Detection:", icon='LIGHT')
        box.prop(settings, "adaptive_use_hotspots")
        if settings.adaptive_use_hotspots:
            box.prop(settings, "adaptive_hotspot_bias")

        layout.separator()

        # Analyze button
        row = layout.row()
        row.scale_y = 1.3
        row.operator("gs_capture.analyze_selected", text="Analyze Selected", icon='VIEWZOOM')

        # Analysis results
        if settings.analysis_vertex_count > 0:
            results_box = layout.box()
            results_box.label(text="Analysis Results:", icon='GRAPH')

            col = results_box.column(align=True)
            col.label(text=f"Quality: {settings.analysis_quality_preset}")
            col.label(text=f"Recommended Cameras: {settings.analysis_recommended_cameras}")
            col.label(text=f"Recommended Resolution: {settings.analysis_recommended_resolution}")
            col.label(text=f"Est. Render Time: {settings.analysis_render_time_estimate}")

            results_box.separator()

            # Mesh stats
            mesh_box = results_box.box()
            mesh_box.label(text="Mesh Analysis:", icon='MESH_DATA')
            mesh_col = mesh_box.column(align=True)
            mesh_col.label(text=f"Vertices: {settings.analysis_vertex_count:,}")
            mesh_col.label(text=f"Faces: {settings.analysis_face_count:,}")
            mesh_col.label(text=f"Surface Area: {settings.analysis_surface_area:.2f} sq units")
            mesh_col.label(text=f"Detail Score: {settings.analysis_detail_score:.2f}")

            # Texture stats
            tex_box = results_box.box()
            tex_box.label(text="Texture Analysis:", icon='TEXTURE')
            tex_col = tex_box.column(align=True)
            tex_col.label(text=f"Max Resolution: {settings.analysis_texture_resolution}px")
            tex_col.label(text=f"Texture Score: {settings.analysis_texture_score:.2f}")

            # Warnings
            if settings.analysis_warnings and settings.analysis_warnings != "None":
                warn_box = results_box.box()
                warn_box.label(text="Warnings:", icon='ERROR')
                for warning in settings.analysis_warnings.split(" | "):
                    if warning.strip():
                        warn_box.label(text=warning.strip(), icon='DOT')

            # Apply button
            results_box.separator()
            row = results_box.row(align=True)
            row.operator("gs_capture.apply_recommendations", text="Apply", icon='CHECKMARK')
            row.operator("gs_capture.export_analysis_report", text="Export Report", icon='FILE_TEXT')
