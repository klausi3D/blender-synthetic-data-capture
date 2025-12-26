"""
Analysis operators for mesh and texture evaluation.
Provides scene analysis and adaptive settings recommendations.
"""

import bpy
import os
import json
from bpy.types import Operator

from ..core.analysis import (
    analyze_mesh_complexity,
    analyze_texture_quality,
    calculate_adaptive_settings,
)


class GSCAPTURE_OT_analyze_selected(Operator):
    """Analyze selected objects to determine optimal capture settings."""
    bl_idname = "gs_capture.analyze_selected"
    bl_label = "Analyze Selected"
    bl_options = {'REGISTER'}

    def execute(self, context):
        settings = context.scene.gs_capture_settings

        selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not selected:
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}

        # Calculate adaptive settings
        result = calculate_adaptive_settings(selected, settings.adaptive_quality_preset)

        # Store results in settings for UI display
        settings.analysis_vertex_count = result.mesh_analysis.vertex_count
        settings.analysis_face_count = result.mesh_analysis.face_count
        settings.analysis_surface_area = result.mesh_analysis.surface_area
        settings.analysis_detail_score = result.mesh_analysis.detail_score

        settings.analysis_texture_resolution = result.texture_analysis.max_resolution
        settings.analysis_texture_score = result.texture_analysis.texture_score

        settings.analysis_recommended_cameras = result.recommended_camera_count
        settings.analysis_recommended_resolution = f"{result.recommended_resolution[0]}x{result.recommended_resolution[1]}"
        settings.analysis_quality_preset = result.quality_preset

        # Render time estimate
        if result.estimated_render_time_minutes > 60:
            hours = result.estimated_render_time_minutes // 60
            mins = result.estimated_render_time_minutes % 60
            settings.analysis_render_time_estimate = f"~{hours}h {mins}m"
        else:
            settings.analysis_render_time_estimate = f"~{result.estimated_render_time_minutes}m"

        # Warnings
        if result.warnings:
            settings.analysis_warnings = " | ".join(result.warnings)
        else:
            settings.analysis_warnings = "None"

        self.report({'INFO'}, f"Analysis complete: {result.quality_preset} quality recommended")
        return {'FINISHED'}


class GSCAPTURE_OT_analyze_scene(Operator):
    """Analyze all collections in the scene for batch capture planning."""
    bl_idname = "gs_capture.analyze_scene"
    bl_label = "Analyze Scene"
    bl_options = {'REGISTER'}

    def execute(self, context):
        settings = context.scene.gs_capture_settings

        # Analyze each collection
        report = {
            'scene_name': bpy.data.filepath or "Untitled",
            'collections': []
        }

        for collection in bpy.data.collections:
            meshes = [obj for obj in collection.objects if obj.type == 'MESH']
            if not meshes:
                continue

            result = calculate_adaptive_settings(meshes, 'AUTO')

            coll_data = {
                'name': collection.name,
                'object_count': len(meshes),
                'total_vertices': result.mesh_analysis.vertex_count,
                'total_faces': result.mesh_analysis.face_count,
                'surface_area': result.mesh_analysis.surface_area,
                'detail_score': result.mesh_analysis.detail_score,
                'recommended_quality': result.quality_preset,
                'recommended_cameras': result.recommended_camera_count,
                'recommended_resolution': f"{result.recommended_resolution[0]}x{result.recommended_resolution[1]}",
                'estimated_render_time_minutes': result.estimated_render_time_minutes,
                'warnings': result.warnings,
            }
            report['collections'].append(coll_data)

        # Save report
        output_path = bpy.path.abspath(settings.output_path)
        os.makedirs(output_path, exist_ok=True)
        report_path = os.path.join(output_path, "scene_analysis_report.json")

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.report({'INFO'}, f"Scene analysis saved to {report_path}")
        return {'FINISHED'}


class GSCAPTURE_OT_export_analysis_report(Operator):
    """Export detailed analysis report as JSON."""
    bl_idname = "gs_capture.export_analysis_report"
    bl_label = "Export Report"
    bl_options = {'REGISTER'}

    def execute(self, context):
        settings = context.scene.gs_capture_settings

        selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not selected:
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}

        # Build detailed report
        report = {
            'analyzed_objects': [],
            'combined_analysis': {},
        }

        for obj in selected:
            mesh_analysis = analyze_mesh_complexity(obj)
            tex_analysis = analyze_texture_quality(obj)

            obj_data = {
                'name': obj.name,
                'mesh': {
                    'vertex_count': mesh_analysis.vertex_count,
                    'face_count': mesh_analysis.face_count,
                    'surface_area': mesh_analysis.surface_area,
                    'detail_score': mesh_analysis.detail_score,
                    'has_ngons': mesh_analysis.has_ngons,
                    'curvature_variance': mesh_analysis.curvature_variance,
                },
                'textures': {
                    'max_resolution': tex_analysis.max_resolution,
                    'total_textures': tex_analysis.total_textures,
                    'has_normal_maps': tex_analysis.has_normal_maps,
                    'has_displacement': tex_analysis.has_displacement,
                    'texture_score': tex_analysis.texture_score,
                    'detail_level': tex_analysis.estimated_detail_level,
                },
            }
            report['analyzed_objects'].append(obj_data)

        # Combined analysis
        result = calculate_adaptive_settings(selected, settings.adaptive_quality_preset)
        report['combined_analysis'] = {
            'total_vertices': result.mesh_analysis.vertex_count,
            'total_faces': result.mesh_analysis.face_count,
            'total_surface_area': result.mesh_analysis.surface_area,
            'max_detail_score': result.mesh_analysis.detail_score,
            'recommended_quality': result.quality_preset,
            'recommended_cameras': result.recommended_camera_count,
            'recommended_resolution': list(result.recommended_resolution),
            'recommended_distance_multiplier': result.recommended_distance_multiplier,
            'estimated_render_time_minutes': result.estimated_render_time_minutes,
            'warnings': result.warnings,
            'hotspots_count': len(result.detail_hotspots),
        }

        # Save report
        output_path = bpy.path.abspath(settings.output_path)
        os.makedirs(output_path, exist_ok=True)
        report_path = os.path.join(output_path, "analysis_report.json")

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.report({'INFO'}, f"Report exported to {report_path}")
        return {'FINISHED'}


class GSCAPTURE_OT_apply_recommendations(Operator):
    """Apply recommended settings from analysis."""
    bl_idname = "gs_capture.apply_recommendations"
    bl_label = "Apply Recommendations"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.gs_capture_settings

        # Apply stored recommendations
        if settings.analysis_recommended_cameras > 0:
            settings.camera_count = settings.analysis_recommended_cameras

        if settings.analysis_recommended_resolution:
            try:
                res_parts = settings.analysis_recommended_resolution.split('x')
                context.scene.render.resolution_x = int(res_parts[0])
                context.scene.render.resolution_y = int(res_parts[1])
            except (ValueError, IndexError):
                pass

        self.report({'INFO'}, "Applied recommended settings")
        return {'FINISHED'}
