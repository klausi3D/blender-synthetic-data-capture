"""
Coverage visualization operators.

Creates visual heatmap showing camera coverage on mesh surfaces.
"""

import bpy
from bpy.types import Operator
from mathutils import Vector, Color
from ..utils.coverage import CoverageAnalyzer


class GSCAPTURE_OT_show_coverage_heatmap(Operator):
    """Visualize camera coverage as vertex color heatmap"""
    bl_idname = "gs_capture.show_coverage_heatmap"
    bl_label = "Show Coverage Heatmap"
    bl_description = "Create vertex color layer showing camera coverage quality"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # Need selected objects and preview cameras
        if not context.selected_objects:
            return False

        # Check for preview cameras (cameras created with GS_Cam_ prefix)
        preview_cams = [obj for obj in bpy.data.objects
                       if obj.type == 'CAMERA' and obj.name.startswith('GS_Cam_')]
        return len(preview_cams) > 0

    def execute(self, context):
        # Get mesh objects
        mesh_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not mesh_objects:
            self.report({'WARNING'}, "No mesh objects selected")
            return {'CANCELLED'}

        # Get preview cameras (cameras created with GS_Cam_ prefix)
        cameras = [obj for obj in bpy.data.objects
                  if obj.type == 'CAMERA' and obj.name.startswith('GS_Cam_')]

        if not cameras:
            self.report({'WARNING'}, "No preview cameras found. Generate cameras first.")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Analyzing coverage with {len(cameras)} cameras...")

        # Analyze coverage
        analyzer = CoverageAnalyzer(mesh_objects, cameras)
        coverage_data = analyzer.calculate_vertex_coverage()
        stats = analyzer.get_coverage_statistics(coverage_data)

        # Apply heatmap to each mesh
        for obj in mesh_objects:
            if obj.name not in coverage_data:
                continue

            self._apply_heatmap(obj, coverage_data[obj.name], stats)

        # Store coverage percentage for UI display
        if stats['total_vertices'] > 0:
            well_covered_pct = (stats['well_covered'] / stats['total_vertices']) * 100
            context.scene.gs_capture_settings.coverage_percentage = well_covered_pct
            context.scene.gs_capture_settings.coverage_analyzed = True

        # Switch viewport shading to show vertex colors
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'SOLID'
                        space.shading.color_type = 'VERTEX'

        rating = analyzer.get_coverage_quality_rating()
        self.report({'INFO'}, f"Coverage: {rating} - {stats['well_covered']}/{stats['total_vertices']} vertices well covered")

        return {'FINISHED'}

    def _apply_heatmap(self, obj, vertex_coverage, stats):
        """Apply coverage heatmap as vertex colors.

        Args:
            obj: Blender mesh object
            vertex_coverage: Dict of vertex_index -> camera_count
            stats: Coverage statistics
        """
        mesh = obj.data

        # Ensure mesh has loops (required for vertex colors)
        if not mesh.loops:
            return

        # Create or get vertex color layer
        color_layer_name = "CoverageHeatmap"
        if color_layer_name not in mesh.color_attributes:
            mesh.color_attributes.new(
                name=color_layer_name,
                type='FLOAT_COLOR',
                domain='CORNER'
            )

        color_attr = mesh.color_attributes[color_layer_name]

        # Determine coverage range for normalization
        max_coverage = max(max(vertex_coverage.values()) if vertex_coverage else 1, 5)

        # Apply colors per-loop (corner)
        for poly in mesh.polygons:
            for loop_idx in poly.loop_indices:
                vert_idx = mesh.loops[loop_idx].vertex_index
                coverage = vertex_coverage.get(vert_idx, 0)

                # Normalize to 0-1 range
                normalized = min(coverage / max_coverage, 1.0)

                # Create color: red (bad) -> yellow -> green (good)
                color = self._coverage_to_color(normalized)
                color_attr.data[loop_idx].color = color

        # Set as active color attribute
        mesh.color_attributes.active = color_attr

    def _coverage_to_color(self, normalized):
        """Convert normalized coverage (0-1) to RGBA color.

        0.0 = Red (no coverage)
        0.5 = Yellow (partial coverage)
        1.0 = Green (full coverage)

        Args:
            normalized: Coverage value 0-1

        Returns:
            RGBA tuple
        """
        if normalized < 0.5:
            # Red to Yellow
            r = 1.0
            g = normalized * 2.0
            b = 0.0
        else:
            # Yellow to Green
            r = 1.0 - (normalized - 0.5) * 2.0
            g = 1.0
            b = 0.0

        return (r, g, b, 1.0)


class GSCAPTURE_OT_clear_coverage_heatmap(Operator):
    """Remove coverage heatmap visualization"""
    bl_idname = "gs_capture.clear_coverage_heatmap"
    bl_label = "Clear Heatmap"
    bl_description = "Remove coverage heatmap vertex colors"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return any(obj.type == 'MESH' for obj in context.selected_objects)

    def execute(self, context):
        removed_count = 0

        for obj in context.selected_objects:
            if obj.type != 'MESH':
                continue

            mesh = obj.data
            if "CoverageHeatmap" in mesh.color_attributes:
                mesh.color_attributes.remove(mesh.color_attributes["CoverageHeatmap"])
                removed_count += 1

        # Reset viewport shading
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.color_type = 'MATERIAL'

        # Clear stored data
        context.scene.gs_capture_settings.coverage_analyzed = False
        context.scene.gs_capture_settings.coverage_percentage = 0.0

        self.report({'INFO'}, f"Cleared heatmap from {removed_count} objects")
        return {'FINISHED'}


class GSCAPTURE_OT_analyze_coverage(Operator):
    """Analyze camera coverage without visual heatmap"""
    bl_idname = "gs_capture.analyze_coverage"
    bl_label = "Analyze Coverage"
    bl_description = "Calculate coverage statistics"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        if not context.selected_objects:
            return False

        preview_cams = [obj for obj in bpy.data.objects
                       if obj.type == 'CAMERA' and obj.name.startswith('GS_Cam_')]
        return len(preview_cams) > 0

    def execute(self, context):
        mesh_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
        cameras = [obj for obj in bpy.data.objects
                  if obj.type == 'CAMERA' and obj.name.startswith('GS_Cam_')]

        if not mesh_objects or not cameras:
            self.report({'WARNING'}, "Need mesh objects and preview cameras")
            return {'CANCELLED'}

        analyzer = CoverageAnalyzer(mesh_objects, cameras)
        stats = analyzer.get_coverage_statistics()
        rating = analyzer.get_coverage_quality_rating()

        # Store for UI
        if stats['total_vertices'] > 0:
            well_covered_pct = (stats['well_covered'] / stats['total_vertices']) * 100
            context.scene.gs_capture_settings.coverage_percentage = well_covered_pct
            context.scene.gs_capture_settings.coverage_analyzed = True

        self.report({'INFO'},
            f"Coverage: {rating} | "
            f"Min: {stats['min']} | Max: {stats['max']} | "
            f"Mean: {stats['mean']:.1f} cameras per vertex"
        )

        return {'FINISHED'}


# Registration
classes = [
    GSCAPTURE_OT_show_coverage_heatmap,
    GSCAPTURE_OT_clear_coverage_heatmap,
    GSCAPTURE_OT_analyze_coverage,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
