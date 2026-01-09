"""
Warnings and analysis panel for GS Capture.

Displays material problems, scene score, and coverage analysis.
"""

import bpy
from bpy.types import Panel


class GSCAPTURE_PT_warnings(Panel):
    """Panel showing scene analysis and warnings."""
    bl_label = "Scene Analysis"
    bl_idname = "GSCAPTURE_PT_warnings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_order = 1  # Show near top

    def draw(self, context):
        layout = self.layout
        settings = context.scene.gs_capture_settings

        # Analyze button
        row = layout.row(align=True)
        row.scale_y = 1.2
        row.operator("gs_capture.analyze_scene_mvp", text="Analyze Scene", icon='VIEWZOOM')

        # Scene Score display
        if settings.scene_analyzed:
            box = layout.box()
            row = box.row()

            # Grade icon
            grade_icons = {
                'EXCELLENT': 'CHECKMARK',
                'GOOD': 'SOLO_ON',
                'FAIR': 'ERROR',
                'POOR': 'CANCEL',
                'NONE': 'QUESTION',
            }
            icon = grade_icons.get(settings.scene_grade, 'QUESTION')
            row.label(text="Scene Score:", icon=icon)

            # Score bar
            score = settings.scene_score
            bar_length = 10
            filled = int(score / 100 * bar_length)
            bar = chr(9608) * filled + chr(9617) * (bar_length - filled)  # Unicode block chars
            row.label(text=f"{bar} {score}%")

            # Grade text
            grade_text = settings.scene_grade.replace('_', ' ').title()
            box.label(text=f"Grade: {grade_text}")

        # Material Warnings
        if settings.material_problems_count > 0:
            box = layout.box()
            row = box.row()

            # Warning icon based on severity
            if settings.material_problems_high > 0:
                row.label(text="Material Issues", icon='ERROR')
            else:
                row.label(text="Material Issues", icon='INFO')

            # Problem counts
            col = box.column(align=True)
            if settings.material_problems_high > 0:
                col.label(text=f"  {settings.material_problems_high} high severity", icon='CANCEL')

            total = settings.material_problems_count
            other = total - settings.material_problems_high
            if other > 0:
                col.label(text=f"  {other} other issues", icon='DOT')

            # Fix/View buttons
            row = box.row(align=True)
            row.operator("gs_capture.show_material_problems", text="Details", icon='RIGHTARROW')
            row.operator("gs_capture.fix_material_problems", text="Fix All", icon='MODIFIER')

        # Coverage display
        if settings.coverage_analyzed:
            box = layout.box()
            row = box.row()
            row.label(text="Camera Coverage", icon='CAMERA_DATA')

            pct = settings.coverage_percentage
            bar_length = 10
            filled = int(pct / 100 * bar_length)
            bar = chr(9608) * filled + chr(9617) * (bar_length - filled)

            row = box.row()
            row.label(text=f"{bar} {pct:.0f}%")

            # Coverage quality label
            if pct >= 90:
                box.label(text="Excellent coverage", icon='CHECKMARK')
            elif pct >= 70:
                box.label(text="Good coverage", icon='SOLO_ON')
            elif pct >= 50:
                box.label(text="Fair coverage - consider more cameras", icon='ERROR')
            else:
                box.label(text="Poor coverage - add more cameras", icon='CANCEL')

        # Heatmap controls
        row = layout.row(align=True)
        row.operator("gs_capture.show_coverage_heatmap", text="Show Heatmap", icon='COLORSET_01_VEC')
        row.operator("gs_capture.clear_coverage_heatmap", text="Clear", icon='X')


class GSCAPTURE_OT_analyze_scene_mvp(bpy.types.Operator):
    """Analyze scene for GS capture suitability"""
    bl_idname = "gs_capture.analyze_scene_mvp"
    bl_label = "Analyze Scene"
    bl_description = "Analyze materials, geometry, and provide recommendations"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return any(obj.type == 'MESH' for obj in context.selected_objects)

    def execute(self, context):
        from ..core.material_analyzer import analyze_objects, get_problem_summary, ProblemSeverity
        from ..core.scene_score import analyze_scene, SceneGrade

        settings = context.scene.gs_capture_settings
        mesh_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']

        if not mesh_objects:
            self.report({'WARNING'}, "No mesh objects selected")
            return {'CANCELLED'}

        # Analyze materials
        problems = analyze_objects(mesh_objects)
        summary = get_problem_summary(problems)

        settings.material_problems_count = summary['total']
        settings.material_problems_high = summary['by_severity']['high']

        # Analyze scene complexity
        scene_result = analyze_scene(mesh_objects, summary['total'])

        settings.scene_score = scene_result.score
        settings.scene_grade = scene_result.grade.value.upper()
        settings.scene_analyzed = True

        # Report
        if summary['total'] > 0:
            self.report({'WARNING'},
                f"Found {summary['total']} material issues. "
                f"Scene score: {scene_result.score}% ({scene_result.grade.value})"
            )
        else:
            self.report({'INFO'},
                f"No material issues. Scene score: {scene_result.score}% ({scene_result.grade.value})"
            )

        return {'FINISHED'}


class GSCAPTURE_OT_show_material_problems(bpy.types.Operator):
    """Show detailed material problem report"""
    bl_idname = "gs_capture.show_material_problems"
    bl_label = "Material Problems"
    bl_description = "Show detailed list of material issues"
    bl_options = {'REGISTER'}

    def execute(self, context):
        return {'FINISHED'}

    def invoke(self, context, event):
        from ..core.material_analyzer import analyze_objects, PROBLEM_TYPES

        mesh_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
        problems = analyze_objects(mesh_objects)

        if not problems:
            self.report({'INFO'}, "No material problems found")
            return {'FINISHED'}

        # Build report
        lines = ["Material Problems Found:", ""]
        for problem in problems:
            ptype = PROBLEM_TYPES.get(problem.problem_type, {})
            lines.append(f"{problem.material_name} ({problem.object_name}):")
            lines.append(f"  Type: {ptype.get('label', problem.problem_type)}")
            lines.append(f"  Severity: {problem.severity.value}")
            lines.append(f"  {problem.description}")
            lines.append(f"  Suggestion: {problem.suggestion}")
            lines.append("")

        # Show in info area
        for line in lines:
            self.report({'INFO'}, line)

        return {'FINISHED'}


class GSCAPTURE_OT_fix_material_problems(bpy.types.Operator):
    """Attempt to fix material problems automatically"""
    bl_idname = "gs_capture.fix_material_problems"
    bl_label = "Fix Material Problems"
    bl_description = "Attempt to automatically fix detected material issues"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.gs_capture_settings.material_problems_count > 0

    def execute(self, context):
        from ..core.material_analyzer import fix_all_problems

        mesh_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
        fixed, unfixable = fix_all_problems(mesh_objects)

        # Re-analyze
        bpy.ops.gs_capture.analyze_scene_mvp()

        if fixed > 0:
            self.report({'INFO'}, f"Fixed {fixed} issues. {unfixable} issues require manual fixes.")
        else:
            self.report({'WARNING'}, f"No issues could be auto-fixed. {unfixable} issues require manual fixes.")

        return {'FINISHED'}


# Registration
classes = [
    GSCAPTURE_PT_warnings,
    GSCAPTURE_OT_analyze_scene_mvp,
    GSCAPTURE_OT_show_material_problems,
    GSCAPTURE_OT_fix_material_problems,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
