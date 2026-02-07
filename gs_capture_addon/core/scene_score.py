"""
Scene Complexity Scorer for Gaussian Splatting.

Calculates a simple complexity score and provides recommendations
for optimal capture settings based on scene geometry and materials.
"""

import bpy
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class SceneGrade(Enum):
    """Overall scene grade for GS capture."""
    EXCELLENT = "excellent"  # Easy to capture, will work great
    GOOD = "good"            # Should work well with default settings
    FAIR = "fair"            # May need adjustments
    POOR = "poor"            # Challenging, may have issues


@dataclass
class SceneMetrics:
    """Raw metrics about the scene."""
    total_vertices: int
    total_faces: int
    total_objects: int
    bounding_box_size: Tuple[float, float, float]
    bounding_box_volume: float
    surface_area_estimate: float
    has_thin_geometry: bool
    material_problem_count: int
    max_dimension: float
    min_dimension: float
    aspect_ratio: float  # max/min dimension ratio


@dataclass
class SceneScore:
    """Complete scene analysis result."""
    grade: SceneGrade
    score: int  # 0-100
    metrics: SceneMetrics
    recommended_cameras: int
    recommendations: List[str]
    warnings: List[str]


def calculate_bounding_box(objects) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Calculate combined bounding box of all objects.

    Returns:
        Tuple of (min_corner, max_corner) as (x, y, z) tuples
    """
    min_coords = [float('inf')] * 3
    max_coords = [float('-inf')] * 3

    for obj in objects:
        if obj.type != 'MESH':
            continue

        mesh = obj.data
        matrix = obj.matrix_world

        for vert in mesh.vertices:
            world_co = matrix @ vert.co
            for i in range(3):
                min_coords[i] = min(min_coords[i], world_co[i])
                max_coords[i] = max(max_coords[i], world_co[i])

    # Handle empty case
    if min_coords[0] == float('inf'):
        return ((0, 0, 0), (1, 1, 1))

    return (tuple(min_coords), tuple(max_coords))


def estimate_surface_area(objects) -> float:
    """Rough estimate of total surface area."""
    total_area = 0.0

    for obj in objects:
        if obj.type != 'MESH':
            continue

        mesh = obj.data
        matrix = obj.matrix_world
        scale = matrix.to_scale()
        scale_factor = abs(scale.x * scale.y * scale.z) ** (2/3)  # Average scale for area

        # Sum face areas
        mesh.calc_loop_triangles()
        for tri in mesh.loop_triangles:
            total_area += tri.area * scale_factor

    return total_area


def check_thin_geometry(objects, threshold: float = 0.01) -> bool:
    """Check if scene has thin geometry that may not reconstruct well."""
    for obj in objects:
        if obj.type != 'MESH':
            continue

        mesh = obj.data

        # Check for very thin edges
        for edge in mesh.edges:
            v1 = mesh.vertices[edge.vertices[0]].co
            v2 = mesh.vertices[edge.vertices[1]].co
            length = (v2 - v1).length

            if length < threshold:
                return True

        # Check for very small faces
        for poly in mesh.polygons:
            if poly.area < threshold * threshold:
                return True

    return False


def analyze_scene(objects, material_problems: int = 0) -> SceneScore:
    """Analyze scene and generate complexity score.

    Args:
        objects: List of Blender objects to analyze
        material_problems: Number of material problems detected (from material_analyzer)

    Returns:
        SceneScore with grade, metrics, and recommendations
    """
    mesh_objects = [obj for obj in objects if obj.type == 'MESH']

    if not mesh_objects:
        return SceneScore(
            grade=SceneGrade.POOR,
            score=0,
            metrics=SceneMetrics(
                total_vertices=0,
                total_faces=0,
                total_objects=0,
                bounding_box_size=(0, 0, 0),
                bounding_box_volume=0,
                surface_area_estimate=0,
                has_thin_geometry=False,
                material_problem_count=0,
                max_dimension=0,
                min_dimension=0,
                aspect_ratio=1,
            ),
            recommended_cameras=0,
            recommendations=["No mesh objects selected"],
            warnings=["Select mesh objects to analyze"],
        )

    # Calculate metrics
    total_verts = sum(len(obj.data.vertices) for obj in mesh_objects)
    total_faces = sum(len(obj.data.polygons) for obj in mesh_objects)

    bbox_min, bbox_max = calculate_bounding_box(mesh_objects)
    bbox_size = tuple(bbox_max[i] - bbox_min[i] for i in range(3))
    bbox_volume = bbox_size[0] * bbox_size[1] * bbox_size[2]

    max_dim = max(bbox_size)
    min_dim = min(d for d in bbox_size if d > 0.001) if any(d > 0.001 for d in bbox_size) else 0.001
    aspect_ratio = max_dim / min_dim if min_dim > 0 else 1

    surface_area = estimate_surface_area(mesh_objects)
    has_thin = check_thin_geometry(mesh_objects)

    metrics = SceneMetrics(
        total_vertices=total_verts,
        total_faces=total_faces,
        total_objects=len(mesh_objects),
        bounding_box_size=bbox_size,
        bounding_box_volume=bbox_volume,
        surface_area_estimate=surface_area,
        has_thin_geometry=has_thin,
        material_problem_count=material_problems,
        max_dimension=max_dim,
        min_dimension=min_dim,
        aspect_ratio=aspect_ratio,
    )

    # Calculate score components (0-100 each)
    scores = {}
    recommendations = []
    warnings = []

    # 1. Geometry complexity (simpler is better for GS)
    # Ideal: 1K-100K faces. Very high poly can work but takes longer
    if total_faces < 100:
        scores['geometry'] = 50  # Too simple, may lack detail
        recommendations.append("Scene has very few faces - ensure adequate detail")
    elif total_faces < 10000:
        scores['geometry'] = 100  # Ideal range
    elif total_faces < 100000:
        scores['geometry'] = 90  # Still good
    elif total_faces < 1000000:
        scores['geometry'] = 70  # High but manageable
        recommendations.append("High poly count - consider decimation for faster training")
    else:
        scores['geometry'] = 50  # Very high
        warnings.append("Very high poly count may slow training significantly")

    # 2. Aspect ratio (extreme ratios are harder)
    if aspect_ratio < 3:
        scores['aspect'] = 100
    elif aspect_ratio < 5:
        scores['aspect'] = 80
        recommendations.append("Consider using multi-ring camera distribution")
    elif aspect_ratio < 10:
        scores['aspect'] = 60
        warnings.append("High aspect ratio - ensure cameras cover all dimensions")
    else:
        scores['aspect'] = 40
        warnings.append("Extreme aspect ratio may cause coverage gaps")

    # 3. Material compatibility
    if material_problems == 0:
        scores['materials'] = 100
    elif material_problems <= 2:
        scores['materials'] = 70
        recommendations.append("Review material warnings before capture")
    elif material_problems <= 5:
        scores['materials'] = 50
        warnings.append("Multiple material issues detected")
    else:
        scores['materials'] = 30
        warnings.append("Many problematic materials - consider material overrides")

    # 4. Thin geometry penalty
    if has_thin:
        scores['thin'] = 50
        warnings.append("Thin geometry detected - may not reconstruct well")
    else:
        scores['thin'] = 100

    # 5. Object count (many objects can be tricky)
    if len(mesh_objects) <= 5:
        scores['objects'] = 100
    elif len(mesh_objects) <= 20:
        scores['objects'] = 85
    elif len(mesh_objects) <= 50:
        scores['objects'] = 70
        recommendations.append("Many objects - ensure all are visible from cameras")
    else:
        scores['objects'] = 55
        warnings.append("Very many objects - consider grouping or simplifying")

    # Calculate weighted average
    weights = {
        'geometry': 0.25,
        'aspect': 0.20,
        'materials': 0.25,
        'thin': 0.15,
        'objects': 0.15,
    }

    final_score = sum(scores[k] * weights[k] for k in scores)
    final_score = int(round(final_score))

    # Determine grade
    if final_score >= 85:
        grade = SceneGrade.EXCELLENT
    elif final_score >= 70:
        grade = SceneGrade.GOOD
    elif final_score >= 50:
        grade = SceneGrade.FAIR
    else:
        grade = SceneGrade.POOR

    # Recommend camera count based on scene complexity
    # Base: 100 cameras for simple scenes
    # Scale up for complexity
    base_cameras = 100

    # Adjust for geometry
    if total_faces > 100000:
        base_cameras += 50

    # Adjust for aspect ratio
    if aspect_ratio > 3:
        base_cameras += int(aspect_ratio * 10)

    # Adjust for surface area (more area = more cameras)
    if surface_area > 100:  # Large scene
        base_cameras += 50

    # Cap at reasonable range
    recommended_cameras = max(50, min(300, base_cameras))

    # Add general recommendations
    if not recommendations and not warnings:
        recommendations.append("Scene looks good for GS capture")

    return SceneScore(
        grade=grade,
        score=final_score,
        metrics=metrics,
        recommended_cameras=recommended_cameras,
        recommendations=recommendations,
        warnings=warnings,
    )


def get_grade_icon(grade: SceneGrade) -> str:
    """Get Blender icon name for grade."""
    icons = {
        SceneGrade.EXCELLENT: 'CHECKMARK',
        SceneGrade.GOOD: 'SOLO_ON',
        SceneGrade.FAIR: 'ERROR',
        SceneGrade.POOR: 'CANCEL',
    }
    return icons.get(grade, 'QUESTION')


def get_grade_color(grade: SceneGrade) -> Tuple[float, float, float, float]:
    """Get RGBA color for grade display."""
    colors = {
        SceneGrade.EXCELLENT: (0.2, 0.8, 0.2, 1.0),  # Green
        SceneGrade.GOOD: (0.6, 0.8, 0.2, 1.0),       # Yellow-green
        SceneGrade.FAIR: (0.9, 0.7, 0.1, 1.0),       # Orange
        SceneGrade.POOR: (0.9, 0.2, 0.2, 1.0),       # Red
    }
    return colors.get(grade, (0.5, 0.5, 0.5, 1.0))


def format_score_display(score: SceneScore) -> str:
    """Format score for UI display."""
    bar_length = 10
    filled = int(score.score / 100 * bar_length)
    empty = bar_length - filled
    bar = "█" * filled + "░" * empty

    grade_text = score.grade.value.capitalize()
    return f"{bar} {score.score}% - {grade_text}"
