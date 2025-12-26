"""
Mesh and texture analysis for adaptive capture settings.
Analyzes scene complexity to recommend optimal capture parameters.
"""

import bpy
import math
from mathutils import Vector


class MeshAnalysis:
    """Results of mesh complexity analysis."""

    def __init__(self):
        self.vertex_count = 0
        self.face_count = 0
        self.surface_area = 0.0
        self.volume = 0.0
        self.bounding_box_volume = 0.0
        self.vertex_density = 0.0  # vertices per unit area
        self.detail_score = 0.0  # 0-1 complexity score
        self.curvature_variance = 0.0
        self.has_ngons = False
        self.edge_length_variance = 0.0

    def __repr__(self):
        return (f"MeshAnalysis(verts={self.vertex_count}, faces={self.face_count}, "
                f"area={self.surface_area:.2f}, detail={self.detail_score:.2f})")


class TextureAnalysis:
    """Results of texture quality analysis."""

    def __init__(self):
        self.max_resolution = 0
        self.total_textures = 0
        self.has_normal_maps = False
        self.has_displacement = False
        self.texture_score = 0.0  # 0-1 quality score
        self.estimated_detail_level = 'LOW'  # LOW, MEDIUM, HIGH, ULTRA

    def __repr__(self):
        return (f"TextureAnalysis(max_res={self.max_resolution}, "
                f"score={self.texture_score:.2f}, level={self.estimated_detail_level})")


class AdaptiveCaptureResult:
    """Recommended capture settings based on analysis."""

    def __init__(self):
        self.recommended_camera_count = 100
        self.recommended_resolution = (1920, 1080)
        self.recommended_distance_multiplier = 2.5
        self.detail_hotspots = []  # List of (position, weight) tuples
        self.quality_preset = 'STANDARD'
        self.estimated_render_time_minutes = 0
        self.warnings = []
        self.mesh_analysis = MeshAnalysis()
        self.texture_analysis = TextureAnalysis()


def calculate_mesh_surface_area(obj):
    """Calculate total surface area of a mesh in world space.

    Args:
        obj: Blender mesh object

    Returns:
        float: Total surface area in world units squared
    """
    if obj.type != 'MESH':
        return 0.0

    mesh = obj.data
    matrix = obj.matrix_world
    total_area = 0.0

    # Need to ensure mesh has calculated loop triangles
    mesh.calc_loop_triangles()

    for tri in mesh.loop_triangles:
        # Get world-space vertices
        v0 = matrix @ mesh.vertices[tri.vertices[0]].co
        v1 = matrix @ mesh.vertices[tri.vertices[1]].co
        v2 = matrix @ mesh.vertices[tri.vertices[2]].co

        # Calculate triangle area using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        area = edge1.cross(edge2).length / 2.0
        total_area += area

    return total_area


def calculate_vertex_curvature_variance(obj):
    """Estimate curvature variance to detect detailed areas.

    Higher variance indicates more complex surface topology.

    Args:
        obj: Blender mesh object

    Returns:
        float: Variance of normal angles between adjacent vertices
    """
    if obj.type != 'MESH':
        return 0.0

    mesh = obj.data

    # Need vertex normals
    if not mesh.vertex_normals:
        return 0.0

    # Sample a subset for performance
    sample_size = min(1000, len(mesh.vertices))
    step = max(1, len(mesh.vertices) // sample_size)

    normal_angles = []

    for i in range(0, len(mesh.vertices), step):
        vert = mesh.vertices[i]
        vert_normal = vert.normal

        # Compare with connected vertices' normals
        for edge in mesh.edges:
            if i in edge.vertices:
                other_idx = edge.vertices[0] if edge.vertices[1] == i else edge.vertices[1]
                if other_idx < len(mesh.vertices):
                    other_normal = mesh.vertices[other_idx].normal
                    # Angle between normals indicates curvature
                    dot = max(-1, min(1, vert_normal.dot(other_normal)))
                    angle = math.acos(dot)
                    normal_angles.append(angle)

    if not normal_angles:
        return 0.0

    # Calculate variance
    mean_angle = sum(normal_angles) / len(normal_angles)
    variance = sum((a - mean_angle) ** 2 for a in normal_angles) / len(normal_angles)

    return variance


def analyze_mesh_complexity(obj):
    """Analyze mesh to determine complexity and detail level.

    Args:
        obj: Blender mesh object

    Returns:
        MeshAnalysis: Analysis results with complexity metrics
    """
    analysis = MeshAnalysis()

    if obj.type != 'MESH':
        return analysis

    mesh = obj.data

    # Basic counts
    analysis.vertex_count = len(mesh.vertices)
    analysis.face_count = len(mesh.polygons)

    # Check for ngons
    analysis.has_ngons = any(len(p.vertices) > 4 for p in mesh.polygons)

    # Surface area
    analysis.surface_area = calculate_mesh_surface_area(obj)

    # Bounding box volume
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    dims = Vector((
        max(v.x for v in bbox) - min(v.x for v in bbox),
        max(v.y for v in bbox) - min(v.y for v in bbox),
        max(v.z for v in bbox) - min(v.z for v in bbox)
    ))
    analysis.bounding_box_volume = dims.x * dims.y * dims.z

    # Vertex density
    if analysis.surface_area > 0:
        analysis.vertex_density = analysis.vertex_count / analysis.surface_area

    # Curvature variance
    analysis.curvature_variance = calculate_vertex_curvature_variance(obj)

    # Calculate detail score (0-1)
    # Based on vertex count, density, and curvature
    vert_score = min(1.0, analysis.vertex_count / 100000)
    density_score = min(1.0, analysis.vertex_density / 1000)
    curvature_score = min(1.0, analysis.curvature_variance * 10)

    analysis.detail_score = (vert_score * 0.4 + density_score * 0.3 + curvature_score * 0.3)

    return analysis


def analyze_texture_quality(obj):
    """Analyze textures to estimate required capture quality.

    Args:
        obj: Blender mesh object

    Returns:
        TextureAnalysis: Analysis results with texture metrics
    """
    analysis = TextureAnalysis()

    if obj.type != 'MESH':
        return analysis

    # Check all materials on the object
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue

        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                analysis.total_textures += 1

                # Track max resolution
                width, height = node.image.size
                max_dim = max(width, height)
                if max_dim > analysis.max_resolution:
                    analysis.max_resolution = max_dim

                # Check for normal maps
                if any(link.to_socket.name.lower() in ['normal', 'bump']
                       for link in node.outputs[0].links):
                    analysis.has_normal_maps = True

            elif node.type == 'DISPLACEMENT':
                analysis.has_displacement = True

    # Calculate texture score
    if analysis.max_resolution >= 4096:
        analysis.texture_score = 1.0
        analysis.estimated_detail_level = 'ULTRA'
    elif analysis.max_resolution >= 2048:
        analysis.texture_score = 0.75
        analysis.estimated_detail_level = 'HIGH'
    elif analysis.max_resolution >= 1024:
        analysis.texture_score = 0.5
        analysis.estimated_detail_level = 'MEDIUM'
    else:
        analysis.texture_score = 0.25
        analysis.estimated_detail_level = 'LOW'

    # Boost for normal maps
    if analysis.has_normal_maps:
        analysis.texture_score = min(1.0, analysis.texture_score + 0.15)

    return analysis


def find_detail_hotspots(objects, num_hotspots=5):
    """Find areas of high detail that need extra camera coverage.

    Identifies regions with high vertex density or curvature for
    biased camera placement.

    Args:
        objects: List of Blender mesh objects
        num_hotspots: Maximum number of hotspots to return

    Returns:
        list: List of (Vector position, float weight) tuples
    """
    hotspots = []

    for obj in objects:
        if obj.type != 'MESH':
            continue

        mesh = obj.data
        matrix = obj.matrix_world

        # Sample vertices and check local density/curvature
        sample_step = max(1, len(mesh.vertices) // 100)

        for i in range(0, len(mesh.vertices), sample_step):
            vert = mesh.vertices[i]
            world_pos = matrix @ vert.co

            # Calculate local detail score based on normal variance
            neighbor_normals = []
            for edge in mesh.edges:
                if i in edge.vertices:
                    other_idx = edge.vertices[0] if edge.vertices[1] == i else edge.vertices[1]
                    if other_idx < len(mesh.vertices):
                        neighbor_normals.append(mesh.vertices[other_idx].normal)

            if neighbor_normals:
                avg_angle = 0
                for n in neighbor_normals:
                    dot = max(-1, min(1, vert.normal.dot(n)))
                    avg_angle += math.acos(dot)
                avg_angle /= len(neighbor_normals)

                # High angle variance = high detail
                if avg_angle > 0.3:  # ~17 degrees
                    hotspots.append((world_pos.copy(), avg_angle))

    # Sort by weight and return top hotspots
    hotspots.sort(key=lambda x: x[1], reverse=True)
    return hotspots[:num_hotspots]


def calculate_adaptive_settings(objects, quality_preset='AUTO'):
    """Calculate recommended capture settings based on scene analysis.

    Args:
        objects: List of Blender mesh objects to analyze
        quality_preset: One of 'AUTO', 'DRAFT', 'STANDARD', 'HIGH', 'ULTRA'

    Returns:
        AdaptiveCaptureResult: Recommended settings
    """
    result = AdaptiveCaptureResult()

    # Aggregate mesh analysis
    total_verts = 0
    total_faces = 0
    total_area = 0
    max_detail = 0

    for obj in objects:
        mesh_analysis = analyze_mesh_complexity(obj)
        total_verts += mesh_analysis.vertex_count
        total_faces += mesh_analysis.face_count
        total_area += mesh_analysis.surface_area
        max_detail = max(max_detail, mesh_analysis.detail_score)

    result.mesh_analysis.vertex_count = total_verts
    result.mesh_analysis.face_count = total_faces
    result.mesh_analysis.surface_area = total_area
    result.mesh_analysis.detail_score = max_detail

    # Aggregate texture analysis
    max_tex_res = 0
    for obj in objects:
        tex_analysis = analyze_texture_quality(obj)
        max_tex_res = max(max_tex_res, tex_analysis.max_resolution)
        if tex_analysis.has_normal_maps:
            result.texture_analysis.has_normal_maps = True
        if tex_analysis.has_displacement:
            result.texture_analysis.has_displacement = True

    result.texture_analysis.max_resolution = max_tex_res

    # Determine quality preset if AUTO
    if quality_preset == 'AUTO':
        combined_score = (max_detail + result.texture_analysis.texture_score) / 2
        if combined_score > 0.8:
            quality_preset = 'ULTRA'
        elif combined_score > 0.6:
            quality_preset = 'HIGH'
        elif combined_score > 0.3:
            quality_preset = 'STANDARD'
        else:
            quality_preset = 'DRAFT'

    result.quality_preset = quality_preset

    # Set recommendations based on preset
    presets = {
        'DRAFT': {
            'cameras': 50,
            'resolution': (1280, 720),
            'distance_mult': 3.0,
        },
        'STANDARD': {
            'cameras': 100,
            'resolution': (1920, 1080),
            'distance_mult': 2.5,
        },
        'HIGH': {
            'cameras': 200,
            'resolution': (2560, 1440),
            'distance_mult': 2.0,
        },
        'ULTRA': {
            'cameras': 300,
            'resolution': (3840, 2160),
            'distance_mult': 1.8,
        },
    }

    preset_config = presets.get(quality_preset, presets['STANDARD'])
    result.recommended_camera_count = preset_config['cameras']
    result.recommended_resolution = preset_config['resolution']
    result.recommended_distance_multiplier = preset_config['distance_mult']

    # Find detail hotspots
    result.detail_hotspots = find_detail_hotspots(objects)

    # Estimate render time (rough approximation)
    # Assumes ~2 seconds per image at 1080p with Cycles
    pixels = result.recommended_resolution[0] * result.recommended_resolution[1]
    base_time = 2.0  # seconds at 1080p
    pixel_factor = pixels / (1920 * 1080)
    result.estimated_render_time_minutes = int(
        (result.recommended_camera_count * base_time * pixel_factor) / 60
    )

    # Add warnings
    if total_verts > 1000000:
        result.warnings.append("Very high vertex count may slow rendering")
    if max_tex_res > 4096:
        result.warnings.append("Large textures may require significant VRAM")
    if result.texture_analysis.has_displacement:
        result.warnings.append("Displacement mapping detected - consider higher camera count")

    return result
