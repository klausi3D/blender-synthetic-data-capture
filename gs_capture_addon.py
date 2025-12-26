# SPDX-License-Identifier: MIT
# Gaussian Splatting Capture Addon for Blender 4.x/5.0
# Generates training images for 3D Gaussian Splatting from Blender scenes

bl_info = {
    "name": "GS Capture - Gaussian Splatting Training Data Generator",
    "author": "Custom",
    "version": (1, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > GS Capture",
    "description": "Generate camera captures for Gaussian Splatting training",
    "category": "Render",
}


def get_eevee_engine_name():
    """Get the correct Eevee engine name for this Blender version."""
    # Blender 4.2+ uses BLENDER_EEVEE_NEXT, but 5.0 reverted to BLENDER_EEVEE
    # Check what's actually available in this build
    import bpy
    # Get available render engines
    scene = bpy.context.scene if hasattr(bpy.context, 'scene') else None
    if scene:
        # Check if BLENDER_EEVEE_NEXT exists by trying to find it in enum items
        try:
            render_prop = scene.bl_rna.properties['render'].fixed_type.properties['engine']
            available_engines = [item.identifier for item in render_prop.enum_items]
            if 'BLENDER_EEVEE_NEXT' in available_engines:
                return 'BLENDER_EEVEE_NEXT'
            elif 'BLENDER_EEVEE' in available_engines:
                return 'BLENDER_EEVEE'
        except:
            pass

    # Fallback: version-based detection
    if bpy.app.version >= (4, 2, 0) and bpy.app.version < (5, 0, 0):
        return 'BLENDER_EEVEE_NEXT'
    else:
        return 'BLENDER_EEVEE'

import bpy
import math
import os
import json
import random
import mathutils
from mathutils import Vector, Matrix
from bpy.props import (
    StringProperty,
    IntProperty,
    FloatProperty,
    BoolProperty,
    EnumProperty,
    PointerProperty,
    CollectionProperty,
)
from bpy.types import PropertyGroup, Operator, Panel, UIList


# =============================================================================
# ADAPTIVE ANALYSIS SYSTEM
# =============================================================================

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
        return f"MeshAnalysis(verts={self.vertex_count}, faces={self.face_count}, area={self.surface_area:.2f}, detail={self.detail_score:.2f})"


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
        return f"TextureAnalysis(max_res={self.max_resolution}, score={self.texture_score:.2f}, level={self.estimated_detail_level})"


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
    """Calculate total surface area of a mesh in world space."""
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
    """Estimate curvature variance to detect detailed areas."""
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
    """Analyze mesh to determine complexity and detail level."""
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
    
    # Vertex density (vertices per unit surface area)
    if analysis.surface_area > 0:
        analysis.vertex_density = analysis.vertex_count / analysis.surface_area
    
    # Curvature variance
    analysis.curvature_variance = calculate_vertex_curvature_variance(obj)
    
    # Edge length variance (indicates detail variation)
    if mesh.edges:
        edge_lengths = []
        sample_size = min(500, len(mesh.edges))
        step = max(1, len(mesh.edges) // sample_size)
        
        for i in range(0, len(mesh.edges), step):
            edge = mesh.edges[i]
            v0 = obj.matrix_world @ mesh.vertices[edge.vertices[0]].co
            v1 = obj.matrix_world @ mesh.vertices[edge.vertices[1]].co
            edge_lengths.append((v1 - v0).length)
        
        if edge_lengths:
            mean_length = sum(edge_lengths) / len(edge_lengths)
            analysis.edge_length_variance = sum((l - mean_length) ** 2 for l in edge_lengths) / len(edge_lengths)
    
    # Calculate overall detail score (0-1)
    # Based on vertex count, density, and curvature
    vertex_score = min(1.0, analysis.vertex_count / 100000)  # Max at 100k verts
    density_score = min(1.0, analysis.vertex_density / 1000)  # Normalized density
    curvature_score = min(1.0, analysis.curvature_variance * 10)  # Normalized curvature
    
    analysis.detail_score = (vertex_score * 0.4 + density_score * 0.3 + curvature_score * 0.3)
    
    return analysis


def analyze_texture_quality(obj):
    """Analyze textures used by an object."""
    analysis = TextureAnalysis()
    
    if obj.type != 'MESH':
        return analysis
    
    max_res = 0
    texture_count = 0
    
    for slot in obj.material_slots:
        if not slot.material or not slot.material.use_nodes:
            continue
        
        for node in slot.material.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                texture_count += 1
                img = node.image
                res = max(img.size[0], img.size[1])
                max_res = max(max_res, res)
                
                # Check for normal/displacement maps by name or connection
                if node.outputs:
                    for link in node.outputs[0].links:
                        to_node = link.to_node
                        if to_node.type == 'NORMAL_MAP':
                            analysis.has_normal_maps = True
                        elif to_node.type == 'DISPLACEMENT':
                            analysis.has_displacement = True
                
                # Also check by image name
                name_lower = img.name.lower()
                if 'normal' in name_lower or 'nrm' in name_lower or 'nor' in name_lower:
                    analysis.has_normal_maps = True
                if 'disp' in name_lower or 'height' in name_lower or 'bump' in name_lower:
                    analysis.has_displacement = True
    
    analysis.max_resolution = max_res
    analysis.total_textures = texture_count
    
    # Calculate texture score
    if max_res >= 4096:
        analysis.texture_score = 1.0
        analysis.estimated_detail_level = 'ULTRA'
    elif max_res >= 2048:
        analysis.texture_score = 0.75
        analysis.estimated_detail_level = 'HIGH'
    elif max_res >= 1024:
        analysis.texture_score = 0.5
        analysis.estimated_detail_level = 'MEDIUM'
    elif max_res > 0:
        analysis.texture_score = 0.25
        analysis.estimated_detail_level = 'LOW'
    else:
        analysis.texture_score = 0.0
        analysis.estimated_detail_level = 'LOW'
    
    # Boost score if normal/displacement maps present
    if analysis.has_normal_maps:
        analysis.texture_score = min(1.0, analysis.texture_score + 0.15)
    if analysis.has_displacement:
        analysis.texture_score = min(1.0, analysis.texture_score + 0.1)
    
    return analysis


def find_detail_hotspots(obj, num_samples=50):
    """Find areas of high detail that need more camera coverage."""
    hotspots = []
    
    if obj.type != 'MESH':
        return hotspots
    
    mesh = obj.data
    matrix = obj.matrix_world
    
    if not mesh.polygons:
        return hotspots
    
    # Calculate face areas and identify small faces (high detail areas)
    face_data = []
    
    mesh.calc_loop_triangles()
    
    for poly in mesh.polygons:
        center = matrix @ poly.center
        # Approximate area from vertices
        if len(poly.vertices) >= 3:
            v0 = matrix @ mesh.vertices[poly.vertices[0]].co
            v1 = matrix @ mesh.vertices[poly.vertices[1]].co
            v2 = matrix @ mesh.vertices[poly.vertices[2]].co
            area = (v1 - v0).cross(v2 - v0).length / 2.0
            face_data.append((center, area, poly.normal.copy()))
    
    if not face_data:
        return hotspots
    
    # Find areas with small faces (high detail) and high curvature
    mean_area = sum(f[1] for f in face_data) / len(face_data)
    
    # Score each face: smaller area = higher detail, use normal variance too
    scored_faces = []
    for center, area, normal in face_data:
        # Inverse area score (smaller = higher score)
        area_score = mean_area / max(area, 0.0001)
        area_score = min(area_score, 10.0)  # Cap at 10x mean
        scored_faces.append((center, area_score))
    
    # Sort by score and take top samples
    scored_faces.sort(key=lambda x: x[1], reverse=True)
    
    # Cluster nearby hotspots
    selected = []
    min_distance = 0.1  # Minimum distance between hotspots
    
    for center, score in scored_faces:
        # Check if too close to existing hotspot
        too_close = False
        for existing, _ in selected:
            if (center - existing).length < min_distance:
                too_close = True
                break
        
        if not too_close:
            selected.append((center, score))
            if len(selected) >= num_samples:
                break
    
    # Normalize weights
    if selected:
        max_score = max(s[1] for s in selected)
        hotspots = [(pos, score / max_score) for pos, score in selected]
    
    return hotspots


def calculate_adaptive_settings(objects, quality_preset='AUTO'):
    """Calculate optimal capture settings based on object analysis."""
    result = AdaptiveCaptureResult()
    
    if not objects:
        return result
    
    # Analyze all objects
    total_mesh = MeshAnalysis()
    total_texture = TextureAnalysis()
    all_hotspots = []
    
    for obj in objects:
        if obj.type != 'MESH':
            continue
        
        mesh_analysis = analyze_mesh_complexity(obj)
        texture_analysis = analyze_texture_quality(obj)
        hotspots = find_detail_hotspots(obj)
        
        # Accumulate mesh stats
        total_mesh.vertex_count += mesh_analysis.vertex_count
        total_mesh.face_count += mesh_analysis.face_count
        total_mesh.surface_area += mesh_analysis.surface_area
        total_mesh.bounding_box_volume += mesh_analysis.bounding_box_volume
        total_mesh.detail_score = max(total_mesh.detail_score, mesh_analysis.detail_score)
        total_mesh.curvature_variance = max(total_mesh.curvature_variance, mesh_analysis.curvature_variance)
        
        # Track best texture quality
        if texture_analysis.max_resolution > total_texture.max_resolution:
            total_texture.max_resolution = texture_analysis.max_resolution
            total_texture.estimated_detail_level = texture_analysis.estimated_detail_level
        total_texture.total_textures += texture_analysis.total_textures
        total_texture.texture_score = max(total_texture.texture_score, texture_analysis.texture_score)
        total_texture.has_normal_maps = total_texture.has_normal_maps or texture_analysis.has_normal_maps
        total_texture.has_displacement = total_texture.has_displacement or texture_analysis.has_displacement
        
        all_hotspots.extend(hotspots)
    
    # Recalculate density for combined mesh
    if total_mesh.surface_area > 0:
        total_mesh.vertex_density = total_mesh.vertex_count / total_mesh.surface_area
    
    result.mesh_analysis = total_mesh
    result.texture_analysis = total_texture
    result.detail_hotspots = all_hotspots
    
    # Determine quality preset if AUTO
    if quality_preset == 'AUTO':
        combined_score = (total_mesh.detail_score * 0.6 + total_texture.texture_score * 0.4)
        
        if combined_score >= 0.8:
            quality_preset = 'ULTRA'
        elif combined_score >= 0.6:
            quality_preset = 'HIGH'
        elif combined_score >= 0.3:
            quality_preset = 'STANDARD'
        else:
            quality_preset = 'DRAFT'
    
    result.quality_preset = quality_preset
    
    # Calculate recommended settings based on preset and analysis
    presets = {
        'DRAFT': {
            'base_cameras': 36,
            'resolution': (1280, 720),
            'distance_mult': 3.0,
        },
        'STANDARD': {
            'base_cameras': 100,
            'resolution': (1920, 1080),
            'distance_mult': 2.5,
        },
        'HIGH': {
            'base_cameras': 200,
            'resolution': (2560, 1440),
            'distance_mult': 2.5,
        },
        'ULTRA': {
            'base_cameras': 400,
            'resolution': (3840, 2160),
            'distance_mult': 2.0,
        },
    }
    
    preset_config = presets[quality_preset]
    
    # Adjust camera count based on surface area and complexity
    base_cameras = preset_config['base_cameras']
    
    # Scale by surface area (more area = more cameras needed)
    # Reference: 1 sq meter needs base cameras, scale from there
    area_factor = math.sqrt(total_mesh.surface_area) / 2.0  # Sqrt to prevent explosion
    area_factor = max(0.5, min(area_factor, 3.0))  # Clamp between 0.5x and 3x
    
    # Scale by detail score
    detail_factor = 0.7 + (total_mesh.detail_score * 0.6)  # 0.7x to 1.3x
    
    # Scale by texture quality
    texture_factor = 0.8 + (total_texture.texture_score * 0.4)  # 0.8x to 1.2x
    
    recommended_cameras = int(base_cameras * area_factor * detail_factor * texture_factor)
    recommended_cameras = max(24, min(recommended_cameras, 500))  # Clamp to reasonable range
    
    result.recommended_camera_count = recommended_cameras
    result.recommended_resolution = preset_config['resolution']
    result.recommended_distance_multiplier = preset_config['distance_mult']
    
    # Estimate render time (very rough)
    # Assume 2 seconds per frame at 1080p with Cycles
    res_factor = (result.recommended_resolution[0] * result.recommended_resolution[1]) / (1920 * 1080)
    result.estimated_render_time_minutes = int((recommended_cameras * 2 * res_factor) / 60)
    
    # Generate warnings
    if total_mesh.vertex_count > 500000:
        result.warnings.append(f"High vertex count ({total_mesh.vertex_count:,}). Consider decimation for faster GS training.")
    
    if total_mesh.vertex_count < 100:
        result.warnings.append("Very low vertex count. Object may lack detail for GS reconstruction.")
    
    if total_texture.max_resolution > 4096:
        result.warnings.append(f"Very high texture resolution ({total_texture.max_resolution}px). May not improve GS quality.")
    
    if total_texture.total_textures == 0:
        result.warnings.append("No textures found. Consider if original materials mode is appropriate.")
    
    if total_mesh.has_ngons:
        result.warnings.append("Mesh contains ngons. Consider triangulating for consistent results.")
    
    return result


def generate_adaptive_camera_positions(center, radius, camera_count, hotspots=None, distribution='FIBONACCI', elevation_limits=(-60, 60)):
    """Generate camera positions with adaptive density based on detail hotspots."""
    
    # Start with base distribution
    if distribution == 'FIBONACCI':
        base_points = fibonacci_sphere_points(camera_count)
    else:
        base_points = fibonacci_sphere_points(camera_count)
    
    # Filter by elevation
    min_y = math.sin(math.radians(elevation_limits[0]))
    max_y = math.sin(math.radians(elevation_limits[1]))
    base_points = [p for p in base_points if min_y <= p.y <= max_y]
    
    if not hotspots or len(hotspots) == 0:
        return base_points
    
    # Bias camera positions toward hotspots
    # Add extra cameras looking at high-detail areas
    hotspot_cameras = int(camera_count * 0.2)  # 20% extra for hotspots
    
    biased_points = list(base_points)
    
    for hotspot_pos, weight in hotspots[:hotspot_cameras]:
        if weight < 0.3:  # Skip low-weight hotspots
            continue
        
        # Calculate direction from center to hotspot
        hotspot_dir = (hotspot_pos - center).normalized()
        
        # Create camera position opposite to hotspot (looking at it)
        # Add some random offset for variety
        import random
        offset_angle = random.uniform(-0.3, 0.3)
        
        # Rotate direction slightly
        rot_axis = Vector((0, 1, 0)).cross(hotspot_dir)
        if rot_axis.length > 0.001:
            rot_axis.normalize()
            rot_mat = Matrix.Rotation(offset_angle, 3, rot_axis)
            cam_dir = rot_mat @ (-hotspot_dir)
        else:
            cam_dir = -hotspot_dir
        
        biased_points.append(cam_dir)
    
    return biased_points


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def fibonacci_sphere_points(num_points, randomize=False):
    """Generate evenly distributed points on a sphere using Fibonacci spiral."""
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # Golden angle
    
    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)
        theta = phi * i
        
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        
        points.append(Vector((x, y, z)))
    
    return points


def get_object_bounds_center_radius(obj):
    """Calculate bounding sphere center and radius for an object."""
    # Get world-space bounding box corners
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    # Calculate center
    center = sum(bbox_corners, Vector()) / 8
    
    # Calculate radius (max distance from center to any corner)
    radius = max((corner - center).length for corner in bbox_corners)
    
    return center, radius


def get_objects_combined_bounds(objects):
    """Get combined bounding sphere for multiple objects."""
    if not objects:
        return Vector((0, 0, 0)), 1.0

    all_corners = []
    for obj in objects:
        if obj.type == 'MESH':
            bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            all_corners.extend(bbox_corners)

    if not all_corners:
        return Vector((0, 0, 0)), 1.0

    # Calculate center
    center = sum(all_corners, Vector()) / len(all_corners)

    # Calculate radius
    radius = max((corner - center).length for corner in all_corners)

    return center, radius


def get_collection_objects(collection, include_nested=True, mesh_only=True):
    """Get all objects from a collection, optionally including nested collections."""
    objects = []

    # Get direct objects
    for obj in collection.objects:
        if mesh_only and obj.type != 'MESH':
            continue
        if obj.visible_get():  # Only include visible objects
            objects.append(obj)

    # Recursively get from child collections
    if include_nested:
        for child_collection in collection.children:
            objects.extend(get_collection_objects(child_collection, include_nested=True, mesh_only=mesh_only))

    return objects


def create_camera_at_position(context, position, look_at, name="GS_Camera"):
    """Create a camera at position, looking at target."""
    # Create camera data
    cam_data = bpy.data.cameras.new(name=name)
    cam_obj = bpy.data.objects.new(name, cam_data)
    
    # Link to scene
    context.scene.collection.objects.link(cam_obj)
    
    # Position camera
    cam_obj.location = position
    
    # Point camera at target
    direction = look_at - position
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()
    
    return cam_obj


def setup_neutral_lighting(context, settings):
    """Configure neutral lighting for capture."""
    scene = context.scene
    world = scene.world
    
    if world is None:
        world = bpy.data.worlds.new("GS_Capture_World")
        scene.world = world
    
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    # Clear existing nodes
    nodes.clear()
    
    # Create background node
    bg_node = nodes.new('ShaderNodeBackground')
    output_node = nodes.new('ShaderNodeOutputWorld')
    
    if settings.lighting_mode == 'WHITE':
        bg_node.inputs['Color'].default_value = (1, 1, 1, 1)
        bg_node.inputs['Strength'].default_value = settings.background_strength
    elif settings.lighting_mode == 'GRAY':
        gray = settings.gray_value
        bg_node.inputs['Color'].default_value = (gray, gray, gray, 1)
        bg_node.inputs['Strength'].default_value = settings.background_strength
    elif settings.lighting_mode == 'HDR':
        # Add environment texture node
        env_node = nodes.new('ShaderNodeTexEnvironment')
        if settings.hdr_path and os.path.exists(bpy.path.abspath(settings.hdr_path)):
            env_node.image = bpy.data.images.load(bpy.path.abspath(settings.hdr_path))
        bg_node.inputs['Strength'].default_value = settings.hdr_strength
        links.new(env_node.outputs['Color'], bg_node.inputs['Color'])
    
    links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    
    # Disable all lights if requested
    if settings.disable_scene_lights:
        for obj in scene.objects:
            if obj.type == 'LIGHT':
                obj.hide_render = True
                obj.hide_viewport = True


def restore_lighting(context, stored_light_states):
    """Restore original lighting state."""
    for obj_name, (hide_render, hide_viewport) in stored_light_states.items():
        if obj_name in bpy.data.objects:
            obj = bpy.data.objects[obj_name]
            obj.hide_render = hide_render
            obj.hide_viewport = hide_viewport


def store_lighting_state(context):
    """Store current lighting state for restoration."""
    states = {}
    for obj in context.scene.objects:
        if obj.type == 'LIGHT':
            states[obj.name] = (obj.hide_render, obj.hide_viewport)
    return states


def override_materials(objects, override_type='DIFFUSE'):
    """Override materials on objects for neutral capture."""
    original_materials = {}
    
    for obj in objects:
        if obj.type != 'MESH':
            continue
        
        original_materials[obj.name] = list(obj.data.materials)
        
        if override_type == 'DIFFUSE':
            # Create neutral diffuse material
            mat_name = "GS_Neutral_Diffuse"
            if mat_name not in bpy.data.materials:
                mat = bpy.data.materials.new(name=mat_name)
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                nodes.clear()
                
                bsdf = nodes.new('ShaderNodeBsdfDiffuse')
                bsdf.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1)
                output = nodes.new('ShaderNodeOutputMaterial')
                links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
            else:
                mat = bpy.data.materials[mat_name]
            
            # Apply to all slots
            obj.data.materials.clear()
            obj.data.materials.append(mat)
        
        elif override_type == 'VERTEX_COLOR':
            # Keep vertex colors if present, otherwise use neutral
            pass  # Implement if needed
    
    return original_materials


def restore_materials(original_materials):
    """Restore original materials."""
    for obj_name, materials in original_materials.items():
        if obj_name in bpy.data.objects:
            obj = bpy.data.objects[obj_name]
            if obj.type == 'MESH':
                obj.data.materials.clear()
                for mat in materials:
                    obj.data.materials.append(mat)


def export_colmap_cameras(cameras, camera_data, output_path, image_width, image_height):
    """Export camera parameters in COLMAP format compatible with LichtFeld Studio."""
    # cameras.txt - camera intrinsics
    cameras_file = os.path.join(output_path, "cameras.txt")
    with open(cameras_file, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")

        # Use PINHOLE model (fx, fy, cx, cy)
        # Get focal length from first camera
        cam = cameras[0]
        cam_obj = cam.data

        # Calculate focal length in pixels
        sensor_width = cam_obj.sensor_width
        focal_mm = cam_obj.lens
        focal_px = (focal_mm / sensor_width) * image_width

        cx = image_width / 2
        cy = image_height / 2

        f.write(f"1 PINHOLE {image_width} {image_height} {focal_px:.6f} {focal_px:.6f} {cx:.6f} {cy:.6f}\n")

    # images.txt - camera extrinsics
    # IMPORTANT: LichtFeld Studio requires even number of non-comment lines
    # Each image needs: pose line + points line (use space " " to prevent removal)
    images_file = os.path.join(output_path, "images.txt")
    with open(images_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for i, cam in enumerate(cameras):
            # Get world matrix
            mat = cam.matrix_world

            # Blender coordinate system: Y-forward, Z-up
            # COLMAP/OpenCV coordinate system: Z-forward, -Y-up
            # We need to convert from Blender to COLMAP

            # Rotation matrix to convert from Blender to COLMAP camera space
            # Blender camera looks down -Z with Y up
            # COLMAP camera looks down +Z with -Y up
            # This flips Y and Z axes
            blender_to_colmap = Matrix((
                (1,  0,  0, 0),
                (0, -1,  0, 0),
                (0,  0, -1, 0),
                (0,  0,  0, 1)
            ))

            # Apply coordinate conversion and invert for world-to-camera
            mat_colmap = blender_to_colmap @ mat.inverted()

            # Extract rotation as quaternion
            rot = mat_colmap.to_quaternion()

            # Extract translation
            trans = mat_colmap.translation

            image_name = f"image_{i:04d}.png"

            f.write(f"{i+1} {rot.w:.6f} {rot.x:.6f} {rot.y:.6f} {rot.z:.6f} ")
            f.write(f"{trans.x:.6f} {trans.y:.6f} {trans.z:.6f} 1 {image_name}\n")
            # Use single space for points line - prevents LichtFeld from removing it as "empty"
            f.write(" \n")

    # points3D.txt - generate initial points for Gaussian Splatting training
    # For synthetic data we generate random points in the scene volume
    points_file = os.path.join(output_path, "points3D.txt")
    _generate_initial_points(cameras, points_file)


def _generate_initial_points(cameras, output_file, num_points=5000):
    """Generate initial 3D points for Gaussian Splatting training.

    For synthetic Blender captures, we generate random points in the scene volume
    since we don't have COLMAP feature matching.
    """
    import random

    # Calculate scene bounds from camera positions
    cam_positions = []
    for cam in cameras:
        pos = cam.matrix_world.translation
        cam_positions.append((pos.x, pos.y, pos.z))

    if not cam_positions:
        # Fallback: create points around origin
        center = (0, 0, 0)
        radius = 5.0
    else:
        # Calculate center by finding where cameras are looking
        # Get camera forward directions and find intersection point
        cam_targets = []
        for cam in cameras:
            pos = cam.matrix_world.translation
            # Camera looks along -Z in local space
            forward = cam.matrix_world.to_3x3() @ Vector((0, 0, -1))
            forward.normalize()
            # Estimate target: cast ray from camera along forward direction
            # Use distance from camera to origin as reference
            dist_to_origin = pos.length
            target = pos + forward * dist_to_origin
            cam_targets.append((target.x, target.y, target.z))

        # Camera center (where cameras are positioned)
        xs = [p[0] for p in cam_positions]
        ys = [p[1] for p in cam_positions]
        zs = [p[2] for p in cam_positions]
        cam_center = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))

        # Target center (where cameras are looking)
        txs = [t[0] for t in cam_targets]
        tys = [t[1] for t in cam_targets]
        tzs = [t[2] for t in cam_targets]
        target_center = (sum(txs)/len(txs), sum(tys)/len(tys), sum(tzs)/len(tzs))

        # Max distance from camera center to any camera
        max_cam_dist = max(
            math.sqrt((p[0]-cam_center[0])**2 + (p[1]-cam_center[1])**2 + (p[2]-cam_center[2])**2)
            for p in cam_positions
        )

        # Center points at where cameras are looking, radius ~30% of camera spread
        center = target_center
        radius = max_cam_dist * 0.4

    # Generate points
    random.seed(42)  # Reproducible
    points = []

    # Mix of volume and surface points
    for i in range(num_points):
        if i < num_points * 0.6:
            # Volume points (60%)
            while True:
                x = (random.random() - 0.5) * 2 * radius + center[0]
                y = (random.random() - 0.5) * 2 * radius + center[1]
                z = (random.random() - 0.5) * 2 * radius + center[2]
                dist = math.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
                if dist <= radius:
                    break
        else:
            # Surface points (40%) - Fibonacci sphere
            phi = math.pi * (3.0 - math.sqrt(5.0))
            idx = i - int(num_points * 0.6)
            total_surface = num_points - int(num_points * 0.6)

            y_norm = 1 - (idx / float(total_surface - 1)) * 2
            r = math.sqrt(1 - y_norm * y_norm)
            theta = phi * idx

            x = center[0] + math.cos(theta) * r * radius * 0.8
            y = center[1] + y_norm * radius * 0.8
            z = center[2] + math.sin(theta) * r * radius * 0.8

        # Random gray color
        r_col = random.randint(100, 200)
        g_col = random.randint(100, 200)
        b_col = random.randint(100, 200)

        points.append((x, y, z, r_col, g_col, b_col))

    # Write points3D.txt
    # Note: Points are generated in Blender world space
    # COLMAP world space has Y and Z flipped compared to Blender
    # Blender: Z-up, Y-forward | COLMAP: Y-up (negated), Z-forward (negated)
    with open(output_file, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")

        for i, (x, y, z, r, g, b) in enumerate(points):
            # Convert from Blender to COLMAP world space
            # Blender (x, y, z) -> COLMAP (x, -y, -z)
            colmap_x = x
            colmap_y = -y
            colmap_z = -z
            # POINT3D_ID X Y Z R G B ERROR (no tracks for synthetic data)
            f.write(f"{i+1} {colmap_x:.6f} {colmap_y:.6f} {colmap_z:.6f} {r} {g} {b} 0.0\n")


def export_transforms_json(cameras, output_path, image_width, image_height):
    """Export camera transforms in NeRF/3DGS JSON format (transforms.json)."""
    transforms = {
        "camera_angle_x": 0,
        "frames": []
    }
    
    # Calculate camera angle from first camera
    if cameras:
        cam = cameras[0].data
        focal_mm = cam.lens
        sensor_width = cam.sensor_width
        transforms["camera_angle_x"] = 2 * math.atan(sensor_width / (2 * focal_mm))
    
    for i, cam in enumerate(cameras):
        mat = cam.matrix_world
        
        # Convert to list format
        transform_matrix = [
            [mat[0][0], mat[0][1], mat[0][2], mat[0][3]],
            [mat[1][0], mat[1][1], mat[1][2], mat[1][3]],
            [mat[2][0], mat[2][1], mat[2][2], mat[2][3]],
            [mat[3][0], mat[3][1], mat[3][2], mat[3][3]],
        ]
        
        frame = {
            "file_path": f"./images/image_{i:04d}",
            "transform_matrix": transform_matrix
        }
        transforms["frames"].append(frame)
    
    # Write transforms.json
    json_path = os.path.join(output_path, "transforms.json")
    with open(json_path, 'w') as f:
        json.dump(transforms, f, indent=2)


# =============================================================================
# PROPERTY GROUPS
# =============================================================================

class GSCaptureObjectItem(PropertyGroup):
    """Item in the object group list."""
    obj: PointerProperty(type=bpy.types.Object)


class GSCaptureObjectGroup(PropertyGroup):
    """Group of objects to capture together."""
    name: StringProperty(name="Group Name", default="Object Group")
    objects: CollectionProperty(type=GSCaptureObjectItem)
    expanded: BoolProperty(default=True)


class GSCaptureSettings(PropertyGroup):
    """Main settings for GS Capture."""
    
    # Output settings
    output_path: StringProperty(
        name="Output Path",
        description="Directory to save rendered images",
        default="//gs_capture/",
        subtype='DIR_PATH'
    )
    
    # Camera settings
    camera_count: IntProperty(
        name="Camera Count",
        description="Number of cameras to generate",
        default=100,
        min=8,
        max=500
    )
    
    camera_distance_mode: EnumProperty(
        name="Distance Mode",
        items=[
            ('AUTO', "Auto", "Automatically calculate distance from object bounds"),
            ('MANUAL', "Manual", "Manually set camera distance"),
        ],
        default='AUTO'
    )
    
    camera_distance: FloatProperty(
        name="Camera Distance",
        description="Distance from object center (manual mode)",
        default=3.0,
        min=0.1,
        soft_max=100.0
    )
    
    camera_distance_multiplier: FloatProperty(
        name="Distance Multiplier",
        description="Multiply auto-calculated distance by this factor",
        default=2.5,
        min=1.0,
        soft_max=10.0
    )
    
    camera_distribution: EnumProperty(
        name="Distribution",
        items=[
            ('FIBONACCI', "Fibonacci Sphere", "Even distribution using Fibonacci spiral"),
            ('HEMISPHERE_TOP', "Hemisphere (Top)", "Upper hemisphere only"),
            ('HEMISPHERE_BOTTOM', "Hemisphere (Bottom)", "Lower hemisphere only"),
            ('RING', "Ring", "Cameras in a horizontal ring"),
            ('MULTI_RING', "Multi-Ring", "Multiple horizontal rings at different heights"),
        ],
        default='FIBONACCI'
    )
    
    ring_count: IntProperty(
        name="Ring Count",
        description="Number of rings for multi-ring distribution",
        default=5,
        min=1,
        max=20
    )
    
    min_elevation: FloatProperty(
        name="Min Elevation",
        description="Minimum elevation angle (degrees)",
        default=-60.0,
        min=-90.0,
        max=90.0
    )
    
    max_elevation: FloatProperty(
        name="Max Elevation",
        description="Maximum elevation angle (degrees)",
        default=60.0,
        min=-90.0,
        max=90.0
    )
    
    focal_length: FloatProperty(
        name="Focal Length",
        description="Camera focal length in mm",
        default=50.0,
        min=1.0,
        soft_max=200.0
    )
    
    # Render settings
    render_resolution_x: IntProperty(
        name="Resolution X",
        description="Render width in pixels",
        default=1920,
        min=64,
        max=8192
    )
    
    render_resolution_y: IntProperty(
        name="Resolution Y",
        description="Render height in pixels",
        default=1080,
        min=64,
        max=8192
    )
    
    render_samples: IntProperty(
        name="Samples",
        description="Render samples (Cycles) or quality (Eevee)",
        default=64,
        min=1,
        max=4096
    )
    
    render_engine: EnumProperty(
        name="Render Engine",
        items=[
            ('CYCLES', "Cycles", "Path tracing renderer"),
            ('EEVEE', "Eevee", "Real-time renderer (faster)"),
        ],
        default='CYCLES'
    )
    
    file_format: EnumProperty(
        name="File Format",
        items=[
            ('PNG', "PNG", "Lossless with alpha support"),
            ('JPEG', "JPEG", "Smaller file size, no alpha"),
            ('OPEN_EXR', "OpenEXR", "HDR format with depth data"),
        ],
        default='PNG'
    )

    transparent_background: BoolProperty(
        name="Transparent Background",
        description="Render with transparent background (alpha channel). Recommended for training clean objects without background contamination",
        default=True
    )

    # Lighting settings
    lighting_mode: EnumProperty(
        name="Lighting Mode",
        items=[
            ('WHITE', "White Background", "Pure white environment"),
            ('GRAY', "Gray Background", "Neutral gray environment"),
            ('HDR', "HDR Environment", "Use an HDR image for lighting"),
            ('KEEP', "Keep Scene Lighting", "Don't modify lighting"),
        ],
        default='WHITE'
    )
    
    background_strength: FloatProperty(
        name="Background Strength",
        description="Environment light strength",
        default=1.0,
        min=0.0,
        soft_max=10.0
    )
    
    gray_value: FloatProperty(
        name="Gray Value",
        description="Gray level (0=black, 1=white)",
        default=0.5,
        min=0.0,
        max=1.0
    )
    
    hdr_path: StringProperty(
        name="HDR Path",
        description="Path to HDR environment image",
        default="",
        subtype='FILE_PATH'
    )
    
    hdr_strength: FloatProperty(
        name="HDR Strength",
        description="HDR environment strength",
        default=1.0,
        min=0.0,
        soft_max=10.0
    )
    
    disable_scene_lights: BoolProperty(
        name="Disable Scene Lights",
        description="Hide all lights in the scene during capture",
        default=True
    )
    
    # Material settings
    material_mode: EnumProperty(
        name="Material Mode",
        items=[
            ('ORIGINAL', "Original Materials", "Keep original materials"),
            ('DIFFUSE', "Neutral Diffuse", "Override with neutral gray diffuse"),
            ('MATCAP', "Matcap", "Use matcap for consistent shading"),
        ],
        default='ORIGINAL'
    )
    
    # Export settings
    export_colmap: BoolProperty(
        name="Export COLMAP",
        description="Export camera data in COLMAP format",
        default=True
    )
    
    export_transforms_json: BoolProperty(
        name="Export transforms.json",
        description="Export camera data in NeRF/3DGS JSON format",
        default=True
    )
    
    export_depth: BoolProperty(
        name="Export Depth Maps",
        description="Render and export depth maps",
        default=False
    )
    
    export_masks: BoolProperty(
        name="Export Object Masks",
        description="Render and export object masks",
        default=False
    )
    
    # Batch settings
    batch_mode: EnumProperty(
        name="Batch Mode",
        items=[
            ('SELECTED', "Selected Objects", "Capture selected objects as one"),
            ('COLLECTION', "Active Collection", "Capture entire collection as one object"),
            ('EACH_SELECTED', "Each Selected", "Capture each selected object separately"),
            ('COLLECTIONS', "By Collection", "Capture each collection separately"),
            ('GROUPS', "Object Groups", "Use defined object groups"),
        ],
        default='SELECTED'
    )

    # Collection selection for COLLECTION mode
    target_collection: StringProperty(
        name="Target Collection",
        description="Collection to capture (leave empty for active collection)",
        default=""
    )

    include_children: BoolProperty(
        name="Include Children",
        description="Include child objects when capturing",
        default=True
    )

    include_nested_collections: BoolProperty(
        name="Include Nested",
        description="Include objects from nested/child collections",
        default=True
    )
    
    # Object groups for batch processing
    object_groups: CollectionProperty(type=GSCaptureObjectGroup)
    active_group_index: IntProperty(default=0)
    
    # Progress tracking
    is_rendering: BoolProperty(default=False)
    render_progress: FloatProperty(default=0.0, min=0.0, max=100.0)
    current_render_info: StringProperty(default="")
    
    # Adaptive capture settings
    use_adaptive_capture: BoolProperty(
        name="Adaptive Capture",
        description="Automatically adjust settings based on mesh/texture analysis",
        default=True
    )
    
    adaptive_quality_preset: EnumProperty(
        name="Quality Preset",
        items=[
            ('AUTO', "Auto Detect", "Automatically determine quality based on analysis"),
            ('DRAFT', "Draft", "Quick capture for testing (36-50 cameras)"),
            ('STANDARD', "Standard", "Balanced quality and speed (100-150 cameras)"),
            ('HIGH', "High", "High quality capture (200-300 cameras)"),
            ('ULTRA', "Ultra", "Maximum quality (400+ cameras)"),
        ],
        default='AUTO'
    )
    
    adaptive_use_hotspots: BoolProperty(
        name="Use Detail Hotspots",
        description="Add extra cameras focused on high-detail areas",
        default=True
    )
    
    adaptive_hotspot_bias: FloatProperty(
        name="Hotspot Bias",
        description="How much to bias camera placement toward detailed areas (0=none, 1=maximum)",
        default=0.3,
        min=0.0,
        max=1.0
    )
    
    # Analysis results display (read-only info)
    analysis_vertex_count: IntProperty(default=0)
    analysis_face_count: IntProperty(default=0)
    analysis_surface_area: FloatProperty(default=0.0)
    analysis_detail_score: FloatProperty(default=0.0)
    analysis_texture_resolution: IntProperty(default=0)
    analysis_texture_score: FloatProperty(default=0.0)
    analysis_recommended_cameras: IntProperty(default=100)
    analysis_recommended_resolution: StringProperty(default="1920x1080")
    analysis_quality_preset: StringProperty(default="STANDARD")
    analysis_warnings: StringProperty(default="")
    analysis_render_time_estimate: StringProperty(default="")


# =============================================================================
# OPERATORS
# =============================================================================

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
        
        # Run adaptive analysis
        result = calculate_adaptive_settings(selected, settings.adaptive_quality_preset)
        
        # Store results in settings for display
        settings.analysis_vertex_count = result.mesh_analysis.vertex_count
        settings.analysis_face_count = result.mesh_analysis.face_count
        settings.analysis_surface_area = result.mesh_analysis.surface_area
        settings.analysis_detail_score = result.mesh_analysis.detail_score
        settings.analysis_texture_resolution = result.texture_analysis.max_resolution
        settings.analysis_texture_score = result.texture_analysis.texture_score
        settings.analysis_recommended_cameras = result.recommended_camera_count
        settings.analysis_recommended_resolution = f"{result.recommended_resolution[0]}x{result.recommended_resolution[1]}"
        settings.analysis_quality_preset = result.quality_preset
        settings.analysis_warnings = " | ".join(result.warnings) if result.warnings else "None"
        settings.analysis_render_time_estimate = f"~{result.estimated_render_time_minutes} min"
        
        self.report({'INFO'}, f"Analysis complete: {result.quality_preset} quality, {result.recommended_camera_count} cameras recommended")
        return {'FINISHED'}


class GSCAPTURE_OT_analyze_scene(Operator):
    """Analyze entire scene to plan batch capture."""
    bl_idname = "gs_capture.analyze_scene"
    bl_label = "Analyze Scene"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        settings = context.scene.gs_capture_settings
        
        # Collect all mesh objects
        all_meshes = [obj for obj in context.scene.objects if obj.type == 'MESH']
        
        if not all_meshes:
            self.report({'ERROR'}, "No mesh objects in scene")
            return {'CANCELLED'}
        
        # Analyze by collection
        collection_reports = []
        total_cameras = 0
        total_render_time = 0
        
        for collection in bpy.data.collections:
            mesh_objects = [obj for obj in collection.objects if obj.type == 'MESH' and obj.visible_get()]
            if not mesh_objects:
                continue
            
            result = calculate_adaptive_settings(mesh_objects, settings.adaptive_quality_preset)
            
            collection_reports.append({
                'name': collection.name,
                'object_count': len(mesh_objects),
                'vertex_count': result.mesh_analysis.vertex_count,
                'face_count': result.mesh_analysis.face_count,
                'surface_area': result.mesh_analysis.surface_area,
                'detail_score': result.mesh_analysis.detail_score,
                'texture_resolution': result.texture_analysis.max_resolution,
                'recommended_cameras': result.recommended_camera_count,
                'recommended_resolution': result.recommended_resolution,
                'quality_preset': result.quality_preset,
                'render_time_minutes': result.estimated_render_time_minutes,
                'warnings': result.warnings
            })
            
            total_cameras += result.recommended_camera_count
            total_render_time += result.estimated_render_time_minutes
        
        # Store scene analysis in a JSON file
        output_path = bpy.path.abspath(settings.output_path)
        os.makedirs(output_path, exist_ok=True)
        
        scene_report = {
            'scene_name': context.scene.name,
            'total_collections': len(collection_reports),
            'total_mesh_objects': len(all_meshes),
            'total_cameras_needed': total_cameras,
            'total_estimated_render_time_minutes': total_render_time,
            'collections': collection_reports
        }
        
        report_path = os.path.join(output_path, "scene_analysis_report.json")
        with open(report_path, 'w') as f:
            json.dump(scene_report, f, indent=2)
        
        self.report({'INFO'}, f"Scene analysis saved to {report_path}. Total: {len(collection_reports)} collections, ~{total_render_time} min render time")
        return {'FINISHED'}


class GSCAPTURE_OT_export_analysis_report(Operator):
    """Export detailed analysis report for current selection."""
    bl_idname = "gs_capture.export_analysis_report"
    bl_label = "Export Analysis Report"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        settings = context.scene.gs_capture_settings
        
        selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not selected:
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}
        
        result = calculate_adaptive_settings(selected, settings.adaptive_quality_preset)
        
        # Build detailed report
        report = {
            'analysis_timestamp': str(bpy.app.version_string),
            'selected_objects': [obj.name for obj in selected],
            'mesh_analysis': {
                'vertex_count': result.mesh_analysis.vertex_count,
                'face_count': result.mesh_analysis.face_count,
                'surface_area': result.mesh_analysis.surface_area,
                'bounding_box_volume': result.mesh_analysis.bounding_box_volume,
                'vertex_density': result.mesh_analysis.vertex_density,
                'detail_score': result.mesh_analysis.detail_score,
                'curvature_variance': result.mesh_analysis.curvature_variance,
                'has_ngons': result.mesh_analysis.has_ngons,
                'edge_length_variance': result.mesh_analysis.edge_length_variance
            },
            'texture_analysis': {
                'max_resolution': result.texture_analysis.max_resolution,
                'total_textures': result.texture_analysis.total_textures,
                'has_normal_maps': result.texture_analysis.has_normal_maps,
                'has_displacement': result.texture_analysis.has_displacement,
                'texture_score': result.texture_analysis.texture_score,
                'estimated_detail_level': result.texture_analysis.estimated_detail_level
            },
            'recommendations': {
                'camera_count': result.recommended_camera_count,
                'resolution': list(result.recommended_resolution),
                'distance_multiplier': result.recommended_distance_multiplier,
                'quality_preset': result.quality_preset,
                'estimated_render_time_minutes': result.estimated_render_time_minutes
            },
            'detail_hotspots_count': len(result.detail_hotspots),
            'warnings': result.warnings
        }
        
        # Save report
        output_path = bpy.path.abspath(settings.output_path)
        os.makedirs(output_path, exist_ok=True)
        
        report_path = os.path.join(output_path, "capture_analysis_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.report({'INFO'}, f"Analysis report saved to {report_path}")
        return {'FINISHED'}


class GSCAPTURE_OT_apply_recommendations(Operator):
    """Apply the recommended settings from analysis."""
    bl_idname = "gs_capture.apply_recommendations"
    bl_label = "Apply Recommendations"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        settings = context.scene.gs_capture_settings
        
        # Apply recommended settings
        settings.camera_count = settings.analysis_recommended_cameras
        
        # Parse resolution string
        try:
            res_parts = settings.analysis_recommended_resolution.split('x')
            settings.render_resolution_x = int(res_parts[0])
            settings.render_resolution_y = int(res_parts[1])
        except:
            pass
        
        self.report({'INFO'}, "Applied recommended settings")
        return {'FINISHED'}


class GSCAPTURE_OT_capture_selected(Operator):
    """Capture images of selected objects for Gaussian Splatting training."""
    bl_idname = "gs_capture.capture_selected"
    bl_label = "Capture Selected"
    bl_options = {'REGISTER', 'UNDO'}
    
    _timer = None
    _cameras = []
    _current_camera_index = 0
    _output_path = ""
    _original_camera = None
    _original_light_states = {}
    _original_materials = {}
    _original_file_format = None
    _original_hide_render = {}  # Store original hide_render state for all objects
    _target_objects = []
    _adaptive_result = None
    _target_format = 'PNG'
    _save_manually = False

    def _get_all_children(self, obj):
        """Recursively get all children of an object."""
        children = []
        for child in obj.children:
            children.append(child)
            children.extend(self._get_all_children(child))
        return children

    def _hide_non_target_objects(self, context, target_objects):
        """Hide all objects except target objects and their children from render."""
        # Build set of objects that should be visible
        visible_objects = set(target_objects)

        # Include children if setting is enabled
        settings = context.scene.gs_capture_settings
        if settings.include_children:
            for obj in target_objects:
                visible_objects.update(self._get_all_children(obj))

        # Store original state and hide non-target objects
        self._original_hide_render = {}
        for obj in context.scene.objects:
            self._original_hide_render[obj.name] = obj.hide_render
            if obj not in visible_objects:
                # Hide from render (but not viewport so user can still see)
                obj.hide_render = True

    def _restore_object_visibility(self, context):
        """Restore original hide_render state for all objects."""
        for obj_name, hide_render in self._original_hide_render.items():
            obj = context.scene.objects.get(obj_name)
            if obj:
                obj.hide_render = hide_render

    def execute(self, context):
        settings = context.scene.gs_capture_settings

        # Get selected objects
        selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not selected:
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}

        self._target_objects = selected

        # Hide all non-target objects from render
        self._hide_non_target_objects(context, selected)
        
        # Run adaptive analysis if enabled
        if settings.use_adaptive_capture:
            self._adaptive_result = calculate_adaptive_settings(selected, settings.adaptive_quality_preset)
            
            # Override settings with adaptive recommendations
            actual_camera_count = self._adaptive_result.recommended_camera_count
            actual_resolution = self._adaptive_result.recommended_resolution
            actual_distance_mult = self._adaptive_result.recommended_distance_multiplier
            
            # Store for UI feedback
            settings.analysis_recommended_cameras = actual_camera_count
            settings.analysis_quality_preset = self._adaptive_result.quality_preset
        else:
            actual_camera_count = settings.camera_count
            actual_resolution = (settings.render_resolution_x, settings.render_resolution_y)
            actual_distance_mult = settings.camera_distance_multiplier
        
        # Setup output directory
        self._output_path = bpy.path.abspath(settings.output_path)
        images_path = os.path.join(self._output_path, "images")
        os.makedirs(images_path, exist_ok=True)
        
        # Store original state
        self._original_camera = context.scene.camera
        self._original_light_states = store_lighting_state(context)
        
        # Calculate bounds
        center, radius = get_objects_combined_bounds(selected)
        
        # Calculate camera distance
        if settings.camera_distance_mode == 'AUTO':
            distance = radius * actual_distance_mult
        else:
            distance = settings.camera_distance
        
        # Generate camera positions (with adaptive hotspots if enabled)
        hotspots = []
        if settings.use_adaptive_capture and settings.adaptive_use_hotspots and self._adaptive_result:
            hotspots = self._adaptive_result.detail_hotspots
        
        if settings.camera_distribution == 'FIBONACCI':
            if hotspots and settings.adaptive_hotspot_bias > 0:
                points = generate_adaptive_camera_positions(
                    center, radius, actual_camera_count, hotspots,
                    'FIBONACCI', (settings.min_elevation, settings.max_elevation)
                )
            else:
                points = fibonacci_sphere_points(actual_camera_count)
                min_y = math.sin(math.radians(settings.min_elevation))
                max_y = math.sin(math.radians(settings.max_elevation))
                points = [p for p in points if min_y <= p.y <= max_y]
        elif settings.camera_distribution == 'HEMISPHERE_TOP':
            points = [p for p in fibonacci_sphere_points(actual_camera_count * 2) if p.y > 0][:actual_camera_count]
        elif settings.camera_distribution == 'HEMISPHERE_BOTTOM':
            points = [p for p in fibonacci_sphere_points(actual_camera_count * 2) if p.y < 0][:actual_camera_count]
        elif settings.camera_distribution == 'RING':
            points = []
            for i in range(actual_camera_count):
                angle = (2 * math.pi * i) / actual_camera_count
                points.append(Vector((math.cos(angle), 0, math.sin(angle))))
        elif settings.camera_distribution == 'MULTI_RING':
            points = []
            cams_per_ring = actual_camera_count // settings.ring_count
            for ring in range(settings.ring_count):
                elevation = settings.min_elevation + (settings.max_elevation - settings.min_elevation) * (ring / max(1, settings.ring_count - 1))
                elevation_rad = math.radians(elevation)
                y = math.sin(elevation_rad)
                horizontal_radius = math.cos(elevation_rad)
                
                for i in range(cams_per_ring):
                    angle = (2 * math.pi * i) / cams_per_ring
                    x = math.cos(angle) * horizontal_radius
                    z = math.sin(angle) * horizontal_radius
                    points.append(Vector((x, y, z)))
        
        # Create cameras
        self._cameras = []
        for i, point in enumerate(points):
            cam_pos = center + point * distance
            cam = create_camera_at_position(context, cam_pos, center, f"GS_Cam_{i:04d}")
            cam.data.lens = settings.focal_length
            self._cameras.append(cam)
        
        # Setup lighting
        if settings.lighting_mode != 'KEEP':
            setup_neutral_lighting(context, settings)
        
        # Override materials if requested
        if settings.material_mode == 'DIFFUSE':
            self._original_materials = override_materials(selected, 'DIFFUSE')
        
        # Configure render settings
        scene = context.scene
        scene.render.resolution_x = settings.render_resolution_x
        scene.render.resolution_y = settings.render_resolution_y

        # Set render engine with version compatibility
        if settings.render_engine == 'EEVEE':
            scene.render.engine = get_eevee_engine_name()
        else:
            scene.render.engine = settings.render_engine

        if settings.render_engine == 'CYCLES':
            scene.cycles.samples = settings.render_samples
        else:
            scene.eevee.taa_render_samples = settings.render_samples

        # Store original image settings to restore later
        self._original_file_format = scene.render.image_settings.file_format

        # Set file format for still images
        # In Blender, when file_format is FFMPEG (video mode), the enum is restricted
        # We'll save images manually after render to avoid this issue
        format_map = {
            'PNG': 'PNG',
            'JPEG': 'JPEG',
            'OPEN_EXR': 'OPEN_EXR',
        }
        self._target_format = format_map.get(settings.file_format, 'PNG')
        self._save_manually = False

        try:
            scene.render.image_settings.file_format = self._target_format
            if self._target_format == 'PNG':
                scene.render.image_settings.color_mode = 'RGBA'

            # Enable transparent background if requested
            if settings.transparent_background and self._target_format in ('PNG', 'OPEN_EXR'):
                scene.render.film_transparent = True
                self._original_film_transparent = False
            else:
                self._original_film_transparent = scene.render.film_transparent
        except TypeError:
            # Scene is in video/FFMPEG mode - we'll save images manually after render
            self._save_manually = True
            self.report({'INFO'}, "Scene in video mode - will save images manually as PNG")
        
        # Setup depth output if requested
        if settings.export_depth:
            scene.use_nodes = True
            scene.view_layers["ViewLayer"].use_pass_z = True
        
        # Start rendering
        self._current_camera_index = 0
        settings.is_rendering = True
        settings.render_progress = 0.0
        
        # Add timer for progress
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        settings = context.scene.gs_capture_settings

        if event.type == 'TIMER':
            if self._current_camera_index >= len(self._cameras):
                # Finished rendering
                self.finish(context)
                return {'FINISHED'}

            # Render current camera
            cam = self._cameras[self._current_camera_index]
            context.scene.camera = cam

            # Set output path
            ext = 'png' if self._target_format == 'PNG' else ('jpg' if self._target_format == 'JPEG' else 'exr')
            image_path = os.path.join(
                self._output_path, "images",
                f"image_{self._current_camera_index:04d}.{ext}"
            )

            if self._save_manually:
                # Render without saving (scene is in video mode)
                bpy.ops.render.render()
                # Save the render result manually
                render_result = bpy.data.images.get('Render Result')
                if render_result:
                    render_result.save_render(filepath=image_path)
            else:
                # Normal render with automatic save
                context.scene.render.filepath = image_path
                bpy.ops.render.render(write_still=True)

            # Update progress
            self._current_camera_index += 1
            settings.render_progress = (self._current_camera_index / len(self._cameras)) * 100
            settings.current_render_info = f"Rendering {self._current_camera_index}/{len(self._cameras)}"

        if event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}
    
    def finish(self, context):
        settings = context.scene.gs_capture_settings
        
        # Export camera data
        if settings.export_colmap:
            colmap_path = os.path.join(self._output_path, "sparse", "0")
            os.makedirs(colmap_path, exist_ok=True)
            export_colmap_cameras(
                self._cameras, None, colmap_path,
                settings.render_resolution_x, settings.render_resolution_y
            )
        
        if settings.export_transforms_json:
            export_transforms_json(
                self._cameras, self._output_path,
                settings.render_resolution_x, settings.render_resolution_y
            )
        
        # Cleanup
        self.cleanup(context)
        
        settings.is_rendering = False
        settings.render_progress = 100.0
        settings.current_render_info = "Complete!"
        
        self.report({'INFO'}, f"Captured {len(self._cameras)} images to {self._output_path}")
    
    def cancel(self, context):
        self.cleanup(context)
        context.scene.gs_capture_settings.is_rendering = False
        self.report({'WARNING'}, "Capture cancelled")
    
    def cleanup(self, context):
        settings = context.scene.gs_capture_settings
        
        # Remove timer
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        
        # Delete created cameras
        for cam in self._cameras:
            bpy.data.cameras.remove(cam.data)
        
        # Restore original camera
        context.scene.camera = self._original_camera
        
        # Restore lighting
        restore_lighting(context, self._original_light_states)

        # Restore materials
        if self._original_materials:
            restore_materials(self._original_materials)

        # Restore film transparency setting
        if hasattr(self, '_original_film_transparent'):
            context.scene.render.film_transparent = self._original_film_transparent

        # Restore object visibility (hide_render state)
        self._restore_object_visibility(context)


class GSCAPTURE_OT_capture_collection(Operator):
    """Capture an entire collection as a single object for Gaussian Splatting training."""
    bl_idname = "gs_capture.capture_collection"
    bl_label = "Capture Collection"
    bl_options = {'REGISTER', 'UNDO'}

    collection_name: StringProperty(
        name="Collection",
        description="Name of collection to capture (empty = active collection)",
        default=""
    )

    def execute(self, context):
        settings = context.scene.gs_capture_settings

        # Determine which collection to use
        if self.collection_name:
            if self.collection_name not in bpy.data.collections:
                self.report({'ERROR'}, f"Collection '{self.collection_name}' not found")
                return {'CANCELLED'}
            collection = bpy.data.collections[self.collection_name]
        elif settings.target_collection:
            if settings.target_collection not in bpy.data.collections:
                self.report({'ERROR'}, f"Collection '{settings.target_collection}' not found")
                return {'CANCELLED'}
            collection = bpy.data.collections[settings.target_collection]
        else:
            # Use active collection from view layer
            collection = context.view_layer.active_layer_collection.collection

        # Get all mesh objects from collection
        mesh_objects = get_collection_objects(
            collection,
            include_nested=settings.include_nested_collections,
            mesh_only=True
        )

        if not mesh_objects:
            self.report({'ERROR'}, f"No visible mesh objects in collection '{collection.name}'")
            return {'CANCELLED'}

        # Select all objects from the collection
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objects:
            obj.select_set(True)

        if mesh_objects:
            context.view_layer.objects.active = mesh_objects[0]

        self.report({'INFO'}, f"Capturing collection '{collection.name}' with {len(mesh_objects)} objects")

        # Run the standard capture on the selected objects
        return bpy.ops.gs_capture.capture_selected()


class GSCAPTURE_OT_batch_capture(Operator):
    """Batch capture multiple objects or collections."""
    bl_idname = "gs_capture.batch_capture"
    bl_label = "Batch Capture"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.gs_capture_settings
        base_output = bpy.path.abspath(settings.output_path)

        if settings.batch_mode == 'COLLECTION':
            # Capture a single collection as one object
            return bpy.ops.gs_capture.capture_collection()

        elif settings.batch_mode == 'EACH_SELECTED':
            # Capture each selected object separately
            selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
            
            for obj in selected:
                # Set output path for this object
                settings.output_path = os.path.join(base_output, obj.name)
                
                # Select only this object
                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(True)
                context.view_layer.objects.active = obj
                
                # Run capture
                bpy.ops.gs_capture.capture_selected()
            
            settings.output_path = base_output
            
        elif settings.batch_mode == 'COLLECTIONS':
            # Capture each collection
            for collection in bpy.data.collections:
                if not collection.objects:
                    continue
                
                mesh_objects = [obj for obj in collection.objects if obj.type == 'MESH']
                if not mesh_objects:
                    continue
                
                settings.output_path = os.path.join(base_output, collection.name)
                
                bpy.ops.object.select_all(action='DESELECT')
                for obj in mesh_objects:
                    obj.select_set(True)
                
                bpy.ops.gs_capture.capture_selected()
            
            settings.output_path = base_output
        
        elif settings.batch_mode == 'GROUPS':
            # Capture defined object groups
            for group in settings.object_groups:
                group_objects = [item.obj for item in group.objects if item.obj]
                mesh_objects = [obj for obj in group_objects if obj.type == 'MESH']
                
                if not mesh_objects:
                    continue
                
                settings.output_path = os.path.join(base_output, group.name)
                
                bpy.ops.object.select_all(action='DESELECT')
                for obj in mesh_objects:
                    obj.select_set(True)
                
                bpy.ops.gs_capture.capture_selected()
            
            settings.output_path = base_output
        
        return {'FINISHED'}


class GSCAPTURE_OT_add_object_group(Operator):
    """Add a new object group."""
    bl_idname = "gs_capture.add_object_group"
    bl_label = "Add Object Group"
    
    def execute(self, context):
        settings = context.scene.gs_capture_settings
        group = settings.object_groups.add()
        group.name = f"Group {len(settings.object_groups)}"
        settings.active_group_index = len(settings.object_groups) - 1
        return {'FINISHED'}


class GSCAPTURE_OT_remove_object_group(Operator):
    """Remove the active object group."""
    bl_idname = "gs_capture.remove_object_group"
    bl_label = "Remove Object Group"
    
    def execute(self, context):
        settings = context.scene.gs_capture_settings
        if settings.object_groups:
            settings.object_groups.remove(settings.active_group_index)
            settings.active_group_index = max(0, settings.active_group_index - 1)
        return {'FINISHED'}


class GSCAPTURE_OT_add_to_group(Operator):
    """Add selected objects to the active group."""
    bl_idname = "gs_capture.add_to_group"
    bl_label = "Add Selected to Group"
    
    def execute(self, context):
        settings = context.scene.gs_capture_settings
        
        if not settings.object_groups:
            self.report({'WARNING'}, "Create a group first")
            return {'CANCELLED'}
        
        group = settings.object_groups[settings.active_group_index]
        
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                # Check if already in group
                existing = [item.obj for item in group.objects]
                if obj not in existing:
                    item = group.objects.add()
                    item.obj = obj
        
        return {'FINISHED'}


class GSCAPTURE_OT_remove_from_group(Operator):
    """Remove object from group."""
    bl_idname = "gs_capture.remove_from_group"
    bl_label = "Remove from Group"
    
    index: IntProperty()
    
    def execute(self, context):
        settings = context.scene.gs_capture_settings
        group = settings.object_groups[settings.active_group_index]
        group.objects.remove(self.index)
        return {'FINISHED'}


class GSCAPTURE_OT_preview_cameras(Operator):
    """Create camera preview without rendering."""
    bl_idname = "gs_capture.preview_cameras"
    bl_label = "Preview Camera Positions"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        settings = context.scene.gs_capture_settings
        
        # Remove existing preview cameras
        bpy.ops.gs_capture.clear_preview()
        
        selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not selected:
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}
        
        center, radius = get_objects_combined_bounds(selected)
        
        if settings.camera_distance_mode == 'AUTO':
            distance = radius * settings.camera_distance_multiplier
        else:
            distance = settings.camera_distance
        
        # Generate a subset of cameras for preview
        preview_count = min(24, settings.camera_count)
        points = fibonacci_sphere_points(preview_count)
        
        # Create preview collection
        preview_collection = bpy.data.collections.new("GS_Camera_Preview")
        context.scene.collection.children.link(preview_collection)
        
        for i, point in enumerate(points):
            cam_pos = center + point * distance
            cam = create_camera_at_position(context, cam_pos, center, f"GS_Preview_{i:04d}")
            cam.data.lens = settings.focal_length
            cam.data.display_size = radius * 0.2
            
            # Move to preview collection
            context.scene.collection.objects.unlink(cam)
            preview_collection.objects.link(cam)
        
        self.report({'INFO'}, f"Created {preview_count} preview cameras")
        return {'FINISHED'}


class GSCAPTURE_OT_clear_preview(Operator):
    """Clear camera preview."""
    bl_idname = "gs_capture.clear_preview"
    bl_label = "Clear Preview"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Remove preview collection and cameras
        if "GS_Camera_Preview" in bpy.data.collections:
            collection = bpy.data.collections["GS_Camera_Preview"]
            for obj in list(collection.objects):
                bpy.data.cameras.remove(obj.data)
            bpy.data.collections.remove(collection)
        
        return {'FINISHED'}


class GSCAPTURE_OT_open_output_folder(Operator):
    """Open the output folder in file browser."""
    bl_idname = "gs_capture.open_output_folder"
    bl_label = "Open Output Folder"
    
    def execute(self, context):
        settings = context.scene.gs_capture_settings
        path = bpy.path.abspath(settings.output_path)
        
        if os.path.exists(path):
            import subprocess
            import sys
            
            if sys.platform == 'win32':
                os.startfile(path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', path])
            else:
                subprocess.run(['xdg-open', path])
        else:
            self.report({'WARNING'}, f"Path does not exist: {path}")
        
        return {'FINISHED'}


# =============================================================================
# UI PANELS
# =============================================================================

class GSCAPTURE_PT_main_panel(Panel):
    """Main panel for GS Capture."""
    bl_label = "GS Capture"
    bl_idname = "GSCAPTURE_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.gs_capture_settings
        
        # Quick capture buttons
        box = layout.box()
        box.label(text="Quick Capture", icon='CAMERA_DATA')
        
        # Show adaptive status
        if settings.use_adaptive_capture:
            row = box.row()
            row.label(text=f"Mode: Adaptive ({settings.adaptive_quality_preset})", icon='AUTO')
            if settings.analysis_recommended_cameras > 0:
                row = box.row()
                row.label(text=f"Cameras: {settings.analysis_recommended_cameras}")
        else:
            row = box.row()
            row.label(text=f"Mode: Manual ({settings.camera_count} cameras)", icon='PREFERENCES')
        
        row = box.row(align=True)
        row.scale_y = 1.5
        row.operator("gs_capture.capture_selected", text="Capture Selected", icon='RENDER_STILL')
        
        if settings.is_rendering:
            box.progress(
                factor=settings.render_progress / 100,
                type='BAR',
                text=settings.current_render_info
            )
        
        row = box.row(align=True)
        row.operator("gs_capture.preview_cameras", text="Preview", icon='OUTLINER_OB_CAMERA')
        row.operator("gs_capture.clear_preview", text="Clear", icon='X')


class GSCAPTURE_PT_output_panel(Panel):
    """Output settings panel."""
    bl_label = "Output"
    bl_idname = "GSCAPTURE_PT_output_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_parent_id = "GSCAPTURE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.gs_capture_settings
        
        layout.prop(settings, "output_path")
        layout.operator("gs_capture.open_output_folder", icon='FILE_FOLDER')
        
        layout.separator()
        layout.label(text="Export Formats:")
        layout.prop(settings, "export_colmap")
        layout.prop(settings, "export_transforms_json")
        layout.prop(settings, "export_depth")
        layout.prop(settings, "export_masks")


class GSCAPTURE_PT_camera_panel(Panel):
    """Camera settings panel."""
    bl_label = "Camera Settings"
    bl_idname = "GSCAPTURE_PT_camera_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_parent_id = "GSCAPTURE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.gs_capture_settings
        
        layout.prop(settings, "camera_count")
        layout.prop(settings, "camera_distribution")
        
        if settings.camera_distribution == 'MULTI_RING':
            layout.prop(settings, "ring_count")
        
        layout.prop(settings, "min_elevation")
        layout.prop(settings, "max_elevation")
        
        layout.separator()
        layout.prop(settings, "camera_distance_mode")
        
        if settings.camera_distance_mode == 'AUTO':
            layout.prop(settings, "camera_distance_multiplier")
        else:
            layout.prop(settings, "camera_distance")
        
        layout.prop(settings, "focal_length")


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


class GSCAPTURE_PT_render_panel(Panel):
    """Render settings panel."""
    bl_label = "Render Settings"
    bl_idname = "GSCAPTURE_PT_render_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_parent_id = "GSCAPTURE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.gs_capture_settings
        
        layout.prop(settings, "render_engine")
        
        row = layout.row()
        row.prop(settings, "render_resolution_x")
        row.prop(settings, "render_resolution_y")
        
        layout.prop(settings, "render_samples")
        layout.prop(settings, "file_format")

        # Show transparent background option (only for formats that support alpha)
        if settings.file_format in ('PNG', 'OPEN_EXR'):
            layout.prop(settings, "transparent_background")


class GSCAPTURE_PT_lighting_panel(Panel):
    """Lighting settings panel."""
    bl_label = "Lighting & Materials"
    bl_idname = "GSCAPTURE_PT_lighting_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS Capture"
    bl_parent_id = "GSCAPTURE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.gs_capture_settings
        
        layout.prop(settings, "lighting_mode")
        
        if settings.lighting_mode == 'WHITE' or settings.lighting_mode == 'GRAY':
            layout.prop(settings, "background_strength")
        
        if settings.lighting_mode == 'GRAY':
            layout.prop(settings, "gray_value")
        
        if settings.lighting_mode == 'HDR':
            layout.prop(settings, "hdr_path")
            layout.prop(settings, "hdr_strength")
        
        if settings.lighting_mode != 'KEEP':
            layout.prop(settings, "disable_scene_lights")
        
        layout.separator()
        layout.prop(settings, "material_mode")


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
                row.prop(group, "expanded", text="", icon='DISCLOSURE_TRI_DOWN' if group.expanded else 'DISCLOSURE_TRI_RIGHT', emboss=False)
                row.prop(group, "name", text="")
                
                if i == settings.active_group_index:
                    row.label(text="", icon='RADIOBUT_ON')
                else:
                    op = row.operator("gs_capture.add_object_group", text="", icon='RADIOBUT_OFF')
                
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


# =============================================================================
# REGISTRATION
# =============================================================================

classes = [
    GSCaptureObjectItem,
    GSCaptureObjectGroup,
    GSCaptureSettings,
    GSCAPTURE_OT_analyze_selected,
    GSCAPTURE_OT_analyze_scene,
    GSCAPTURE_OT_export_analysis_report,
    GSCAPTURE_OT_apply_recommendations,
    GSCAPTURE_OT_capture_selected,
    GSCAPTURE_OT_capture_collection,
    GSCAPTURE_OT_batch_capture,
    GSCAPTURE_OT_add_object_group,
    GSCAPTURE_OT_remove_object_group,
    GSCAPTURE_OT_add_to_group,
    GSCAPTURE_OT_remove_from_group,
    GSCAPTURE_OT_preview_cameras,
    GSCAPTURE_OT_clear_preview,
    GSCAPTURE_OT_open_output_folder,
    GSCAPTURE_PT_main_panel,
    GSCAPTURE_PT_output_panel,
    GSCAPTURE_PT_adaptive_panel,
    GSCAPTURE_PT_camera_panel,
    GSCAPTURE_PT_render_panel,
    GSCAPTURE_PT_lighting_panel,
    GSCAPTURE_PT_batch_panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.gs_capture_settings = PointerProperty(type=GSCaptureSettings)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.gs_capture_settings


if __name__ == "__main__":
    register()
