"""
Camera distribution and positioning for multi-view capture.
Provides various camera placement strategies for optimal 3DGS coverage.
"""

import bpy
import math
import random
from mathutils import Vector


def fibonacci_sphere_points(n):
    """Generate evenly distributed points on a sphere using Fibonacci spiral.

    This algorithm provides near-optimal uniform distribution of points
    on a sphere surface, ideal for 360-degree camera coverage.

    Args:
        n: Number of points to generate

    Returns:
        list: List of Vector points on unit sphere
    """
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # Golden angle in radians

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2 if n > 1 else 0  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)

        theta = phi * i

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append(Vector((x, y, z)))

    return points


def generate_ring_points(count, elevation_deg=0):
    """Generate points in a ring at specified elevation.

    Args:
        count: Number of points in the ring
        elevation_deg: Elevation angle in degrees (0 = equator)

    Returns:
        list: List of Vector points on unit sphere
    """
    points = []
    elevation_rad = math.radians(elevation_deg)
    y = math.sin(elevation_rad)
    horizontal_radius = math.cos(elevation_rad)

    for i in range(count):
        angle = (2 * math.pi * i) / count
        x = math.cos(angle) * horizontal_radius
        z = math.sin(angle) * horizontal_radius
        points.append(Vector((x, y, z)))

    return points


def generate_multi_ring_points(total_count, ring_count, min_elevation, max_elevation):
    """Generate points distributed across multiple horizontal rings.

    Args:
        total_count: Total number of points
        ring_count: Number of rings to distribute points across
        min_elevation: Minimum elevation angle in degrees
        max_elevation: Maximum elevation angle in degrees

    Returns:
        list: List of Vector points on unit sphere
    """
    points = []
    cams_per_ring = total_count // ring_count

    for ring in range(ring_count):
        if ring_count > 1:
            t = ring / (ring_count - 1)
        else:
            t = 0.5
        elevation = min_elevation + (max_elevation - min_elevation) * t
        ring_points = generate_ring_points(cams_per_ring, elevation)
        points.extend(ring_points)

    return points


def generate_hemisphere_points(count, top=True):
    """Generate points on a hemisphere.

    Args:
        count: Approximate number of points (may generate more then filter)
        top: If True, top hemisphere (y > 0), else bottom

    Returns:
        list: List of Vector points on unit hemisphere
    """
    # Generate more points and filter to hemisphere
    all_points = fibonacci_sphere_points(count * 2)

    if top:
        filtered = [p for p in all_points if p.y > 0]
    else:
        filtered = [p for p in all_points if p.y < 0]

    return filtered[:count]


def generate_camera_positions(distribution, count, min_elevation=-90, max_elevation=90,
                               ring_count=5, hotspots=None, hotspot_bias=0.0):
    """Generate camera positions based on distribution type.

    Args:
        distribution: One of 'FIBONACCI', 'HEMISPHERE_TOP', 'HEMISPHERE_BOTTOM',
                     'RING', 'MULTI_RING'
        count: Number of cameras
        min_elevation: Minimum elevation angle in degrees
        max_elevation: Maximum elevation angle in degrees
        ring_count: Number of rings for MULTI_RING distribution
        hotspots: Optional list of (position, weight) tuples for biased placement
        hotspot_bias: Strength of hotspot bias (0-1)

    Returns:
        list: List of Vector points on unit sphere
    """
    if distribution == 'FIBONACCI':
        points = fibonacci_sphere_points(count)
        # Filter by elevation
        min_y = math.sin(math.radians(min_elevation))
        max_y = math.sin(math.radians(max_elevation))
        points = [p for p in points if min_y <= p.y <= max_y]

    elif distribution == 'HEMISPHERE_TOP':
        points = generate_hemisphere_points(count, top=True)

    elif distribution == 'HEMISPHERE_BOTTOM':
        points = generate_hemisphere_points(count, top=False)

    elif distribution == 'RING':
        # Single ring at average elevation
        avg_elevation = (min_elevation + max_elevation) / 2
        points = generate_ring_points(count, avg_elevation)

    elif distribution == 'MULTI_RING':
        points = generate_multi_ring_points(count, ring_count, min_elevation, max_elevation)

    else:
        # Default to fibonacci
        points = fibonacci_sphere_points(count)

    # Apply hotspot bias if provided
    if hotspots and hotspot_bias > 0:
        points = apply_hotspot_bias(points, hotspots, hotspot_bias)

    return points


def apply_hotspot_bias(points, hotspots, bias_strength):
    """Bias camera distribution toward detail hotspots.

    Adds additional camera positions near areas of high detail.

    Args:
        points: List of Vector camera direction points
        hotspots: List of (Vector position, float weight) tuples
        bias_strength: How much to bias (0-1)

    Returns:
        list: Modified list of camera positions
    """
    if not hotspots or bias_strength <= 0:
        return points

    # Calculate how many extra cameras to add toward hotspots
    extra_count = int(len(points) * bias_strength * 0.3)  # 30% extra at max bias

    extra_points = []
    for hotspot_pos, weight in hotspots:
        # Add cameras looking toward this hotspot
        # Distribute around the hotspot direction
        hotspot_dir = hotspot_pos.normalized()

        # Number of extra cameras proportional to weight
        n_extra = max(1, int(extra_count * weight / sum(w for _, w in hotspots)))

        for i in range(n_extra):
            # Perturb direction slightly for variety
            offset = Vector((
                random.gauss(0, 0.1),
                random.gauss(0, 0.1),
                random.gauss(0, 0.1)
            ))
            new_dir = (hotspot_dir + offset).normalized()
            # Flip direction - we want cameras looking AT the hotspot
            extra_points.append(-new_dir)

    # Combine and normalize
    all_points = points + extra_points
    return all_points


def generate_adaptive_camera_positions(center, radius, count, hotspots,
                                        distribution='FIBONACCI',
                                        elevation_range=(-60, 60)):
    """Generate camera positions with adaptive biasing toward hotspots.

    Args:
        center: Vector center of the scene
        radius: Distance from center
        count: Number of cameras
        hotspots: List of (position, weight) tuples
        distribution: Distribution type
        elevation_range: (min, max) elevation in degrees

    Returns:
        list: List of Vector points on unit sphere
    """
    min_elev, max_elev = elevation_range

    # Generate base distribution
    points = generate_camera_positions(
        distribution, count,
        min_elevation=min_elev,
        max_elevation=max_elev
    )

    if not hotspots:
        return points

    # Bias toward hotspots
    # For each hotspot, add more cameras looking at it
    total_weight = sum(w for _, w in hotspots)
    if total_weight == 0:
        return points

    biased_points = list(points)

    for hotspot_pos, weight in hotspots:
        # Calculate direction from center to hotspot
        hotspot_local = hotspot_pos - center
        if hotspot_local.length < 0.001:
            continue

        hotspot_dir = hotspot_local.normalized()

        # Add cameras opposite to hotspot direction (looking at it)
        cam_dir = -hotspot_dir

        # Number of extra cameras based on weight
        n_extra = int((weight / total_weight) * count * 0.2)

        for i in range(n_extra):
            # Add variation
            offset = Vector((
                random.gauss(0, 0.15),
                random.gauss(0, 0.15),
                random.gauss(0, 0.15)
            ))
            varied_dir = (cam_dir + offset).normalized()
            biased_points.append(varied_dir)

    return biased_points


def get_objects_combined_bounds(objects):
    """Calculate combined bounding sphere for multiple objects.

    Args:
        objects: List of Blender objects

    Returns:
        tuple: (Vector center, float radius)
    """
    if not objects:
        return Vector((0, 0, 0)), 1.0

    # Get all bounding box corners in world space
    all_corners = []
    for obj in objects:
        if hasattr(obj, 'bound_box'):
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ Vector(corner)
                all_corners.append(world_corner)

    if not all_corners:
        return Vector((0, 0, 0)), 1.0

    # Calculate center
    center = Vector((0, 0, 0))
    for corner in all_corners:
        center += corner
    center /= len(all_corners)

    # Calculate radius
    radius = 0
    for corner in all_corners:
        dist = (corner - center).length
        if dist > radius:
            radius = dist

    # Ensure minimum radius
    radius = max(radius, 0.1)

    return center, radius


def create_camera_at_position(context, position, target, name="GS_Camera"):
    """Create a camera at the given position looking at target.

    Args:
        context: Blender context
        position: Vector camera position
        target: Vector point to look at
        name: Camera name

    Returns:
        Camera object
    """
    # Create camera data
    cam_data = bpy.data.cameras.new(name=name)

    # Create camera object
    cam_obj = bpy.data.objects.new(name=name, object_data=cam_data)

    # Link to scene
    context.scene.collection.objects.link(cam_obj)

    # Position camera
    cam_obj.location = position

    # Point at target using track-to constraint logic
    direction = target - position
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    return cam_obj


def delete_gs_cameras(context):
    """Delete all cameras created by GS Capture.

    Args:
        context: Blender context
    """
    # Find and delete all GS cameras
    cameras_to_delete = [obj for obj in context.scene.objects
                         if obj.type == 'CAMERA' and obj.name.startswith('GS_')]

    for cam in cameras_to_delete:
        bpy.data.objects.remove(cam, do_unlink=True)
