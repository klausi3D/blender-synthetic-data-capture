"""
Coverage analysis for camera placement validation.
Calculates how well cameras cover the target mesh surfaces.
"""

import bpy
import math
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from typing import List, Dict, Tuple, Optional


class CoverageAnalyzer:
    """Analyze camera coverage of mesh surfaces."""

    def __init__(self, mesh_objects: List, cameras: List):
        """Initialize coverage analyzer.

        Args:
            mesh_objects: List of mesh objects to analyze
            cameras: List of camera objects
        """
        self.mesh_objects = mesh_objects
        self.cameras = cameras
        self.bvh_trees = {}

        # Build BVH trees for each mesh
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for obj in mesh_objects:
            if obj.type == 'MESH':
                self.bvh_trees[obj.name] = BVHTree.FromObject(
                    obj, depsgraph
                )

    def _build_camera_cache(self) -> List[Dict]:
        """Precompute camera data used for visibility tests."""
        cache = []
        for cam in self.cameras:
            cam_matrix = cam.matrix_world
            cam_pos = cam_matrix.translation
            cam_forward = (cam_matrix.to_3x3() @ Vector((0, 0, -1))).normalized()
            cam_data = cam.data
            is_persp = cam_data.type == 'PERSP'

            fov_cos = None
            fov_sin = None
            if is_persp:
                half_angle = cam_data.angle / 2.0
                fov_angle = half_angle * 1.2
                fov_cos = math.cos(fov_angle)
                fov_sin = math.sin(fov_angle)

            cache.append({
                'camera': cam,
                'pos': cam_pos,
                'forward': cam_forward,
                'is_persp': is_persp,
                'fov_cos': fov_cos,
                'fov_sin': fov_sin,
            })

        return cache

    def _get_object_bounds(self, obj) -> Tuple[Vector, float]:
        """Compute a conservative world-space bounding sphere for an object."""
        matrix = obj.matrix_world
        bbox = [matrix @ Vector(corner) for corner in obj.bound_box]
        if not bbox:
            return Vector(), 0.0

        center = sum(bbox, Vector()) / len(bbox)
        radius = max((v - center).length for v in bbox)
        return center, radius

    def _prefilter_cameras_for_object(
        self,
        obj_center: Vector,
        obj_radius: float,
        camera_cache: List[Dict]
    ) -> List[Dict]:
        """Cull cameras that cannot possibly see any point on the object."""
        if not camera_cache:
            return []

        if obj_radius <= 0.0:
            return list(camera_cache)

        candidates = []
        for cam_info in camera_cache:
            if not cam_info['is_persp']:
                candidates.append(cam_info)
                continue

            to_center = obj_center - cam_info['pos']
            dist = to_center.length

            # Camera inside or too close to bounds: cannot safely cull.
            if dist <= obj_radius or dist <= 1e-9:
                candidates.append(cam_info)
                continue

            to_center_norm = to_center / dist
            dot = cam_info['forward'].dot(to_center_norm)

            # Angular radius of the bounding sphere.
            sin_ang = obj_radius / dist
            if sin_ang >= 1.0:
                candidates.append(cam_info)
                continue

            cos_ang = math.sqrt(max(0.0, 1.0 - (sin_ang * sin_ang)))

            # cos(fov + angular_radius) via cos(A+B) = cosA cosB - sinA sinB
            threshold = (cam_info['fov_cos'] * cos_ang) - (cam_info['fov_sin'] * sin_ang)

            # Conservative cull to avoid false negatives on boundary.
            if dot < threshold - 1e-6:
                continue

            candidates.append(cam_info)

        return candidates

    def calculate_vertex_coverage(self, min_angle: float = 15.0) -> Dict[str, Dict[int, int]]:
        """Calculate how many cameras see each vertex.

        Args:
            min_angle: Minimum angle between surface normal and view direction
                      for a camera to "see" a vertex (in degrees)

        Returns:
            Dict mapping object_name -> {vertex_index -> camera_count}
        """
        coverage = {}
        min_dot = math.cos(math.radians(90 - min_angle))
        camera_cache = self._build_camera_cache()
        bvh_items = list(self.bvh_trees.items())

        for obj in self.mesh_objects:
            if obj.type != 'MESH':
                continue

            mesh = obj.data
            matrix = obj.matrix_world
            normal_matrix = matrix.to_3x3()
            coverage[obj.name] = {}

            if not camera_cache:
                for vert in mesh.vertices:
                    coverage[obj.name][vert.index] = 0
                continue

            obj_center, obj_radius = self._get_object_bounds(obj)
            candidate_cameras = self._prefilter_cameras_for_object(
                obj_center, obj_radius, camera_cache
            )

            if not candidate_cameras:
                for vert in mesh.vertices:
                    coverage[obj.name][vert.index] = 0
                continue

            for vert in mesh.vertices:
                world_pos = matrix @ vert.co
                world_normal = (normal_matrix @ vert.normal).normalized()

                visible_count = 0
                for cam_info in candidate_cameras:
                    if self._is_visible_from_camera(
                        world_pos, world_normal, cam_info, obj, min_dot, bvh_items
                    ):
                        visible_count += 1

                coverage[obj.name][vert.index] = visible_count

        return coverage

    def _is_visible_from_camera(
        self,
        point: Vector,
        normal: Vector,
        cam_info: Dict,
        source_obj,
        min_dot: float,
        bvh_items: List[Tuple[str, BVHTree]]
    ) -> bool:
        """Check if point is visible from camera.

        Args:
            point: World-space point position
            normal: World-space surface normal
            camera: Precomputed camera info
            source_obj: Object the point belongs to
            min_dot: Minimum dot product for visibility

        Returns:
            True if point is visible from camera
        """
        cam_pos = cam_info['pos']
        source_name = source_obj.name
        to_cam = cam_pos - point
        distance = to_cam.length
        if distance <= 1e-9:
            return False

        view_dir = to_cam / distance

        # Check if camera is in front of surface (dot product with normal)
        dot = normal.dot(view_dir)
        if dot < min_dot:
            return False

        # Check if point is in camera's field of view
        if cam_info['is_persp']:
            cam_forward = cam_info['forward']
            view_dot = -cam_forward.dot(view_dir)

            # Check against camera FOV
            if view_dot < cam_info['fov_cos']:  # 20% margin
                return False

        # Check for occlusion
        direction = -view_dir

        for obj_name, bvh in bvh_items:
            hit, loc, norm, idx = bvh.ray_cast(cam_pos, direction.normalized())
            if hit:
                hit_dist = (loc - cam_pos).length
                # Allow small tolerance for the source object's own surface
                if obj_name == source_name:
                    if hit_dist < distance - 0.01:
                        return False
                else:
                    if hit_dist < distance - 0.001:
                        return False

        return True

    def get_coverage_statistics(
        self,
        coverage: Optional[Dict] = None
    ) -> Dict:
        """Get coverage statistics.

        Args:
            coverage: Pre-calculated coverage data, or None to calculate

        Returns:
            Dict with coverage statistics
        """
        if coverage is None:
            coverage = self.calculate_vertex_coverage()

        all_counts = []
        for obj_coverage in coverage.values():
            all_counts.extend(obj_coverage.values())

        if not all_counts:
            return {
                'min': 0,
                'max': 0,
                'mean': 0,
                'median': 0,
                'poorly_covered': 0,
                'well_covered': 0,
            }

        all_counts.sort()
        n = len(all_counts)

        return {
            'min': min(all_counts),
            'max': max(all_counts),
            'mean': sum(all_counts) / n,
            'median': all_counts[n // 2],
            'poorly_covered': sum(1 for c in all_counts if c < 3),
            'well_covered': sum(1 for c in all_counts if c >= 5),
            'total_vertices': n,
        }

    def get_poorly_covered_vertices(
        self,
        min_coverage: int = 3,
        coverage: Optional[Dict] = None
    ) -> List[Tuple[str, int, Vector]]:
        """Find vertices with insufficient camera coverage.

        Args:
            min_coverage: Minimum acceptable camera count
            coverage: Pre-calculated coverage data, or None to calculate

        Returns:
            List of (object_name, vertex_index, world_position) tuples
        """
        if coverage is None:
            coverage = self.calculate_vertex_coverage()

        poorly_covered = []

        for obj in self.mesh_objects:
            if obj.name not in coverage:
                continue

            obj_coverage = coverage[obj.name]
            mesh = obj.data
            matrix = obj.matrix_world

            for vert_idx, count in obj_coverage.items():
                if count < min_coverage:
                    world_pos = matrix @ mesh.vertices[vert_idx].co
                    poorly_covered.append((obj.name, vert_idx, world_pos))

        return poorly_covered

    def suggest_additional_cameras(
        self,
        target_coverage: int = 5,
        max_suggestions: int = 10,
        camera_distance: float = None
    ) -> List[Tuple[Vector, Vector]]:
        """Suggest positions for additional cameras.

        Args:
            target_coverage: Desired minimum coverage per vertex
            max_suggestions: Maximum number of suggestions
            camera_distance: Distance from object center for cameras

        Returns:
            List of (position, look_at) Vector tuples
        """
        coverage = self.calculate_vertex_coverage()
        poorly_covered = self.get_poorly_covered_vertices(target_coverage, coverage)

        if not poorly_covered:
            return []

        # Cluster poorly covered vertices
        clusters = self._cluster_positions([p[2] for p in poorly_covered])

        suggestions = []

        # Calculate object center and radius
        all_positions = [p[2] for p in poorly_covered]
        if all_positions:
            center = sum(all_positions, Vector()) / len(all_positions)

            if camera_distance is None:
                # Estimate distance from existing cameras
                if self.cameras:
                    cam_dists = [(c.matrix_world.translation - center).length
                                 for c in self.cameras]
                    camera_distance = sum(cam_dists) / len(cam_dists)
                else:
                    camera_distance = 5.0

        for cluster_center in clusters[:max_suggestions]:
            # Calculate optimal camera position
            # Place camera opposite to cluster, at specified distance

            direction = (cluster_center - center).normalized()
            if direction.length < 0.001:
                direction = Vector((0, 0, 1))

            cam_pos = cluster_center - direction * camera_distance
            suggestions.append((cam_pos, cluster_center))

        return suggestions

    def _cluster_positions(
        self,
        positions: List[Vector],
        cluster_radius: float = 0.5
    ) -> List[Vector]:
        """Simple clustering of positions.

        Args:
            positions: List of world-space positions
            cluster_radius: Maximum distance for same cluster

        Returns:
            List of cluster center positions
        """
        if not positions:
            return []

        if cluster_radius <= 0.0:
            return list(positions)

        # Spatial hash for faster neighbor lookup (preserves greedy clustering)
        cell_size = cluster_radius
        inv_cell_size = 1.0 / cell_size

        def _cell_key(pos: Vector) -> Tuple[int, int, int]:
            return (
                int(math.floor(pos.x * inv_cell_size)),
                int(math.floor(pos.y * inv_cell_size)),
                int(math.floor(pos.z * inv_cell_size)),
            )

        grid: Dict[Tuple[int, int, int], List[int]] = {}
        for idx, pos in enumerate(positions):
            key = _cell_key(pos)
            grid.setdefault(key, []).append(idx)

        clusters = []
        used = set()

        for i, pos in enumerate(positions):
            if i in used:
                continue

            # Find all positions within cluster_radius
            cluster_positions = [pos]
            used.add(i)

            key = _cell_key(pos)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                        for j in grid.get(neighbor_key, ()):
                            if j in used:
                                continue
                            other = positions[j]
                            if (pos - other).length < cluster_radius:
                                cluster_positions.append(other)
                                used.add(j)

            # Calculate cluster center
            center = sum(cluster_positions, Vector()) / len(cluster_positions)
            clusters.append(center)

        # Sort by size (largest clusters first)
        return clusters

    def get_coverage_quality_rating(self) -> str:
        """Get overall coverage quality rating.

        Returns:
            Rating string: 'EXCELLENT', 'GOOD', 'FAIR', 'POOR'
        """
        stats = self.get_coverage_statistics()

        if stats['total_vertices'] == 0:
            return 'UNKNOWN'

        poorly_ratio = stats['poorly_covered'] / stats['total_vertices']

        if poorly_ratio < 0.05 and stats['min'] >= 3:
            return 'EXCELLENT'
        elif poorly_ratio < 0.15 and stats['min'] >= 2:
            return 'GOOD'
        elif poorly_ratio < 0.30:
            return 'FAIR'
        else:
            return 'POOR'


def analyze_coverage(mesh_objects: List, cameras: List) -> Dict:
    """Convenience function to analyze coverage.

    Args:
        mesh_objects: List of mesh objects
        cameras: List of camera objects

    Returns:
        Dict with coverage analysis results
    """
    analyzer = CoverageAnalyzer(mesh_objects, cameras)
    coverage = analyzer.calculate_vertex_coverage()
    stats = analyzer.get_coverage_statistics(coverage)
    rating = analyzer.get_coverage_quality_rating()

    return {
        'coverage': coverage,
        'statistics': stats,
        'rating': rating,
        'poorly_covered': analyzer.get_poorly_covered_vertices(3, coverage),
        'suggestions': analyzer.suggest_additional_cameras(),
    }
