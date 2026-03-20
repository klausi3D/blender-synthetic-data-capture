"""Proxy-hull export utilities for post-training splat cleanup."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import bmesh
import bpy
from mathutils import Vector


def _dedupe_points(points, precision=6):
    """Deduplicate world-space points with configurable rounding precision."""
    deduped = []
    seen = set()
    for point in points:
        key = (
            round(float(point.x), precision),
            round(float(point.y), precision),
            round(float(point.z), precision),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(Vector(key))
    return deduped


def _compute_bounds(points):
    """Compute axis-aligned bounds for a list of vectors."""
    min_x = min(point.x for point in points)
    min_y = min(point.y for point in points)
    min_z = min(point.z for point in points)
    max_x = max(point.x for point in points)
    max_y = max(point.y for point in points)
    max_z = max(point.z for point in points)
    return {
        "min": [float(min_x), float(min_y), float(min_z)],
        "max": [float(max_x), float(max_y), float(max_z)],
    }


def _compute_convex_hull_planes(points):
    """Compute convex-hull plane equations from world-space points."""
    bm = bmesh.new()
    try:
        for point in points:
            bm.verts.new(point)

        bm.verts.ensure_lookup_table()
        bmesh.ops.convex_hull(bm, input=list(bm.verts), use_existing_faces=False)
        bm.normal_update()

        planes = []
        seen = set()
        for face in bm.faces:
            if len(face.verts) < 3:
                continue
            normal = face.normal.copy()
            if normal.length < 1e-9:
                continue
            normal.normalize()
            d = -normal.dot(face.verts[0].co)
            key = (
                round(float(normal.x), 8),
                round(float(normal.y), 8),
                round(float(normal.z), 8),
                round(float(d), 8),
            )
            if key in seen:
                continue
            seen.add(key)
            planes.append([float(normal.x), float(normal.y), float(normal.z), float(d)])

        return planes
    finally:
        bm.free()


def build_proxy_hulls_payload(target_objects, recommended_margin=0.01):
    """Build proxy-hull payload for the selected capture objects."""
    depsgraph = None
    if hasattr(bpy.context, "evaluated_depsgraph_get"):
        try:
            depsgraph = bpy.context.evaluated_depsgraph_get()
        except Exception:
            depsgraph = None

    object_entries = []
    skipped = []

    for obj in target_objects:
        if obj is None or obj.type != "MESH":
            skipped.append({
                "name": getattr(obj, "name", "<unknown>"),
                "reason": "object_is_not_mesh",
            })
            continue

        eval_obj = obj
        mesh = None
        try:
            eval_obj = obj.evaluated_get(depsgraph) if depsgraph else obj
            mesh = eval_obj.to_mesh()
            if mesh is None or len(mesh.vertices) < 4:
                skipped.append({"name": obj.name, "reason": "insufficient_vertices"})
                continue

            matrix_world = eval_obj.matrix_world.copy()
            points = [matrix_world @ vertex.co for vertex in mesh.vertices]
            points = _dedupe_points(points)
            if len(points) < 4:
                skipped.append({"name": obj.name, "reason": "degenerate_geometry"})
                continue

            planes = _compute_convex_hull_planes(points)
            if len(planes) < 4:
                skipped.append({"name": obj.name, "reason": "convex_hull_failed"})
                continue

            bounds = _compute_bounds(points)
            object_entries.append(
                {
                    "name": obj.name,
                    "source_vertex_count": int(len(mesh.vertices)),
                    "deduped_vertex_count": int(len(points)),
                    "plane_count": int(len(planes)),
                    "planes": planes,
                    "bounds": bounds,
                }
            )
        except Exception as exc:
            skipped.append({"name": obj.name, "reason": f"exception:{type(exc).__name__}"})
        finally:
            if eval_obj is not None and mesh is not None:
                try:
                    eval_obj.to_mesh_clear()
                except Exception:
                    pass

    global_bounds = None
    if object_entries:
        min_corner = [min(entry["bounds"]["min"][axis] for entry in object_entries) for axis in range(3)]
        max_corner = [max(entry["bounds"]["max"][axis] for entry in object_entries) for axis in range(3)]
        global_bounds = {"min": min_corner, "max": max_corner}

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_system": "BLENDER_WORLD",
        "recommended_margin": float(recommended_margin),
        "object_count": int(len(object_entries)),
        "objects": object_entries,
        "global_bounds": global_bounds,
        "skipped_objects": skipped,
    }


def export_proxy_hulls(output_path, target_objects, recommended_margin=0.01):
    """Export proxy hulls to `<output_path>/cleanup/proxy_hulls.json`."""
    cleanup_dir = os.path.join(output_path, "cleanup")
    os.makedirs(cleanup_dir, exist_ok=True)

    payload = build_proxy_hulls_payload(target_objects, recommended_margin=recommended_margin)
    if payload["object_count"] <= 0:
        raise RuntimeError("No valid mesh objects were available for proxy hull export")

    artifact_path = os.path.join(cleanup_dir, "proxy_hulls.json")
    with open(artifact_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return artifact_path, payload
