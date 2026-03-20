#!/usr/bin/env python3
"""
Blender Scene Introspection Script
====================================
Runs inside Blender to inspect a .blend file and emit structured metadata.

Invocation:
    blender file.blend --background --python tools/introspect_blend.py -- [flags]

Flags:
    --adaptive              Run adaptive analysis (slower, needs full scene load)
    --quality-preset PRESET Quality preset for adaptive analysis (AUTO|DRAFT|STANDARD|HIGH|ULTRA)

Output is JSON between sentinel markers on stdout (Blender prints startup
noise that must be filtered out).
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    print("ERROR: This script must be run inside Blender.", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Sentinels for extracting JSON from Blender's noisy stdout
SENTINEL_BEGIN = "---INTROSPECT_JSON_BEGIN---"
SENTINEL_END = "---INTROSPECT_JSON_END---"


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Introspect a .blend file")
    p.add_argument("--adaptive", action="store_true",
                   help="Run adaptive analysis for camera/resolution recommendations")
    p.add_argument("--quality-preset", default="AUTO",
                   choices=["AUTO", "DRAFT", "STANDARD", "HIGH", "ULTRA"],
                   help="Quality preset for adaptive analysis")
    return p.parse_args(argv)


def ensure_addon_available() -> None:
    """Put repo root on sys.path so gs_capture_addon is importable."""
    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def collect_scene_info() -> list:
    """Gather per-scene metadata."""
    scenes = []
    for scene in bpy.data.scenes:
        # Walk the scene's root collection tree
        collection_names = []
        for coll in bpy.data.collections:
            collection_names.append(coll.name)

        mesh_count = sum(
            1 for obj in scene.objects if obj.type == "MESH"
        )
        scenes.append({
            "name": scene.name,
            "collection_names": collection_names,
            "total_mesh_objects": mesh_count,
        })
    return scenes


def collect_collection_info(adaptive: bool, quality_preset: str) -> list:
    """Gather per-collection metadata, optionally with adaptive analysis."""
    adaptive_func = None
    if adaptive:
        try:
            ensure_addon_available()
            from gs_capture_addon.core.analysis import calculate_adaptive_settings
            adaptive_func = calculate_adaptive_settings
        except ImportError:
            print("[introspect] WARNING: Could not import adaptive analysis",
                  file=sys.stderr)

    collections = []
    for coll in bpy.data.collections:
        meshes = [obj for obj in coll.all_objects if obj.type == "MESH"]
        mesh_count = len(meshes)

        total_vertices = 0
        for obj in meshes:
            if obj.data:
                total_vertices += len(obj.data.vertices)

        object_names = [obj.name for obj in meshes]

        adaptive_info = None
        if adaptive_func and meshes:
            try:
                result = adaptive_func(meshes, quality_preset)
                adaptive_info = {
                    "recommended_cameras": result.recommended_camera_count,
                    "recommended_resolution": list(result.recommended_resolution),
                    "recommended_distance_multiplier": result.recommended_distance_multiplier,
                    "quality_preset": result.quality_preset,
                    "estimated_render_time_minutes": result.estimated_render_time_minutes,
                    "warnings": result.warnings,
                }
            except Exception as e:
                print(f"[introspect] WARNING: Adaptive analysis failed for "
                      f"{coll.name}: {e}", file=sys.stderr)

        collections.append({
            "name": coll.name,
            "mesh_count": mesh_count,
            "total_vertices": total_vertices,
            "object_names": object_names,
            "adaptive": adaptive_info,
        })

    return collections


def main() -> None:
    args = parse_args()

    blend_path = bpy.data.filepath or ""
    file_size = 0
    if blend_path and os.path.exists(blend_path):
        file_size = os.path.getsize(blend_path)

    payload = {
        "version": 1,
        "blend_file": blend_path,
        "blender_version": bpy.app.version_string,
        "file_size_bytes": file_size,
        "scenes": collect_scene_info(),
        "collections": collect_collection_info(args.adaptive, args.quality_preset),
    }

    # Emit JSON between sentinels so the orchestrator can extract it
    # from Blender's noisy stdout
    print(SENTINEL_BEGIN, flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    print(SENTINEL_END, flush=True)

    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
