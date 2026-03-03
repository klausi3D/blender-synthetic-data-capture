#!/usr/bin/env python3
"""Smoke test for ASSET_LIBRARY batch capture workflow."""

from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "training_out" / "smoke_feature_verification"
ASSET_LIBRARY_ROOT = OUT_ROOT / "asset_library_source"
CAPTURE_OUTPUT_ROOT = OUT_ROOT / "asset_library_capture"
REPORT_PATH = OUT_ROOT / "asset_library_capture_report.json"
LIBRARY_NAME = "GS_Smoke_AssetLibrary"

STATE = {
    "phase": "init",
    "start_time": time.time(),
    "report": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "blender_version": bpy.app.version_string,
        "platform": sys.platform,
        "events": [],
        "checks": {},
        "details": {},
        "success": False,
        "errors": [],
    },
}


def log(message: str) -> None:
    print(f"[GS_ASSET_SMOKE] {message}")
    STATE["report"]["events"].append(message)


def pick_fast_engine(scene) -> None:
    engine_prop = scene.render.bl_rna.properties.get("engine")
    ids = {item.identifier for item in engine_prop.enum_items}
    if "BLENDER_EEVEE_NEXT" in ids:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in ids:
        scene.render.engine = "BLENDER_EEVEE"
    elif "CYCLES" in ids:
        scene.render.engine = "CYCLES"


def register_addon() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import gs_capture_addon
    try:
        gs_capture_addon.unregister()
    except Exception:
        pass
    gs_capture_addon.register()
    log("Addon registered")


def reset_scene() -> None:
    scene = bpy.context.scene
    try:
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 5.0))

    pick_fast_engine(scene)
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = 8
    if hasattr(scene, "cycles"):
        scene.cycles.samples = 16

    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    log("Scene reset")


def create_cube_mesh_object(name: str):
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    verts = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    return obj, mesh


def write_mesh_asset(path: Path, object_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj, mesh = create_cube_mesh_object(object_name)
    try:
        bpy.data.libraries.write(str(path), {obj})
    finally:
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj)
        if mesh.name in bpy.data.meshes and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def write_non_mesh_asset(path: Path, object_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    camera = bpy.data.cameras.new(f"{object_name}_Camera")
    obj = bpy.data.objects.new(object_name, camera)
    try:
        bpy.data.libraries.write(str(path), {obj})
    finally:
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj)
        if camera.name in bpy.data.cameras and camera.users == 0:
            bpy.data.cameras.remove(camera)


def setup_asset_library_sources() -> list[Path]:
    shutil.rmtree(ASSET_LIBRARY_ROOT, ignore_errors=True)
    ASSET_LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)

    target_paths = [
        ASSET_LIBRARY_ROOT / "assets_main" / "asset_a_ok.blend",
        ASSET_LIBRARY_ROOT / "assets_main" / "asset_b_fail_nomesh.blend",
        ASSET_LIBRARY_ROOT / "nested" / "asset_c_ok.blend",
    ]
    skipped_path = ASSET_LIBRARY_ROOT / "nested" / "skip_mesh.blend"

    write_mesh_asset(target_paths[0], "AssetA")
    write_non_mesh_asset(target_paths[1], "AssetB_NoMesh")
    write_mesh_asset(target_paths[2], "AssetC")
    write_mesh_asset(skipped_path, "AssetSkipped")
    log(f"Created asset library sources at {ASSET_LIBRARY_ROOT}")
    return target_paths


def configure_asset_library_preference() -> None:
    libraries = bpy.context.preferences.filepaths.asset_libraries
    for library in list(libraries):
        if library.name == LIBRARY_NAME:
            libraries.remove(library)

    library = libraries.new(name=LIBRARY_NAME)
    library.path = str(ASSET_LIBRARY_ROOT)
    log(f"Configured asset library preference: {LIBRARY_NAME}")


def configure_capture_settings() -> None:
    settings = bpy.context.scene.gs_capture_settings
    settings.output_path = str(CAPTURE_OUTPUT_ROOT)
    settings.batch_mode = "ASSET_LIBRARY"
    settings.asset_library_name = LIBRARY_NAME
    settings.asset_library_recursive = True
    settings.asset_library_filename_filter = "asset_"

    settings.camera_count = 3
    settings.camera_distribution = "FIBONACCI"
    settings.render_speed_preset = "FAST"
    settings.transparent_background = False
    settings.material_mode = "ORIGINAL"
    settings.lighting_mode = "KEEP"
    settings.use_adaptive_capture = False
    settings.export_colmap = False
    settings.export_transforms_json = False
    settings.export_depth = False
    settings.export_normals = False
    settings.export_masks = False
    settings.enable_checkpoints = False
    settings.auto_resume = False
    settings.cancel_requested = False
    settings.batch_cancel_requested = False


def start_batch_capture() -> bool:
    result = bpy.ops.gs_capture.batch_capture()
    log(f"Batch start result: {result}")
    return "RUNNING_MODAL" in set(result)


def nonempty_files(path: Path, pattern: str) -> list[Path]:
    if not path.exists():
        return []
    matches = []
    for candidate in sorted(path.glob(pattern)):
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                matches.append(candidate)
        except OSError:
            continue
    return matches


def validate_outputs(expected_targets: list[Path]) -> tuple[dict, dict]:
    checks = {}
    details = {}
    manifest_path = CAPTURE_OUTPUT_ROOT / "asset_library_manifest.json"
    checks["manifest_exists"] = manifest_path.exists()
    if not manifest_path.exists():
        details["manifest_error"] = "missing"
        return checks, details

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files", [])
    summary = manifest.get("summary", {})
    details["manifest_summary"] = summary

    expected_relative = {
        str(path.relative_to(ASSET_LIBRARY_ROOT)).replace("\\", "/")
        for path in expected_targets
    }
    seen_relative = {
        str(entry.get("relative_blend_file", "")).replace("\\", "/")
        for entry in entries
    }
    checks["manifest_has_expected_files"] = seen_relative == expected_relative
    checks["manifest_total_is_three"] = summary.get("total_files") == 3
    checks["manifest_succeeded_is_two"] = summary.get("succeeded") == 2
    checks["manifest_failed_is_one"] = summary.get("failed") == 1

    entries_by_relative = {}
    for entry in entries:
        key = str(entry.get("relative_blend_file", "")).replace("\\", "/")
        entries_by_relative[key] = entry
    fail_entry = entries_by_relative.get("assets_main/asset_b_fail_nomesh.blend")
    checks["non_mesh_asset_failed"] = bool(fail_entry and fail_entry.get("status") == "failed")

    success_relatives = [
        "assets_main/asset_a_ok.blend",
        "nested/asset_c_ok.blend",
    ]
    all_success_outputs_ok = True
    success_output_counts = {}
    for relative_path in success_relatives:
        entry = entries_by_relative.get(relative_path, {})
        output_subfolder = str(entry.get("output_subfolder", ""))
        output_path = CAPTURE_OUTPUT_ROOT / output_subfolder
        images = nonempty_files(output_path / "images", "image_*.png")
        success_output_counts[relative_path] = len(images)
        if entry.get("status") != "succeeded" or len(images) == 0:
            all_success_outputs_ok = False
    checks["success_outputs_exist"] = all_success_outputs_ok
    checks["continued_after_failure"] = (
        entries_by_relative.get("nested/asset_c_ok.blend", {}).get("status") == "succeeded"
    )

    skipped_entry = entries_by_relative.get("nested/skip_mesh.blend")
    checks["filename_filter_excluded_skip_file"] = skipped_entry is None

    details["success_output_counts"] = success_output_counts
    details["entries"] = entries
    return checks, details


def write_and_quit() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(STATE["report"], indent=2), encoding="utf-8")
    log(f"Report written: {REPORT_PATH}")
    bpy.ops.wm.quit_blender()


def tick():
    try:
        if time.time() - STATE["start_time"] > 900:
            raise TimeoutError("Asset-library smoke timed out")

        if STATE["phase"] == "init":
            shutil.rmtree(CAPTURE_OUTPUT_ROOT, ignore_errors=True)
            register_addon()
            reset_scene()
            expected_targets = setup_asset_library_sources()
            configure_asset_library_preference()
            configure_capture_settings()
            if not start_batch_capture():
                raise RuntimeError("Batch capture did not start")

            STATE["expected_targets"] = expected_targets
            STATE["phase"] = "running"
            return 0.2

        settings = bpy.context.scene.gs_capture_settings
        if STATE["phase"] == "running":
            if settings.batch_running or settings.is_rendering:
                return 0.2

            checks, details = validate_outputs(STATE.get("expected_targets", []))
            STATE["report"]["checks"] = checks
            STATE["report"]["details"] = details
            STATE["report"]["success"] = bool(checks) and all(checks.values())
            write_and_quit()
            return None

        return None

    except Exception as exc:
        STATE["report"]["errors"].append(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        write_and_quit()
        return None


if __name__ == "__main__":
    bpy.app.timers.register(tick, first_interval=0.5)
