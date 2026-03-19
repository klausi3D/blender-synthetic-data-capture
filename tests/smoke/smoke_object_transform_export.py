#!/usr/bin/env python3
"""Smoke test for object_transforms.json sidecar export."""

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
CASE_DIR = OUT_ROOT / "object_transform_export"
REPORT_PATH = OUT_ROOT / "object_transform_export_report.json"


STATE = {
    "phase": "init",
    "start_time": time.time(),
    "expected_object_names": [],
    "capture_started_at": None,
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
    print(f"[GS_OBJ_XFORM] {message}")
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


def ensure_addon_registered() -> None:
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    import gs_capture_addon

    try:
        gs_capture_addon.unregister()
    except Exception:
        pass

    gs_capture_addon.register()
    log("Addon registered")


def clear_scene() -> None:
    try:
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def setup_scene() -> None:
    scene = bpy.context.scene
    clear_scene()

    bpy.ops.mesh.primitive_cube_add(size=1.8, location=(1.2, -0.6, 0.9))
    cube = bpy.context.active_object
    cube.rotation_euler = (0.23, 0.41, -0.14)
    cube.scale = (1.3, 0.85, 1.6)

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.7, location=(-1.4, 1.1, 1.3))
    sphere = bpy.context.active_object
    sphere.rotation_euler = (-0.19, 0.32, 0.27)
    sphere.scale = (0.9, 1.15, 0.75)

    bpy.ops.object.camera_add(location=(4.2, -4.2, 3.2), rotation=(1.12, 0.0, 0.79))
    scene.camera = bpy.context.active_object
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 6.0))

    bpy.ops.object.select_all(action="DESELECT")
    cube.select_set(True)
    sphere.select_set(True)
    bpy.context.view_layer.objects.active = cube
    STATE["expected_object_names"] = sorted([cube.name, sphere.name])

    pick_fast_engine(scene)
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = 8
    if hasattr(scene, "cycles"):
        scene.cycles.samples = 16

    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    log(f"Scene ready with objects: {STATE['expected_object_names']}")


def configure_settings() -> None:
    settings = bpy.context.scene.gs_capture_settings
    if not hasattr(settings, "export_object_transforms"):
        raise RuntimeError("Addon build missing export_object_transforms setting")

    settings.output_path = str(CASE_DIR)
    settings.target_collection = ""
    settings.use_adaptive_capture = False
    settings.camera_count = 8
    settings.camera_distribution = "FIBONACCI"
    settings.render_speed_preset = "FAST"
    settings.transparent_background = False

    settings.export_colmap = False
    settings.export_transforms_json = False
    settings.export_depth = False
    settings.export_normals = False
    settings.export_masks = False
    settings.export_object_transforms = True
    settings.object_transform_target_preset = "UNITY"
    settings.enable_checkpoints = False
    settings.auto_resume = False
    settings.cancel_requested = False


def is_number(value: object) -> bool:
    return isinstance(value, (int, float))


def is_float_list(value: object, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(is_number(item) for item in value)
    )


def is_matrix_4x4(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(is_float_list(row, 4) for row in value)
    )


def has_valid_trs(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        is_float_list(payload.get("location"), 3)
        and is_float_list(payload.get("rotation_quaternion_wxyz"), 4)
        and is_float_list(payload.get("rotation_euler_xyz_radians"), 3)
        and is_float_list(payload.get("scale"), 3)
    )


def start_capture() -> None:
    result = bpy.ops.gs_capture.capture_selected()
    result_set = set(result)
    if "RUNNING_MODAL" not in result_set:
        raise RuntimeError(f"Capture failed to start: {result}")
    STATE["capture_started_at"] = time.time()
    log("Object transform smoke capture started")


def evaluate_results() -> None:
    sidecar_path = CASE_DIR / "object_transforms.json"
    checks = {
        "sidecar_exists": sidecar_path.exists() and sidecar_path.is_file() and sidecar_path.stat().st_size > 0,
        "schema_basics_valid": False,
        "target_profile_valid": False,
        "conversion_metadata_valid": False,
        "objects_present": False,
        "selected_objects_exported": False,
        "object_schema_valid": False,
    }
    details = {
        "expected_object_names": list(STATE["expected_object_names"]),
        "exported_object_names": [],
        "object_count": 0,
    }

    if checks["sidecar_exists"]:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        source_metadata = data.get("source_metadata")
        target_profile = data.get("target_profile")
        conversion_metadata = data.get("conversion_metadata")
        objects = data.get("objects")

        checks["schema_basics_valid"] = (
            data.get("schema_version") == 1
            and isinstance(source_metadata, dict)
            and isinstance(target_profile, dict)
            and isinstance(conversion_metadata, dict)
            and isinstance(objects, list)
        )
        checks["target_profile_valid"] = (
            isinstance(target_profile, dict)
            and target_profile.get("id") == "UNITY"
            and isinstance(target_profile.get("handedness"), str)
        )
        checks["conversion_metadata_valid"] = (
            isinstance(conversion_metadata, dict)
            and is_matrix_4x4(conversion_metadata.get("conversion_matrix"))
            and isinstance(conversion_metadata.get("rule"), str)
        )
        checks["objects_present"] = isinstance(objects, list) and len(objects) >= len(
            STATE["expected_object_names"]
        )

        exported_names = sorted(
            [
                entry.get("name")
                for entry in objects
                if isinstance(entry, dict) and isinstance(entry.get("name"), str)
            ]
        )
        details["exported_object_names"] = exported_names
        details["object_count"] = len(objects) if isinstance(objects, list) else 0
        checks["selected_objects_exported"] = all(
            name in exported_names for name in STATE["expected_object_names"]
        )

        valid_entries = []
        if isinstance(objects, list):
            expected_name_set = set(STATE["expected_object_names"])
            for entry in objects:
                if not isinstance(entry, dict):
                    continue
                if entry.get("name") not in expected_name_set:
                    continue
                valid_entries.append(
                    is_matrix_4x4(entry.get("source_matrix_world"))
                    and is_matrix_4x4(entry.get("target_matrix_world"))
                    and has_valid_trs(entry.get("source_trs"))
                    and has_valid_trs(entry.get("target_trs"))
                )
        checks["object_schema_valid"] = bool(valid_entries) and all(valid_entries)

    STATE["report"]["checks"] = checks
    STATE["report"]["details"] = details
    STATE["report"]["success"] = all(checks.values()) and not STATE["report"]["errors"]
    log(f"Object transform smoke complete: success={STATE['report']['success']}")


def write_and_quit() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(STATE["report"], indent=2), encoding="utf-8")
    log(f"Report written: {REPORT_PATH}")
    bpy.ops.wm.quit_blender()


def tick():
    try:
        if time.time() - STATE["start_time"] > 900:
            raise TimeoutError("Object transform smoke timed out")

        if STATE["phase"] == "init":
            OUT_ROOT.mkdir(parents=True, exist_ok=True)
            if REPORT_PATH.exists():
                REPORT_PATH.unlink()
            shutil.rmtree(CASE_DIR, ignore_errors=True)
            ensure_addon_registered()
            setup_scene()
            configure_settings()
            start_capture()
            STATE["phase"] = "wait_capture"
            return 0.2

        if STATE["phase"] == "wait_capture":
            started_at = STATE["capture_started_at"] or STATE["start_time"]
            if time.time() - started_at > 300:
                raise TimeoutError("Capture did not complete in time")

            settings = bpy.context.scene.gs_capture_settings
            if settings.is_rendering:
                return 0.2

            evaluate_results()
            STATE["phase"] = "done"
            write_and_quit()
            return None

        if STATE["phase"] == "done":
            return None

        raise RuntimeError(f"Unknown phase: {STATE['phase']}")

    except Exception as exc:
        STATE["report"]["errors"].append(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        STATE["report"]["success"] = False
        write_and_quit()
        return None


if __name__ == "__main__":
    bpy.app.timers.register(tick, first_interval=0.5)
