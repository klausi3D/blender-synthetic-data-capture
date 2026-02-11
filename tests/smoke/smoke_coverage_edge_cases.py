#!/usr/bin/env python3
"""Smoke test coverage for edge-case camera/coverage behavior."""

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
EMPTY_CASE_DIR = OUT_ROOT / "empty_camera_list"
REPORT_PATH = OUT_ROOT / "coverage_edge_cases_report.json"


STATE = {
    "phase": "init",
    "start_time": time.time(),
    "empty_case_started_at": None,
    "report": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "blender_version": bpy.app.version_string,
        "platform": sys.platform,
        "events": [],
        "cases": {},
        "checks": {},
        "success": False,
        "errors": [],
    },
}


def log(message: str) -> None:
    print(f"[GS_COVERAGE_EDGE] {message}")
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


def clear_scene() -> None:
    try:
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def register_addon() -> None:
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


def setup_empty_camera_scene() -> None:
    scene = bpy.context.scene
    clear_scene()

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    bpy.ops.object.camera_add(location=(4.2, -4.2, 3.2), rotation=(1.12, 0.0, 0.79))
    scene.camera = bpy.context.active_object
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 6.0))

    bpy.ops.object.select_all(action="DESELECT")
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube

    pick_fast_engine(scene)
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"


def configure_empty_camera_settings() -> None:
    settings = bpy.context.scene.gs_capture_settings
    settings.output_path = str(EMPTY_CASE_DIR)
    settings.target_collection = ""
    settings.use_adaptive_capture = False
    settings.camera_count = 8
    settings.camera_distribution = "FIBONACCI"
    # Intentionally inverted range: this should produce an empty camera list.
    settings.min_elevation = 60.0
    settings.max_elevation = -60.0
    settings.render_speed_preset = "FAST"
    settings.export_colmap = False
    settings.export_transforms_json = False
    settings.export_depth = False
    settings.export_normals = False
    settings.export_masks = False
    settings.enable_checkpoints = False
    settings.auto_resume = False
    settings.cancel_requested = False


def run_empty_camera_case() -> None:
    from gs_capture_addon.core.camera import generate_camera_positions

    shutil.rmtree(EMPTY_CASE_DIR, ignore_errors=True)
    setup_empty_camera_scene()
    configure_empty_camera_settings()

    settings = bpy.context.scene.gs_capture_settings
    generated_points = generate_camera_positions(
        settings.camera_distribution,
        settings.camera_count,
        min_elevation=settings.min_elevation,
        max_elevation=settings.max_elevation,
        ring_count=settings.ring_count,
    )

    case = {
        "output_dir": str(EMPTY_CASE_DIR),
        "generated_camera_count": len(generated_points),
        "start_result": [],
        "capture_started": False,
        "images_count": 0,
        "last_capture_images": None,
        "last_capture_success": None,
        "checks": {},
        "success": False,
    }
    STATE["report"]["cases"]["empty_camera_list"] = case

    result = bpy.ops.gs_capture.capture_selected()
    case["start_result"] = list(result)
    case["capture_started"] = "RUNNING_MODAL" in set(result)
    if not case["capture_started"]:
        raise RuntimeError(f"Empty camera smoke did not start capture: {result}")

    STATE["empty_case_started_at"] = time.time()
    log("Empty camera-list capture started")


def finalize_empty_camera_case() -> None:
    case = STATE["report"]["cases"]["empty_camera_list"]
    settings = bpy.context.scene.gs_capture_settings

    images_dir = EMPTY_CASE_DIR / "images"
    images = list(images_dir.glob("image_*.png")) if images_dir.exists() else []

    case["images_count"] = len(images)
    case["last_capture_images"] = int(settings.last_capture_images)
    case["last_capture_success"] = bool(settings.last_capture_success)

    checks = {
        "generated_camera_count_is_zero": case["generated_camera_count"] == 0,
        "capture_started": case["capture_started"] is True,
        "rendered_images_is_zero": case["images_count"] == 0,
        "last_capture_images_is_zero": case["last_capture_images"] == 0,
        "last_capture_marked_successful": case["last_capture_success"] is True,
    }
    case["checks"] = checks
    case["success"] = all(checks.values())
    log(f"Empty camera-list case complete: success={case['success']}")


def run_high_poly_coverage_skip_case() -> None:
    from gs_capture_addon.core.validation import CoverageValidator

    scene = bpy.context.scene
    clear_scene()
    pick_fast_engine(scene)

    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=450,
        y_subdivisions=450,
        size=2.0,
        location=(0.0, 0.0, 0.0),
    )
    grid = bpy.context.active_object

    camera_positions = (
        (5.0, -5.0, 4.0),
        (-5.0, -5.0, 4.0),
        (0.0, 6.5, 3.0),
    )
    cameras = []
    for index, location in enumerate(camera_positions):
        bpy.ops.object.camera_add(location=location, rotation=(1.15, 0.0, 0.8))
        cam = bpy.context.active_object
        cam.name = f"SmokeCoverageCam_{index:02d}"
        cameras.append(cam)

    scene.camera = cameras[0]

    bpy.ops.object.select_all(action="DESELECT")
    grid.select_set(True)
    bpy.context.view_layer.objects.active = grid

    settings = scene.gs_capture_settings
    settings.target_collection = ""

    result = CoverageValidator().validate(bpy.context, settings, cameras)
    issues = [
        {
            "level": issue.level.value,
            "category": issue.category,
            "message": issue.message,
        }
        for issue in result.issues
    ]
    skip_messages = [
        issue["message"]
        for issue in issues
        if issue["category"] == "coverage"
        and "Coverage validation skipped for" in issue["message"]
    ]

    vertex_count = len(grid.data.vertices) if grid.data else 0
    checks = {
        "vertex_threshold_exceeded": vertex_count > 200000,
        "coverage_skip_message_present": bool(skip_messages),
    }

    case = {
        "vertex_count": vertex_count,
        "camera_count": len(cameras),
        "coverage_issues": issues,
        "skip_messages": skip_messages,
        "checks": checks,
        "success": all(checks.values()),
    }
    STATE["report"]["cases"]["high_poly_coverage_skip"] = case
    log(f"High-poly coverage-skip case complete: success={case['success']}")


def write_and_quit() -> None:
    empty_ok = STATE["report"]["cases"].get("empty_camera_list", {}).get("success") is True
    high_poly_ok = STATE["report"]["cases"].get("high_poly_coverage_skip", {}).get("success") is True

    STATE["report"]["checks"] = {
        "empty_camera_list_smoke": empty_ok,
        "high_poly_coverage_skip_smoke": high_poly_ok,
    }
    STATE["report"]["success"] = all(STATE["report"]["checks"].values()) and not STATE["report"]["errors"]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(STATE["report"], indent=2), encoding="utf-8")
    log(f"Report written: {REPORT_PATH}")
    bpy.ops.wm.quit_blender()


def tick():
    try:
        if time.time() - STATE["start_time"] > 600:
            raise TimeoutError("Coverage edge-case smoke timed out")

        if STATE["phase"] == "init":
            OUT_ROOT.mkdir(parents=True, exist_ok=True)
            if REPORT_PATH.exists():
                REPORT_PATH.unlink()
            register_addon()
            run_empty_camera_case()
            STATE["phase"] = "empty_wait"
            return 0.2

        if STATE["phase"] == "empty_wait":
            started_at = STATE["empty_case_started_at"] or STATE["start_time"]
            if time.time() - started_at > 180:
                raise TimeoutError("Empty camera-list capture did not complete in time")

            settings = bpy.context.scene.gs_capture_settings
            if settings.is_rendering:
                return 0.2

            finalize_empty_camera_case()
            run_high_poly_coverage_skip_case()
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
        write_and_quit()
        return None


if __name__ == "__main__":
    bpy.app.timers.register(tick, first_interval=0.5)
