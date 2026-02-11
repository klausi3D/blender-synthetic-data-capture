#!/usr/bin/env python3
"""Smoke test for optional COLMAP binary export artifacts."""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import time
import traceback
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "training_out" / "smoke_feature_verification"
CASE_DIR = OUT_ROOT / "colmap_binary"
REPORT_PATH = OUT_ROOT / "colmap_binary_report.json"


STATE = {
    "phase": "init",
    "start_time": time.time(),
    "capture_started_at": None,
    "report": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "blender_version": bpy.app.version_string,
        "platform": sys.platform,
        "events": [],
        "checks": {},
        "success": False,
        "errors": [],
    },
}


def log(message: str) -> None:
    print(f"[GS_COLMAP_BIN] {message}")
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

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    bpy.ops.object.camera_add(location=(4.2, -4.2, 3.2), rotation=(1.12, 0.0, 0.79))
    scene.camera = bpy.context.active_object
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 6.0))

    bpy.ops.object.select_all(action="DESELECT")
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube

    pick_fast_engine(scene)
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = 8
    if hasattr(scene, "cycles"):
        scene.cycles.samples = 16

    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"


def configure_settings() -> None:
    settings = bpy.context.scene.gs_capture_settings
    settings.output_path = str(CASE_DIR)
    settings.target_collection = ""
    settings.use_adaptive_capture = False
    settings.camera_count = 8
    settings.camera_distribution = "FIBONACCI"
    settings.render_speed_preset = "FAST"
    settings.transparent_background = False

    settings.export_colmap = True
    settings.export_colmap_binary = True
    settings.colmap_initial_point_count = 1500
    settings.colmap_point_sampling = "SURFACE_FALLBACK"

    settings.export_transforms_json = False
    settings.export_depth = False
    settings.export_normals = False
    settings.export_masks = False
    settings.enable_checkpoints = False
    settings.auto_resume = False
    settings.cancel_requested = False


def read_u64_header(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        return -1
    return int(struct.unpack("<Q", header)[0])


def start_capture() -> None:
    result = bpy.ops.gs_capture.capture_selected()
    result_set = set(result)
    if "RUNNING_MODAL" not in result_set:
        raise RuntimeError(f"Capture failed to start: {result}")
    STATE["capture_started_at"] = time.time()
    log("Binary COLMAP smoke capture started")


def evaluate_results() -> None:
    colmap_dir = CASE_DIR / "sparse" / "0"
    images_dir = CASE_DIR / "images"

    cameras_txt = colmap_dir / "cameras.txt"
    images_txt = colmap_dir / "images.txt"
    points_txt = colmap_dir / "points3D.txt"
    cameras_bin = colmap_dir / "cameras.bin"
    images_bin = colmap_dir / "images.bin"
    points_bin = colmap_dir / "points3D.bin"
    validation_report = CASE_DIR / "validation_report.json"
    rendered_images = len(list(images_dir.glob("image_*.png")))

    checks = {
        "rendered_images_nonzero": rendered_images > 0,
        "text_colmap_exists": all(path.exists() for path in (cameras_txt, images_txt, points_txt)),
        "binary_colmap_exists": all(path.exists() for path in (cameras_bin, images_bin, points_bin)),
        "validation_report_exists": validation_report.exists(),
    }

    if checks["binary_colmap_exists"]:
        checks["cameras_bin_count_valid"] = read_u64_header(cameras_bin) == 1
        checks["images_bin_count_valid"] = read_u64_header(images_bin) == rendered_images
        checks["points_bin_count_valid"] = read_u64_header(points_bin) >= 100
    else:
        checks["cameras_bin_count_valid"] = False
        checks["images_bin_count_valid"] = False
        checks["points_bin_count_valid"] = False

    STATE["report"]["checks"] = checks
    STATE["report"]["success"] = all(checks.values()) and not STATE["report"]["errors"]
    log(f"Binary COLMAP smoke complete: success={STATE['report']['success']}")


def write_and_quit() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(STATE["report"], indent=2), encoding="utf-8")
    log(f"Report written: {REPORT_PATH}")
    bpy.ops.wm.quit_blender()


def tick():
    try:
        if time.time() - STATE["start_time"] > 600:
            raise TimeoutError("COLMAP binary smoke timed out")

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
            if time.time() - started_at > 240:
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
