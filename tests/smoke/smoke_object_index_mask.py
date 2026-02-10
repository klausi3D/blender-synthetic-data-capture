#!/usr/bin/env python3
"""Isolated object-index mask smoke test."""

from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "training_out" / "smoke_feature_verification" / "object_index_only"
REPORT_PATH = ROOT / "training_out" / "smoke_feature_verification" / "object_index_only_report.json"

STATE = {
    "phase": "init",
    "start_time": time.time(),
    "report": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "blender_version": bpy.app.version_string,
        "platform": sys.platform,
        "events": [],
        "images_count": 0,
        "masks_count": 0,
        "success": False,
        "errors": [],
    },
}


def log(msg: str) -> None:
    print(f"[GS_OBJMASK] {msg}")
    STATE["report"]["events"].append(msg)


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


def setup_scene() -> None:
    scene = bpy.context.scene
    try:
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    bpy.ops.object.camera_add(location=(4.2, -4.2, 3.2), rotation=(1.12, 0.0, 0.79))
    scene.camera = bpy.context.active_object
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 6.0))
    bpy.ops.object.select_all(action="DESELECT")
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube

    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    log("Scene ready")


def configure() -> None:
    settings = bpy.context.scene.gs_capture_settings
    settings.output_path = str(OUT_DIR)
    settings.camera_count = 8
    settings.render_speed_preset = "QUALITY"  # Force Cycles for pass support.
    settings.use_adaptive_capture = False
    settings.export_colmap = False
    settings.export_transforms_json = False
    settings.export_depth = False
    settings.export_normals = False
    settings.export_masks = True
    settings.mask_source = "OBJECT_INDEX"
    settings.mask_format = "STANDARD"
    settings.enable_checkpoints = False
    settings.auto_resume = False
    settings.cancel_requested = False


def start_capture() -> bool:
    result = bpy.ops.gs_capture.capture_selected()
    log(f"Capture start result: {result}")
    return "RUNNING_MODAL" in set(result)


def write_and_quit() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(STATE["report"], indent=2), encoding="utf-8")
    log(f"Report written: {REPORT_PATH}")
    bpy.ops.wm.quit_blender()


def tick():
    try:
        if time.time() - STATE["start_time"] > 900:
            raise TimeoutError("Object-index smoke timed out")

        if STATE["phase"] == "init":
            shutil.rmtree(OUT_DIR, ignore_errors=True)
            register_addon()
            setup_scene()
            configure()
            if not start_capture():
                raise RuntimeError("Object-index capture failed to start")
            STATE["phase"] = "running"
            return 0.2

        settings = bpy.context.scene.gs_capture_settings
        if STATE["phase"] == "running":
            if settings.is_rendering:
                return 0.2

            images = list((OUT_DIR / "images").glob("image_*.png")) if (OUT_DIR / "images").exists() else []
            masks = []
            if (OUT_DIR / "masks").exists():
                masks.extend((OUT_DIR / "masks").glob("mask_*.png"))
                masks.extend((OUT_DIR / "masks").glob("mask_*.exr"))
            STATE["report"]["images_count"] = len(images)
            STATE["report"]["masks_count"] = len(masks)
            STATE["report"]["success"] = len(images) > 0 and len(masks) == len(images)
            write_and_quit()
            return None

        return None

    except Exception as e:
        STATE["report"]["errors"].append(
            {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        )
        write_and_quit()
        return None


if __name__ == "__main__":
    bpy.app.timers.register(tick, first_interval=0.5)
