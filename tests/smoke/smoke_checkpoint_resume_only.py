#!/usr/bin/env python3
"""Isolated checkpoint resume smoke test for GS Capture."""

from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "training_out" / "smoke_feature_verification" / "checkpoint_only"
REPORT_PATH = ROOT / "training_out" / "smoke_feature_verification" / "checkpoint_only_report.json"

STATE = {
    "phase": "init",
    "start_time": time.time(),
    "cancel_sent": False,
    "report": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "blender_version": bpy.app.version_string,
        "platform": sys.platform,
        "events": [],
        "checkpoint_exists_after_cancel": False,
        "checkpoint_exists_after_resume": None,
        "images_count": 0,
        "success": False,
        "errors": [],
    },
}


def log(msg: str) -> None:
    print(f"[GS_CHECKPOINT] {msg}")
    STATE["report"]["events"].append(msg)


def pick_engine(scene) -> None:
    engine_prop = scene.render.bl_rna.properties.get("engine")
    ids = {item.identifier for item in engine_prop.enum_items}
    if "BLENDER_EEVEE_NEXT" in ids:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in ids:
        scene.render.engine = "BLENDER_EEVEE"


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

    pick_engine(scene)
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    log("Scene ready")


def configure() -> None:
    settings = bpy.context.scene.gs_capture_settings
    settings.output_path = str(OUT_DIR)
    settings.camera_count = 6
    settings.render_speed_preset = "FAST"
    settings.use_adaptive_capture = False
    settings.export_colmap = False
    settings.export_transforms_json = False
    settings.export_depth = False
    settings.export_normals = False
    settings.export_masks = False
    settings.enable_checkpoints = True
    settings.checkpoint_interval = 1
    settings.auto_resume = True
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
        if time.time() - STATE["start_time"] > 600:
            raise TimeoutError("Checkpoint smoke timed out")

        if STATE["phase"] == "init":
            shutil.rmtree(OUT_DIR, ignore_errors=True)
            register_addon()
            setup_scene()
            configure()
            if not start_capture():
                raise RuntimeError("Initial capture failed to start")
            STATE["phase"] = "running_initial"
            return 0.2

        settings = bpy.context.scene.gs_capture_settings

        if STATE["phase"] == "running_initial":
            if settings.is_rendering:
                if not STATE["cancel_sent"] and settings.capture_current >= 2:
                    bpy.ops.gs_capture.cancel_capture()
                    STATE["cancel_sent"] = True
                    log("Cancel requested")
                return 0.2

            checkpoint_path = OUT_DIR / ".gs_capture_checkpoint.json"
            STATE["report"]["checkpoint_exists_after_cancel"] = checkpoint_path.exists()
            if not STATE["cancel_sent"]:
                raise RuntimeError("Capture finished before cancel was sent")
            if not start_capture():
                raise RuntimeError("Resume capture failed to start")
            STATE["phase"] = "running_resume"
            return 0.2

        if STATE["phase"] == "running_resume":
            if settings.is_rendering:
                return 0.2

            checkpoint_path = OUT_DIR / ".gs_capture_checkpoint.json"
            images_dir = OUT_DIR / "images"
            images = list(images_dir.glob("image_*.png")) if images_dir.exists() else []
            STATE["report"]["checkpoint_exists_after_resume"] = checkpoint_path.exists()
            STATE["report"]["images_count"] = len(images)
            STATE["report"]["success"] = (
                STATE["report"]["checkpoint_exists_after_cancel"]
                and (not STATE["report"]["checkpoint_exists_after_resume"])
                and len(images) >= 6
            )
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
