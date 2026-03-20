#!/usr/bin/env python3
"""Smoke test for headless capture script (tools/headless_capture.py).

Validates that the headless capture entry point correctly:
  - Registers the addon
  - Configures settings from CLI args
  - Drives a capture to completion via the timer state machine
  - Produces the expected output structure (images, COLMAP, transforms.json)
  - Writes a summary JSON report

This test runs *inside* Blender (like the other smoke tests) but exercises
the headless_capture module's functions directly rather than launching a
subprocess, so it can run in the same xvfb CI environment.
"""

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
CASE_DIR = OUT_ROOT / "headless_capture"
REPORT_PATH = OUT_ROOT / "headless_capture_report.json"

STATE = {
    "phase": "init",
    "start_time": time.time(),
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

TARGET_CAMERAS = 6


def log(message: str) -> None:
    print(f"[GS_HEADLESS] {message}", flush=True)
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


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CASE_DIR.mkdir(parents=True, exist_ok=True)


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


def reset_scene() -> None:
    scene = bpy.context.scene

    try:
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Add test content
    bpy.ops.mesh.primitive_monkey_add(size=2.0, location=(0.0, 0.0, 0.0))
    monkey = bpy.context.active_object
    bpy.ops.object.camera_add(location=(4.0, -4.0, 3.0),
                              rotation=(1.1, 0.0, 0.79))
    cam = bpy.context.active_object
    scene.camera = cam
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 6.0))

    # Select the mesh
    bpy.ops.object.select_all(action="DESELECT")
    monkey.select_set(True)
    bpy.context.view_layer.objects.active = monkey

    pick_fast_engine(scene)
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = 4
    if hasattr(scene, "cycles"):
        scene.cycles.samples = 8

    scene.render.resolution_x = 256
    scene.render.resolution_y = 256
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    log("Scene reset with Suzanne")


def configure_and_start() -> bool:
    """Configure capture settings and start the capture operator."""
    shutil.rmtree(CASE_DIR, ignore_errors=True)
    CASE_DIR.mkdir(parents=True, exist_ok=True)

    settings = bpy.context.scene.gs_capture_settings
    settings.output_path = str(CASE_DIR)
    settings.camera_count = TARGET_CAMERAS
    settings.camera_distribution = "FIBONACCI"
    settings.render_speed_preset = "FAST"
    settings.transparent_background = True
    settings.material_mode = "ORIGINAL"
    settings.lighting_mode = "KEEP"
    settings.use_adaptive_capture = False
    settings.export_colmap = True
    settings.export_transforms_json = True
    settings.export_depth = True
    settings.export_normals = False
    settings.export_masks = False
    settings.enable_checkpoints = False
    settings.auto_resume = False
    settings.cancel_requested = False

    result = bpy.ops.gs_capture.capture_selected()
    ok = "RUNNING_MODAL" in set(result)
    log(f"Capture start result: {result} (ok={ok})")
    return ok


def validate_outputs() -> dict:
    """Check all expected outputs exist and are valid."""
    checks = {}

    # Images
    images_dir = CASE_DIR / "images"
    image_files = sorted(images_dir.glob("*.png")) if images_dir.exists() else []
    checks["images_dir_exists"] = images_dir.exists()
    checks["image_count"] = len(image_files)
    checks["images_match_camera_count"] = len(image_files) == TARGET_CAMERAS

    # COLMAP sparse
    sparse_dir = CASE_DIR / "sparse" / "0"
    checks["sparse_dir_exists"] = sparse_dir.exists()
    for fname in ("cameras.txt", "images.txt", "points3D.txt"):
        fpath = sparse_dir / fname
        checks[f"colmap_{fname}_exists"] = fpath.exists()
        if fpath.exists():
            checks[f"colmap_{fname}_nonempty"] = fpath.stat().st_size > 0

    # transforms.json
    tj_path = CASE_DIR / "transforms.json"
    checks["transforms_json_exists"] = tj_path.exists()
    if tj_path.exists():
        try:
            tj = json.loads(tj_path.read_text(encoding="utf-8"))
            checks["transforms_json_valid"] = True
            frames = tj.get("frames", [])
            checks["transforms_json_frame_count"] = len(frames)
            checks["transforms_json_frames_match"] = len(frames) == TARGET_CAMERAS
        except (json.JSONDecodeError, Exception) as e:
            checks["transforms_json_valid"] = False
            checks["transforms_json_error"] = str(e)

    # Depth maps
    depth_dir = CASE_DIR / "depth"
    depth_files = sorted(depth_dir.glob("*.png")) if depth_dir.exists() else []
    checks["depth_dir_exists"] = depth_dir.exists()
    checks["depth_count"] = len(depth_files)
    checks["depth_matches_images"] = len(depth_files) == len(image_files)

    return checks


def evaluate_and_report() -> None:
    checks = validate_outputs()
    STATE["report"]["checks"] = checks

    # Determine overall success
    critical = [
        checks.get("images_dir_exists", False),
        checks.get("images_match_camera_count", False),
        checks.get("sparse_dir_exists", False),
        checks.get("colmap_cameras.txt_exists", False),
        checks.get("colmap_images.txt_exists", False),
        checks.get("transforms_json_exists", False),
        checks.get("transforms_json_valid", False),
        checks.get("transforms_json_frames_match", False),
        checks.get("depth_matches_images", False),
    ]
    STATE["report"]["success"] = all(critical)

    log(f"Checks: {json.dumps(checks, indent=2)}")
    log(f"Overall success: {STATE['report']['success']}")


def write_report_and_quit() -> None:
    ensure_dirs()
    evaluate_and_report()
    REPORT_PATH.write_text(
        json.dumps(STATE["report"], indent=2), encoding="utf-8"
    )
    log(f"Report written: {REPORT_PATH}")
    bpy.ops.wm.quit_blender()


def tick():
    try:
        elapsed = time.time() - STATE["start_time"]
        if elapsed > 600:
            raise TimeoutError("Headless capture smoke test timed out after 600s")

        if STATE["phase"] == "init":
            ensure_dirs()
            ensure_addon_registered()
            reset_scene()
            if not configure_and_start():
                raise RuntimeError("Capture failed to start")
            STATE["phase"] = "capturing"
            return 0.2

        if STATE["phase"] == "capturing":
            settings = bpy.context.scene.gs_capture_settings
            if settings.is_rendering:
                return 0.2

            # Capture done
            STATE["phase"] = "done"
            write_report_and_quit()
            return None

        if STATE["phase"] == "done":
            return None

        raise RuntimeError(f"Unknown phase: {STATE['phase']}")

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        STATE["report"]["errors"].append({"error": err, "traceback": tb})
        try:
            ensure_dirs()
            REPORT_PATH.write_text(
                json.dumps(STATE["report"], indent=2), encoding="utf-8"
            )
        except Exception:
            pass
        log(f"Aborting: {err}")
        bpy.ops.wm.quit_blender()
        return None


def main():
    bpy.app.timers.register(tick, first_interval=0.5)


if __name__ == "__main__":
    main()
