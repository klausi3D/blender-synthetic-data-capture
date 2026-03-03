#!/usr/bin/env python3
"""Smoke test for proxy-hull splat cleanup integration."""

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
CAPTURE_DIR = OUT_ROOT / "cleanup_proxy_hull_capture"
TRAIN_DIR = OUT_ROOT / "cleanup_proxy_hull_training"
REPORT_PATH = OUT_ROOT / "splat_cleanup_proxy_hull_report.json"

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
    print(f"[GS_CLEANUP_SMOKE] {message}")
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

    scene.render.resolution_x = 384
    scene.render.resolution_y = 384
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"


def configure_capture_settings() -> None:
    settings = bpy.context.scene.gs_capture_settings
    settings.output_path = str(CAPTURE_DIR)
    settings.use_adaptive_capture = False
    settings.camera_count = 4
    settings.camera_distribution = "FIBONACCI"
    settings.render_speed_preset = "FAST"
    settings.transparent_background = False

    settings.export_colmap = False
    settings.export_transforms_json = False
    settings.export_depth = False
    settings.export_normals = False
    settings.export_masks = False
    settings.enable_checkpoints = False
    settings.auto_resume = False
    settings.cancel_requested = False

    settings.cleanup_export_proxy_hulls = True
    settings.cleanup_hull_margin = 0.0


def start_capture() -> None:
    result = bpy.ops.gs_capture.capture_selected()
    if "RUNNING_MODAL" not in set(result):
        raise RuntimeError(f"Capture failed to start: {result}")
    STATE["capture_started_at"] = time.time()
    log("Capture started")


def write_dummy_training_ply(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 5",
                "property float x",
                "property float y",
                "property float z",
                "property float opacity",
                "end_header",
                "0.0 0.0 0.0 0.8",
                "0.4 -0.2 0.3 0.6",
                "-0.5 0.5 -0.4 0.2",
                "1.9 0.0 0.0 0.1",
                "0.0 0.0 -1.8 0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def read_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            decoded = line.decode("ascii", errors="replace").strip()
            if decoded.startswith("element vertex "):
                return int(decoded.split()[-1])
            if decoded == "end_header":
                break
    return -1


def run_cleanup_and_evaluate() -> None:
    settings = bpy.context.scene.gs_capture_settings

    hull_path = CAPTURE_DIR / "cleanup" / "proxy_hulls.json"
    write_dummy_training_ply(TRAIN_DIR / "point_cloud" / "iteration_20" / "point_cloud.ply")

    settings.training_data_path = str(CAPTURE_DIR)
    settings.training_output_path = str(TRAIN_DIR)
    settings.cleanup_enable_auto = False
    settings.cleanup_hull_margin = 0.0
    settings.cleanup_max_removal_ratio = 0.9
    settings.cleanup_prefer_cleaned_import = True

    cleanup_result = bpy.ops.gs_capture.run_splat_cleanup()

    cleaned_path = TRAIN_DIR / "point_cloud" / "iteration_20" / "point_cloud.cleaned.ply"
    cleanup_report_path = TRAIN_DIR / "cleanup_report.json"

    object_count = -1
    if hull_path.exists():
        try:
            hull_payload = json.loads(hull_path.read_text(encoding="utf-8"))
            object_count = int(hull_payload.get("object_count", 0))
        except Exception:
            object_count = -1

    removed_points = -1
    proxy_hull_used = ""
    if cleanup_report_path.exists():
        try:
            cleanup_payload = json.loads(cleanup_report_path.read_text(encoding="utf-8"))
            removed_points = int(cleanup_payload.get("removed_points", -1))
            proxy_hull_used = str(cleanup_payload.get("proxy_hull_path", ""))
        except Exception:
            removed_points = -1

    checks = {
        "capture_finished": not bpy.context.scene.gs_capture_settings.is_rendering,
        "proxy_hulls_exists": hull_path.exists(),
        "proxy_hulls_has_objects": object_count > 0,
        "cleanup_operator_finished": "FINISHED" in set(cleanup_result),
        "cleaned_ply_exists": cleaned_path.exists(),
        "cleanup_report_exists": cleanup_report_path.exists(),
        "removed_points_positive": removed_points > 0,
        "kept_points_expected": read_vertex_count(cleaned_path) == 3,
        "proxy_hull_path_used": proxy_hull_used.replace("\\", "/").endswith("cleanup/proxy_hulls.json"),
    }

    STATE["report"]["checks"] = checks
    STATE["report"]["success"] = all(checks.values()) and not STATE["report"]["errors"]
    log(f"Cleanup smoke complete: success={STATE['report']['success']}")


def write_and_quit() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(STATE["report"], indent=2), encoding="utf-8")
    log(f"Report written: {REPORT_PATH}")
    bpy.ops.wm.quit_blender()


def tick():
    try:
        if time.time() - STATE["start_time"] > 600:
            raise TimeoutError("Cleanup smoke timed out")

        if STATE["phase"] == "init":
            OUT_ROOT.mkdir(parents=True, exist_ok=True)
            if REPORT_PATH.exists():
                REPORT_PATH.unlink()
            shutil.rmtree(CAPTURE_DIR, ignore_errors=True)
            shutil.rmtree(TRAIN_DIR, ignore_errors=True)
            ensure_addon_registered()
            setup_scene()
            configure_capture_settings()
            start_capture()
            STATE["phase"] = "wait_capture"
            return 0.2

        if STATE["phase"] == "wait_capture":
            started_at = STATE["capture_started_at"] or STATE["start_time"]
            if time.time() - started_at > 300:
                raise TimeoutError("Capture did not complete in time")

            if bpy.context.scene.gs_capture_settings.is_rendering:
                return 0.2

            run_cleanup_and_evaluate()
            STATE["phase"] = "done"
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
        STATE["report"]["success"] = False
        write_and_quit()
        return None


if __name__ == "__main__":
    bpy.app.timers.register(tick, first_interval=0.5)
