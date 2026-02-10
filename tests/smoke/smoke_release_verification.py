#!/usr/bin/env python3
"""
GS Capture release smoke verification.

Run this script from Blender (GUI mode, not background) because the capture
operator is modal and requires the event loop.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "training_out" / "smoke_feature_verification"
REPORT_PATH = OUT_ROOT / "smoke_report.json"


STATE = {
    "phase": "init",
    "start_time": time.time(),
    "cancel_sent": False,
    "case1_target_images": 6,
    "case2_target_images": 3,
    "report": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "blender_version": bpy.app.version_string,
            "platform": sys.platform,
        },
        "timeline": [],
        "cases": {},
        "checks": {},
        "errors": [],
    },
}


def log(message: str) -> None:
    print(f"[GS_SMOKE] {message}")
    STATE["report"]["timeline"].append(message)


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def nonempty_files(pattern: str) -> list[str]:
    matches = []
    for path in glob.glob(pattern):
        try:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                matches.append(path)
        except OSError:
            continue
    return sorted(matches)


def count_non_comment_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                count += 1
    return count


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

    # Ensure clean registration state.
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

    # Remove all objects.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Add simple test content.
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    bpy.ops.object.camera_add(location=(4.2, -4.2, 3.2), rotation=(1.12, 0.0, 0.79))
    cam = bpy.context.active_object
    scene.camera = cam
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 6.0))

    # Select target mesh object.
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

    log("Scene reset")


def configure_settings(
    output_dir: Path,
    camera_count: int,
    export_depth: bool,
    export_normals: bool,
    export_masks: bool,
    mask_source: str,
    enable_checkpoints: bool,
    auto_resume: bool,
    render_speed_preset: str = "FAST",
) -> None:
    settings = bpy.context.scene.gs_capture_settings
    settings.output_path = str(output_dir)
    settings.camera_count = camera_count
    settings.camera_distribution = "FIBONACCI"
    settings.render_speed_preset = render_speed_preset
    settings.transparent_background = (mask_source == "ALPHA")
    settings.material_mode = "ORIGINAL"
    settings.lighting_mode = "KEEP"
    settings.use_adaptive_capture = False
    settings.export_colmap = True
    settings.export_transforms_json = True
    settings.export_depth = export_depth
    settings.export_normals = export_normals
    settings.export_masks = export_masks
    settings.mask_source = mask_source
    settings.mask_format = "STANDARD"
    settings.enable_checkpoints = enable_checkpoints
    settings.checkpoint_interval = 1
    settings.auto_resume = auto_resume
    settings.cancel_requested = False


def start_capture(case_name: str) -> bool:
    result = bpy.ops.gs_capture.capture_selected()
    ok = "RUNNING_MODAL" in set(result)
    STATE["report"]["cases"].setdefault(case_name, {})["start_result"] = list(result)
    if ok:
        log(f"Capture started: {case_name}")
    else:
        log(f"Capture did not start cleanly: {case_name} -> {result}")
    return ok


def run_case1_prepare() -> bool:
    out_dir = OUT_ROOT / "case1_resume_object_index"
    shutil.rmtree(out_dir, ignore_errors=True)
    reset_scene()
    configure_settings(
        output_dir=out_dir,
        camera_count=STATE["case1_target_images"],
        export_depth=True,
        export_normals=True,
        export_masks=True,
        mask_source="OBJECT_INDEX",
        enable_checkpoints=True,
        auto_resume=True,
        render_speed_preset="QUALITY",
    )
    STATE["case1_dir"] = out_dir
    return start_capture("case1_initial_with_cancel")


def run_case1_resume() -> bool:
    # Re-select the cube in case context changed.
    cube = bpy.data.objects.get("Cube")
    if cube is not None:
        bpy.ops.object.select_all(action="DESELECT")
        cube.select_set(True)
        bpy.context.view_layer.objects.active = cube
    return start_capture("case1_resume")


def collect_case1_results() -> None:
    out_dir = Path(STATE["case1_dir"])
    checkpoint_path = out_dir / ".gs_capture_checkpoint.json"
    images = nonempty_files(str(out_dir / "images" / "image_*.png"))
    depth = (
        nonempty_files(str(out_dir / "depth" / "depth_*.png")) +
        nonempty_files(str(out_dir / "depth" / "depth_*.exr"))
    )
    normals = nonempty_files(str(out_dir / "normals" / "normal_*.exr"))
    masks = (
        nonempty_files(str(out_dir / "masks" / "mask_*.png")) +
        nonempty_files(str(out_dir / "masks" / "mask_*.exr"))
    )

    transforms_path = out_dir / "transforms.json"
    transforms_ok = False
    transforms_frames = 0
    transforms_error = ""
    if transforms_path.exists():
        try:
            data = json.loads(transforms_path.read_text(encoding="utf-8"))
            frames = data.get("frames", [])
            transforms_frames = len(frames)
            transforms_ok = isinstance(frames, list) and transforms_frames > 0
        except Exception as e:
            transforms_error = str(e)

    colmap_dir = out_dir / "sparse" / "0"
    cameras_txt = colmap_dir / "cameras.txt"
    images_txt = colmap_dir / "images.txt"
    points3d_txt = colmap_dir / "points3D.txt"
    colmap_present = cameras_txt.exists() and images_txt.exists() and points3d_txt.exists()
    colmap_counts = {
        "cameras_lines": count_non_comment_lines(cameras_txt),
        "images_lines": count_non_comment_lines(images_txt),
        "points3D_lines": count_non_comment_lines(points3d_txt),
    }

    case = {
        "output_dir": str(out_dir),
        "checkpoint_exists_after_resume": checkpoint_path.exists(),
        "images_count": len(images),
        "depth_count": len(depth),
        "normals_count": len(normals),
        "object_index_masks_count": len(masks),
        "transforms_exists": transforms_path.exists(),
        "transforms_ok": transforms_ok,
        "transforms_frames": transforms_frames,
        "transforms_error": transforms_error,
        "colmap_present": colmap_present,
        "colmap_counts": colmap_counts,
    }
    STATE["report"]["cases"]["case1_final"] = case


def run_case2_prepare() -> bool:
    out_dir = OUT_ROOT / "case2_alpha_mask"
    shutil.rmtree(out_dir, ignore_errors=True)
    reset_scene()
    configure_settings(
        output_dir=out_dir,
        camera_count=STATE["case2_target_images"],
        export_depth=False,
        export_normals=False,
        export_masks=True,
        mask_source="ALPHA",
        enable_checkpoints=False,
        auto_resume=False,
        render_speed_preset="QUALITY",
    )
    STATE["case2_dir"] = out_dir
    return start_capture("case2_alpha_mask")


def collect_case2_results() -> None:
    out_dir = Path(STATE["case2_dir"])
    images = nonempty_files(str(out_dir / "images" / "image_*.png"))
    masks = nonempty_files(str(out_dir / "masks" / "mask_*.png"))
    case = {
        "output_dir": str(out_dir),
        "images_count": len(images),
        "alpha_masks_count": len(masks),
    }
    STATE["report"]["cases"]["case2_final"] = case


def evaluate_checks() -> None:
    from gs_capture_addon.utils.paths import validate_path_length

    c1_partial = STATE["report"]["cases"].get("case1_after_cancel", {})
    c1_final = STATE["report"]["cases"].get("case1_final", {})
    c2_final = STATE["report"]["cases"].get("case2_final", {})

    checkpoint_resume_ok = (
        c1_partial.get("checkpoint_exists_after_cancel") is True
        and c1_final.get("checkpoint_exists_after_resume") is False
        and c1_final.get("images_count", 0) >= STATE["case1_target_images"]
    )

    long_path = "C:/" + ("x" * 400)
    long_ok, _, long_error = validate_path_length(long_path)
    windows_warning_ok = (sys.platform == "win32" and (not long_ok) and ("MAX_PATH" in long_error))

    transforms_ok = (
        c1_final.get("transforms_exists")
        and c1_final.get("transforms_ok")
        and c1_final.get("transforms_frames", 0) >= STATE["case1_target_images"]
    )

    colmap_ok = (
        c1_final.get("colmap_present")
        and c1_final.get("colmap_counts", {}).get("cameras_lines", 0) > 0
        and c1_final.get("colmap_counts", {}).get("images_lines", 0) > 0
    )

    STATE["report"]["checks"] = {
        "clean_capture_test": c1_final.get("images_count", 0) >= STATE["case1_target_images"],
        "mask_export_alpha_object_index": (
            c1_final.get("object_index_masks_count", 0) >= STATE["case1_target_images"]
            and c2_final.get("alpha_masks_count", 0) >= STATE["case2_target_images"]
        ),
        "depth_export": c1_final.get("depth_count", 0) >= STATE["case1_target_images"],
        "normal_export": c1_final.get("normals_count", 0) >= STATE["case1_target_images"],
        "checkpoint_resume": checkpoint_resume_ok,
        "windows_path_warnings": windows_warning_ok,
        # Inference based on exported file structure and parseability.
        "colmap_loads_in_3dgs_inferred": colmap_ok,
        "transforms_json_works_inferred": transforms_ok,
    }


def write_report_and_quit() -> None:
    ensure_dirs()
    evaluate_checks()
    REPORT_PATH.write_text(json.dumps(STATE["report"], indent=2), encoding="utf-8")
    log(f"Report written: {REPORT_PATH}")
    bpy.ops.wm.quit_blender()


def tick():
    try:
        elapsed = time.time() - STATE["start_time"]
        if elapsed > 900:
            raise TimeoutError("Smoke test timed out after 900 seconds")

        if STATE["phase"] == "init":
            ensure_dirs()
            ensure_addon_registered()
            if not run_case1_prepare():
                raise RuntimeError("Case 1 initial capture failed to start")
            STATE["phase"] = "case1_cancel_running"
            return 0.2

        settings = bpy.context.scene.gs_capture_settings

        if STATE["phase"] == "case1_cancel_running":
            if settings.is_rendering:
                if not STATE["cancel_sent"] and settings.capture_current >= 2:
                    result = bpy.ops.gs_capture.cancel_capture()
                    STATE["cancel_sent"] = True
                    log(f"Cancel requested for checkpoint test: {result}")
                return 0.2

            out_dir = Path(STATE["case1_dir"])
            checkpoint_path = out_dir / ".gs_capture_checkpoint.json"
            images_partial = nonempty_files(str(out_dir / "images" / "image_*.png"))
            STATE["report"]["cases"]["case1_after_cancel"] = {
                "output_dir": str(out_dir),
                "cancel_sent": STATE["cancel_sent"],
                "checkpoint_exists_after_cancel": checkpoint_path.exists(),
                "partial_images_count": len(images_partial),
            }

            if not STATE["cancel_sent"]:
                raise RuntimeError("Capture finished before cancellation; checkpoint resume not exercised")

            if not run_case1_resume():
                raise RuntimeError("Case 1 resume capture failed to start")

            STATE["phase"] = "case1_resume_running"
            return 0.2

        if STATE["phase"] == "case1_resume_running":
            if settings.is_rendering:
                return 0.2

            collect_case1_results()
            if not run_case2_prepare():
                raise RuntimeError("Case 2 alpha mask capture failed to start")
            STATE["phase"] = "case2_running"
            return 0.2

        if STATE["phase"] == "case2_running":
            if settings.is_rendering:
                return 0.2

            collect_case2_results()
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
            REPORT_PATH.write_text(json.dumps(STATE["report"], indent=2), encoding="utf-8")
        except Exception:
            pass
        log(f"Aborting due to error: {err}")
        bpy.ops.wm.quit_blender()
        return None


def main():
    # Start after startup context is fully ready.
    bpy.app.timers.register(tick, first_interval=0.5)


if __name__ == "__main__":
    main()
