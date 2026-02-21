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
        "platform_requirements": {},
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


def first_non_comment_line(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    return ""


def extract_export_ids(paths: list[str], prefix: str) -> set[str]:
    ids: set[str] = set()
    stem_prefix = f"{prefix}_"
    for path in paths:
        stem = Path(path).stem
        if stem.startswith(stem_prefix):
            ids.add(stem[len(stem_prefix):])
    return ids


def validate_export_file_signature(path: Path) -> tuple[bool, str]:
    try:
        with path.open("rb") as f:
            header = f.read(32)
    except OSError as e:
        return False, f"io_error:{e}"

    suffix = path.suffix.lower()
    if suffix == ".png":
        if len(header) < 24:
            return False, "png_header_too_short"
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            return False, "png_signature_missing"
        if header[12:16] != b"IHDR":
            return False, "png_ihdr_missing"
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        if width <= 0 or height <= 0:
            return False, "png_invalid_dimensions"
        return True, ""

    if suffix == ".exr":
        if len(header) < 4 or header[:4] != b"\x76\x2f\x31\x01":
            return False, "exr_signature_missing"
        return True, ""

    return False, f"unsupported_extension:{suffix}"


def assess_artifact_sanity(paths: list[str], sample_limit: int = 2) -> dict:
    samples = []
    sample_errors = []
    for sample_path in paths[:sample_limit]:
        path = Path(sample_path)
        signature_ok, signature_error = validate_export_file_signature(path)
        sample = {"file": path.name, "signature_ok": signature_ok}
        if not signature_ok:
            sample["error"] = signature_error
            sample_errors.append(f"{path.name}:{signature_error}")
        samples.append(sample)

    return {
        "count": len(paths),
        "sample_checked": len(samples),
        "sample_signatures_ok": bool(samples) and not sample_errors,
        "sample_errors": sample_errors,
        "samples": samples,
    }


def looks_like_colmap_camera_line(line: str) -> bool:
    parts = line.split()
    return len(parts) >= 5 and parts[0].isdigit() and parts[2].isdigit() and parts[3].isdigit()


def looks_like_colmap_image_line(line: str) -> bool:
    parts = line.split()
    return len(parts) >= 10 and parts[0].isdigit() and parts[8].isdigit()


def validate_transforms_frames(frames: object) -> dict:
    result = {
        "frame_count": 0,
        "frame_list_ok": False,
        "sample_frame_schema_ok": False,
        "sample_frame_error": "",
    }
    if not isinstance(frames, list):
        result["sample_frame_error"] = "frames_is_not_list"
        return result

    result["frame_count"] = len(frames)
    result["frame_list_ok"] = True
    if not frames:
        result["sample_frame_error"] = "frames_empty"
        return result

    first = frames[0]
    if not isinstance(first, dict):
        result["sample_frame_error"] = "first_frame_not_object"
        return result

    file_path = first.get("file_path")
    matrix = first.get("transform_matrix")
    matrix_ok = (
        isinstance(matrix, list)
        and len(matrix) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in matrix)
    )
    file_path_ok = isinstance(file_path, str) and bool(file_path.strip())
    if file_path_ok and matrix_ok:
        result["sample_frame_schema_ok"] = True
    else:
        result["sample_frame_error"] = "first_frame_schema_invalid"
    return result


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

    image_ids = extract_export_ids(images, "image")
    depth_ids = extract_export_ids(depth, "depth")
    normal_ids = extract_export_ids(normals, "normal")
    mask_ids = extract_export_ids(masks, "mask")

    artifact_sanity = {
        "images": assess_artifact_sanity(images),
        "depth": assess_artifact_sanity(depth),
        "normals": assess_artifact_sanity(normals),
        "object_index_masks": assess_artifact_sanity(masks),
    }
    id_alignment = {
        "depth_covers_images": bool(image_ids) and image_ids.issubset(depth_ids),
        "normals_cover_images": bool(image_ids) and image_ids.issubset(normal_ids),
        "masks_cover_images": bool(image_ids) and image_ids.issubset(mask_ids),
        "depth_missing_image_ids": sorted(image_ids - depth_ids),
        "normal_missing_image_ids": sorted(image_ids - normal_ids),
        "mask_missing_image_ids": sorted(image_ids - mask_ids),
    }

    transforms_path = out_dir / "transforms.json"
    transforms_ok = False
    transforms_frames = 0
    transforms_sample_frame_schema_ok = False
    transforms_error = ""
    if transforms_path.exists():
        try:
            data = json.loads(transforms_path.read_text(encoding="utf-8"))
            frames = data.get("frames", [])
            transforms_detail = validate_transforms_frames(frames)
            transforms_frames = transforms_detail["frame_count"]
            transforms_sample_frame_schema_ok = transforms_detail["sample_frame_schema_ok"]
            transforms_ok = (
                transforms_detail["frame_list_ok"]
                and transforms_frames > 0
                and transforms_sample_frame_schema_ok
            )
            transforms_error = transforms_detail["sample_frame_error"]
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
    first_camera_line = first_non_comment_line(cameras_txt)
    first_image_line = first_non_comment_line(images_txt)
    colmap_text_sanity = {
        "camera_line_schema_ok": looks_like_colmap_camera_line(first_camera_line),
        "image_line_schema_ok": looks_like_colmap_image_line(first_image_line),
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
        "transforms_sample_frame_schema_ok": transforms_sample_frame_schema_ok,
        "transforms_error": transforms_error,
        "colmap_present": colmap_present,
        "colmap_counts": colmap_counts,
        "colmap_text_sanity": colmap_text_sanity,
        "artifact_sanity": artifact_sanity,
        "id_alignment": id_alignment,
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
    image_ids = extract_export_ids(images, "image")
    mask_ids = extract_export_ids(masks, "mask")
    artifact_sanity = {
        "images": assess_artifact_sanity(images),
        "alpha_masks": assess_artifact_sanity(masks),
    }
    case = {
        "output_dir": str(out_dir),
        "images_count": len(images),
        "alpha_masks_count": len(masks),
        "artifact_sanity": artifact_sanity,
        "id_alignment": {
            "masks_cover_images": bool(image_ids) and image_ids.issubset(mask_ids),
            "mask_missing_image_ids": sorted(image_ids - mask_ids),
        },
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

    is_windows = sys.platform == "win32"
    if is_windows:
        long_path = "C:/" + ("x" * 400)
        long_ok, _, long_error = validate_path_length(long_path)
        windows_warning_ok: bool | None = (not long_ok) and ("MAX_PATH" in (long_error or ""))
        windows_path_requirement = {
            "required": True,
            "status": "required",
            "result": windows_warning_ok,
            "detail": long_error,
        }
    else:
        windows_warning_ok = None
        windows_path_requirement = {
            "required": False,
            "status": "skipped_non_windows",
            "result": None,
            "detail": "validate_path_length MAX_PATH behavior is Windows-only",
        }

    transforms_ok = (
        c1_final.get("transforms_exists")
        and c1_final.get("transforms_ok")
        and c1_final.get("transforms_frames", 0) >= STATE["case1_target_images"]
        and c1_final.get("transforms_sample_frame_schema_ok") is True
    )

    colmap_ok = (
        c1_final.get("colmap_present")
        and c1_final.get("colmap_counts", {}).get("cameras_lines", 0) > 0
        and c1_final.get("colmap_counts", {}).get("images_lines", 0) > 0
        and c1_final.get("colmap_text_sanity", {}).get("camera_line_schema_ok") is True
        and c1_final.get("colmap_text_sanity", {}).get("image_line_schema_ok") is True
    )

    c1_artifacts = c1_final.get("artifact_sanity", {})
    c2_artifacts = c2_final.get("artifact_sanity", {})

    STATE["report"]["checks"] = {
        "clean_capture_test": (
            c1_final.get("images_count", 0) >= STATE["case1_target_images"]
            and c1_artifacts.get("images", {}).get("sample_signatures_ok") is True
        ),
        "mask_export_alpha_object_index": (
            c1_final.get("object_index_masks_count", 0) >= STATE["case1_target_images"]
            and c2_final.get("alpha_masks_count", 0) >= STATE["case2_target_images"]
            and c1_artifacts.get("object_index_masks", {}).get("sample_signatures_ok") is True
            and c2_artifacts.get("alpha_masks", {}).get("sample_signatures_ok") is True
            and c1_final.get("id_alignment", {}).get("masks_cover_images") is True
            and c2_final.get("id_alignment", {}).get("masks_cover_images") is True
        ),
        "depth_export": (
            c1_final.get("depth_count", 0) >= STATE["case1_target_images"]
            and c1_artifacts.get("depth", {}).get("sample_signatures_ok") is True
            and c1_final.get("id_alignment", {}).get("depth_covers_images") is True
        ),
        "normal_export": (
            c1_final.get("normals_count", 0) >= STATE["case1_target_images"]
            and c1_artifacts.get("normals", {}).get("sample_signatures_ok") is True
            and c1_final.get("id_alignment", {}).get("normals_cover_images") is True
        ),
        "checkpoint_resume": checkpoint_resume_ok,
        "windows_path_warnings": windows_warning_ok,
        # Inference based on exported file structure and parseability.
        "colmap_loads_in_3dgs_inferred": colmap_ok,
        "transforms_json_works_inferred": transforms_ok,
    }
    STATE["report"]["platform_requirements"] = {
        "windows_path_warnings": windows_path_requirement,
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
