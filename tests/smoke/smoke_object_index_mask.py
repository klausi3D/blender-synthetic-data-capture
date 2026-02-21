#!/usr/bin/env python3
"""Isolated object-index mask smoke test."""

from __future__ import annotations

import json
import re
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
        "checks": {},
        "details": {},
        "success": False,
        "errors": [],
    },
}


def log(msg: str) -> None:
    print(f"[GS_OBJMASK] {msg}")
    STATE["report"]["events"].append(msg)


def sorted_nonempty_matches(directory: Path, pattern: str) -> list[Path]:
    if not directory.exists():
        return []
    matches = []
    for path in sorted(directory.glob(pattern)):
        try:
            if path.is_file() and path.stat().st_size > 0:
                matches.append(path)
        except OSError:
            continue
    return matches


def extract_export_ids(paths: list[Path], prefix: str) -> set[str]:
    ids: set[str] = set()
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{4}})(?:\d{{4}})?$")
    for path in paths:
        stem = path.stem
        match = pattern.match(stem)
        if match:
            ids.add(match.group(1))
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


def assess_artifact_sanity(paths: list[Path], sample_limit: int = 2) -> dict:
    samples = []
    sample_errors = []
    for path in paths[:sample_limit]:
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


def sampled_mask_has_signal(mask_paths: list[Path], max_masks: int = 2, max_samples: int = 4096) -> tuple[bool, str]:
    if not mask_paths:
        return False, "no_mask_files"

    last_error = "sampled_masks_have_no_signal"
    for path in mask_paths[:max_masks]:
        image = None
        try:
            image = bpy.data.images.load(str(path), check_existing=False)
            width, height = image.size
            pixel_count = int(width) * int(height)
            channels = max(1, int(getattr(image, "channels", 4)))
            if pixel_count <= 0:
                last_error = f"{path.name}:invalid_image_size"
                continue

            step = max(1, pixel_count // max_samples)
            min_value = float("inf")
            max_value = float("-inf")
            for index in range(0, pixel_count, step):
                base = index * channels
                for channel in range(channels):
                    value = float(image.pixels[base + channel])
                    min_value = min(min_value, value)
                    max_value = max(max_value, value)
                if (max_value - min_value) > 0.01:
                    return True, f"{path.name}:value_range={max_value - min_value:.4f}"

            # Some valid object-index outputs can be nearly flat but still non-zero.
            if max_value > 0.01:
                return True, f"{path.name}:nonzero_flat_signal={max_value:.4f}"

            if min_value == float("inf"):
                last_error = f"{path.name}:no_sampled_pixels"
            else:
                last_error = f"{path.name}:value_range={max_value - min_value:.4f};max={max_value:.4f}"
        except Exception as e:
            last_error = f"{path.name}:load_error:{e}"
        finally:
            if image is not None:
                try:
                    bpy.data.images.remove(image)
                except Exception:
                    pass

    return False, last_error


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

            images = sorted_nonempty_matches(OUT_DIR / "images", "image_*.png")
            masks = sorted_nonempty_matches(OUT_DIR / "masks", "mask_*.png")
            masks.extend(sorted_nonempty_matches(OUT_DIR / "masks", "mask_*.exr"))

            image_ids = extract_export_ids(images, "image")
            mask_ids = extract_export_ids(masks, "mask")
            missing_mask_ids = sorted(image_ids - mask_ids)
            orphan_mask_ids = sorted(mask_ids - image_ids)

            image_sanity = assess_artifact_sanity(images)
            mask_sanity = assess_artifact_sanity(masks)
            mask_signal_ok, mask_signal_detail = sampled_mask_has_signal(masks)

            checks = {
                "images_present": len(images) > 0,
                "masks_present": len(masks) > 0,
                "mask_count_matches_images": len(masks) == len(images),
                "mask_ids_match_images": bool(image_ids) and not missing_mask_ids and not orphan_mask_ids,
                "artifact_signatures_valid": (
                    image_sanity.get("sample_signatures_ok") is True
                    and mask_sanity.get("sample_signatures_ok") is True
                ),
                "mask_content_has_signal": mask_signal_ok,
            }

            STATE["report"]["images_count"] = len(images)
            STATE["report"]["masks_count"] = len(masks)
            STATE["report"]["checks"] = checks
            STATE["report"]["details"] = {
                "images_sanity": image_sanity,
                "masks_sanity": mask_sanity,
                "missing_mask_ids": missing_mask_ids,
                "orphan_mask_ids": orphan_mask_ids,
                "mask_content_detail": mask_signal_detail,
            }
            STATE["report"]["success"] = all(checks.values())
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
