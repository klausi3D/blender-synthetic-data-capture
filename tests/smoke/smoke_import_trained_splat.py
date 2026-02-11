#!/usr/bin/env python3
"""Smoke test for importing a trained splat (.ply) via training output operator."""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "training_out" / "smoke_feature_verification"
CASE_DIR = OUT_ROOT / "import_trained_splat"
REPORT_PATH = OUT_ROOT / "import_trained_splat_report.json"


def write_dummy_ply(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "0.0 0.0 0.0",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "3 0 1 2",
            ]
        ),
        encoding="utf-8",
    )


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


def reset_scene() -> None:
    try:
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))


def run() -> dict:
    report = {
        "blender_version": bpy.app.version_string,
        "checks": {},
        "success": False,
        "errors": [],
    }

    shutil.rmtree(CASE_DIR, ignore_errors=True)
    CASE_DIR.mkdir(parents=True, exist_ok=True)

    ensure_addon_registered()
    reset_scene()

    dummy_ply = CASE_DIR / "point_cloud" / "iteration_42" / "point_cloud.ply"
    write_dummy_ply(dummy_ply)

    settings = bpy.context.scene.gs_capture_settings
    settings.training_backend = "gaussian_splatting"
    settings.training_output_path = str(CASE_DIR)
    settings.training_import_location = (1.0, 2.0, 3.0)
    settings.training_import_uniform_scale = 2.0
    settings.training_import_replace_selection = True

    before_ids = {obj.as_pointer() for obj in bpy.data.objects}
    result = bpy.ops.gs_capture.open_training_output(action="IMPORT_SPLAT")
    imported = [obj for obj in bpy.data.objects if obj.as_pointer() not in before_ids]

    report["result"] = list(result)
    report["imported_count"] = len(imported)

    first_obj = imported[0] if imported else None
    if first_obj is not None:
        report["first_import_name"] = first_obj.name
        report["first_import_location"] = tuple(float(v) for v in first_obj.location)
        report["first_import_scale"] = tuple(float(v) for v in first_obj.scale)

    checks = {
        "operator_finished": "FINISHED" in set(result),
        "imported_object_detected": len(imported) >= 1,
        "location_applied": bool(
            first_obj
            and tuple(round(v, 3) for v in first_obj.location) == (1.0, 2.0, 3.0)
        ),
        "uniform_scale_applied": bool(
            first_obj
            and tuple(round(v, 3) for v in first_obj.scale) == (2.0, 2.0, 2.0)
        ),
    }
    report["checks"] = checks
    report["success"] = all(checks.values())
    return report


if __name__ == "__main__":
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        final_report = run()
    except Exception as exc:
        final_report = {
            "blender_version": bpy.app.version_string,
            "checks": {},
            "success": False,
            "errors": [
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            ],
        }

    REPORT_PATH.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"[GS_IMPORT_SPLAT] Report written: {REPORT_PATH}")
    bpy.ops.wm.quit_blender()
