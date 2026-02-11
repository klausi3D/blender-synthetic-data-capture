#!/usr/bin/env python3
"""Verify Blender smoke-test reports produced in CI."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "training_out" / "smoke_feature_verification"


def load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    failures: list[str] = []

    release_report = load_report(REPORT_ROOT / "smoke_report.json")
    checkpoint_report = load_report(REPORT_ROOT / "checkpoint_only_report.json")
    object_index_report = load_report(REPORT_ROOT / "object_index_only_report.json")
    coverage_edge_report = load_report(REPORT_ROOT / "coverage_edge_cases_report.json")
    colmap_binary_report = load_report(REPORT_ROOT / "colmap_binary_report.json")
    import_splat_report = load_report(REPORT_ROOT / "import_trained_splat_report.json")

    if release_report.get("errors"):
        failures.append("release smoke report contains errors")

    checks = release_report.get("checks", {})
    required_release_checks = [
        "clean_capture_test",
        "mask_export_alpha_object_index",
        "depth_export",
        "normal_export",
        "windows_path_warnings",
        "colmap_loads_in_3dgs_inferred",
        "transforms_json_works_inferred",
    ]
    for key in required_release_checks:
        if checks.get(key) is not True:
            failures.append(f"release check failed: {key}")

    if checkpoint_report.get("errors"):
        failures.append("checkpoint-only report contains errors")
    if checkpoint_report.get("success") is not True:
        failures.append("checkpoint-only smoke test failed")

    if object_index_report.get("errors"):
        failures.append("object-index-only report contains errors")
    if object_index_report.get("success") is not True:
        failures.append("object-index-only smoke test failed")

    if coverage_edge_report.get("errors"):
        failures.append("coverage-edge report contains errors")
    if coverage_edge_report.get("success") is not True:
        failures.append("coverage-edge smoke test failed")

    coverage_checks = coverage_edge_report.get("checks", {})
    required_coverage_checks = [
        "empty_camera_list_smoke",
        "high_poly_coverage_skip_smoke",
    ]
    for key in required_coverage_checks:
        if coverage_checks.get(key) is not True:
            failures.append(f"coverage-edge check failed: {key}")

    if colmap_binary_report.get("errors"):
        failures.append("colmap-binary report contains errors")
    if colmap_binary_report.get("success") is not True:
        failures.append("colmap-binary smoke test failed")

    binary_checks = colmap_binary_report.get("checks", {})
    required_binary_checks = [
        "rendered_images_nonzero",
        "text_colmap_exists",
        "binary_colmap_exists",
        "validation_report_exists",
        "cameras_bin_count_valid",
        "images_bin_count_valid",
        "points_bin_count_valid",
    ]
    for key in required_binary_checks:
        if binary_checks.get(key) is not True:
            failures.append(f"colmap-binary check failed: {key}")

    if import_splat_report.get("errors"):
        failures.append("import-splat report contains errors")
    if import_splat_report.get("success") is not True:
        failures.append("import-splat smoke test failed")

    import_checks = import_splat_report.get("checks", {})
    required_import_checks = [
        "operator_finished",
        "imported_object_detected",
        "location_applied",
        "uniform_scale_applied",
    ]
    for key in required_import_checks:
        if import_checks.get(key) is not True:
            failures.append(f"import-splat check failed: {key}")

    if failures:
        print("Smoke verification failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Smoke verification passed.")
    print(f"- release checks: {checks}")
    print(f"- checkpoint-only success: {checkpoint_report.get('success')}")
    print(f"- object-index-only success: {object_index_report.get('success')}")
    print(f"- coverage-edge checks: {coverage_checks}")
    print(f"- colmap-binary checks: {binary_checks}")
    print(f"- import-splat checks: {import_checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
