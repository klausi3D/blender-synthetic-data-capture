#!/usr/bin/env python3
"""Verify Blender smoke-test reports produced in CI."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "training_out" / "smoke_feature_verification"


def load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_windows_path_requirement(release_report: dict, failures: list[str]) -> None:
    checks = release_report.get("checks", {})
    platform_name = release_report.get("environment", {}).get("platform")
    requirement = release_report.get("platform_requirements", {}).get("windows_path_warnings")

    if not isinstance(requirement, dict):
        failures.append("release report missing platform_requirements.windows_path_warnings")
        return

    required = requirement.get("required")
    status = requirement.get("status")
    result = checks.get("windows_path_warnings")

    if required is True:
        if platform_name != "win32":
            failures.append(
                f"release platform mismatch: windows_path_warnings marked required on {platform_name!r}"
            )
        if status != "required":
            failures.append("release windows_path_warnings status should be 'required'")
        if result is not True:
            failures.append("release check failed: windows_path_warnings (required on Windows)")
        return

    if required is False:
        if platform_name == "win32":
            failures.append("release platform mismatch: windows_path_warnings skipped on win32")
        if status != "skipped_non_windows":
            failures.append("release windows_path_warnings status should be 'skipped_non_windows'")
        if result is not None:
            failures.append("release windows_path_warnings should be null when skipped on non-Windows")
        return

    failures.append("release windows_path_warnings requirement must be explicit boolean")


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
        "colmap_loads_in_3dgs_inferred",
        "transforms_json_works_inferred",
    ]
    for key in required_release_checks:
        if checks.get(key) is not True:
            failures.append(f"release check failed: {key}")
    verify_windows_path_requirement(release_report, failures)

    if checkpoint_report.get("errors"):
        failures.append("checkpoint-only report contains errors")
    if checkpoint_report.get("success") is not True:
        failures.append("checkpoint-only smoke test failed")

    if object_index_report.get("errors"):
        failures.append("object-index-only report contains errors")
    if object_index_report.get("success") is not True:
        failures.append("object-index-only smoke test failed")
    object_index_checks = object_index_report.get("checks", {})
    required_object_index_checks = [
        "images_present",
        "masks_present",
        "mask_count_matches_images",
        "mask_ids_match_images",
        "artifact_signatures_valid",
        "mask_content_has_foreground_background",
    ]
    for key in required_object_index_checks:
        if object_index_checks.get(key) is not True:
            failures.append(f"object-index check failed: {key}")

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
    print(f"- object-index-only checks: {object_index_checks}")
    print(f"- coverage-edge checks: {coverage_checks}")
    print(f"- colmap-binary checks: {binary_checks}")
    print(f"- import-splat checks: {import_checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
