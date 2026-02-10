#!/usr/bin/env python3
"""Verify Blender smoke-test reports produced in CI."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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

    if failures:
        print("Smoke verification failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Smoke verification passed.")
    print(f"- release checks: {checks}")
    print(f"- checkpoint-only success: {checkpoint_report.get('success')}")
    print(f"- object-index-only success: {object_index_report.get('success')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

