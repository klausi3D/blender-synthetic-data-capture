#!/usr/bin/env python3
"""Unit tests for pure Python parts of gs_capture_addon/core/validation.py.

Tests cover dataclasses, properties, add_* methods, and dict converters.
Blender-dependent validators (SceneValidator, etc.) are not tested here.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ── Mock bpy and coverage module so validation.py can import ─────────────

bpy_mock = types.ModuleType("bpy")
sys.modules["bpy"] = bpy_mock

# Mock the coverage module that validation.py imports
coverage_mock = types.ModuleType("gs_capture_addon.utils.coverage")
coverage_mock.CoverageAnalyzer = type("CoverageAnalyzer", (), {})
for pkg in ("gs_capture_addon", "gs_capture_addon.utils"):
    if pkg not in sys.modules:
        sys.modules[pkg] = types.ModuleType(pkg)
sys.modules["gs_capture_addon.utils.coverage"] = coverage_mock

# Also register the core package
if "gs_capture_addon.core" not in sys.modules:
    sys.modules["gs_capture_addon.core"] = types.ModuleType("gs_capture_addon.core")

# Load validation module
MODULE_PATH = ROOT / "gs_capture_addon" / "core" / "validation.py"
spec = importlib.util.spec_from_file_location("gs_capture_addon.core.validation", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module spec for {MODULE_PATH}")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class _Failures:
    """Simple test framework (no pytest dependency)."""
    def __init__(self):
        self.failures = []
        self.passed = 0
        self.total = 0

    def check(self, name: str, condition: bool, detail: str = ""):
        self.total += 1
        if condition:
            self.passed += 1
        else:
            msg = f"FAIL: {name}"
            if detail:
                msg += f" — {detail}"
            self.failures.append(msg)
            print(f"  {msg}")

    def report(self) -> int:
        if self.failures:
            print(f"\n{len(self.failures)} of {self.total} checks FAILED:")
            for f in self.failures:
                print(f"  - {f}")
            return 1
        print(f"\nAll {self.total} checks passed.")
        return 0


# ── Tests ────────────────────────────────────────────────────────────────


def test_validation_issue_construction(F: _Failures) -> None:
    """Test ValidationIssue dataclass construction."""
    issue = validation.ValidationIssue(
        level=validation.ValidationLevel.ERROR,
        category="scene",
        message="No objects",
        suggestion="Add objects",
        auto_fixable=True,
        fix_operator="fix.op",
    )
    F.check("issue level", issue.level == validation.ValidationLevel.ERROR)
    F.check("issue category", issue.category == "scene")
    F.check("issue message", issue.message == "No objects")
    F.check("issue suggestion", issue.suggestion == "Add objects")
    F.check("issue auto_fixable", issue.auto_fixable is True)
    F.check("issue fix_operator", issue.fix_operator == "fix.op")

    # Defaults
    issue2 = validation.ValidationIssue(
        level=validation.ValidationLevel.INFO,
        category="test",
        message="msg",
    )
    F.check("default suggestion empty", issue2.suggestion == "")
    F.check("default auto_fixable False", issue2.auto_fixable is False)
    F.check("default fix_operator empty", issue2.fix_operator == "")


def test_validation_result_can_proceed(F: _Failures) -> None:
    """Test can_proceed property."""
    result = validation.ValidationResult()
    F.check("empty result can proceed", result.can_proceed is True)

    result.add_warning("test", "a warning")
    F.check("warnings only can proceed", result.can_proceed is True)

    result.add_info("test", "some info")
    F.check("warnings+info can proceed", result.can_proceed is True)

    result.add_error("test", "an error")
    F.check("with error cannot proceed", result.can_proceed is False)


def test_validation_result_counts(F: _Failures) -> None:
    """Test error/warning/info count properties."""
    result = validation.ValidationResult()
    F.check("initial error_count 0", result.error_count == 0)
    F.check("initial warning_count 0", result.warning_count == 0)
    F.check("initial info_count 0", result.info_count == 0)

    result.add_error("a", "err1")
    result.add_error("b", "err2")
    result.add_warning("c", "warn1")
    result.add_info("d", "info1")
    result.add_info("e", "info2")
    result.add_info("f", "info3")

    F.check("error_count 2", result.error_count == 2)
    F.check("warning_count 1", result.warning_count == 1)
    F.check("info_count 3", result.info_count == 3)
    F.check("total issues 6", len(result.issues) == 6)


def test_add_error(F: _Failures) -> None:
    """Test add_error method."""
    result = validation.ValidationResult()
    result.add_error("cat", "msg", suggestion="fix", auto_fixable=True, fix_operator="op.fix")
    F.check("add_error adds one issue", len(result.issues) == 1)
    issue = result.issues[0]
    F.check("add_error level is ERROR", issue.level == validation.ValidationLevel.ERROR)
    F.check("add_error category", issue.category == "cat")
    F.check("add_error message", issue.message == "msg")
    F.check("add_error suggestion", issue.suggestion == "fix")
    F.check("add_error auto_fixable", issue.auto_fixable is True)
    F.check("add_error fix_operator", issue.fix_operator == "op.fix")


def test_add_warning(F: _Failures) -> None:
    """Test add_warning method."""
    result = validation.ValidationResult()
    result.add_warning("cat", "msg")
    issue = result.issues[0]
    F.check("add_warning level is WARNING", issue.level == validation.ValidationLevel.WARNING)


def test_add_info(F: _Failures) -> None:
    """Test add_info method."""
    result = validation.ValidationResult()
    result.add_info("cat", "msg")
    issue = result.issues[0]
    F.check("add_info level is INFO", issue.level == validation.ValidationLevel.INFO)
    F.check("add_info suggestion empty", issue.suggestion == "")


def test_validation_issue_to_dict(F: _Failures) -> None:
    """Test conversion of ValidationIssue to dict."""
    # Full issue
    issue = validation.ValidationIssue(
        level=validation.ValidationLevel.ERROR,
        category="scene",
        message="No objects",
        suggestion="Add objects",
        auto_fixable=True,
        fix_operator="fix.op",
    )
    d = validation.validation_issue_to_dict(issue)
    F.check("dict level", d["level"] == "error")
    F.check("dict category", d["category"] == "scene")
    F.check("dict message", d["message"] == "No objects")
    F.check("dict suggestion", d["suggestion"] == "Add objects")
    F.check("dict auto_fixable", d["auto_fixable"] is True)
    F.check("dict fix_operator", d["fix_operator"] == "fix.op")

    # Minimal issue (optional fields omitted)
    issue2 = validation.ValidationIssue(
        level=validation.ValidationLevel.INFO,
        category="test",
        message="info msg",
    )
    d2 = validation.validation_issue_to_dict(issue2)
    F.check("minimal dict has level", d2["level"] == "info")
    F.check("minimal dict no suggestion key", "suggestion" not in d2)
    F.check("minimal dict no auto_fixable key", "auto_fixable" not in d2)
    F.check("minimal dict no fix_operator key", "fix_operator" not in d2)


def test_validation_result_to_dict(F: _Failures) -> None:
    """Test conversion of ValidationResult to dict."""
    # None result
    d_none = validation.validation_result_to_dict(None)
    F.check("None result available=False", d_none["available"] is False)

    # Empty result
    result = validation.ValidationResult()
    d = validation.validation_result_to_dict(result)
    F.check("empty available=True", d["available"] is True)
    F.check("empty can_proceed=True", d["can_proceed"] is True)
    F.check("empty counts.errors=0", d["counts"]["errors"] == 0)
    F.check("empty counts.total=0", d["counts"]["total"] == 0)
    F.check("empty issues empty list", d["issues"] == [])

    # Populated result
    result.add_error("a", "err")
    result.add_warning("b", "warn")
    d2 = validation.validation_result_to_dict(result)
    F.check("populated can_proceed=False", d2["can_proceed"] is False)
    F.check("populated counts.errors=1", d2["counts"]["errors"] == 1)
    F.check("populated counts.warnings=1", d2["counts"]["warnings"] == 1)
    F.check("populated counts.total=2", d2["counts"]["total"] == 2)
    F.check("populated issues length=2", len(d2["issues"]) == 2)


def main() -> int:
    F = _Failures()

    test_validation_issue_construction(F)
    test_validation_result_can_proceed(F)
    test_validation_result_counts(F)
    test_add_error(F)
    test_add_warning(F)
    test_add_info(F)
    test_validation_issue_to_dict(F)
    test_validation_result_to_dict(F)

    print()
    return F.report()


if __name__ == "__main__":
    raise SystemExit(main())
