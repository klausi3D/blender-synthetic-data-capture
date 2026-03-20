#!/usr/bin/env python3
"""Unit tests for gs_capture_addon/utils/paths.py (no Blender dependency)."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "gs_capture_addon" / "utils" / "paths.py"

spec = importlib.util.spec_from_file_location("paths", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module spec for {MODULE_PATH}")

paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)


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


def test_validate_path_length(F: _Failures) -> None:
    """Test path length validation."""
    # Normal path
    valid, norm, err = paths.validate_path_length("/some/normal/path")
    F.check("normal path is valid", valid)
    F.check("normal path no error", err == "")

    # Empty path
    valid, norm, err = paths.validate_path_length("")
    F.check("empty path invalid", not valid)
    F.check("empty path error message", "empty" in err.lower())

    # At limit (Windows check only applies on win32)
    long_path = "C:\\" + "a" * 256
    valid, norm, err = paths.validate_path_length(long_path)
    if sys.platform == "win32":
        F.check("at-limit path valid on Windows", valid)
    else:
        F.check("long path valid on non-Windows", valid)

    # Over limit on Windows
    over_path = "C:\\" + "a" * 300
    valid, norm, err = paths.validate_path_length(over_path, max_len=260)
    if sys.platform == "win32":
        F.check("over-limit path invalid on Windows", not valid)
        F.check("over-limit error mentions MAX_PATH", "MAX_PATH" in err)
    else:
        F.check("over-limit path still valid on non-Windows", valid)


def test_normalize_path(F: _Failures) -> None:
    """Test path normalization."""
    # Empty path
    F.check("empty returns empty", paths.normalize_path("") == "")

    # Home expansion
    result = paths.normalize_path("~/test")
    F.check("home expanded", "~" not in result or result.startswith("~"))
    # More precisely: os.path.expanduser should have run
    expected = os.path.normpath(os.path.expanduser("~/test"))
    if len(expected) > 1:
        expected = expected.rstrip(os.sep)
    F.check("home expansion correct", result == expected)

    # Trailing slash removal
    result = paths.normalize_path("/tmp/test/")
    F.check("trailing slash removed", not result.endswith(os.sep) or result == os.sep)

    # Root path preserved
    if sys.platform != "win32":
        result = paths.normalize_path("/")
        F.check("root path preserved", result == "/")


def test_validate_directory(F: _Failures) -> None:
    """Test directory validation."""
    # Empty path
    valid, norm, err = paths.validate_directory("")
    F.check("empty dir invalid", not valid)

    # Existing directory
    with tempfile.TemporaryDirectory(prefix="gs_test_") as tmpdir:
        valid, norm, err = paths.validate_directory(tmpdir, must_exist=True)
        F.check("existing dir valid", valid)
        F.check("existing dir no error", err == "")

    # Non-existing with must_exist
    valid, norm, err = paths.validate_directory("/nonexistent/path/xyz", must_exist=True)
    F.check("nonexistent must_exist invalid", not valid)
    F.check("nonexistent error message", "does not exist" in err.lower())

    # Create mode
    with tempfile.TemporaryDirectory(prefix="gs_test_") as tmpdir:
        new_dir = os.path.join(tmpdir, "new_subdir")
        valid, norm, err = paths.validate_directory(new_dir, create=True)
        F.check("create mode valid", valid)
        F.check("create mode dir exists", os.path.isdir(new_dir))


def test_validate_file(F: _Failures) -> None:
    """Test file validation."""
    # Empty path
    valid, norm, err = paths.validate_file("")
    F.check("empty file invalid", not valid)

    # Existing file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        tf.write(b"test")
        tf_path = tf.name
    try:
        valid, norm, err = paths.validate_file(tf_path, must_exist=True)
        F.check("existing file valid", valid)
        F.check("existing file no error", err == "")
    finally:
        os.unlink(tf_path)

    # Non-existing with must_exist
    valid, norm, err = paths.validate_file("/nonexistent/file.txt", must_exist=True)
    F.check("nonexistent file invalid", not valid)

    # Non-existing without must_exist
    valid, norm, err = paths.validate_file("/nonexistent/file.txt", must_exist=False)
    F.check("nonexistent file valid when not required", valid)


def test_get_conda_base(F: _Failures) -> None:
    """Test conda base detection (returns string, may be empty)."""
    result = paths.get_conda_base()
    F.check("get_conda_base returns string", isinstance(result, str))
    # If found, verify it has envs subdir
    if result:
        F.check("conda base has envs dir", os.path.isdir(os.path.join(result, "envs")))


def test_get_conda_python(F: _Failures) -> None:
    """Test conda Python path resolution."""
    # Empty env name
    F.check("empty env returns empty", paths.get_conda_python("") == "")

    # Non-existing base
    F.check("bad base returns empty", paths.get_conda_python("test_env", conda_base="/nonexistent") == "")


def test_get_conda_script(F: _Failures) -> None:
    """Test conda script path resolution."""
    # Empty args
    F.check("empty env returns empty", paths.get_conda_script("", "script") == "")
    F.check("empty script returns empty", paths.get_conda_script("env", "") == "")

    # Non-existing base
    F.check("bad base returns empty", paths.get_conda_script("env", "script", conda_base="/nonexistent") == "")


def test_get_conda_executable(F: _Failures) -> None:
    """Test conda executable detection."""
    result = paths.get_conda_executable(conda_base="/nonexistent")
    F.check("conda exe returns string", isinstance(result, str))
    # Falls back to shutil.which("conda")


def test_check_disk_space(F: _Failures) -> None:
    """Test disk space checking."""
    # Sufficient space (current dir should have some space)
    has_space, free_gb, err = paths.check_disk_space(os.getcwd(), required_gb=0.001)
    F.check("current dir has space", has_space)
    F.check("free_gb is positive", free_gb > 0)
    F.check("no error on success", err == "")

    # Insufficient space (require absurd amount)
    has_space, free_gb, err = paths.check_disk_space(os.getcwd(), required_gb=999999)
    F.check("insufficient space detected", not has_space)
    F.check("insufficient space error", "insufficient" in err.lower())


def main() -> int:
    F = _Failures()

    test_validate_path_length(F)
    test_normalize_path(F)
    test_validate_directory(F)
    test_validate_file(F)
    test_get_conda_base(F)
    test_get_conda_python(F)
    test_get_conda_script(F)
    test_get_conda_executable(F)
    test_check_disk_space(F)

    print()
    return F.report()


if __name__ == "__main__":
    raise SystemExit(main())
