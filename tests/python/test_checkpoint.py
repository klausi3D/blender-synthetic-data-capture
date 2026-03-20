#!/usr/bin/env python3
"""Unit tests for gs_capture_addon/utils/checkpoint.py (no Blender dependency)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Load paths module first (dependency of checkpoint)
paths_path = ROOT / "gs_capture_addon" / "utils" / "paths.py"
spec = importlib.util.spec_from_file_location("gs_capture_addon.utils.paths", paths_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module spec for {paths_path}")
paths_mod = importlib.util.module_from_spec(spec)
sys.modules["gs_capture_addon.utils.paths"] = paths_mod
# Also register the parent packages so relative imports resolve
for pkg in ("gs_capture_addon", "gs_capture_addon.utils"):
    if pkg not in sys.modules:
        import types
        sys.modules[pkg] = types.ModuleType(pkg)
spec.loader.exec_module(paths_mod)

# Now load checkpoint
checkpoint_path = ROOT / "gs_capture_addon" / "utils" / "checkpoint.py"
spec2 = importlib.util.spec_from_file_location("gs_capture_addon.utils.checkpoint", checkpoint_path)
if spec2 is None or spec2.loader is None:
    raise RuntimeError(f"Failed to load module spec for {checkpoint_path}")
checkpoint = importlib.util.module_from_spec(spec2)
sys.modules["gs_capture_addon.utils.checkpoint"] = checkpoint
spec2.loader.exec_module(checkpoint)


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


def test_get_checkpoint_path(F: _Failures) -> None:
    """Test checkpoint path generation."""
    result = checkpoint.get_checkpoint_path("/some/output")
    expected_name = ".gs_capture_checkpoint.json"
    F.check("checkpoint path ends correctly", result.endswith(expected_name))
    F.check("checkpoint path includes output dir", "/some/output" in result.replace("\\", "/"))


def test_save_and_load_checkpoint(F: _Failures) -> None:
    """Test saving and loading checkpoint data."""
    with tempfile.TemporaryDirectory(prefix="gs_ckpt_test_") as tmpdir:
        data = {
            "current_index": 5,
            "total_cameras": 20,
            "completed_images": [0, 1, 2, 3, 4],
            "settings_hash": "abc123",
        }
        # Save
        success, err = checkpoint.save_checkpoint(tmpdir, data)
        F.check("save succeeds", success)
        F.check("save no error", err is None)

        # Verify file exists
        cp_path = checkpoint.get_checkpoint_path(tmpdir)
        F.check("checkpoint file exists", os.path.isfile(cp_path))

        # Verify it's valid JSON
        with open(cp_path, "r") as f:
            saved = json.load(f)
        F.check("saved data has current_index", saved["current_index"] == 5)
        F.check("saved data has updated_at", "updated_at" in saved)

        # Load
        loaded, err = checkpoint.load_checkpoint(tmpdir)
        F.check("load succeeds", loaded is not None)
        F.check("load no error", err is None)
        F.check("loaded current_index", loaded["current_index"] == 5)
        F.check("loaded total_cameras", loaded["total_cameras"] == 20)
        F.check("loaded completed_images", loaded["completed_images"] == [0, 1, 2, 3, 4])


def test_load_checkpoint_missing(F: _Failures) -> None:
    """Test loading from directory with no checkpoint."""
    with tempfile.TemporaryDirectory(prefix="gs_ckpt_test_") as tmpdir:
        loaded, err = checkpoint.load_checkpoint(tmpdir)
        F.check("missing checkpoint returns None", loaded is None)
        F.check("missing checkpoint no error", err is None)


def test_load_checkpoint_corrupted(F: _Failures) -> None:
    """Test loading corrupted checkpoint JSON."""
    with tempfile.TemporaryDirectory(prefix="gs_ckpt_test_") as tmpdir:
        cp_path = checkpoint.get_checkpoint_path(tmpdir)
        with open(cp_path, "w") as f:
            f.write("{invalid json!!!")
        loaded, err = checkpoint.load_checkpoint(tmpdir)
        F.check("corrupted returns None", loaded is None)
        F.check("corrupted has error", err is not None)
        F.check("corrupted error mentions failure", "failed" in err.lower())


def test_load_checkpoint_missing_fields(F: _Failures) -> None:
    """Test loading checkpoint with missing required fields."""
    with tempfile.TemporaryDirectory(prefix="gs_ckpt_test_") as tmpdir:
        cp_path = checkpoint.get_checkpoint_path(tmpdir)
        with open(cp_path, "w") as f:
            json.dump({"current_index": 0}, f)  # Missing total_cameras and completed_images
        loaded, err = checkpoint.load_checkpoint(tmpdir)
        F.check("missing fields returns None", loaded is None)
        F.check("missing fields has error", err is not None)
        F.check("error mentions missing field", "missing" in err.lower())


def test_clear_checkpoint(F: _Failures) -> None:
    """Test clearing checkpoint file."""
    with tempfile.TemporaryDirectory(prefix="gs_ckpt_test_") as tmpdir:
        # Save first
        data = {"current_index": 0, "total_cameras": 10, "completed_images": []}
        checkpoint.save_checkpoint(tmpdir, data)
        cp_path = checkpoint.get_checkpoint_path(tmpdir)
        F.check("file exists before clear", os.path.isfile(cp_path))

        # Clear
        success, err = checkpoint.clear_checkpoint(tmpdir)
        F.check("clear succeeds", success)
        F.check("clear no error", err is None)
        F.check("file gone after clear", not os.path.exists(cp_path))

    # Clear on already-empty dir
    with tempfile.TemporaryDirectory(prefix="gs_ckpt_test_") as tmpdir:
        success, err = checkpoint.clear_checkpoint(tmpdir)
        F.check("clear on empty succeeds", success)
        F.check("clear on empty no error", err is None)


def test_create_checkpoint(F: _Failures) -> None:
    """Test checkpoint dictionary creation."""
    cp = checkpoint.create_checkpoint(
        current_index=3,
        total_cameras=50,
        settings_hash="hash123",
        settings_hash_legacy="legacy456",
        completed_images=[0, 1, 2],
    )
    F.check("has current_index", cp["current_index"] == 3)
    F.check("has total_cameras", cp["total_cameras"] == 50)
    F.check("has settings_hash", cp["settings_hash"] == "hash123")
    F.check("has settings_hash_legacy", cp["settings_hash_legacy"] == "legacy456")
    F.check("has completed_images", cp["completed_images"] == [0, 1, 2])
    F.check("has started_at", "started_at" in cp)
    F.check("has updated_at", "updated_at" in cp)
    F.check("has version", cp["version"] == "2.0.0")

    # Defaults
    cp2 = checkpoint.create_checkpoint(0, 10)
    F.check("default cameras_data empty list", cp2["cameras_data"] == [])
    F.check("default completed_images empty list", cp2["completed_images"] == [])
    F.check("default settings_hash is None", cp2["settings_hash"] is None)
    F.check("no legacy key when None", "settings_hash_legacy" not in cp2)


def test_get_missing_images(F: _Failures) -> None:
    """Test detection of missing image files."""
    with tempfile.TemporaryDirectory(prefix="gs_ckpt_test_") as tmpdir:
        # No images dir at all
        missing = checkpoint.get_missing_images(tmpdir, 5)
        F.check("all missing when no dir", missing == [0, 1, 2, 3, 4])

        # Create images dir with some files
        images_dir = os.path.join(tmpdir, "images")
        os.makedirs(images_dir)
        for i in [0, 2, 4]:
            with open(os.path.join(images_dir, f"image_{i:04d}.png"), "w") as f:
                f.write("fake")

        missing = checkpoint.get_missing_images(tmpdir, 5, extension="png")
        F.check("missing indices correct", missing == [1, 3])

        # All present
        for i in [1, 3]:
            with open(os.path.join(images_dir, f"image_{i:04d}.png"), "w") as f:
                f.write("fake")
        missing = checkpoint.get_missing_images(tmpdir, 5, extension="png")
        F.check("none missing when all present", missing == [])


def test_validate_checkpoint(F: _Failures) -> None:
    """Test checkpoint validation against current settings."""
    with tempfile.TemporaryDirectory(prefix="gs_ckpt_test_") as tmpdir:
        # No checkpoint on disk
        valid, cp, err = checkpoint.validate_checkpoint(tmpdir, "hash123")
        F.check("no checkpoint is invalid", not valid)
        F.check("no checkpoint error", "no checkpoint" in err.lower())

    # Hash mismatch with provided data
    cp_data = {
        "current_index": 5,
        "total_cameras": 20,
        "settings_hash": "old_hash",
        "completed_images": [],
    }
    valid, cp, err = checkpoint.validate_checkpoint(
        "/tmp", "new_hash", checkpoint_data=cp_data
    )
    F.check("hash mismatch invalid", not valid)
    F.check("hash mismatch error", "changed" in err.lower())

    # Valid checkpoint with matching hash
    cp_data2 = {
        "current_index": 5,
        "total_cameras": 20,
        "settings_hash": "matching_hash",
        "completed_images": [],
    }
    valid, cp, err = checkpoint.validate_checkpoint(
        "/tmp", "matching_hash", checkpoint_data=cp_data2
    )
    F.check("matching hash valid", valid)
    F.check("matching hash no error", err is None)


def test_settings_hash_matches(F: _Failures) -> None:
    """Test settings hash matching logic."""
    # Direct match
    cp = {"settings_hash": "abc123"}
    F.check("direct match", checkpoint.settings_hash_matches(cp, "abc123"))

    # Legacy match
    cp2 = {"settings_hash": "old", "settings_hash_legacy": "legacy789"}
    F.check("legacy match", checkpoint.settings_hash_matches(cp2, "legacy789"))

    # Mismatch
    F.check("mismatch", not checkpoint.settings_hash_matches(cp, "wrong"))

    # Legacy param match
    cp3 = {"settings_hash": "stored"}
    F.check("legacy param match", checkpoint.settings_hash_matches(cp3, "nope", legacy_settings_hash="stored"))

    # Empty/None guards
    F.check("None checkpoint", not checkpoint.settings_hash_matches(None, "abc"))
    F.check("empty hash", not checkpoint.settings_hash_matches(cp, ""))

    # v1 key match
    cp4 = {"settings_hash_v1": "v1hash"}
    F.check("v1 key match", checkpoint.settings_hash_matches(cp4, "v1hash"))


def main() -> int:
    F = _Failures()

    test_get_checkpoint_path(F)
    test_save_and_load_checkpoint(F)
    test_load_checkpoint_missing(F)
    test_load_checkpoint_corrupted(F)
    test_load_checkpoint_missing_fields(F)
    test_clear_checkpoint(F)
    test_create_checkpoint(F)
    test_get_missing_images(F)
    test_validate_checkpoint(F)
    test_settings_hash_matches(F)

    print()
    return F.report()


if __name__ == "__main__":
    raise SystemExit(main())
