#!/usr/bin/env python3
"""Unit tests for tools/capture_config_schema.py (no Blender dependency)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Load module via importlib
MODULE_PATH = ROOT / "tools" / "capture_config_schema.py"
spec = importlib.util.spec_from_file_location("capture_config_schema", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module spec for {MODULE_PATH}")
schema = importlib.util.module_from_spec(spec)
sys.modules["tools.capture_config_schema"] = schema
spec.loader.exec_module(schema)


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

def test_capture_exports_defaults(F):
    """Test CaptureExportsConfig default values."""
    exports = schema.CaptureExportsConfig()
    F.check("exports colmap default True", exports.colmap is True)
    F.check("exports colmap_binary default False", exports.colmap_binary is False)
    F.check("exports transforms_json default True", exports.transforms_json is True)
    F.check("exports depth default False", exports.depth is False)
    F.check("exports normals default False", exports.normals is False)
    F.check("exports masks default False", exports.masks is False)
    F.check("exports mask_source default ALPHA", exports.mask_source == "ALPHA")


def test_capture_config_defaults(F):
    """Test CaptureConfig default values."""
    cfg = schema.CaptureConfig()
    F.check("capture cameras default 100", cfg.cameras == 100)
    F.check("capture distribution default FIBONACCI", cfg.distribution == "FIBONACCI")
    F.check("capture preset default None", cfg.preset is None)
    F.check("capture engine default None", cfg.engine is None)
    F.check("capture speed_preset BALANCED", cfg.speed_preset == "BALANCED")
    F.check("capture resolution default None", cfg.resolution is None)
    F.check("capture transparent_background True", cfg.transparent_background is True)
    F.check("capture checkpoints True", cfg.checkpoints is True)
    F.check("capture auto_resume True", cfg.auto_resume is True)
    F.check("capture exports is CaptureExportsConfig",
            isinstance(cfg.exports, schema.CaptureExportsConfig))


def test_training_config_defaults(F):
    """Test TrainingConfig default values."""
    cfg = schema.TrainingConfig()
    F.check("training enabled default False", cfg.enabled is False)
    F.check("training backend default original", cfg.backend == "original")
    F.check("training iterations 30000", cfg.iterations == 30000)
    F.check("training save_iterations", cfg.save_iterations == [7000, 15000, 30000])
    F.check("training white_background True", cfg.white_background is True)
    F.check("training gpu_id 0", cfg.gpu_id == 0)
    F.check("training extra_args None", cfg.extra_args is None)


def test_cleanup_config_defaults(F):
    """Test CleanupConfig default values."""
    cfg = schema.CleanupConfig()
    F.check("cleanup enabled default False", cfg.enabled is False)
    F.check("cleanup proxy_hulls True", cfg.proxy_hulls is True)
    F.check("cleanup hull_margin 0.01", cfg.hull_margin == 0.01)
    F.check("cleanup max_removal_ratio 0.70", cfg.max_removal_ratio == 0.70)


def test_job_config_defaults(F):
    """Test JobConfig default construction."""
    job = schema.JobConfig()
    F.check("job blender_path default", job.blender_path == "blender")
    F.check("job scene empty", job.scene == "")
    F.check("job output_base empty", job.output_base == "")
    F.check("job capture is CaptureConfig", isinstance(job.capture, schema.CaptureConfig))
    F.check("job items empty list", job.items == [])
    F.check("job training is TrainingConfig", isinstance(job.training, schema.TrainingConfig))
    F.check("job cleanup is CleanupConfig", isinstance(job.cleanup, schema.CleanupConfig))
    F.check("job timeout_per_capture 3600", job.timeout_per_capture == 3600)
    F.check("job dry_run False", job.dry_run is False)


def test_dict_to_dataclass_basic(F):
    """Test _dict_to_dataclass with simple fields."""
    data = {"cameras": 200, "distribution": "RING", "speed_preset": "FAST"}
    cfg = schema._dict_to_dataclass(schema.CaptureConfig, data)
    F.check("d2d cameras", cfg.cameras == 200)
    F.check("d2d distribution", cfg.distribution == "RING")
    F.check("d2d speed_preset", cfg.speed_preset == "FAST")
    # Unspecified fields keep defaults
    F.check("d2d preset default", cfg.preset is None)


def test_dict_to_dataclass_ignores_unknown(F):
    """Test that unknown keys are silently ignored."""
    data = {"cameras": 50, "unknown_field": "should_be_ignored"}
    cfg = schema._dict_to_dataclass(schema.CaptureConfig, data)
    F.check("d2d unknown ignored", cfg.cameras == 50)
    F.check("d2d no unknown attr", not hasattr(cfg, "unknown_field"))


def test_dict_to_dataclass_nested(F):
    """Test recursive nested dataclass population."""
    data = {
        "scene": "test.blend",
        "capture": {
            "cameras": 50,
            "exports": {"depth": True, "normals": True},
        },
        "training": {"enabled": True, "iterations": 10000},
    }
    job = schema._dict_to_dataclass(schema.JobConfig, data)
    F.check("nested scene", job.scene == "test.blend")
    F.check("nested capture cameras", job.capture.cameras == 50)
    F.check("nested exports depth", job.capture.exports.depth is True)
    F.check("nested exports normals", job.capture.exports.normals is True)
    F.check("nested exports colmap default", job.capture.exports.colmap is True)
    F.check("nested training enabled", job.training.enabled is True)
    F.check("nested training iterations", job.training.iterations == 10000)


def test_dict_to_dataclass_items_list(F):
    """Test list of nested dataclass items."""
    data = {
        "items": [
            {"type": "collection", "name": "Chairs"},
            {"type": "collection", "name": "Tables", "cameras": 200},
        ],
    }
    job = schema._dict_to_dataclass(schema.JobConfig, data)
    F.check("items length 2", len(job.items) == 2)
    F.check("items[0] name", job.items[0].name == "Chairs")
    F.check("items[1] name", job.items[1].name == "Tables")
    F.check("items[1] cameras", job.items[1].cameras == 200)


def test_load_job_config_json(F):
    """Test loading a job config from a JSON file."""
    config_data = {
        "blender_path": "/usr/bin/blender",
        "scene": "/path/to/scene.blend",
        "output_base": "/output",
        "capture": {
            "cameras": 75,
            "distribution": "HEMISPHERE_TOP",
        },
        "training": {"enabled": True, "backend": "nerfstudio"},
        "cleanup": {"enabled": True, "hull_margin": 0.05},
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(config_data, f)
        config_path = f.name

    try:
        job = schema.load_job_config(config_path)
        F.check("load blender_path", job.blender_path == "/usr/bin/blender")
        F.check("load scene", job.scene == "/path/to/scene.blend")
        F.check("load capture cameras", job.capture.cameras == 75)
        F.check("load capture distribution", job.capture.distribution == "HEMISPHERE_TOP")
        F.check("load training enabled", job.training.enabled is True)
        F.check("load training backend", job.training.backend == "nerfstudio")
        F.check("load cleanup enabled", job.cleanup.enabled is True)
        F.check("load cleanup hull_margin", job.cleanup.hull_margin == 0.05)
    finally:
        os.unlink(config_path)


def test_job_config_to_dict(F):
    """Test serialization of JobConfig to dict."""
    job = schema.JobConfig(
        scene="test.blend",
        output_base="/out",
        capture=schema.CaptureConfig(cameras=50),
        training=schema.TrainingConfig(enabled=True),
    )
    d = schema.job_config_to_dict(job)
    F.check("to_dict is dict", isinstance(d, dict))
    F.check("to_dict scene", d["scene"] == "test.blend")
    F.check("to_dict output_base", d["output_base"] == "/out")
    F.check("to_dict capture.cameras", d["capture"]["cameras"] == 50)
    F.check("to_dict training.enabled", d["training"]["enabled"] is True)
    F.check("to_dict cleanup nested", isinstance(d["cleanup"], dict))
    F.check("to_dict exports nested", isinstance(d["capture"]["exports"], dict))


def test_job_config_roundtrip(F):
    """Test dict → dataclass → dict round-trip."""
    original = {
        "blender_path": "blender",
        "scene": "my.blend",
        "output_base": "/tmp/out",
        "capture": {"cameras": 120, "exports": {"depth": True}},
        "items": [{"name": "Chair"}],
        "training": {"backend": "gsplat", "iterations": 5000},
        "cleanup": {"hull_margin": 0.1},
    }
    job = schema._dict_to_dataclass(schema.JobConfig, original)
    d = schema.job_config_to_dict(job)
    F.check("roundtrip scene", d["scene"] == "my.blend")
    F.check("roundtrip cameras", d["capture"]["cameras"] == 120)
    F.check("roundtrip depth", d["capture"]["exports"]["depth"] is True)
    F.check("roundtrip items[0]", d["items"][0]["name"] == "Chair")
    F.check("roundtrip training backend", d["training"]["backend"] == "gsplat")
    F.check("roundtrip cleanup margin", d["cleanup"]["hull_margin"] == 0.1)


def main() -> int:
    F = _Failures()

    test_capture_exports_defaults(F)
    test_capture_config_defaults(F)
    test_training_config_defaults(F)
    test_cleanup_config_defaults(F)
    test_job_config_defaults(F)
    test_dict_to_dataclass_basic(F)
    test_dict_to_dataclass_ignores_unknown(F)
    test_dict_to_dataclass_nested(F)
    test_dict_to_dataclass_items_list(F)
    test_load_job_config_json(F)
    test_job_config_to_dict(F)
    test_job_config_roundtrip(F)

    print()
    return F.report()


if __name__ == "__main__":
    raise SystemExit(main())
