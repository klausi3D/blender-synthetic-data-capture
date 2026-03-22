#!/usr/bin/env python3
"""Unit tests for the library capture tool (no Blender dependency)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- Load modules via importlib to avoid __init__.py issues ---

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load capture_config_schema first (dependency of library_config_schema)
capture_schema = _load_module(
    "tools.capture_config_schema",
    ROOT / "tools" / "capture_config_schema.py",
)

# Load library_config_schema
library_schema = _load_module(
    "tools.library_config_schema",
    ROOT / "tools" / "library_config_schema.py",
)

# Load pipeline (dependency of library_capture)
pipeline_mod = _load_module(
    "tools.pipeline",
    ROOT / "tools" / "pipeline.py",
)

# Patch pipeline into the module path so library_capture can import from it
sys.modules["pipeline"] = pipeline_mod

# Load library_capture
library_capture = _load_module(
    "tools.library_capture",
    ROOT / "tools" / "library_capture.py",
)


# ── Helpers ──────────────────────────────────────────────────────────────

BLEND_HEADER = b"BLENDER"


def make_blend_file(path: Path) -> None:
    """Create a minimal file with a valid .blend header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write minimum valid-looking .blend header (7 bytes + padding)
    path.write_bytes(BLEND_HEADER + b"\x00" * 100)


def make_summary(path: Path, success: bool = True, images: int = 10) -> None:
    """Write a headless_capture_summary.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "success": success,
        "total_images": images,
    }), encoding="utf-8")


# ── Tests ────────────────────────────────────────────────────────────────

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


def test_discover_blend_files(F: _Failures) -> None:
    """Test .blend file discovery with filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create test files
        make_blend_file(root / "chair.blend")
        make_blend_file(root / "table.blend")
        make_blend_file(root / "chair_backup.blend")
        make_blend_file(root / "subdir" / "lamp.blend")
        (root / "readme.txt").write_text("not a blend")

        # Basic discovery (recursive)
        config = {"library_path": str(root), "recursive": True,
                  "filters": {"include_files": ["*.blend"], "exclude_files": []}}
        found = library_capture.discover_blend_files(config)
        names = {p.name for p in found}
        F.check("discover_recursive_count", len(found) == 4,
                f"expected 4, got {len(found)}: {names}")
        F.check("discover_includes_subdir", "lamp.blend" in names)

        # Non-recursive
        config["recursive"] = False
        found = library_capture.discover_blend_files(config)
        names = {p.name for p in found}
        F.check("discover_nonrecursive", len(found) == 3,
                f"expected 3, got {len(found)}")
        F.check("discover_no_subdir", "lamp.blend" not in names)

        # Exclude pattern
        config["recursive"] = True
        config["filters"]["exclude_files"] = ["*_backup*"]
        found = library_capture.discover_blend_files(config)
        names = {p.name for p in found}
        F.check("discover_exclude_backup", "chair_backup.blend" not in names)
        F.check("discover_exclude_keeps_others", len(found) == 3)


def test_discover_explicit_files(F: _Failures) -> None:
    """Test explicit blend_files list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        make_blend_file(root / "a.blend")
        make_blend_file(root / "b.blend")

        config = {"blend_files": [str(root / "a.blend"), str(root / "b.blend"),
                                   str(root / "nonexistent.blend")]}
        found = library_capture.discover_blend_files(config)
        F.check("explicit_files_count", len(found) == 2,
                f"expected 2, got {len(found)}")


def test_sentinel_json_extraction(F: _Failures) -> None:
    """Test sentinel-based JSON extraction from noisy output."""
    SENTINEL_BEGIN = library_capture.SENTINEL_BEGIN
    SENTINEL_END = library_capture.SENTINEL_END

    # Normal case
    output = (f"Blender startup noise...\n"
              f"{SENTINEL_BEGIN}\n"
              f'{{"version": 1, "collections": []}}\n'
              f"{SENTINEL_END}\n"
              f"More noise\n")
    result = library_capture.extract_sentinel_json(output)
    F.check("sentinel_normal", result is not None)
    F.check("sentinel_version", result.get("version") == 1)

    # No sentinels
    result = library_capture.extract_sentinel_json("just noise")
    F.check("sentinel_missing", result is None)

    # Malformed JSON
    output = f"{SENTINEL_BEGIN}\nnot json\n{SENTINEL_END}\n"
    result = library_capture.extract_sentinel_json(output)
    F.check("sentinel_bad_json", result is None)


def test_is_collection_captured(F: _Failures) -> None:
    """Test skip-if-already-captured logic."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Not captured (no summary)
        F.check("not_captured_no_file",
                not library_capture.is_collection_captured(root, "model", "Furniture"))

        # Captured successfully
        summary_path = root / "model" / "Furniture" / "headless_capture_summary.json"
        make_summary(summary_path, success=True, images=10)
        F.check("captured_success",
                library_capture.is_collection_captured(root, "model", "Furniture"))

        # Captured but failed
        make_summary(summary_path, success=False, images=0)
        F.check("captured_failed",
                not library_capture.is_collection_captured(root, "model", "Furniture"))

        # Captured success but 0 images
        make_summary(summary_path, success=True, images=0)
        F.check("captured_no_images",
                not library_capture.is_collection_captured(root, "model", "Furniture"))


def test_filter_collections(F: _Failures) -> None:
    """Test collection filtering logic."""
    collections = [
        library_capture.CollectionInfo(name="Furniture", mesh_count=5, total_vertices=1000),
        library_capture.CollectionInfo(name="Lights", mesh_count=0, total_vertices=0),
        library_capture.CollectionInfo(name="Camera", mesh_count=0, total_vertices=0),
        library_capture.CollectionInfo(name="Architecture", mesh_count=3, total_vertices=500),
        library_capture.CollectionInfo(name="Helpers", mesh_count=1, total_vertices=10),
    ]

    # Default: min_mesh_count=1, no include/exclude
    config = {"filters": {"min_mesh_count": 1, "include_collections": [],
                           "exclude_collections": []}}
    result = library_capture.filter_collections(collections, config)
    names = [c.name for c in result]
    F.check("filter_min_mesh", len(result) == 3,
            f"expected 3 (Furniture, Architecture, Helpers), got {names}")

    # Exclude Camera* and Light*
    config["filters"]["exclude_collections"] = ["Camera*", "Light*"]
    result = library_capture.filter_collections(collections, config)
    names = [c.name for c in result]
    F.check("filter_exclude", len(result) == 3,
            f"expected 3, got {names}")

    # Exclude + include only Furniture
    config["filters"]["include_collections"] = ["Furniture"]
    result = library_capture.filter_collections(collections, config)
    names = [c.name for c in result]
    F.check("filter_include_only", names == ["Furniture"],
            f"expected ['Furniture'], got {names}")


def test_build_job_config(F: _Failures) -> None:
    """Test pipeline job config generation."""
    blend_path = Path("/assets/chair.blend")

    config = {
        "blender_path": "blender",
        "output_base": "/output",
        "capture": {"cameras": 100, "preset": "3dgs"},
        "training": {"enabled": True, "backend": "original"},
        "cleanup": {"enabled": False},
    }
    output_base = Path("/output/chair_scene")
    item = {"type": "collection", "name": "Chair", "cameras": 160}

    job = library_capture.build_job_config(blend_path, config, output_base, item=item)

    F.check("job_scene", job["scene"] == str(blend_path))
    F.check("job_output", job["output_base"] == str(output_base))
    F.check("job_capture_cameras", job["capture"]["cameras"] == 160)
    F.check("job_no_batch_mode", "batch_mode" not in job["capture"])
    F.check("job_items", job["items"][0]["name"] == "Chair")
    F.check("job_training", job["training"]["enabled"] is True)


def test_output_key_uniqueness(F: _Failures) -> None:
    """Ensure same-stem files in different paths do not collide."""
    key_a = library_capture._blend_output_key(Path("/assets/a/chair.blend"), library_path="/assets")
    key_b = library_capture._blend_output_key(Path("/assets/b/chair.blend"), library_path="/assets")
    F.check("blend_key_unique", key_a != key_b, f"keys collided: {key_a}")


def test_generate_jobs_collection_targets(F: _Failures) -> None:
    """Collection granularity should generate explicit per-collection jobs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        blend_path = Path("/library/scene.blend")
        info = library_capture.BlendFileInfo(
            path=str(blend_path),
            collections=[
                library_capture.CollectionInfo(name="Keep", mesh_count=2, object_names=["ObjA"]),
                library_capture.CollectionInfo(name="Lights", mesh_count=4, object_names=["ObjB"]),
                library_capture.CollectionInfo(name="Empty", mesh_count=0, object_names=[]),
            ],
        )

        config = {
            "output_base": tmpdir,
            "library_path": "/library",
            "granularity": "COLLECTIONS",
            "skip_existing": False,
            "filters": {
                "include_collections": [],
                "exclude_collections": ["Light*"],
                "min_mesh_count": 1,
            },
            "capture": {"cameras": 100},
            "training": {},
            "cleanup": {},
        }

        jobs = library_capture.generate_jobs([blend_path], {blend_path: info}, config)
        F.check("collection_jobs_count", len(jobs) == 1, f"expected 1, got {len(jobs)}")
        if jobs:
            job = jobs[0]
            F.check("collection_job_item_type", job["items"][0]["type"] == "collection")
            F.check("collection_job_item_name", job["items"][0]["name"] == "Keep")
            F.check("collection_job_output_segment", "/collections/" in job["output_base"])


def test_generate_jobs_skip_existing(F: _Failures) -> None:
    """skip_existing should apply to exact target outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        blend_path = Path("/library/asset.blend")
        info = library_capture.BlendFileInfo(
            path=str(blend_path),
            collections=[
                library_capture.CollectionInfo(name="Furniture", mesh_count=3, object_names=["Chair"]),
            ],
        )

        config = {
            "output_base": tmpdir,
            "library_path": "/library",
            "granularity": "COLLECTIONS",
            "skip_existing": True,
            "filters": {"include_collections": [], "exclude_collections": [], "min_mesh_count": 1},
            "capture": {"cameras": 50},
            "training": {},
            "cleanup": {},
        }

        jobs = library_capture.generate_jobs([blend_path], {blend_path: info}, config)
        F.check("skip_existing_initial_jobs", len(jobs) == 1, f"expected 1, got {len(jobs)}")
        if jobs:
            summary_path = Path(jobs[0]["output_base"]) / "headless_capture_summary.json"
            make_summary(summary_path, success=True, images=12)
            jobs_after = library_capture.generate_jobs([blend_path], {blend_path: info}, config)
            F.check("skip_existing_after_summary", len(jobs_after) == 0, f"expected 0, got {len(jobs_after)}")


def test_generate_jobs_each_object_targets(F: _Failures) -> None:
    """Each-object granularity should emit one job per unique object name."""
    blend_path = Path("/library/object_scene.blend")
    info = library_capture.BlendFileInfo(
        path=str(blend_path),
        collections=[
            library_capture.CollectionInfo(name="Main", mesh_count=2, object_names=["A", "B", "A"]),
        ],
    )
    config = {
        "output_base": "/tmp/out",
        "library_path": "/library",
        "granularity": "EACH_OBJECT",
        "skip_existing": False,
        "filters": {"include_collections": [], "exclude_collections": [], "min_mesh_count": 1},
        "capture": {"cameras": 30},
        "training": {},
        "cleanup": {},
    }

    jobs = library_capture.generate_jobs([blend_path], {blend_path: info}, config)
    names = sorted(job["items"][0]["name"] for job in jobs)
    F.check("each_object_job_count", len(jobs) == 2, f"expected 2, got {len(jobs)}")
    F.check("each_object_names", names == ["A", "B"], f"unexpected names: {names}")
    F.check(
        "each_object_output_segment",
        all("/objects/" in job["output_base"] for job in jobs),
        "expected all outputs under /objects/",
    )


def test_parse_asset_catalog(F: _Failures) -> None:
    """Test blender_assets.cats.txt parsing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # No catalog file
        result = library_capture.parse_asset_catalog(root)
        F.check("catalog_missing", result == {})

        # Valid catalog
        cat_file = root / "blender_assets.cats.txt"
        cat_file.write_text(
            "VERSION 1\n"
            "# comment line\n"
            "\n"
            "abc-123:props/furniture:Furniture\n"
            "def-456:props/lighting:Lighting\n",
            encoding="utf-8",
        )
        result = library_capture.parse_asset_catalog(root)
        F.check("catalog_count", len(result) == 2,
                f"expected 2, got {len(result)}")
        F.check("catalog_value", result.get("abc-123") == "props/furniture")


def test_library_config_schema(F: _Failures) -> None:
    """Test loading a library config from YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.json"
        config_path.write_text(
            json.dumps({
                "library_path": "/assets",
                "output_base": "/output",
                "granularity": "SCENE",
                "skip_existing": False,
                "filters": {
                    "exclude_files": ["*_old*"],
                    "min_mesh_count": 2,
                },
                "capture": {
                    "cameras": 200,
                },
            }),
            encoding="utf-8",
        )

        try:
            cfg = library_schema.load_library_config(str(config_path))
            F.check("schema_library_path", cfg.library_path == "/assets")
            F.check("schema_output_base", cfg.output_base == "/output")
            F.check("schema_granularity", cfg.granularity == "SCENE")
            F.check("schema_skip_existing", cfg.skip_existing is False)
            F.check("schema_filters_exclude",
                    cfg.filters.exclude_files == ["*_old*"])
            F.check("schema_filters_min_mesh", cfg.filters.min_mesh_count == 2)
            F.check("schema_capture_cameras", cfg.capture.cameras == 200)
        except Exception as e:
            F.check("schema_load", False, str(e))


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    F = _Failures()

    print("test_discover_blend_files")
    test_discover_blend_files(F)

    print("test_discover_explicit_files")
    test_discover_explicit_files(F)

    print("test_sentinel_json_extraction")
    test_sentinel_json_extraction(F)

    print("test_is_collection_captured")
    test_is_collection_captured(F)

    print("test_filter_collections")
    test_filter_collections(F)

    print("test_build_job_config")
    test_build_job_config(F)

    print("test_output_key_uniqueness")
    test_output_key_uniqueness(F)

    print("test_generate_jobs_collection_targets")
    test_generate_jobs_collection_targets(F)

    print("test_generate_jobs_skip_existing")
    test_generate_jobs_skip_existing(F)

    print("test_generate_jobs_each_object_targets")
    test_generate_jobs_each_object_targets(F)

    print("test_parse_asset_catalog")
    test_parse_asset_catalog(F)

    print("test_library_config_schema")
    test_library_config_schema(F)

    return F.report()


if __name__ == "__main__":
    raise SystemExit(main())
