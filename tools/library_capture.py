#!/usr/bin/env python3
"""
GS Blender — Asset Library Batch Capture
==========================================
Scans a folder (or explicit list) of .blend files, introspects each to
discover collections/objects, and runs the full capture → training → cleanup
pipeline for every item.

Usage:
    python tools/library_capture.py --config jobs/example_library.yaml
    python tools/library_capture.py --library-path /assets --output /data
    python tools/library_capture.py --config lib.yaml --dry-run
    python tools/library_capture.py --config lib.yaml --introspect-only
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Import pipeline for delegation
sys.path.insert(0, str(SCRIPT_DIR))
from pipeline import find_blender, run_pipeline


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str, level: str = "INFO") -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def log_section(title: str) -> None:
    log(f"{'=' * 60}")
    log(f"  {title}")
    log(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Introspection data structures
# ---------------------------------------------------------------------------
@dataclass
class CollectionInfo:
    name: str
    mesh_count: int = 0
    total_vertices: int = 0
    object_names: List[str] = field(default_factory=list)
    adaptive: Optional[dict] = None


@dataclass
class SceneInfo:
    name: str
    collection_names: List[str] = field(default_factory=list)
    total_mesh_objects: int = 0


@dataclass
class BlendFileInfo:
    path: str = ""
    blender_version: str = ""
    file_size_bytes: int = 0
    scenes: List[SceneInfo] = field(default_factory=list)
    collections: List[CollectionInfo] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class JobResult:
    blend_path: str = ""
    success: bool = False
    elapsed_seconds: float = 0.0
    collections_captured: List[str] = field(default_factory=list)
    collections_skipped: List[str] = field(default_factory=list)
    collections_failed: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    """Load a YAML or JSON config file as a raw dict."""
    config_path = Path(path)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: Cannot read config file '{path}': {exc}", file=sys.stderr)
        raise SystemExit(1)

    yaml_module = None
    try:
        import yaml as yaml_module
    except ImportError:
        yaml_module = None

    if yaml_module is not None:
        try:
            return yaml_module.safe_load(text) or {}
        except Exception as exc:
            print(f"ERROR: Failed to parse YAML config '{path}': {exc}", file=sys.stderr)
            raise SystemExit(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Failed to parse config '{path}': {exc}", file=sys.stderr)
        raise SystemExit(1)


def config_get(config: dict, *keys, default=None):
    """Nested dict access with default."""
    d = config
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


# ---------------------------------------------------------------------------
# Sentinel JSON extraction
# ---------------------------------------------------------------------------
SENTINEL_BEGIN = "---INTROSPECT_JSON_BEGIN---"
SENTINEL_END = "---INTROSPECT_JSON_END---"


def extract_sentinel_json(output: str) -> Optional[dict]:
    """Extract JSON payload from between sentinel markers in stdout."""
    begin = output.find(SENTINEL_BEGIN)
    end = output.find(SENTINEL_END)
    if begin == -1 or end == -1 or end <= begin:
        return None
    json_str = output[begin + len(SENTINEL_BEGIN):end].strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Phase 1: Discovery
# ---------------------------------------------------------------------------
def parse_asset_catalog(library_path: Path) -> Dict[str, str]:
    """Parse blender_assets.cats.txt if present. Returns {uuid: path}."""
    cat_file = library_path / "blender_assets.cats.txt"
    if not cat_file.exists():
        return {}
    catalogs = {}
    for line in cat_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("VERSION"):
            continue
        parts = line.split(":", 2)
        if len(parts) >= 2:
            catalogs[parts[0]] = parts[1]
    return catalogs


def matches_any_glob(name: str, patterns: List[str]) -> bool:
    """Check if name matches any of the glob patterns."""
    return any(fnmatch(name, pat) for pat in patterns)


def discover_blend_files(config: dict) -> List[Path]:
    """Find .blend files from library_path or explicit blend_files list."""
    # Explicit file list
    explicit = config.get("blend_files", [])
    if explicit:
        files = []
        for p in explicit:
            path = Path(p)
            if path.exists() and path.suffix.lower() == ".blend":
                files.append(path.resolve())
            else:
                log(f"Skipping non-existent or non-.blend: {p}", "WARN")
        return sorted(files)

    # Folder scan
    library_path = Path(config.get("library_path", ""))
    if not library_path.exists():
        log(f"Library path does not exist: {library_path}", "ERROR")
        return []

    recursive = config.get("recursive", True)
    filters = config.get("filters", {})
    include_patterns = filters.get("include_files", ["*.blend"])
    exclude_patterns = filters.get("exclude_files", [])

    found = []
    if recursive:
        candidates = library_path.rglob("*.blend")
    else:
        candidates = library_path.glob("*.blend")

    for path in candidates:
        name = path.name
        if not matches_any_glob(name, include_patterns):
            continue
        if exclude_patterns and matches_any_glob(name, exclude_patterns):
            continue
        found.append(path.resolve())

    return sorted(found)


# ---------------------------------------------------------------------------
# Phase 2: Introspection
# ---------------------------------------------------------------------------
def introspect_blend_file(blender_exe: str, blend_path: Path,
                          adaptive: bool = False,
                          quality_preset: str = "AUTO") -> BlendFileInfo:
    """Launch Blender to introspect a single .blend file."""
    introspect_script = SCRIPT_DIR / "introspect_blend.py"
    if not introspect_script.exists():
        return BlendFileInfo(
            path=str(blend_path),
            error=f"introspect_blend.py not found at {introspect_script}",
        )

    cmd = [blender_exe, str(blend_path), "--background",
           "--python", str(introspect_script), "--"]
    if adaptive:
        cmd.append("--adaptive")
        cmd.extend(["--quality-preset", quality_preset])

    # Use xvfb-run on Linux if available
    if platform.system() == "Linux" and shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a"] + cmd

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace",
        )
        payload = extract_sentinel_json(result.stdout)
        if payload is None:
            return BlendFileInfo(
                path=str(blend_path),
                error=f"No valid JSON in introspection output (exit={result.returncode})",
            )

        # Parse into dataclass
        info = BlendFileInfo(
            path=payload.get("blend_file", str(blend_path)),
            blender_version=payload.get("blender_version", ""),
            file_size_bytes=payload.get("file_size_bytes", 0),
        )
        for s in payload.get("scenes", []):
            info.scenes.append(SceneInfo(
                name=s["name"],
                collection_names=s.get("collection_names", []),
                total_mesh_objects=s.get("total_mesh_objects", 0),
            ))
        for c in payload.get("collections", []):
            info.collections.append(CollectionInfo(
                name=c["name"],
                mesh_count=c.get("mesh_count", 0),
                total_vertices=c.get("total_vertices", 0),
                object_names=c.get("object_names", []),
                adaptive=c.get("adaptive"),
            ))
        return info

    except subprocess.TimeoutExpired:
        return BlendFileInfo(
            path=str(blend_path),
            error="Introspection timed out (120s)",
        )
    except Exception as e:
        return BlendFileInfo(
            path=str(blend_path),
            error=str(e),
        )


def introspect_all(blender_exe: str, blend_files: List[Path],
                   config: dict) -> Dict[Path, BlendFileInfo]:
    """Introspect all discovered .blend files."""
    adaptive = config.get("adaptive", False)
    quality = config.get("adaptive_quality_preset", "AUTO")
    results = {}

    for i, path in enumerate(blend_files, 1):
        log(f"Introspecting [{i}/{len(blend_files)}]: {path.name}")
        info = introspect_blend_file(blender_exe, path, adaptive, quality)
        if info.error:
            log(f"  ERROR: {info.error}", "WARN")
        else:
            mesh_colls = [c for c in info.collections if c.mesh_count > 0]
            log(f"  {len(info.collections)} collections, "
                f"{len(mesh_colls)} with meshes")
        results[path] = info

    return results


# ---------------------------------------------------------------------------
# Phase 3: Job generation
# ---------------------------------------------------------------------------
def _sanitize_path_component(value: str, fallback: str = "item") -> str:
    cleaned = []
    for char in str(value):
        if ord(char) < 128 and (char.isalnum() or char in {"-", "_", "."}):
            cleaned.append(char)
        else:
            cleaned.append("_")
    normalized = "".join(cleaned).strip(" ._")
    return normalized or fallback


def _stable_digest(value: str, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _blend_output_key(blend_path: Path, library_path: str = "") -> str:
    resolved = blend_path.resolve()
    source = resolved.as_posix()

    relative_no_suffix = None
    if library_path:
        try:
            relative_no_suffix = resolved.relative_to(Path(library_path).resolve()).with_suffix("")
        except Exception:
            relative_no_suffix = None

    if relative_no_suffix is None:
        label = _sanitize_path_component(blend_path.stem, fallback="blend")
    else:
        parts = [_sanitize_path_component(part, fallback="part") for part in relative_no_suffix.parts]
        label = "__".join(parts) if parts else _sanitize_path_component(blend_path.stem, fallback="blend")

    return f"{label}__{_stable_digest(source)}"


def _target_output_key(name: str) -> str:
    normalized = _sanitize_path_component(name, fallback="target")
    return f"{normalized}__{_stable_digest(str(name), length=8)}"


def _is_capture_completed(output_dir: Path) -> bool:
    summary_path = output_dir / "headless_capture_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(summary.get("success")) and int(summary.get("total_images", 0)) > 0


def is_collection_captured(output_base: Path, blend_stem: str, collection_name: str) -> bool:
    """Backward-compatible helper used by tests."""
    return _is_capture_completed(output_base / blend_stem / collection_name)


def is_scene_captured(output_base: Path, blend_stem: str) -> bool:
    """Backward-compatible helper used by tests."""
    return _is_capture_completed(output_base / blend_stem)


def filter_collections(collections: List[CollectionInfo],
                       config: dict) -> List[CollectionInfo]:
    """Apply collection filters."""
    filters = config.get("filters", {})
    include = filters.get("include_collections", [])
    exclude = filters.get("exclude_collections", [])
    min_meshes = filters.get("min_mesh_count", 1)

    result = []
    for coll in collections:
        if coll.mesh_count < min_meshes:
            continue
        if include and not matches_any_glob(coll.name, include):
            continue
        if exclude and matches_any_glob(coll.name, exclude):
            continue
        result.append(coll)
    return result


def build_job_config(
    blend_path: Path,
    config: dict,
    output_base: Path,
    item: Optional[dict] = None,
    cameras_override: Optional[int] = None,
) -> dict:
    """Build a pipeline.py-compatible job config dict for one explicit target."""
    capture = dict(config.get("capture", {}))
    capture.pop("batch_mode", None)

    if cameras_override is None and item is not None:
        item_cameras = item.get("cameras")
        if isinstance(item_cameras, int) and item_cameras > 0:
            cameras_override = item_cameras
    if cameras_override is not None:
        capture["cameras"] = int(cameras_override)

    job_label = blend_path.name
    items = None
    if item:
        items = [item]
        job_label = f"{blend_path.name}:{item.get('type', 'item')}:{item.get('name', '')}"

    job_id = _stable_digest(f"{blend_path.resolve()}|{output_base}|{item or ''}", length=16)
    job = {
        "job_id": job_id,
        "job_label": job_label,
        "blender_path": config.get("blender_path", "blender"),
        "scene": str(blend_path),
        "output_base": str(output_base),
        "capture": capture,
        "training": config.get("training", {}),
        "cleanup": config.get("cleanup", {}),
        "timeout_per_capture": config.get("timeout_per_file", 7200),
    }
    if items:
        job["items"] = items
    return job


def generate_jobs(blend_files: List[Path],
                  introspection: Dict[Path, BlendFileInfo],
                  config: dict) -> List[dict]:
    """Generate pipeline job configs for all .blend files."""
    output_base = Path(config.get("output_base", ""))
    skip_existing = config.get("skip_existing", True)
    granularity = config.get("granularity", "COLLECTIONS")
    library_path = config.get("library_path", "")

    jobs = []
    total_skipped = 0
    total_targets = 0

    for blend_path in blend_files:
        info = introspection.get(blend_path)
        if not info or info.error:
            log(f"Skipping {blend_path.name} (introspection failed)", "WARN")
            continue

        blend_key = _blend_output_key(blend_path, library_path=library_path)
        blend_root = output_base / blend_key

        # Filter collections
        filtered = filter_collections(info.collections, config)
        if not filtered and granularity in {"COLLECTIONS", "EACH_OBJECT"}:
            log(f"Skipping {blend_path.name} (no collections pass filters)")
            continue

        if granularity == "SCENE":
            scene_output = blend_root / "scene"
            if skip_existing and _is_capture_completed(scene_output):
                log(f"Skipping {blend_path.name} (scene target already captured)")
                total_skipped += 1
                continue
            jobs.append(build_job_config(blend_path, config, scene_output))
            total_targets += 1
            continue

        if granularity == "COLLECTIONS":
            for coll in filtered:
                coll_output = blend_root / "collections" / _target_output_key(coll.name)
                if skip_existing and _is_capture_completed(coll_output):
                    total_skipped += 1
                    continue
                item = {"type": "collection", "name": coll.name}
                cameras_override = None
                if config.get("adaptive", False):
                    recommended = (coll.adaptive or {}).get("recommended_cameras")
                    if isinstance(recommended, int) and recommended > 0:
                        item["cameras"] = recommended
                        cameras_override = recommended
                jobs.append(
                    build_job_config(
                        blend_path,
                        config,
                        coll_output,
                        item=item,
                        cameras_override=cameras_override,
                    )
                )
                total_targets += 1
            continue

        if granularity == "EACH_OBJECT":
            object_names = []
            for coll in filtered:
                object_names.extend(coll.object_names)
            unique_object_names = sorted(set(name for name in object_names if name))
            if not unique_object_names:
                log(f"Skipping {blend_path.name} (no object targets discovered)", "WARN")
                continue

            for object_name in unique_object_names:
                object_output = blend_root / "objects" / _target_output_key(object_name)
                if skip_existing and _is_capture_completed(object_output):
                    total_skipped += 1
                    continue
                item = {"type": "object", "name": object_name}
                jobs.append(build_job_config(blend_path, config, object_output, item=item))
                total_targets += 1
            continue

        log(f"Skipping {blend_path.name} (unsupported granularity: {granularity})", "WARN")

    log(f"Generated {len(jobs)} jobs covering {total_targets} targets "
        f"({total_skipped} skipped)")
    return jobs


# ---------------------------------------------------------------------------
# Phase 4: Execution
# ---------------------------------------------------------------------------
def write_temp_config(job_config: dict, output_dir: Path) -> Path:
    """Write a temporary YAML config file for pipeline.py."""
    output_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(job_config.get("job_id", "")).strip()
    if not job_id:
        fallback = f"{job_config.get('scene', '')}|{job_config.get('output_base', '')}"
        job_id = _stable_digest(fallback, length=16)
    config_path = output_dir / f"_generated_{job_id}.yaml"

    try:
        import yaml
        text = yaml.dump(job_config, default_flow_style=False, sort_keys=False)
    except ImportError:
        text = json.dumps(job_config, indent=2)

    config_path.write_text(text, encoding="utf-8")
    return config_path


def execute_job(job_config: dict, config: dict,
                gpu_id: int = 0, dry_run: bool = False) -> JobResult:
    """Execute a single pipeline job. Returns JobResult."""
    blend_path = job_config.get("scene", "")
    job_label = job_config.get("job_label", Path(blend_path).stem or "job")
    start = time.monotonic()

    result = JobResult(blend_path=blend_path)

    if dry_run:
        log(f"[DRY RUN] Would process: {job_label}")
        log(f"  Scene: {blend_path}")
        log(f"  Output: {job_config.get('output_base')}")
        capture = job_config.get("capture", {})
        log(f"  Cameras: {capture.get('cameras', 'default')}, "
            f"Mode: {capture.get('batch_mode', 'SCENE')}")
        result.success = True
        result.elapsed_seconds = 0.0
        return result

    # Write temp config
    output_base = Path(config.get("output_base", "."))
    config_path = write_temp_config(job_config, output_base / "_configs")

    # Set GPU for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        ok = run_pipeline(str(config_path), dry_run=False)
        result.success = ok
        if not ok:
            result.error = "pipeline execution returned failure"
    except Exception as e:
        result.success = False
        result.error = str(e)
        log(f"Pipeline error for {job_label}: {e}", "ERROR")

    result.elapsed_seconds = round(time.monotonic() - start, 2)
    return result


def execute_all(jobs: List[dict], config: dict, dry_run: bool = False) -> List[JobResult]:
    """Execute all jobs, optionally in parallel."""
    max_workers = config.get("max_workers", 1)
    gpu_ids = list(config.get("gpu_ids", [0]) or [0])
    results = []

    if max_workers <= 1:
        # Sequential execution
        for i, job in enumerate(jobs, 1):
            job_label = job.get("job_label", Path(job.get("scene", "")).stem or "job")
            log(f"Processing [{i}/{len(jobs)}]: {job_label}")
            gpu_id = gpu_ids[0] if gpu_ids else 0
            result = execute_job(job, config, gpu_id, dry_run)
            results.append(result)
            if result.success:
                log(f"  Done in {result.elapsed_seconds:.0f}s")
            else:
                log(f"  FAILED: {result.error or 'unknown error'}", "ERROR")
    else:
        # Parallel execution with ProcessPoolExecutor
        log(f"Running with {max_workers} parallel workers")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, job in enumerate(jobs):
                gpu_id = gpu_ids[i % len(gpu_ids)]
                future = executor.submit(execute_job, job, config, gpu_id, dry_run)
                futures[future] = job

            for future in as_completed(futures):
                job = futures[future]
                job_label = job.get("job_label", Path(job.get("scene", "")).stem or "job")
                try:
                    result = future.result()
                    results.append(result)
                    status = "OK" if result.success else "FAILED"
                    log(f"  {job_label}: {status} ({result.elapsed_seconds:.0f}s)")
                except Exception as e:
                    results.append(JobResult(
                        blend_path=job.get("scene", ""),
                        success=False,
                        error=str(e),
                    ))
                    log(f"  {job_label}: EXCEPTION: {e}", "ERROR")

    return results


# ---------------------------------------------------------------------------
# Phase 5: Reporting
# ---------------------------------------------------------------------------
def write_library_report(output_base: Path, blend_files: List[Path],
                         introspection: Dict[Path, BlendFileInfo],
                         results: List[JobResult],
                         elapsed: float) -> Path:
    """Write aggregate JSON report."""
    output_base.mkdir(parents=True, exist_ok=True)
    report_path = output_base / "library_report.json"

    succeeded = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    report = {
        "success": failed == 0 and succeeded > 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed, 2),
        "summary": {
            "blend_files_discovered": len(blend_files),
            "jobs_executed": len(results),
            "succeeded": succeeded,
            "failed": failed,
        },
        "files": [],
    }

    for r in results:
        entry = {
            "blend_file": r.blend_path,
            "success": r.success,
            "elapsed_seconds": r.elapsed_seconds,
        }
        if r.error:
            entry["error"] = r.error

        # Add introspection info
        path = Path(r.blend_path)
        info = introspection.get(path)
        if info and not info.error:
            entry["collections"] = [
                {"name": c.name, "mesh_count": c.mesh_count,
                 "total_vertices": c.total_vertices}
                for c in info.collections if c.mesh_count > 0
            ]

        report["files"].append(entry)

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Library report: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def run_library(config: dict, dry_run: bool = False,
                introspect_only: bool = False) -> bool:
    """Main entry point: discover → introspect → generate → execute → report."""
    log_section("GS Blender Library Capture")

    # Find Blender
    blender_exe = find_blender(config.get("blender_path", "blender"))
    if not blender_exe:
        log("Blender executable not found", "ERROR")
        return False
    log(f"Blender: {blender_exe}")

    pipeline_start = time.monotonic()

    # Phase 1: Discover
    log_section("Phase 1: Discovery")
    blend_files = discover_blend_files(config)
    if not blend_files:
        log("No .blend files found", "ERROR")
        return False
    log(f"Found {len(blend_files)} .blend file(s)")
    for f in blend_files:
        log(f"  {f}")

    # Phase 2: Introspect
    log_section("Phase 2: Introspection")
    introspection = introspect_all(blender_exe, blend_files, config)

    if introspect_only:
        log_section("Introspection Results")
        for path, info in introspection.items():
            log(f"\n{path.name}:")
            if info.error:
                log(f"  ERROR: {info.error}")
                continue
            for coll in info.collections:
                if coll.mesh_count > 0:
                    adaptive_str = ""
                    if coll.adaptive:
                        adaptive_str = (f" → recommended: "
                                        f"{coll.adaptive['recommended_cameras']} cameras, "
                                        f"{coll.adaptive['quality_preset']}")
                    log(f"  {coll.name}: {coll.mesh_count} meshes, "
                        f"{coll.total_vertices} verts{adaptive_str}")
        return True

    # Phase 3: Generate jobs
    log_section("Phase 3: Job Generation")
    jobs = generate_jobs(blend_files, introspection, config)
    if not jobs:
        log("No jobs to execute (all filtered or already captured)")
        return True

    # Phase 4: Execute
    log_section("Phase 4: Execution")
    results = execute_all(jobs, config, dry_run)

    # Phase 5: Report
    log_section("Phase 5: Report")
    output_base = Path(config.get("output_base", "."))
    elapsed = time.monotonic() - pipeline_start
    write_library_report(output_base, blend_files, introspection, results, elapsed)

    succeeded = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    log_section("Library Capture Complete")
    log(f"Total time: {elapsed:.0f}s")
    log(f"Results: {succeeded} succeeded, {failed} failed out of {len(results)} jobs")

    return failed == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="GS Blender asset library batch capture",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to library YAML config")
    parser.add_argument("--library-path", type=str, default=None,
                        help="Path to folder of .blend files (alternative to config)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (alternative to config)")
    parser.add_argument("--blender-path", type=str, default=None,
                        help="Path to Blender executable")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without executing")
    parser.add_argument("--introspect-only", action="store_true",
                        help="Only introspect files, don't capture")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Re-capture even if output already exists")
    parser.add_argument("--adaptive", action="store_true",
                        help="Enable adaptive analysis for camera/resolution")
    parser.add_argument("--granularity", type=str, default=None,
                        choices=["COLLECTIONS", "SCENE", "EACH_OBJECT"],
                        help="Capture granularity (default: COLLECTIONS)")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Parallel Blender instances (default: 1)")

    args = parser.parse_args()

    # Build config from file or CLI args
    if args.config:
        config = load_config(args.config)
    else:
        config = {}

    # CLI args override config
    if args.library_path:
        config["library_path"] = args.library_path
    if args.output:
        config["output_base"] = args.output
    if args.blender_path:
        config["blender_path"] = args.blender_path
    if args.no_skip_existing:
        config["skip_existing"] = False
    if args.adaptive:
        config["adaptive"] = True
    if args.granularity:
        config["granularity"] = args.granularity
    if args.max_workers is not None:
        config["max_workers"] = args.max_workers

    # Validate required fields
    if not config.get("library_path") and not config.get("blend_files"):
        log("ERROR: --library-path or config with library_path/blend_files is required")
        sys.exit(1)
    if not config.get("output_base"):
        log("ERROR: --output or config with output_base is required")
        sys.exit(1)

    ok = run_library(config, dry_run=args.dry_run,
                     introspect_only=args.introspect_only)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
