#!/usr/bin/env python3
"""
Headless Capture Script for GS Capture Addon
=============================================
Drives multi-view captures from the command line without a GUI.

Invocation:
    blender scene.blend --python tools/headless_capture.py -- [args]

Examples:
    # Single capture with CLI args
    blender sponza.blend --python tools/headless_capture.py -- \
        --output /tmp/capture --cameras 100 --preset 3dgs

    # Batch: all collections
    blender sponza.blend --python tools/headless_capture.py -- \
        --output /tmp/capture --batch-mode COLLECTIONS --cameras 150

    # Config file mode
    blender sponza.blend --python tools/headless_capture.py -- \
        --config jobs/my_job.yaml

The script uses bpy.app.timers to drive the modal capture operator,
following the proven pattern from tests/smoke/smoke_release_verification.py.
"""

import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Blender bootstrap — must be importable inside Blender's embedded Python
# ---------------------------------------------------------------------------
try:
    import bpy
except ImportError:
    print("ERROR: This script must be run inside Blender.", file=sys.stderr)
    print("Usage: blender scene.blend --python tools/headless_capture.py -- [args]")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_SECONDS = 3600  # 1 hour max per capture
TICK_INTERVAL = 0.3     # seconds between polls


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[headless-capture] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Argument parsing  (reads sys.argv after the "--" separator)
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    # Blender passes everything after "--" to the script.
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    p = argparse.ArgumentParser(
        description="Headless GS Capture — config-driven multi-view rendering",
    )

    # Config file mode (takes precedence over individual args)
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML job config file (overrides CLI args)")

    # Output
    p.add_argument("--output", type=str, default=None,
                   help="Output directory for captured data")

    # Camera settings
    p.add_argument("--cameras", type=int, default=None,
                   help="Number of cameras to generate")
    p.add_argument("--distribution", type=str, default=None,
                   choices=["FIBONACCI", "HEMISPHERE_TOP", "HEMISPHERE_BOTTOM",
                            "RING", "MULTI_RING"],
                   help="Camera distribution pattern")
    p.add_argument("--min-elevation", type=float, default=None)
    p.add_argument("--max-elevation", type=float, default=None)
    p.add_argument("--focal-length", type=float, default=None)
    p.add_argument("--distance-multiplier", type=float, default=None)

    # Preset
    p.add_argument("--preset", type=str, default=None,
                   help="Framework preset name (3dgs, nerfstudio, gsplat, "
                        "instant_ngp, postshot, polycam, luma_ai)")

    # Render settings
    p.add_argument("--speed-preset", type=str, default=None,
                   choices=["FAST", "BALANCED", "QUALITY", "CUSTOM"],
                   help="Render speed/quality preset")
    p.add_argument("--resolution", type=str, default=None,
                   help="Resolution as WIDTHxHEIGHT, e.g. 1920x1080")
    p.add_argument("--engine", type=str, default=None,
                   choices=["EEVEE", "CYCLES"],
                   help="Render engine override")

    # Export flags
    p.add_argument("--export-colmap", action="store_true", default=None)
    p.add_argument("--no-export-colmap", dest="export_colmap", action="store_false")
    p.add_argument("--export-colmap-binary", action="store_true", default=None)
    p.add_argument("--export-transforms-json", action="store_true", default=None)
    p.add_argument("--no-export-transforms-json", dest="export_transforms_json",
                   action="store_false")
    p.add_argument("--export-depth", action="store_true", default=None)
    p.add_argument("--export-normals", action="store_true", default=None)
    p.add_argument("--export-masks", action="store_true", default=None)
    p.add_argument("--mask-source", type=str, default=None,
                   choices=["ALPHA", "OBJECT_INDEX"])

    # Background / lighting / material
    p.add_argument("--transparent-background", action="store_true", default=None)
    p.add_argument("--no-transparent-background", dest="transparent_background",
                   action="store_false")
    p.add_argument("--lighting-mode", type=str, default=None,
                   choices=["WHITE", "GRAY", "HDR", "KEEP"])
    p.add_argument("--material-mode", type=str, default=None,
                   choices=["ORIGINAL", "DIFFUSE", "VERTEX_COLOR", "MATCAP"])

    # Checkpoint / resume
    p.add_argument("--enable-checkpoints", action="store_true", default=None)
    p.add_argument("--no-checkpoints", dest="enable_checkpoints",
                   action="store_false")
    p.add_argument("--auto-resume", action="store_true", default=None)

    # Batch mode
    p.add_argument("--batch-mode", type=str, default=None,
                   choices=["SCENE", "SELECTED", "EACH_SELECTED",
                            "COLLECTION", "COLLECTIONS", "GROUPS"],
                   help="Batch capture mode")
    p.add_argument("--collections", nargs="*", default=None,
                   help="Specific collection name(s) to capture")

    # Timeout
    p.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS,
                   help="Max seconds before aborting (default: 3600)")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Addon registration
# ---------------------------------------------------------------------------
def ensure_addon_registered() -> None:
    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    import gs_capture_addon

    try:
        gs_capture_addon.unregister()
    except Exception:
        pass

    gs_capture_addon.register()
    log("Addon registered")


# ---------------------------------------------------------------------------
# Preset name normalization
# ---------------------------------------------------------------------------
PRESET_ALIASES = {
    "3dgs": "GAUSSIAN_SPLATTING",
    "gaussian_splatting": "GAUSSIAN_SPLATTING",
    "nerfstudio": "NERFSTUDIO",
    "instant_ngp": "INSTANT_NGP",
    "postshot": "POSTSHOT",
    "polycam": "POLYCAM",
    "luma_ai": "LUMA_AI",
    "gsplat": "GSPLAT",
}


def resolve_preset(name: str) -> str:
    key = name.lower().strip()
    if key in PRESET_ALIASES:
        return PRESET_ALIASES[key]
    upper = key.upper()
    if upper in PRESET_ALIASES.values():
        return upper
    raise ValueError(f"Unknown preset: {name!r}. "
                     f"Valid: {', '.join(sorted(PRESET_ALIASES.keys()))}")


# ---------------------------------------------------------------------------
# YAML config loading (optional dependency)
# ---------------------------------------------------------------------------
def load_yaml_config(config_path: str) -> dict:
    """Load a YAML job config, return the capture section as a flat dict."""
    if not Path(config_path).exists():
        log(f"ERROR: Config file not found: {config_path}")
        return {}
    try:
        import yaml
    except ImportError:
        log("WARNING: PyYAML not available — trying json fallback")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            log(f"ERROR: Failed to parse config '{config_path}': {exc}")
            return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, Exception) as exc:
        log(f"ERROR: Failed to parse config '{config_path}': {exc}")
        return {}

    return data or {}


def merge_config_into_args(args: argparse.Namespace, config: dict) -> None:
    """Overlay YAML config values onto argparse namespace (CLI args win)."""
    capture = config.get("capture", {})
    exports = capture.get("exports", {})

    mapping = {
        "output": config.get("output_base"),
        "cameras": capture.get("cameras"),
        "distribution": capture.get("distribution"),
        "preset": capture.get("preset"),
        "speed_preset": capture.get("speed_preset"),
        "engine": capture.get("engine"),
        "resolution": (f"{capture['resolution'][0]}x{capture['resolution'][1]}"
                       if isinstance(capture.get("resolution"), (list, tuple))
                       else capture.get("resolution")),
        "min_elevation": capture.get("min_elevation"),
        "max_elevation": capture.get("max_elevation"),
        "focal_length": capture.get("focal_length"),
        "distance_multiplier": capture.get("distance_multiplier"),
        "transparent_background": capture.get("transparent_background"),
        "lighting_mode": capture.get("lighting_mode"),
        "material_mode": capture.get("material_mode"),
        "enable_checkpoints": capture.get("checkpoints"),
        "auto_resume": capture.get("auto_resume"),
        "export_colmap": exports.get("colmap"),
        "export_colmap_binary": exports.get("colmap_binary"),
        "export_transforms_json": exports.get("transforms_json"),
        "export_depth": exports.get("depth"),
        "export_normals": exports.get("normals"),
        "export_masks": exports.get("masks"),
        "mask_source": exports.get("mask_source"),
        "batch_mode": capture.get("batch_mode"),
    }

    for attr, value in mapping.items():
        if value is not None and getattr(args, attr, None) is None:
            setattr(args, attr, value)

    # Items list from config (collections/objects to capture)
    if not args.collections and config.get("items"):
        collection_names = [
            item["name"] for item in config["items"]
            if item.get("type") == "collection"
        ]
        if collection_names:
            args.collections = collection_names
            if args.batch_mode is None:
                args.batch_mode = "COLLECTIONS"


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------
def apply_render_engine(scene, engine_name: str) -> None:
    """Set the render engine, handling Blender version differences."""
    engine_prop = scene.render.bl_rna.properties.get("engine")
    ids = {item.identifier for item in engine_prop.enum_items}

    if engine_name == "EEVEE":
        if "BLENDER_EEVEE_NEXT" in ids:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        elif "BLENDER_EEVEE" in ids:
            scene.render.engine = "BLENDER_EEVEE"
        else:
            log(f"WARNING: EEVEE not available, keeping {scene.render.engine}")
    elif engine_name == "CYCLES":
        if "CYCLES" in ids:
            scene.render.engine = "CYCLES"
        else:
            log("WARNING: Cycles not available, keeping current engine")


def configure_scene(args: argparse.Namespace) -> None:
    """Apply all CLI/config settings to the scene and GS Capture settings."""
    scene = bpy.context.scene
    settings = scene.gs_capture_settings

    # --- Apply framework preset first (individual args override after) ---
    if args.preset:
        preset_id = resolve_preset(args.preset)
        from gs_capture_addon.core.presets import get_preset, apply_preset_to_settings
        preset = get_preset(preset_id)
        if preset:
            apply_preset_to_settings(preset, settings, scene)
            log(f"Applied preset: {preset.name}")
        else:
            log(f"WARNING: Preset {preset_id!r} not found, skipping")

    # --- Output ---
    if args.output:
        settings.output_path = str(Path(args.output).resolve())

    # --- Cameras ---
    if args.cameras is not None:
        settings.camera_count = args.cameras
    if args.distribution:
        settings.camera_distribution = args.distribution
    if args.min_elevation is not None:
        settings.min_elevation = args.min_elevation
    if args.max_elevation is not None:
        settings.max_elevation = args.max_elevation
    if args.focal_length is not None:
        settings.focal_length = args.focal_length
    if args.distance_multiplier is not None:
        settings.camera_distance_multiplier = args.distance_multiplier

    # --- Render speed preset ---
    if args.speed_preset:
        settings.render_speed_preset = args.speed_preset

    # --- Engine ---
    if args.engine:
        apply_render_engine(scene, args.engine)

    # --- Resolution ---
    if args.resolution:
        try:
            w, h = args.resolution.lower().split("x")
            scene.render.resolution_x = int(w)
            scene.render.resolution_y = int(h)
        except ValueError:
            log(f"WARNING: Invalid resolution {args.resolution!r}, expected WxH")

    # --- Export flags ---
    if args.export_colmap is not None:
        settings.export_colmap = args.export_colmap
    if args.export_colmap_binary is not None:
        settings.export_colmap_binary = args.export_colmap_binary
    if args.export_transforms_json is not None:
        settings.export_transforms_json = args.export_transforms_json
    if args.export_depth is not None:
        settings.export_depth = args.export_depth
    if args.export_normals is not None:
        settings.export_normals = args.export_normals
    if args.export_masks is not None:
        settings.export_masks = args.export_masks
    if args.mask_source:
        settings.mask_source = args.mask_source

    # --- Background / lighting / material ---
    if args.transparent_background is not None:
        settings.transparent_background = args.transparent_background
    if args.lighting_mode:
        settings.lighting_mode = args.lighting_mode
    if args.material_mode:
        settings.material_mode = args.material_mode

    # --- Checkpoints ---
    if args.enable_checkpoints is not None:
        settings.enable_checkpoints = args.enable_checkpoints
    if args.auto_resume is not None:
        settings.auto_resume = args.auto_resume

    # --- Batch mode ---
    if args.batch_mode:
        settings.batch_mode = args.batch_mode

    # Ensure cancel state is clear
    settings.cancel_requested = False
    settings.batch_cancel_requested = False

    log(f"Output: {settings.output_path}")
    log(f"Cameras: {settings.camera_count} ({settings.camera_distribution})")
    log(f"Engine: {scene.render.engine}")
    log(f"Resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
    log(f"Exports: COLMAP={settings.export_colmap} "
        f"transforms.json={settings.export_transforms_json} "
        f"depth={settings.export_depth} normals={settings.export_normals} "
        f"masks={settings.export_masks}")


# ---------------------------------------------------------------------------
# Object / collection selection
# ---------------------------------------------------------------------------
def select_targets(args: argparse.Namespace) -> bool:
    """Select the right objects/collections for capture. Returns True if ready."""
    scene = bpy.context.scene
    settings = scene.gs_capture_settings

    # If specific collections are requested, select the first one for
    # single-collection mode, or set up batch for multiple.
    if args.collections and len(args.collections) == 1:
        coll_name = args.collections[0]
        coll = bpy.data.collections.get(coll_name)
        if not coll:
            log(f"ERROR: Collection {coll_name!r} not found")
            return False
        settings.batch_mode = "COLLECTION"
        settings.target_collection = coll_name
        # Select meshes in collection so single capture works
        bpy.ops.object.select_all(action="DESELECT")
        selected_any = False
        for obj in coll.all_objects:
            if obj.type == "MESH" and obj.visible_get():
                obj.select_set(True)
                if not bpy.context.view_layer.objects.active:
                    bpy.context.view_layer.objects.active = obj
                selected_any = True
        if not selected_any:
            log(f"ERROR: No visible meshes in collection {coll_name!r}")
            return False
        log(f"Selected collection: {coll_name}")
        return True

    # Batch mode for multiple collections or all-collections
    if args.batch_mode == "COLLECTIONS":
        # batch operator handles this; just make sure there's a scene camera
        if not scene.camera:
            bpy.ops.object.camera_add(location=(5, -5, 4),
                                      rotation=(1.1, 0.0, 0.8))
            scene.camera = bpy.context.active_object
            log("Added default camera (none found)")
        return True

    if args.batch_mode in ("COLLECTION", "SELECTED", "EACH_SELECTED",
                           "SCENE", "GROUPS"):
        return True

    # Default: select all visible meshes (single capture of entire scene)
    bpy.ops.object.select_all(action="DESELECT")
    selected_any = False
    for obj in scene.objects:
        if obj.type == "MESH" and obj.visible_get():
            obj.select_set(True)
            if not bpy.context.view_layer.objects.active:
                bpy.context.view_layer.objects.active = obj
            selected_any = True

    if not selected_any:
        log("ERROR: No visible mesh objects in scene")
        return False

    log(f"Selected {len(bpy.context.selected_objects)} mesh objects")
    return True


# ---------------------------------------------------------------------------
# Capture invocation
# ---------------------------------------------------------------------------
def start_capture(args: argparse.Namespace) -> bool:
    """Invoke the appropriate capture operator. Returns True if started."""
    use_batch = args.batch_mode in ("COLLECTIONS", "EACH_SELECTED", "GROUPS")

    if use_batch:
        result = bpy.ops.gs_capture.batch_capture()
        op_name = "batch_capture"
    else:
        result = bpy.ops.gs_capture.capture_selected()
        op_name = "capture_selected"

    ok = "RUNNING_MODAL" in set(result)
    if ok:
        log(f"Capture started via {op_name}")
    else:
        log(f"ERROR: {op_name} returned {result} (expected RUNNING_MODAL)")
    return ok


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------
def validate_outputs(output_dir: str) -> dict:
    """Check that expected output files exist. Returns a summary dict."""
    out = Path(output_dir)
    result = {
        "output_dir": str(out),
        "exists": out.exists(),
        "images": [],
        "has_colmap": False,
        "has_transforms_json": False,
        "has_depth": False,
        "has_normals": False,
        "has_masks": False,
    }

    if not out.exists():
        return result

    # Images
    images_dir = out / "images"
    if images_dir.exists():
        result["images"] = sorted(
            p.name for p in images_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".exr")
        )

    # COLMAP
    sparse_dir = out / "sparse" / "0"
    if sparse_dir.exists():
        result["has_colmap"] = all(
            (sparse_dir / f).exists()
            for f in ("cameras.txt", "images.txt", "points3D.txt")
        )

    # transforms.json
    result["has_transforms_json"] = (out / "transforms.json").exists()

    # Depth / normals / masks
    for sub, key in [("depth", "has_depth"), ("normals", "has_normals"),
                     ("masks", "has_masks")]:
        d = out / sub
        if d.exists() and any(d.iterdir()):
            result[key] = True

    return result


def collect_batch_outputs(base_dir: str) -> list:
    """For batch captures, validate each sub-folder."""
    base = Path(base_dir)
    results = []
    if not base.exists():
        return results
    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / "images").exists():
            results.append(validate_outputs(str(child)))
    # If no sub-folders found, try the base itself
    if not results:
        results.append(validate_outputs(base_dir))
    return results


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def write_summary(output_dir: str, validations: list, elapsed: float,
                  success: bool) -> Path:
    """Write a JSON summary of the capture run."""
    report_path = Path(output_dir) / "headless_capture_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "success": success,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "captures": validations,
        "total_images": sum(len(v.get("images", [])) for v in validations),
    }
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Summary written: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Timer-based state machine (drives the modal operator)
# ---------------------------------------------------------------------------
STATE = {
    "phase": "init",
    "start_time": 0.0,
    "args": None,
    "timeout": TIMEOUT_SECONDS,
    "tick_count": 0,
}


def tick() -> float | None:
    """Called repeatedly by bpy.app.timers. Returns next interval or None."""
    try:
        elapsed = time.time() - STATE["start_time"]
        args = STATE["args"]

        if elapsed > STATE["timeout"]:
            raise TimeoutError(
                f"Headless capture timed out after {STATE['timeout']}s"
            )

        # ---- INIT: register addon, configure, start capture ----
        if STATE["phase"] == "init":
            ensure_addon_registered()
            configure_scene(args)

            if not select_targets(args):
                raise RuntimeError("Failed to select capture targets")

            if not start_capture(args):
                raise RuntimeError("Capture operator failed to start")

            STATE["phase"] = "capturing"
            return TICK_INTERVAL

        # ---- CAPTURING: wait for completion ----
        if STATE["phase"] == "capturing":
            settings = bpy.context.scene.gs_capture_settings
            STATE["tick_count"] += 1

            # Still running?
            if settings.is_rendering or settings.batch_running:
                # Progress logging every ~10 ticks (~3 seconds)
                current = settings.capture_current
                total = settings.capture_total
                if total > 0 and current > 0 and STATE["tick_count"] % 10 == 0:
                    pct = (current / total) * 100
                    log(f"Progress: {current}/{total} ({pct:.0f}%) "
                        f"elapsed={elapsed:.0f}s")
                return TICK_INTERVAL

            # Capture finished
            STATE["phase"] = "done"
            success = settings.last_capture_success
            output_dir = settings.output_path

            if success:
                log("Capture completed successfully")
            else:
                log("WARNING: Capture finished but last_capture_success=False")

            # Validate outputs
            use_batch = (args.batch_mode in
                         ("COLLECTIONS", "EACH_SELECTED", "GROUPS"))
            if use_batch:
                validations = collect_batch_outputs(output_dir)
            else:
                validations = [validate_outputs(output_dir)]

            total_images = sum(len(v.get("images", [])) for v in validations)
            log(f"Total images captured: {total_images}")

            for v in validations:
                d = v["output_dir"]
                n = len(v.get("images", []))
                log(f"  {d}: {n} images, "
                    f"COLMAP={v['has_colmap']}, "
                    f"transforms.json={v['has_transforms_json']}")

            write_summary(output_dir, validations, elapsed, success)

            if success and total_images > 0:
                log("Headless capture finished — exiting Blender")
                bpy.ops.wm.quit_blender()
                return None
            else:
                raise RuntimeError(
                    f"Capture produced {total_images} images "
                    f"(success={success})"
                )

        if STATE["phase"] == "done":
            return None

        raise RuntimeError(f"Unknown phase: {STATE['phase']}")

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        log(f"FATAL: {err}")
        log(tb)

        # Try to write error summary
        try:
            args = STATE.get("args")
            output_dir = getattr(args, "output", None) or str(Path(tempfile.gettempdir()) / "headless_capture_error")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            error_report = Path(output_dir) / "headless_capture_summary.json"
            error_report.write_text(json.dumps({
                "success": False,
                "error": err,
                "traceback": tb,
                "elapsed_seconds": round(time.time() - STATE["start_time"], 2),
            }, indent=2), encoding="utf-8")
        except Exception:
            pass

        bpy.ops.wm.quit_blender()
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Load YAML config if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = REPO_ROOT / config_path
        if not config_path.exists():
            log(f"ERROR: Config file not found: {config_path}")
            bpy.ops.wm.quit_blender()
            return
        config = load_yaml_config(str(config_path))
        merge_config_into_args(args, config)
        log(f"Loaded config: {config_path}")

    # Validate required args
    if not args.output:
        log("ERROR: --output is required (or set output_base in config)")
        bpy.ops.wm.quit_blender()
        return

    STATE["args"] = args
    STATE["start_time"] = time.time()
    STATE["timeout"] = args.timeout

    log("Starting headless capture pipeline")
    log(f"Blender {bpy.app.version_string}")

    # Start after Blender's startup context is fully ready.
    bpy.app.timers.register(tick, first_interval=0.5)


if __name__ == "__main__":
    main()
