#!/usr/bin/env python3
"""
GS Blender — End-to-End Pipeline Orchestrator
==============================================
Standalone Python script (no Blender dependency) that orchestrates:
  .blend → headless capture → GS training → splat cleanup

Usage:
    python tools/pipeline.py --config jobs/my_job.yaml
    python tools/pipeline.py --config jobs/a.yaml jobs/b.yaml
    python tools/pipeline.py --config jobs/my_job.yaml --dry-run
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


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
# Blender auto-detection
# ---------------------------------------------------------------------------
def find_blender(configured_path: str = "blender") -> Optional[str]:
    """Locate the Blender executable."""
    # Explicit path
    if configured_path and configured_path != "blender":
        p = Path(configured_path)
        if p.exists():
            return str(p)

    # On PATH
    found = shutil.which("blender")
    if found:
        return found

    # Common install locations
    candidates = []
    if platform.system() == "Windows":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        base = Path(pf) / "Blender Foundation"
        if base.exists():
            for d in sorted(base.iterdir(), reverse=True):
                exe = d / "blender.exe"
                if exe.exists():
                    candidates.append(str(exe))
    elif platform.system() == "Darwin":
        app = Path("/Applications/Blender.app/Contents/MacOS/Blender")
        if app.exists():
            candidates.append(str(app))
    else:
        for p in ("/usr/bin/blender", "/snap/bin/blender"):
            if Path(p).exists():
                candidates.append(p)

    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    """Load a YAML or JSON config file."""
    config_path = Path(path)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: Cannot read config file '{path}': {exc}", file=sys.stderr)
        raise SystemExit(1)
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        pass
    except Exception as exc:
        print(f"ERROR: Failed to parse YAML config '{path}': {exc}", file=sys.stderr)
        raise SystemExit(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Failed to parse config '{path}': {exc}", file=sys.stderr)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Phase 1: Capture
# ---------------------------------------------------------------------------
def run_capture(blender_exe: str, config: dict, config_path: str,
                dry_run: bool = False, env: dict = None) -> bool:
    """Launch Blender with headless_capture.py for a single job config."""
    scene_path = config.get("scene", "")
    output_base = config.get("output_base", "")
    timeout = config.get("timeout_per_capture", 3600)

    if not scene_path:
        log("No 'scene' specified in config — skipping capture phase", "WARN")
        return True

    if not Path(scene_path).exists():
        if dry_run:
            log(f"[DRY RUN] Scene file not found (ok in dry-run): {scene_path}")
        else:
            log(f"Scene file not found: {scene_path}", "ERROR")
            return False

    headless_script = SCRIPT_DIR / "headless_capture.py"
    if not headless_script.exists():
        log(f"headless_capture.py not found at {headless_script}", "ERROR")
        return False

    # Build command
    cmd = [blender_exe, str(scene_path), "--background",
           "--python", str(headless_script), "--"]
    cmd.extend(["--config", str(Path(config_path).resolve())])
    cmd.extend(["--timeout", str(timeout)])

    # Override output if specified
    if output_base:
        cmd.extend(["--output", str(output_base)])

    if dry_run:
        log(f"[DRY RUN] Would run: {' '.join(cmd)}")
        return True

    log(f"Launching capture: {scene_path}")
    log(f"Command: {' '.join(cmd)}")

    start = time.monotonic()
    try:
        # Use xvfb-run on Linux CI
        if platform.system() == "Linux" and shutil.which("xvfb-run"):
            cmd = ["xvfb-run", "-a"] + cmd
            log("Using xvfb-run for headless display")

        result = subprocess.run(
            cmd,
            timeout=timeout + 60,  # grace period on top of internal timeout
            capture_output=False,   # let output stream to console
            env=env,
        )
        elapsed = time.monotonic() - start

        if result.returncode == 0:
            log(f"Capture completed in {elapsed:.0f}s")
            return True
        else:
            log(f"Capture failed (exit code {result.returncode}) "
                f"after {elapsed:.0f}s", "ERROR")
            return False

    except subprocess.TimeoutExpired:
        log(f"Capture timed out after {timeout + 60}s", "ERROR")
        return False
    except Exception as e:
        log(f"Capture error: {e}", "ERROR")
        return False


# ---------------------------------------------------------------------------
# Phase 2: Training
# ---------------------------------------------------------------------------
def find_capture_folders(output_base: str) -> List[Path]:
    """Find all sub-folders that contain captured training data."""
    base = Path(output_base)
    folders = []

    if not base.exists():
        return folders

    # Check if base itself is a capture folder
    if (base / "images").exists():
        folders.append(base)
        return folders

    # Check sub-folders
    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / "images").exists():
            folders.append(child)

    return folders


def find_gs_training_script() -> Optional[Path]:
    """Locate the training pipeline script."""
    candidates = [
        SCRIPT_DIR / "gs_training_pipeline.py",
        SCRIPT_DIR / "train_gs_simple.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_training_for_folder(folder: Path, training_config: dict,
                            output_base: str, dry_run: bool = False) -> bool:
    """Run GS training for a single capture folder."""
    backend = training_config.get("backend") or training_config.get("implementation", "original")
    iterations = training_config.get("iterations", 30000)
    save_iters = training_config.get("save_iterations", [7000, 15000, 30000])
    white_bg = training_config.get("white_background", True)
    gpu_id = training_config.get("gpu_id", 0)
    extra_args = training_config.get("extra_args", "")

    model_output = Path(output_base) / "trained" / folder.name

    if dry_run:
        log(f"[DRY RUN] Would train: {folder} → {model_output}")
        return True

    # Try the full pipeline script first
    training_script = find_gs_training_script()

    if training_script and training_script.name == "gs_training_pipeline.py":
        cmd = [
            sys.executable, str(training_script),
            "--input", str(folder),
            "--output", str(model_output),
            "--implementation", backend,
            "--iterations", str(iterations),
        ]
        if white_bg:
            cmd.append("--white-background")
    elif training_script and training_script.name == "train_gs_simple.py":
        cmd = [
            sys.executable, str(training_script),
            str(folder),
            "--output", str(model_output),
            "--iterations", str(iterations),
        ]
    else:
        # Direct launch — look for gaussian-splatting train.py
        gs_path = training_config.get("gaussian_splatting_path", "")
        if not gs_path:
            env_path = os.environ.get("GAUSSIAN_SPLATTING_PATH", "")
            if env_path:
                gs_path = env_path
            else:
                # Search common locations
                for candidate in [
                    Path.home() / "gaussian-splatting",
                    Path("./gaussian-splatting"),
                ]:
                    if (candidate / "train.py").exists():
                        gs_path = str(candidate)
                        break

        if not gs_path or not (Path(gs_path) / "train.py").exists():
            log(f"Cannot find training backend for {folder.name}", "ERROR")
            return False

        cmd = [
            sys.executable, str(Path(gs_path) / "train.py"),
            "-s", str(folder),
            "-m", str(model_output),
            "--iterations", str(iterations),
        ]
        if save_iters:
            cmd.extend(["--save_iterations"] + [str(i) for i in save_iters])
        if white_bg:
            cmd.append("--white_background")

    if extra_args:
        cmd.extend(extra_args.split())

    # Set CUDA device
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log(f"Training: {folder.name} (backend={backend}, iters={iterations})")
    log(f"  Command: {' '.join(cmd)}")

    # Training timeout: 4 hours default (training can be very long)
    training_timeout = training_config.get("timeout", 14400)

    start = time.monotonic()
    try:
        result = subprocess.run(cmd, env=env, capture_output=False,
                                timeout=training_timeout)
        elapsed = time.monotonic() - start

        if result.returncode == 0:
            log(f"Training completed: {folder.name} in {elapsed:.0f}s")
            return True
        else:
            log(f"Training failed: {folder.name} "
                f"(exit={result.returncode}, {elapsed:.0f}s)", "ERROR")
            return False
    except subprocess.TimeoutExpired:
        log(f"Training timed out for {folder.name} after {training_timeout}s",
            "ERROR")
        return False
    except Exception as e:
        log(f"Training error for {folder.name}: {e}", "ERROR")
        return False


def run_training(config: dict, dry_run: bool = False) -> List[Path]:
    """Run training for all captured folders. Returns list of trained model dirs."""
    training_config = config.get("training", {})
    if not training_config.get("enabled", False):
        log("Training disabled — skipping")
        return []

    output_base = config.get("output_base", "")
    capture_folders = find_capture_folders(output_base)

    if not capture_folders:
        log("No capture folders found — skipping training", "WARN")
        return []

    log(f"Found {len(capture_folders)} capture folder(s) to train")
    trained = []

    for folder in capture_folders:
        success = run_training_for_folder(
            folder, training_config, output_base, dry_run
        )
        if success:
            model_dir = Path(output_base) / "trained" / folder.name
            trained.append(model_dir)

    return trained


# ---------------------------------------------------------------------------
# Phase 3: Cleanup
# ---------------------------------------------------------------------------
def find_ply_model(model_dir: Path) -> Optional[Path]:
    """Find the trained .ply file in a model output directory."""
    # Common locations for the final model
    candidates = [
        model_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        model_dir / "point_cloud" / "iteration_15000" / "point_cloud.ply",
        model_dir / "point_cloud" / "iteration_7000" / "point_cloud.ply",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Search for any .ply in the model dir
    plys = sorted(model_dir.rglob("*.ply"), key=lambda p: p.stat().st_mtime,
                  reverse=True)
    return plys[0] if plys else None


def run_cleanup(config: dict, trained_dirs: List[Path],
                dry_run: bool = False) -> List[Path]:
    """Run proxy-hull cleanup on trained models. Returns cleaned PLY paths."""
    cleanup_config = config.get("cleanup", {})
    if not cleanup_config.get("enabled", False):
        log("Cleanup disabled — skipping")
        return []

    output_base = config.get("output_base", "")
    cleaned = []

    for model_dir in trained_dirs:
        ply_path = find_ply_model(model_dir)
        if not ply_path:
            log(f"No .ply found in {model_dir} — skipping cleanup", "WARN")
            continue

        # Find proxy hulls — they're in the capture folder
        capture_name = model_dir.name
        proxy_hull_path = Path(output_base) / capture_name / "proxy_hulls.json"
        if not proxy_hull_path.exists():
            proxy_hull_path = Path(output_base) / capture_name / "proxy_hulls" / "proxy_hulls.json"
        if not proxy_hull_path.exists():
            log(f"No proxy hulls found for {capture_name} — skipping cleanup",
                "WARN")
            continue

        if dry_run:
            log(f"[DRY RUN] Would clean: {ply_path}")
            cleaned.append(ply_path.with_suffix(".cleaned.ply"))
            continue

        log(f"Cleanup: {capture_name}")
        margin = cleanup_config.get("hull_margin", 0.01)
        max_ratio = cleanup_config.get("max_removal_ratio", 0.70)

        try:
            # Import splat_cleanup — it's pure Python, no Blender needed
            cleanup_module_path = (REPO_ROOT / "gs_capture_addon" / "core"
                                   / "splat_cleanup.py")
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))

            from gs_capture_addon.core.splat_cleanup import run_proxy_hull_cleanup

            result = run_proxy_hull_cleanup(
                model_path=str(ply_path),
                proxy_hull_path=str(proxy_hull_path),
                hull_margin=margin,
                max_removal_ratio=max_ratio,
            )
            log(f"Cleanup done: {capture_name} — {result}")
            cleaned_path = ply_path.with_suffix(".cleaned.ply")
            if cleaned_path.exists():
                cleaned.append(cleaned_path)
            else:
                # The cleanup function may use a different naming scheme
                cleaned.append(ply_path)
        except Exception as e:
            log(f"Cleanup failed for {capture_name}: {e}", "ERROR")

    return cleaned


# ---------------------------------------------------------------------------
# Pipeline report
# ---------------------------------------------------------------------------
def write_pipeline_report(config: dict, phases: dict, elapsed: float) -> Path:
    """Write a JSON report summarizing the pipeline run."""
    output_base = Path(config.get("output_base", "."))
    output_base.mkdir(parents=True, exist_ok=True)
    report_path = output_base / "pipeline_report.json"

    report = {
        "success": phases.get("overall_success", False),
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phases": {},
    }

    for phase_name, phase_data in phases.items():
        if phase_name == "overall_success":
            continue
        report["phases"][phase_name] = phase_data

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Pipeline report: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(config_path: str, dry_run: bool = False) -> bool:
    """Execute the full pipeline for a single config file."""
    config = load_config(config_path)
    if dry_run:
        config["dry_run"] = True

    log_section(f"Pipeline: {Path(config_path).name}")

    # Resolve Blender
    blender_exe = find_blender(config.get("blender_path", "blender"))
    if not blender_exe and config.get("scene"):
        log("Blender executable not found — cannot run capture phase", "ERROR")
        if not dry_run:
            return False

    if blender_exe:
        log(f"Blender: {blender_exe}")

    phases = {}
    pipeline_start = time.monotonic()

    # --- Phase 1: Capture ---
    if config.get("scene"):
        log_section("Phase 1: Capture")
        capture_start = time.monotonic()
        capture_ok = run_capture(blender_exe, config, config_path, dry_run)
        phases["capture"] = {
            "success": capture_ok,
            "elapsed_seconds": round(time.monotonic() - capture_start, 2),
            "scene": config.get("scene"),
            "output": config.get("output_base"),
        }
        if not capture_ok and not dry_run:
            log("Capture failed — aborting pipeline", "ERROR")
            phases["overall_success"] = False
            write_pipeline_report(config, phases,
                                  time.monotonic() - pipeline_start)
            return False
    else:
        log("No scene specified — skipping capture (training-only mode)")

    # --- Phase 2: Training ---
    training_config = config.get("training", {})
    if training_config.get("enabled", False):
        log_section("Phase 2: Training")
        train_start = time.monotonic()
        trained_dirs = run_training(config, dry_run)
        phases["training"] = {
            "success": len(trained_dirs) > 0 or dry_run,
            "elapsed_seconds": round(time.monotonic() - train_start, 2),
            "trained_models": [str(d) for d in trained_dirs],
        }
    else:
        trained_dirs = []

    # --- Phase 3: Cleanup ---
    cleanup_config = config.get("cleanup", {})
    if cleanup_config.get("enabled", False) and trained_dirs:
        log_section("Phase 3: Cleanup")
        cleanup_start = time.monotonic()
        cleaned = run_cleanup(config, trained_dirs, dry_run)
        phases["cleanup"] = {
            "success": len(cleaned) > 0 or dry_run,
            "elapsed_seconds": round(time.monotonic() - cleanup_start, 2),
            "cleaned_models": [str(p) for p in cleaned],
        }

    # --- Report ---
    total_elapsed = time.monotonic() - pipeline_start
    phases["overall_success"] = all(
        p.get("success", True) for p in phases.values()
        if isinstance(p, dict)
    )
    write_pipeline_report(config, phases, total_elapsed)

    log_section("Pipeline Complete")
    log(f"Total time: {total_elapsed:.0f}s")
    log(f"Success: {phases['overall_success']}")

    return phases["overall_success"]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="GS Blender end-to-end pipeline orchestrator",
    )
    parser.add_argument(
        "--config", nargs="+", required=True,
        help="Path(s) to YAML job config file(s)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without executing",
    )

    args = parser.parse_args()

    log_section("GS Blender Pipeline")
    log(f"Jobs: {len(args.config)}")
    if args.dry_run:
        log("DRY RUN MODE — no actual work will be performed")

    all_ok = True
    for config_path in args.config:
        if not Path(config_path).exists():
            log(f"Config not found: {config_path}", "ERROR")
            all_ok = False
            continue

        ok = run_pipeline(config_path, dry_run=args.dry_run)
        if not ok:
            all_ok = False

    if all_ok:
        log("All jobs completed successfully")
    else:
        log("Some jobs failed", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
