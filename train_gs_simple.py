#!/usr/bin/env python3
"""
Simple 3DGS Training Script for Blender GS Capture Output
==========================================================
Optimized for synthetic data with known cameras (transforms.json).
No COLMAP needed - directly uses Blender's camera exports.

Usage:
    python train_gs_simple.py /path/to/captures
    python train_gs_simple.py /path/to/captures --iterations 50000 --output ./trained
    python train_gs_simple.py /path/to/parent_folder --batch
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default paths - modify these for your system
DEFAULT_GS_PATH = os.path.expanduser("~/gaussian-splatting")
DEFAULT_OUTPUT = "./trained_splats"

# Training defaults optimized for Blender synthetic captures
DEFAULT_ITERATIONS = 30000
DEFAULT_SAVE_ITERATIONS = [7000, 15000, 30000]


# =============================================================================
# UTILITIES
# =============================================================================

def find_gaussian_splatting() -> Optional[Path]:
    """Find the gaussian-splatting installation."""
    search_paths = [
        Path(DEFAULT_GS_PATH),
        Path.home() / "gaussian-splatting",
        Path.home() / "repos" / "gaussian-splatting",
        Path.home() / "code" / "gaussian-splatting",
        Path("/opt/gaussian-splatting"),
        Path("./gaussian-splatting"),
    ]
    
    # Also check environment variable
    env_path = os.environ.get("GAUSSIAN_SPLATTING_PATH")
    if env_path:
        search_paths.insert(0, Path(env_path))
    
    for path in search_paths:
        if path.exists() and (path / "train.py").exists():
            return path
    
    return None


def find_nerfstudio() -> bool:
    """Check if nerfstudio is available."""
    return shutil.which("ns-train") is not None


def validate_capture_folder(folder: Path) -> Tuple[bool, str]:
    """Validate that a folder contains valid GS Capture output."""
    # Check for images
    images_folder = folder / "images"
    if not images_folder.exists():
        # Maybe images are in root
        images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
        if not images:
            return False, "No images found"
    else:
        images = list(images_folder.glob("*.png")) + list(images_folder.glob("*.jpg"))
        if not images:
            return False, "No images in 'images' folder"
    
    # Check for transforms.json (preferred) or COLMAP data
    has_transforms = (folder / "transforms.json").exists()
    has_colmap = (folder / "sparse" / "0" / "cameras.bin").exists() or \
                 (folder / "sparse" / "0" / "cameras.txt").exists()
    
    if not has_transforms and not has_colmap:
        return False, "No camera data (transforms.json or COLMAP sparse)"
    
    return True, f"Valid ({len(images)} images, {'transforms.json' if has_transforms else 'COLMAP'})"


def get_pending_folders(base_path: Path, output_base: Path) -> List[Path]:
    """Find folders that haven't been processed yet."""
    pending = []
    
    for item in sorted(base_path.iterdir()):
        if not item.is_dir():
            continue
        
        valid, msg = validate_capture_folder(item)
        if not valid:
            print(f"  Skipping {item.name}: {msg}")
            continue
        
        # Check if already processed
        output_folder = output_base / item.name / "splat"
        if output_folder.exists():
            # Check for completion
            final_ply = None
            for save_iter in reversed(DEFAULT_SAVE_ITERATIONS):
                check_path = output_folder / "point_cloud" / f"iteration_{save_iter}" / "point_cloud.ply"
                if check_path.exists():
                    final_ply = check_path
                    break
            
            if final_ply:
                print(f"  Already complete: {item.name}")
                continue
        
        pending.append(item)
        print(f"  Queued: {item.name} ({msg})")
    
    return pending


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_with_original_gs(
    data_folder: Path,
    output_folder: Path,
    gs_path: Path,
    iterations: int = DEFAULT_ITERATIONS,
    white_bg: bool = True,
    resolution: int = -1,
    gpu_id: int = 0,
    extra_args: List[str] = None
) -> bool:
    """Train using the original 3DGS implementation."""
    
    train_script = gs_path / "train.py"
    
    cmd = [
        sys.executable, str(train_script),
        "-s", str(data_folder),
        "-m", str(output_folder),
        "--iterations", str(iterations),
        "--save_iterations", " ".join(str(i) for i in DEFAULT_SAVE_ITERATIONS),
        "--test_iterations", " ".join(str(i) for i in DEFAULT_SAVE_ITERATIONS),
    ]
    
    if white_bg:
        cmd.append("--white_background")
    
    if resolution > 0:
        cmd.extend(["--resolution", str(resolution)])
    
    if extra_args:
        cmd.extend(extra_args)
    
    # Set environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"\n{'='*60}")
    print(f"Training: {data_folder.name}")
    print(f"Output: {output_folder}")
    print(f"Iterations: {iterations}")
    print(f"{'='*60}\n")
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            cwd=str(gs_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.wait()
        
        if process.returncode != 0:
            print(f"\nTraining failed with return code {process.returncode}")
            return False
        
        print(f"\nTraining complete: {data_folder.name}")
        return True
        
    except Exception as e:
        print(f"\nError during training: {e}")
        return False


def train_with_nerfstudio(
    data_folder: Path,
    output_folder: Path,
    iterations: int = DEFAULT_ITERATIONS,
    white_bg: bool = True,
    gpu_id: int = 0
) -> bool:
    """Train using nerfstudio's splatfacto."""
    
    # Determine data format
    has_transforms = (data_folder / "transforms.json").exists()
    
    cmd = [
        "ns-train", "splatfacto",
        "--data", str(data_folder),
        "--output-dir", str(output_folder.parent),
        "--experiment-name", output_folder.name,
        "--max-num-iterations", str(iterations),
        "--viewer.quit-on-train-completion", "True",
    ]
    
    if has_transforms:
        cmd.extend(["--pipeline.datamanager.dataparser", "nerfstudio-data"])
    
    if white_bg:
        cmd.extend(["--pipeline.model.background-color", "white"])
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"\n{'='*60}")
    print(f"Training with nerfstudio: {data_folder.name}")
    print(f"{'='*60}\n")
    
    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.wait()
        return process.returncode == 0
        
    except Exception as e:
        print(f"\nError: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train 3D Gaussian Splatting models from Blender GS Capture output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single capture
  python train_gs_simple.py ./gs_capture/MyObject
  
  # Train with more iterations
  python train_gs_simple.py ./gs_capture/MyObject --iterations 50000
  
  # Batch process all captures in a folder
  python train_gs_simple.py ./gs_capture --batch
  
  # Specify output location
  python train_gs_simple.py ./gs_capture --batch --output ./my_splats
  
  # Use nerfstudio instead of original 3DGS
  python train_gs_simple.py ./gs_capture/MyObject --nerfstudio

Environment Variables:
  GAUSSIAN_SPLATTING_PATH - Path to gaussian-splatting repo
  CUDA_VISIBLE_DEVICES - GPU selection (overridden by --gpu)
        """
    )
    
    parser.add_argument("input", help="Input folder (capture) or parent folder (with --batch)")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--batch", "-b", action="store_true", help="Process all subfolders")
    parser.add_argument("--iterations", "-i", type=int, default=DEFAULT_ITERATIONS, help="Training iterations")
    parser.add_argument("--resolution", "-r", type=int, default=-1, help="Training resolution (-1 for auto)")
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU ID")
    parser.add_argument("--no-white-bg", action="store_true", help="Disable white background")
    parser.add_argument("--nerfstudio", action="store_true", help="Use nerfstudio instead of original 3DGS")
    parser.add_argument("--gs-path", help="Path to gaussian-splatting repo")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    
    args = parser.parse_args()
    
    input_path = Path(args.input).resolve()
    output_base = Path(args.output).resolve()
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Detect implementation
    use_nerfstudio = args.nerfstudio
    gs_path = None
    
    if use_nerfstudio:
        if not find_nerfstudio():
            print("Error: nerfstudio not found. Install with: pip install nerfstudio")
            sys.exit(1)
        print("Using nerfstudio splatfacto")
    else:
        gs_path = Path(args.gs_path) if args.gs_path else find_gaussian_splatting()
        if not gs_path:
            # Fall back to nerfstudio if available
            if find_nerfstudio():
                print("Original 3DGS not found, falling back to nerfstudio")
                use_nerfstudio = True
            else:
                print("Error: gaussian-splatting not found")
                print("Either:")
                print("  1. Clone https://github.com/graphdeco-inria/gaussian-splatting")
                print("  2. Set GAUSSIAN_SPLATTING_PATH environment variable")
                print("  3. Use --gs-path argument")
                print("  4. Install nerfstudio: pip install nerfstudio")
                sys.exit(1)
        else:
            print(f"Using original 3DGS from: {gs_path}")
    
    # Collect folders to process
    folders_to_process = []
    
    if args.batch:
        print(f"\nScanning: {input_path}")
        folders_to_process = get_pending_folders(input_path, output_base)
    else:
        valid, msg = validate_capture_folder(input_path)
        if valid:
            folders_to_process = [input_path]
            print(f"Single capture: {msg}")
        else:
            print(f"Error: {msg}")
            sys.exit(1)
    
    if not folders_to_process:
        print("\nNo folders to process!")
        sys.exit(0)
    
    print(f"\n{len(folders_to_process)} folder(s) to process")
    
    if args.dry_run:
        print("\nDry run - would process:")
        for folder in folders_to_process:
            print(f"  - {folder.name}")
        sys.exit(0)
    
    # Process each folder
    results = {"success": [], "failed": []}
    start_time = datetime.now()
    
    for i, folder in enumerate(folders_to_process):
        print(f"\n[{i+1}/{len(folders_to_process)}] Processing {folder.name}")
        
        output_folder = output_base / folder.name / "splat"
        output_folder.mkdir(parents=True, exist_ok=True)
        
        if use_nerfstudio:
            success = train_with_nerfstudio(
                folder,
                output_folder,
                iterations=args.iterations,
                white_bg=not args.no_white_bg,
                gpu_id=args.gpu
            )
        else:
            success = train_with_original_gs(
                folder,
                output_folder,
                gs_path,
                iterations=args.iterations,
                white_bg=not args.no_white_bg,
                resolution=args.resolution,
                gpu_id=args.gpu
            )
        
        if success:
            results["success"].append(folder.name)
        else:
            results["failed"].append(folder.name)
    
    # Summary
    elapsed = datetime.now() - start_time
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {elapsed}")
    print(f"Successful: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    
    if results["failed"]:
        print(f"\nFailed:")
        for name in results["failed"]:
            print(f"  - {name}")
    
    print(f"\nOutput location: {output_base}")


if __name__ == "__main__":
    main()
