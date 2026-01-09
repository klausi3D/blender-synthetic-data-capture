"""
Nerfstudio backend for splatfacto training.
https://docs.nerf.studio/
"""

import os
import re
import shutil
from typing import Optional, List, Dict

from .base import TrainingBackend, TrainingConfig, TrainingProgress, TrainingStatus
from ...utils.paths import normalize_path, get_conda_base, get_conda_script, get_conda_executable


class NerfstudioBackend(TrainingBackend):
    """Nerfstudio splatfacto training backend."""

    name = "Nerfstudio"
    description = "Nerfstudio framework with splatfacto"
    website = "https://docs.nerf.studio/"
    install_instructions = """
To install Nerfstudio:

1. Create conda environment:
   conda create -n nerfstudio python=3.10
   conda activate nerfstudio

2. Install PyTorch (CUDA 11.8):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

3. Install Nerfstudio:
   pip install nerfstudio

4. Verify installation:
   ns-train --help
"""

    def __init__(self, conda_env: Optional[str] = None):
        """Initialize backend.

        Args:
            conda_env: Conda environment name for nerfstudio
        """
        self._conda_env = conda_env

    def is_available(self) -> bool:
        """Check if Nerfstudio is installed."""
        # First check if ns-train is directly in PATH
        if shutil.which("ns-train") is not None:
            return True

        # Check conda environment from preferences
        conda_env = self._get_conda_env()
        ns_train_path = get_conda_script(conda_env, "ns-train")
        if ns_train_path:
            return True
        return False

    def _get_conda_env(self) -> Optional[str]:
        """Get conda environment name from preferences."""
        if self._conda_env:
            return self._conda_env
        try:
            import bpy
            prefs = bpy.context.preferences.addons.get('gs_capture_addon')
            if prefs and prefs.preferences:
                return prefs.preferences.nerfstudio_env
        except Exception:
            pass
        return "nerfstudio"  # Default

    def _find_conda(self) -> Optional[str]:
        """Find conda executable."""
        return get_conda_executable()

    def get_install_path(self) -> Optional[str]:
        """Get ns-train executable path."""
        direct = shutil.which("ns-train")
        if direct:
            return normalize_path(direct)

        # Check in conda environment
        conda_env = self._get_conda_env()
        ns_train_path = get_conda_script(conda_env, "ns-train")
        if ns_train_path:
            return normalize_path(ns_train_path)
        return None

    def validate_data(self, data_path: str) -> tuple:
        """Validate transforms.json data format."""
        transforms_file = os.path.join(data_path, "transforms.json")

        if not os.path.exists(transforms_file):
            return False, "Missing transforms.json"

        # Check for images directory
        images_dir = os.path.join(data_path, "images")
        if not os.path.exists(images_dir):
            return False, "Missing images directory"

        # Count images
        image_count = len([f for f in os.listdir(images_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if image_count == 0:
            return False, "No images found"

        return True, f"Valid dataset with {image_count} images"

    def get_command(self, config: TrainingConfig) -> List[str]:
        """Build training command."""
        # Find conda and environment
        conda_path = self._find_conda()
        conda_env = self._get_conda_env()

        if not conda_path:
            raise RuntimeError("Could not find conda. Please install Miniconda or Anaconda.")

        # Use conda run to execute ns-train in the correct environment
        cmd = [
            conda_path, "run", "-n", conda_env, "--no-capture-output",
            "ns-train", "splatfacto",
            "--data", config.data_path,
            "--output-dir", config.output_path,
            "--max-num-iterations", str(config.iterations),
        ]

        # Pipeline config
        if config.white_background:
            cmd.extend([
                "--pipeline.datamanager.dataparser.white-background", "True"
            ])

        # Extra args
        cmd.extend(config.extra_args)

        return cmd

    def parse_output(self, line: str) -> Optional[TrainingProgress]:
        """Parse training output line."""
        progress = TrainingProgress(status=TrainingStatus.RUNNING)

        # Match Nerfstudio progress
        # Format: "Step: 1000/30000 | Loss: 0.0234 | PSNR: 25.67"
        step_match = re.search(r'Step:\s*(\d+)/(\d+)', line)
        if step_match:
            progress.iteration = int(step_match.group(1))
            progress.total_iterations = int(step_match.group(2))

        # Match loss
        loss_match = re.search(r'Loss:\s*([\d.]+)', line)
        if loss_match:
            progress.loss = float(loss_match.group(1))

        # Match PSNR
        psnr_match = re.search(r'PSNR:\s*([\d.]+)', line)
        if psnr_match:
            progress.psnr = float(psnr_match.group(1))

        # Match tqdm-style progress
        # Format: "45%|████████  | 13500/30000 [02:15<02:45]"
        tqdm_match = re.search(r'(\d+)%\|[█▏▎▍▌▋▊▉ ]+\|\s*(\d+)/(\d+)', line)
        if tqdm_match:
            progress.iteration = int(tqdm_match.group(2))
            progress.total_iterations = int(tqdm_match.group(3))

        # Check for completion
        if "Training finished" in line or "Saving final checkpoint" in line:
            progress.status = TrainingStatus.COMPLETED

        # Check for errors
        if "error" in line.lower() or "exception" in line.lower():
            progress.error = line

        if progress.iteration > 0 or progress.error:
            progress.message = line.strip()
            return progress

        return None

    def get_output_files(self, output_path: str, iteration: int) -> Dict[str, str]:
        """Get paths to output files."""
        # Nerfstudio uses a different structure
        # outputs/<project>/<method>/<timestamp>/
        return {
            'checkpoint': os.path.join(output_path, "nerfstudio_models"),
            'config': os.path.join(output_path, "config.yml"),
        }

    def get_final_model_path(self, output_path: str) -> Optional[str]:
        """Get path to final trained model."""
        # Look for exported splat
        exports_dir = os.path.join(output_path, "exports")
        if os.path.exists(exports_dir):
            for f in os.listdir(exports_dir):
                if f.endswith(".ply"):
                    return os.path.join(exports_dir, f)

        return None

    def export_splat(self, output_path: str) -> Optional[str]:
        """Export trained model to PLY.

        Args:
            output_path: Training output directory

        Returns:
            Path to exported PLY or None
        """
        # ns-export gaussian-splat --load-config <config> --output-dir <output>
        import subprocess

        config_path = os.path.join(output_path, "config.yml")
        export_dir = os.path.join(output_path, "exports")

        if not os.path.exists(config_path):
            return None

        try:
            subprocess.run([
                "ns-export", "gaussian-splat",
                "--load-config", config_path,
                "--output-dir", export_dir,
            ], check=True, capture_output=True)

            # Find exported file
            for f in os.listdir(export_dir):
                if f.endswith(".ply"):
                    return os.path.join(export_dir, f)
        except subprocess.CalledProcessError:
            pass

        return None
