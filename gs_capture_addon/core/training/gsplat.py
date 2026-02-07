"""
gsplat backend - optimized CUDA Gaussian Splatting.
https://github.com/nerfstudio-project/gsplat
"""

import os
import re
import shutil
from typing import Optional, List, Dict

from .base import TrainingBackend, TrainingConfig, TrainingProgress, TrainingStatus
from ...utils.paths import normalize_path, get_conda_base, get_conda_python


class GsplatBackend(TrainingBackend):
    """gsplat training backend."""

    name = "gsplat"
    description = "Optimized CUDA Gaussian Splatting by Nerfstudio"
    website = "https://github.com/nerfstudio-project/gsplat"
    install_instructions = """
To install gsplat:

1. Create conda environment:
   conda create -n gsplat python=3.10
   conda activate gsplat

2. Install PyTorch (CUDA 11.8):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

3. Install gsplat:
   pip install gsplat

4. For training scripts, clone the examples:
   git clone https://github.com/nerfstudio-project/gsplat.git
   cd gsplat/examples
"""

    def __init__(self, examples_path: Optional[str] = None, conda_env: Optional[str] = None):
        """Initialize backend.

        Args:
            examples_path: Path to gsplat examples directory
            conda_env: Conda environment name for gsplat
        """
        self._examples_path = examples_path
        self._conda_env = conda_env

    def is_available(self) -> bool:
        """Check if gsplat is installed."""
        # Check if gsplat conda env exists and has python
        conda_env = self._get_conda_env()
        python_path = get_conda_python(conda_env)
        if python_path:
            return True
        return False

    def _get_conda_env(self) -> str:
        """Get conda environment name from preferences."""
        if self._conda_env:
            return self._conda_env
        try:
            import bpy
            prefs = bpy.context.preferences.addons.get('gs_capture_addon')
            if prefs and prefs.preferences:
                return prefs.preferences.gsplat_env
        except Exception:
            pass
        return "gsplat"  # Default

    def _get_conda_python_path(self) -> Optional[str]:
        """Get Python executable from conda environment."""
        conda_env = self._get_conda_env()
        python_path = get_conda_python(conda_env)
        if python_path:
            return python_path
        return None

    def get_install_path(self) -> Optional[str]:
        """Get gsplat examples path."""
        def _validated_examples_path(path: Optional[str]) -> Optional[str]:
            if not path:
                return None
            normalized = normalize_path(path)
            trainer_script = os.path.join(normalized, "simple_trainer.py")
            if os.path.exists(trainer_script):
                return normalized
            return None

        # Check explicitly provided path
        if self._examples_path:
            validated = _validated_examples_path(self._examples_path)
            if validated:
                return validated

        # Check addon preferences
        try:
            import bpy
            prefs = bpy.context.preferences.addons.get('gs_capture_addon')
            if prefs and prefs.preferences:
                pref_path = getattr(prefs.preferences, 'gsplat_examples_path', '')
                validated = _validated_examples_path(pref_path)
                if validated:
                    return validated
        except Exception:
            pass

        # Check environment variable
        env_path = os.environ.get("GSPLAT_EXAMPLES_PATH")
        validated = _validated_examples_path(env_path)
        if validated:
            return validated

        # Search common locations
        search_paths = [
            os.path.expanduser("~/gsplat/examples"),
            os.path.expanduser("~/repos/gsplat/examples"),
            "/opt/gsplat/examples",
        ]

        for path in search_paths:
            validated = _validated_examples_path(path)
            if validated:
                return validated

        return None

    def validate_data(self, data_path: str) -> tuple:
        """Validate data format (supports both COLMAP and transforms.json)."""
        # Check for transforms.json
        transforms_file = os.path.join(data_path, "transforms.json")
        has_transforms = os.path.exists(transforms_file)

        # Check for COLMAP
        sparse_path = os.path.join(data_path, "sparse", "0")
        has_colmap = os.path.exists(sparse_path)

        if not has_transforms and not has_colmap:
            return False, "Need either transforms.json or COLMAP sparse/0/"

        # Check for images
        images_dir = os.path.join(data_path, "images")
        if not os.path.exists(images_dir):
            return False, "Missing images directory"

        image_count = len([f for f in os.listdir(images_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if image_count == 0:
            return False, "No images found"

        format_type = "transforms.json" if has_transforms else "COLMAP"
        return True, f"Valid dataset ({format_type}) with {image_count} images"

    def get_command(self, config: TrainingConfig) -> List[str]:
        """Build training command."""
        examples_path = self.get_install_path()

        # Get conda Python
        python_exe = self._get_conda_python_path()
        if not python_exe:
            raise RuntimeError(
                "Could not find Python in conda environment. "
                "Please ensure the 'gsplat' conda environment exists."
            )

        if not examples_path:
            raise RuntimeError(
                "gsplat examples path not found. Set 'gsplat Examples Path' in the addon "
                "preferences to the gsplat/examples directory (must contain simple_trainer.py), "
                "or set the GSPLAT_EXAMPLES_PATH environment variable to that directory."
            )

        # Use gsplat example trainer
        trainer_script = os.path.join(examples_path, "simple_trainer.py")
        cmd = [
            python_exe,
            trainer_script,
            "--data_dir", config.data_path,
            "--result_dir", config.output_path,
            "--iterations", str(config.iterations),
        ]

        # Extra args
        cmd.extend(config.extra_args)

        return cmd

    def parse_output(self, line: str) -> Optional[TrainingProgress]:
        """Parse training output line."""
        progress = TrainingProgress(status=TrainingStatus.RUNNING)

        # Match iteration progress
        iter_match = re.search(r'(?:step|iter(?:ation)?)\s*[=:]\s*(\d+)', line, re.I)
        if iter_match:
            progress.iteration = int(iter_match.group(1))

        # Match loss
        loss_match = re.search(r'loss\s*[=:]\s*([\d.e+-]+)', line, re.I)
        if loss_match:
            try:
                progress.loss = float(loss_match.group(1))
            except ValueError:
                pass

        # Match PSNR
        psnr_match = re.search(r'psnr\s*[=:]\s*([\d.]+)', line, re.I)
        if psnr_match:
            progress.psnr = float(psnr_match.group(1))

        # Check for completion
        if "training complete" in line.lower() or "finished" in line.lower():
            progress.status = TrainingStatus.COMPLETED

        # Check for errors
        if "error" in line.lower() or "exception" in line.lower():
            progress.error = line

        if progress.iteration > 0 or progress.error:
            progress.message = line.strip()
            return progress

        return None

    def get_final_model_path(self, output_path: str) -> Optional[str]:
        """Get path to final trained model."""
        # Look for PLY files in output
        for root, dirs, files in os.walk(output_path):
            for f in files:
                if f.endswith(".ply"):
                    return os.path.join(root, f)
        return None

    def estimate_training_time(self, config: TrainingConfig) -> float:
        """Estimate training time (gsplat is faster than original)."""
        # gsplat is typically 2-3x faster than original 3DGS
        return config.iterations * 0.04

    def estimate_vram_usage(self, config: TrainingConfig) -> float:
        """Estimate VRAM usage."""
        # gsplat is more memory efficient
        return 6.0
