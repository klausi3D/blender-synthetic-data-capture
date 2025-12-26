"""
gsplat backend - optimized CUDA Gaussian Splatting.
https://github.com/nerfstudio-project/gsplat
"""

import os
import re
import shutil
import sys
from typing import Optional, List, Dict

from .base import TrainingBackend, TrainingConfig, TrainingProgress, TrainingStatus


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

    def __init__(self, examples_path: Optional[str] = None):
        """Initialize backend.

        Args:
            examples_path: Path to gsplat examples directory
        """
        self._examples_path = examples_path

    def is_available(self) -> bool:
        """Check if gsplat is installed."""
        try:
            import gsplat
            return True
        except ImportError:
            return False

    def get_install_path(self) -> Optional[str]:
        """Get gsplat examples path."""
        if self._examples_path:
            return self._examples_path

        # Check environment variable
        env_path = os.environ.get("GSPLAT_EXAMPLES_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        # Search common locations
        search_paths = [
            os.path.expanduser("~/gsplat/examples"),
            os.path.expanduser("~/repos/gsplat/examples"),
            "/opt/gsplat/examples",
        ]

        for path in search_paths:
            if os.path.exists(os.path.join(path, "simple_trainer.py")):
                return path

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

        if examples_path:
            # Use gsplat example trainer
            trainer_script = os.path.join(examples_path, "simple_trainer.py")
            cmd = [
                sys.executable,
                trainer_script,
                "--data_dir", config.data_path,
                "--result_dir", config.output_path,
                "--iterations", str(config.iterations),
            ]
        else:
            # Fallback: assume gsplat is importable and use inline training
            cmd = [
                sys.executable, "-c",
                f"""
import gsplat
# Minimal training script
print("gsplat training not implemented without examples")
"""
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
