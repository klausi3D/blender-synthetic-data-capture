"""
Original 3D Gaussian Splatting backend (Kerbl et al., SIGGRAPH 2023).
https://github.com/graphdeco-inria/gaussian-splatting
"""

import os
import re
import sys
from typing import Optional, List, Dict

from .base import TrainingBackend, TrainingConfig, TrainingProgress, TrainingStatus


class GaussianSplattingBackend(TrainingBackend):
    """Original 3DGS training backend."""

    name = "3D Gaussian Splatting"
    description = "Original implementation by Kerbl et al."
    website = "https://github.com/graphdeco-inria/gaussian-splatting"
    install_instructions = """
To install 3D Gaussian Splatting:

1. Clone the repository:
   git clone https://github.com/graphdeco-inria/gaussian-splatting.git

2. Create conda environment:
   conda create -n gaussian_splatting python=3.8
   conda activate gaussian_splatting

3. Install dependencies:
   pip install -r requirements.txt

4. Install submodules:
   pip install submodules/diff-gaussian-rasterization
   pip install submodules/simple-knn

5. Set the path in addon preferences.
"""

    def __init__(self, install_path: Optional[str] = None):
        """Initialize backend.

        Args:
            install_path: Path to gaussian-splatting repository
        """
        self._install_path = install_path

    def is_available(self) -> bool:
        """Check if 3DGS is installed."""
        path = self.get_install_path()
        if path is None:
            return False

        train_script = os.path.join(path, "train.py")
        return os.path.exists(train_script)

    def get_install_path(self) -> Optional[str]:
        """Find 3DGS installation."""
        if self._install_path:
            return self._install_path

        # Check environment variable
        env_path = os.environ.get("GAUSSIAN_SPLATTING_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        # Search common locations
        search_paths = [
            os.path.expanduser("~/gaussian-splatting"),
            os.path.expanduser("~/repos/gaussian-splatting"),
            os.path.expanduser("~/code/gaussian-splatting"),
            "/opt/gaussian-splatting",
            "C:/gaussian-splatting",
            "D:/gaussian-splatting",
        ]

        for path in search_paths:
            if os.path.exists(os.path.join(path, "train.py")):
                return path

        return None

    def validate_data(self, data_path: str) -> tuple:
        """Validate COLMAP data format."""
        # Check for COLMAP sparse reconstruction
        sparse_path = os.path.join(data_path, "sparse", "0")

        if not os.path.exists(sparse_path):
            return False, "Missing sparse/0 directory (COLMAP format required)"

        # Check for camera files
        cameras_txt = os.path.join(sparse_path, "cameras.txt")
        cameras_bin = os.path.join(sparse_path, "cameras.bin")

        if not os.path.exists(cameras_txt) and not os.path.exists(cameras_bin):
            return False, "Missing cameras.txt or cameras.bin"

        # Check for images
        images_txt = os.path.join(sparse_path, "images.txt")
        images_bin = os.path.join(sparse_path, "images.bin")

        if not os.path.exists(images_txt) and not os.path.exists(images_bin):
            return False, "Missing images.txt or images.bin"

        # Check for actual images
        images_dir = os.path.join(data_path, "images")
        if not os.path.exists(images_dir):
            return False, "Missing images directory"

        image_count = len([f for f in os.listdir(images_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if image_count == 0:
            return False, "No images found in images directory"

        return True, f"Valid dataset with {image_count} images"

    def get_command(self, config: TrainingConfig) -> List[str]:
        """Build training command."""
        install_path = self.get_install_path()
        if not install_path:
            raise RuntimeError("3DGS not installed")

        train_script = os.path.join(install_path, "train.py")

        cmd = [
            sys.executable,
            train_script,
            "-s", config.data_path,
            "-m", config.output_path,
            "--iterations", str(config.iterations),
        ]

        # Save iterations
        if config.save_iterations:
            cmd.extend(["--save_iterations"] + [str(i) for i in config.save_iterations])

        # Test iterations
        if config.test_iterations:
            cmd.extend(["--test_iterations"] + [str(i) for i in config.test_iterations])

        # White background
        if config.white_background:
            cmd.append("--white_background")

        # Resolution
        if config.resolution > 0:
            cmd.extend(["-r", str(config.resolution)])

        # Densification
        cmd.extend([
            "--densify_from_iter", str(config.densify_from_iter),
            "--densify_until_iter", str(config.densify_until_iter),
            "--densification_interval", str(config.densification_interval),
        ])

        # Extra args
        cmd.extend(config.extra_args)

        return cmd

    def parse_output(self, line: str) -> Optional[TrainingProgress]:
        """Parse training output line."""
        progress = TrainingProgress(status=TrainingStatus.RUNNING)

        # Match iteration progress
        # Format: "[ITER 1000] Evaluating train: L1 0.0234 PSNR 25.67"
        iter_match = re.search(r'\[ITER\s+(\d+)\]', line)
        if iter_match:
            progress.iteration = int(iter_match.group(1))

        # Match loss
        loss_match = re.search(r'L1\s+([\d.]+)', line)
        if loss_match:
            progress.loss = float(loss_match.group(1))

        # Match PSNR
        psnr_match = re.search(r'PSNR\s+([\d.]+)', line)
        if psnr_match:
            progress.psnr = float(psnr_match.group(1))

        # Match training progress bar
        # Format: "Training progress: 45%|████████  | 13500/30000 [02:15<02:45, 100.0it/s]"
        bar_match = re.search(r'(\d+)/(\d+)\s+\[[\d:]+<([\d:]+)', line)
        if bar_match:
            progress.iteration = int(bar_match.group(1))
            progress.total_iterations = int(bar_match.group(2))

            # Parse ETA
            eta_str = bar_match.group(3)
            parts = eta_str.split(':')
            if len(parts) == 2:
                progress.eta_seconds = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                progress.eta_seconds = (int(parts[0]) * 3600 +
                                        int(parts[1]) * 60 + int(parts[2]))

        # Check for completion
        if "Training complete" in line or "Saving final model" in line:
            progress.status = TrainingStatus.COMPLETED

        # Check for errors
        if "error" in line.lower() or "exception" in line.lower():
            progress.error = line

        # Only return if we found relevant info
        if progress.iteration > 0 or progress.error:
            progress.message = line.strip()
            return progress

        return None

    def get_output_files(self, output_path: str, iteration: int) -> Dict[str, str]:
        """Get paths to output files."""
        point_cloud_dir = os.path.join(
            output_path, "point_cloud", f"iteration_{iteration}"
        )

        return {
            'point_cloud': os.path.join(point_cloud_dir, "point_cloud.ply"),
            'cameras': os.path.join(output_path, "cameras.json"),
            'config': os.path.join(output_path, "cfg_args"),
        }

    def get_final_model_path(self, output_path: str) -> Optional[str]:
        """Get path to final trained model."""
        # Find highest iteration
        pc_base = os.path.join(output_path, "point_cloud")
        if not os.path.exists(pc_base):
            return None

        iterations = []
        for d in os.listdir(pc_base):
            if d.startswith("iteration_"):
                try:
                    iterations.append(int(d.replace("iteration_", "")))
                except ValueError:
                    pass

        if not iterations:
            return None

        max_iter = max(iterations)
        return os.path.join(pc_base, f"iteration_{max_iter}", "point_cloud.ply")

    def estimate_training_time(self, config: TrainingConfig) -> float:
        """Estimate training time."""
        # Rough estimates based on RTX 3090
        # ~0.05s per iteration for small scenes, ~0.2s for large
        base_time = config.iterations * 0.1
        return base_time

    def estimate_vram_usage(self, config: TrainingConfig) -> float:
        """Estimate VRAM usage."""
        # Base: ~6GB, scales with image count and resolution
        return 8.0
