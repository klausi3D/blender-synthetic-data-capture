"""
Gaussian Splatting Lightning backend with mask support.
https://github.com/yzslab/gaussian-splatting-lightning
"""

import os
import re
import shutil
import sys
import warnings
from typing import Optional, List, Dict

from .base import TrainingBackend, TrainingConfig, TrainingProgress, TrainingStatus
from ...utils.paths import normalize_path, get_conda_base, get_conda_python


class GSLightningBackend(TrainingBackend):
    """Gaussian Splatting Lightning training backend with mask support."""

    name = "GS-Lightning"
    description = "Gaussian Splatting Lightning with mask support"
    website = "https://github.com/yzslab/gaussian-splatting-lightning"
    install_instructions = """
To install Gaussian Splatting Lightning:

1. Clone the repository:
   git clone https://github.com/yzslab/gaussian-splatting-lightning.git
   cd gaussian-splatting-lightning

2. Create conda environment:
   conda create -n gs_lightning python=3.10
   conda activate gs_lightning

3. Install PyTorch (CUDA 11.8):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

4. Install requirements:
   pip install -r requirements.txt

5. Set the path in addon preferences.

MASK SUPPORT:
- Masks should be single channel PNG
- Black (0) = masked out, White (255) = trained
- Filename: {image_name}.png (e.g., image_0001.png.png)
- Use --data.parser.mask_dir to specify mask directory
"""

    def __init__(self, install_path: Optional[str] = None, conda_env: Optional[str] = None):
        """Initialize backend.

        Args:
            install_path: Path to gaussian-splatting-lightning repository
            conda_env: Conda environment name
        """
        self._install_path = install_path
        self._conda_env = conda_env

    def is_available(self) -> bool:
        """Check if GS-Lightning is installed."""
        path = self.get_install_path()
        if path is None:
            return False

        main_script = os.path.join(path, "main.py")
        return os.path.exists(main_script)

    def get_install_path(self) -> Optional[str]:
        """Find GS-Lightning installation."""
        if self._install_path:
            return normalize_path(self._install_path)

        # Check addon preferences first
        try:
            import bpy
            prefs = bpy.context.preferences.addons.get('gs_capture_addon')
            if prefs and prefs.preferences:
                pref_path = getattr(prefs.preferences, 'gs_lightning_path', '')
                if pref_path:
                    # Use normalize_path for cross-platform handling
                    pref_path = normalize_path(pref_path)
                    if os.path.exists(os.path.join(pref_path, "main.py")):
                        return pref_path
        except Exception:
            pass

        # Check environment variable
        env_path = os.environ.get("GS_LIGHTNING_PATH")
        if env_path:
            env_path = normalize_path(env_path)
            if os.path.exists(env_path):
                return env_path

        # Search common locations
        if sys.platform == 'win32':
            search_paths = [
                os.path.expanduser("~/gaussian-splatting-lightning"),
                os.path.expanduser("~/repos/gaussian-splatting-lightning"),
                "C:/gaussian-splatting-lightning",
                "D:/gaussian-splatting-lightning",
            ]
        else:
            search_paths = [
                os.path.expanduser("~/gaussian-splatting-lightning"),
                os.path.expanduser("~/repos/gaussian-splatting-lightning"),
                os.path.expanduser("~/code/gaussian-splatting-lightning"),
                "/opt/gaussian-splatting-lightning",
            ]

        for path in search_paths:
            normalized = normalize_path(path)
            if os.path.exists(os.path.join(normalized, "main.py")):
                return normalized

        return None

    def _get_conda_env(self) -> str:
        """Get conda environment name from preferences."""
        if self._conda_env:
            return self._conda_env
        try:
            import bpy
            prefs = bpy.context.preferences.addons.get('gs_capture_addon')
            if prefs and prefs.preferences:
                return getattr(prefs.preferences, 'gs_lightning_env', 'gs_lightning')
        except Exception:
            pass
        return "gs_lightning"

    def _get_conda_python_path(self) -> Optional[str]:
        """Get Python executable from conda environment."""
        conda_env = self._get_conda_env()
        python_path = get_conda_python(conda_env)
        if python_path:
            return python_path
        return None

    def validate_data(self, data_path: str) -> tuple:
        """Validate COLMAP data format with optional masks."""
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
        images_dir = os.path.join(data_path, "images")
        if not os.path.exists(images_dir):
            return False, "Missing images directory"

        image_files = [f for f in os.listdir(images_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_count = len(image_files)

        if image_count == 0:
            return False, "No images found in images directory"

        # Check for masks (optional but recommended)
        masks_dir = os.path.join(data_path, "masks")
        has_masks = False
        mask_count = 0

        if os.path.exists(masks_dir):
            # Count masks with GSL naming convention
            for img_file in image_files:
                mask_file = os.path.join(masks_dir, f"{img_file}.png")
                if os.path.exists(mask_file):
                    mask_count += 1

            if mask_count > 0:
                has_masks = True

        status = f"Valid dataset with {image_count} images"
        if has_masks:
            status += f" and {mask_count} masks"
        else:
            status += " (no masks found - consider enabling mask export)"

        return True, status

    def get_command(self, config: TrainingConfig) -> List[str]:
        """Build training command."""
        install_path = self.get_install_path()
        if not install_path:
            raise RuntimeError("GS-Lightning not installed")

        main_script = os.path.join(install_path, "main.py")

        # Get Python from conda environment
        python_exe = self._get_conda_python_path()
        if not python_exe:
            raise RuntimeError(
                "Could not find Python in conda environment. "
                "Please ensure the 'gs_lightning' conda environment exists."
            )

        cmd = [
            python_exe,
            main_script,
            "fit",
            "--data.parser", "Colmap",
            "--data.path", config.data_path,
            "--trainer.default_root_dir", config.output_path,
            "--trainer.max_steps", str(config.iterations),
        ]

        # Point to sparse/0/ subfolder (our COLMAP export structure)
        sparse_dir = os.path.join(config.data_path, "sparse", "0")
        if os.path.exists(sparse_dir):
            cmd.extend(["--data.parser.sparse_dir", sparse_dir])

        # Check for masks directory - only add if masks actually exist and match images
        masks_dir = os.path.join(config.data_path, "masks")
        mask_dir_to_use = None
        if os.path.isdir(masks_dir):
            images_dir = os.path.join(config.data_path, "images")
            if os.path.isdir(images_dir):
                image_files = [
                    f for f in os.listdir(images_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                mask_files = {
                    f.lower() for f in os.listdir(masks_dir)
                    if f.lower().endswith('.png')
                }
                if image_files and mask_files:
                    # GS-Lightning expects: image_0001.png.png (image filename + .png)
                    matching_masks = sum(
                        1 for img in image_files
                        if f"{img}.png".lower() in mask_files
                    )
                    if matching_masks > 0:
                        mask_dir_to_use = masks_dir
                    else:
                        warnings.warn(
                            "GS-Lightning masks directory found but no masks matched "
                            "the expected naming (image filename + .png). "
                            "Skipping --data.parser.mask_dir."
                        )

        if mask_dir_to_use:
            cmd.extend(["--data.parser.mask_dir", mask_dir_to_use])

        # Save checkpoints
        if config.save_iterations and len(config.save_iterations) > 1:
            # NOTE: GS-Lightning checkpoint scheduling is not wired up here yet.
            warnings.warn(
                "GS-Lightning save iterations are not configurable via this addon yet. "
                "Default checkpointing will be used. "
                "If you need custom save intervals, pass the appropriate flags in Extra Arguments."
            )

        # White background - not a standard flag, skip for now
        # if config.white_background:
        #     cmd.append("--data.parser.white_background")

        # Extra args
        cmd.extend(config.extra_args)

        return cmd

    def parse_output(self, line: str) -> Optional[TrainingProgress]:
        """Parse training output line.

        GS-Lightning uses PyTorch Lightning progress bars with format like:
        Epoch 0:   4%|3  | 10/280 [00:04<02:10, 2.07it/s, train/loss=0.708, train/ssim=0.051]
        """
        progress = TrainingProgress(status=TrainingStatus.RUNNING)

        # Match step progress from Lightning format: "| 10/280 [" or just step count
        # Also match global_step if present
        step_match = re.search(r'\|\s*(\d+)/\d+\s*\[', line)
        if step_match:
            progress.iteration = int(step_match.group(1))
        else:
            # Fallback: match "Step: 1000" or "step=1000" or "global_step=1000"
            step_match = re.search(r'(?:global_)?[Ss]tep[=:\s]+(\d+)', line)
            if step_match:
                progress.iteration = int(step_match.group(1))

        # Match loss - GS-Lightning uses "train/loss=0.708" format
        loss_match = re.search(r'(?:train/)?loss[=:\s]*([\d.e+-]+)', line, re.I)
        if loss_match:
            try:
                progress.loss = float(loss_match.group(1))
            except ValueError:
                pass

        # Match PSNR - check for val/psnr or just psnr
        psnr_match = re.search(r'(?:val/)?psnr[=:\s]*([\d.]+)', line, re.I)
        if psnr_match:
            try:
                progress.psnr = float(psnr_match.group(1))
            except ValueError:
                pass

        # Match SSIM as additional metric (GS-Lightning shows this)
        ssim_match = re.search(r'(?:train/)?ssim[=:\s]*([\d.]+)', line, re.I)
        if ssim_match:
            # Store in message since we don't have dedicated field
            pass

        # Check for completion
        if "training finished" in line.lower() or "fit finished" in line.lower():
            progress.status = TrainingStatus.COMPLETED
        # Lightning also shows "Trainer.fit stopped" on completion
        if "trainer.fit stopped" in line.lower():
            progress.status = TrainingStatus.COMPLETED

        # Check for errors
        if "error" in line.lower() or "exception" in line.lower():
            if "cuda" in line.lower() or "memory" in line.lower() or "oom" in line.lower():
                progress.error = line
        # Also catch CUDA out of memory errors
        if "out of memory" in line.lower():
            progress.error = line
            progress.status = TrainingStatus.FAILED

        if progress.iteration > 0 or progress.loss > 0 or progress.error:
            progress.message = line.strip()
            return progress

        return None

    def get_output_files(self, output_path: str, iteration: int) -> Dict[str, str]:
        """Get paths to output files."""
        return {
            'checkpoint': os.path.join(output_path, "checkpoints"),
            'config': os.path.join(output_path, "config.yaml"),
        }

    def get_final_model_path(self, output_path: str) -> Optional[str]:
        """Get path to final trained model."""
        # Look for PLY files in output
        for root, dirs, files in os.walk(output_path):
            for f in files:
                if f.endswith(".ply"):
                    return os.path.join(root, f)

        # Check point_cloud directory
        pc_path = os.path.join(output_path, "point_cloud")
        if os.path.exists(pc_path):
            for f in os.listdir(pc_path):
                if f.endswith(".ply"):
                    return os.path.join(pc_path, f)

        return None

    def estimate_training_time(self, config: TrainingConfig) -> float:
        """Estimate training time in seconds."""
        # Similar to original 3DGS
        return config.iterations * 0.1

    def estimate_vram_usage(self, config: TrainingConfig) -> float:
        """Estimate VRAM usage in GB."""
        return 8.0
