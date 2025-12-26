#!/usr/bin/env python3
"""
Gaussian Splatting Training Pipeline Automation
================================================
Automates the full pipeline from images to trained 3DGS models.

Supports:
- Batch processing of multiple folders
- COLMAP sparse reconstruction (or skip if cameras known)
- Multiple 3DGS implementations (original, gsplat, nerfstudio)
- Queue management with progress tracking
- GPU memory management
- Resume capability

Requirements:
- COLMAP (for unknown cameras)
- CUDA-capable GPU
- One of: original 3DGS, gsplat, or nerfstudio

Usage:
    python gs_training_pipeline.py --input /path/to/folders --output /path/to/output
    python gs_training_pipeline.py --config pipeline_config.yaml
"""

import os
import sys
import json
import yaml
import shutil
import logging
import argparse
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ColmapConfig:
    """COLMAP processing configuration."""
    enabled: bool = True
    binary_path: str = "colmap"  # Or full path
    
    # Feature extraction
    camera_model: str = "PINHOLE"  # PINHOLE, SIMPLE_RADIAL, OPENCV, etc.
    single_camera: bool = True  # All images from same camera
    
    # Matching
    matcher: str = "exhaustive"  # exhaustive, sequential, vocab_tree
    vocab_tree_path: str = ""  # For vocab_tree matcher
    
    # Reconstruction
    ba_refine_focal_length: bool = False  # Don't refine if known
    ba_refine_principal_point: bool = False
    ba_refine_extra_params: bool = False
    
    # Performance
    gpu_index: str = "0"
    num_threads: int = -1  # -1 = auto
    
    # For synthetic data with known cameras
    skip_if_transforms_exist: bool = True  # Skip COLMAP if transforms.json exists


@dataclass
class TrainingConfig:
    """3DGS training configuration."""
    implementation: str = "original"  # original, gsplat, nerfstudio
    
    # Paths (auto-detected if not set)
    gaussian_splatting_path: str = ""
    gsplat_path: str = ""
    nerfstudio_path: str = ""
    
    # Training parameters
    iterations: int = 30000
    save_iterations: List[int] = field(default_factory=lambda: [7000, 15000, 30000])
    test_iterations: List[int] = field(default_factory=lambda: [7000, 15000, 30000])
    
    # Densification
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold: float = 0.0002
    
    # Learning rates
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    
    # Performance
    white_background: bool = True  # For objects on white bg
    resolution: int = -1  # -1 = auto, or specific resolution
    
    # GPU
    gpu_id: int = 0
    
    # Nerfstudio specific
    nerfstudio_method: str = "splatfacto"  # splatfacto, splatfacto-big


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    # Input/Output
    input_folders: List[str] = field(default_factory=list)
    output_base: str = "./gs_output"
    
    # Processing
    colmap: ColmapConfig = field(default_factory=ColmapConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Queue management
    max_concurrent: int = 1  # Usually 1 for GPU tasks
    retry_failed: bool = True
    max_retries: int = 2
    
    # Logging
    log_file: str = "pipeline.log"
    verbose: bool = True
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_file: str = "pipeline_checkpoint.json"


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(config: PipelineConfig) -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger("gs_pipeline")
    logger.setLevel(logging.DEBUG if config.verbose else logging.INFO)
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
    console.setFormatter(console_format)
    logger.addHandler(console)
    
    # File handler
    if config.log_file:
        log_path = Path(config.output_base) / config.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_executable(name: str, search_paths: List[str] = None) -> Optional[str]:
    """Find executable in PATH or specific locations."""
    # Check PATH first
    result = shutil.which(name)
    if result:
        return result
    
    # Check common locations
    common_paths = [
        f"/usr/local/bin/{name}",
        f"/usr/bin/{name}",
        f"~/bin/{name}",
        f"~/.local/bin/{name}",
    ]
    
    if search_paths:
        common_paths = search_paths + common_paths
    
    for path in common_paths:
        expanded = os.path.expanduser(path)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return expanded
    
    return None


def run_command(cmd: List[str], cwd: str = None, env: Dict = None, 
                logger: logging.Logger = None, timeout: int = None) -> subprocess.CompletedProcess:
    """Run a command with logging."""
    if logger:
        logger.debug(f"Running: {' '.join(cmd)}")
    
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if logger:
            if result.stdout:
                logger.debug(f"STDOUT: {result.stdout[-1000:]}")  # Last 1000 chars
            if result.returncode != 0 and result.stderr:
                logger.error(f"STDERR: {result.stderr[-1000:]}")
        
        return result
    except subprocess.TimeoutExpired:
        if logger:
            logger.error(f"Command timed out after {timeout}s")
        raise


def check_gpu_memory() -> Dict[str, Any]:
    """Check available GPU memory."""
    if not TORCH_AVAILABLE:
        return {"available": False}
    
    if not torch.cuda.is_available():
        return {"available": False}
    
    try:
        gpu_count = torch.cuda.device_count()
        gpus = []
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            free_memory = torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
            total_memory = props.total_memory
            
            gpus.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": total_memory / (1024**3),
                "free_memory_gb": free_memory / (1024**3),
            })
        
        return {"available": True, "gpus": gpus}
    except Exception as e:
        return {"available": False, "error": str(e)}


def detect_input_format(folder: Path) -> Dict[str, Any]:
    """Detect the format of input data in a folder."""
    result = {
        "has_images": False,
        "has_transforms_json": False,
        "has_colmap_sparse": False,
        "image_count": 0,
        "image_folder": None,
        "format": "unknown"
    }
    
    # Check for images folder
    images_folder = folder / "images"
    if images_folder.exists():
        result["image_folder"] = images_folder
        images = list(images_folder.glob("*.png")) + list(images_folder.glob("*.jpg"))
        result["image_count"] = len(images)
        result["has_images"] = len(images) > 0
    else:
        # Check root folder for images
        images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
        if images:
            result["image_folder"] = folder
            result["image_count"] = len(images)
            result["has_images"] = True
    
    # Check for transforms.json (NeRF/3DGS format)
    transforms_path = folder / "transforms.json"
    if transforms_path.exists():
        result["has_transforms_json"] = True
    
    # Check for COLMAP sparse folder
    sparse_folder = folder / "sparse" / "0"
    if (sparse_folder / "cameras.bin").exists() or (sparse_folder / "cameras.txt").exists():
        result["has_colmap_sparse"] = True
    
    # Determine format
    if result["has_transforms_json"]:
        result["format"] = "transforms_json"
    elif result["has_colmap_sparse"]:
        result["format"] = "colmap"
    elif result["has_images"]:
        result["format"] = "images_only"
    
    return result


# =============================================================================
# COLMAP PROCESSING
# =============================================================================

class ColmapProcessor:
    """Handles COLMAP sparse reconstruction."""
    
    def __init__(self, config: ColmapConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.colmap_path = find_executable(config.binary_path)
        
        if not self.colmap_path:
            self.logger.warning("COLMAP not found in PATH. Install or set binary_path.")
    
    def is_available(self) -> bool:
        """Check if COLMAP is available."""
        return self.colmap_path is not None
    
    def process(self, input_folder: Path, output_folder: Path) -> bool:
        """Run COLMAP reconstruction."""
        if not self.is_available():
            self.logger.error("COLMAP not available")
            return False
        
        # Create output structure
        database_path = output_folder / "database.db"
        sparse_path = output_folder / "sparse"
        sparse_path.mkdir(parents=True, exist_ok=True)
        
        # Detect image folder
        input_info = detect_input_format(input_folder)
        image_path = input_info["image_folder"]
        
        if not image_path or not input_info["has_images"]:
            self.logger.error(f"No images found in {input_folder}")
            return False
        
        self.logger.info(f"Processing {input_info['image_count']} images with COLMAP")
        
        # 1. Feature extraction
        self.logger.info("Step 1/3: Feature extraction...")
        cmd = [
            self.colmap_path, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--ImageReader.camera_model", self.config.camera_model,
            "--ImageReader.single_camera", "1" if self.config.single_camera else "0",
            "--SiftExtraction.use_gpu", "1",
            "--SiftExtraction.gpu_index", self.config.gpu_index,
        ]
        
        result = run_command(cmd, logger=self.logger, timeout=3600)
        if result.returncode != 0:
            self.logger.error("Feature extraction failed")
            return False
        
        # 2. Feature matching
        self.logger.info("Step 2/3: Feature matching...")
        if self.config.matcher == "exhaustive":
            cmd = [
                self.colmap_path, "exhaustive_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "1",
                "--SiftMatching.gpu_index", self.config.gpu_index,
            ]
        elif self.config.matcher == "sequential":
            cmd = [
                self.colmap_path, "sequential_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "1",
            ]
        elif self.config.matcher == "vocab_tree":
            cmd = [
                self.colmap_path, "vocab_tree_matcher",
                "--database_path", str(database_path),
                "--VocabTreeMatching.vocab_tree_path", self.config.vocab_tree_path,
                "--SiftMatching.use_gpu", "1",
            ]
        else:
            self.logger.error(f"Unknown matcher: {self.config.matcher}")
            return False
        
        result = run_command(cmd, logger=self.logger, timeout=7200)
        if result.returncode != 0:
            self.logger.error("Feature matching failed")
            return False
        
        # 3. Sparse reconstruction
        self.logger.info("Step 3/3: Sparse reconstruction...")
        cmd = [
            self.colmap_path, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--output_path", str(sparse_path),
            "--Mapper.ba_refine_focal_length", "1" if self.config.ba_refine_focal_length else "0",
            "--Mapper.ba_refine_principal_point", "1" if self.config.ba_refine_principal_point else "0",
            "--Mapper.ba_refine_extra_params", "1" if self.config.ba_refine_extra_params else "0",
        ]
        
        result = run_command(cmd, logger=self.logger, timeout=7200)
        if result.returncode != 0:
            self.logger.error("Sparse reconstruction failed")
            return False
        
        # Verify output
        model_path = sparse_path / "0"
        if not model_path.exists():
            self.logger.error("No reconstruction model created")
            return False
        
        self.logger.info("COLMAP processing complete")
        return True
    
    def convert_to_transforms_json(self, colmap_folder: Path, output_path: Path) -> bool:
        """Convert COLMAP output to transforms.json format."""
        # This would use the COLMAP model reader
        # For now, we assume the training script handles COLMAP format directly
        self.logger.info("COLMAP to transforms.json conversion not implemented - using COLMAP format directly")
        return True


# =============================================================================
# 3DGS TRAINING
# =============================================================================

class GaussianSplatTrainer:
    """Handles Gaussian Splatting training."""
    
    def __init__(self, config: TrainingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._detect_installation()
    
    def _detect_installation(self):
        """Detect available 3DGS installations."""
        self.available_implementations = []
        
        # Check original 3DGS
        if self.config.gaussian_splatting_path:
            gs_path = Path(self.config.gaussian_splatting_path)
        else:
            # Common locations
            possible_paths = [
                Path.home() / "gaussian-splatting",
                Path.home() / "repos" / "gaussian-splatting",
                Path("/opt/gaussian-splatting"),
                Path("./gaussian-splatting"),
            ]
            gs_path = None
            for p in possible_paths:
                if (p / "train.py").exists():
                    gs_path = p
                    break
        
        if gs_path and (gs_path / "train.py").exists():
            self.original_gs_path = gs_path
            self.available_implementations.append("original")
            self.logger.info(f"Found original 3DGS at {gs_path}")
        else:
            self.original_gs_path = None
        
        # Check gsplat
        try:
            import gsplat
            self.available_implementations.append("gsplat")
            self.logger.info("Found gsplat installation")
        except ImportError:
            pass
        
        # Check nerfstudio
        if find_executable("ns-train"):
            self.available_implementations.append("nerfstudio")
            self.logger.info("Found nerfstudio installation")
    
    def is_available(self) -> bool:
        """Check if any training implementation is available."""
        return len(self.available_implementations) > 0
    
    def get_implementation(self) -> str:
        """Get the implementation to use."""
        if self.config.implementation in self.available_implementations:
            return self.config.implementation
        elif self.available_implementations:
            return self.available_implementations[0]
        return None
    
    def train(self, data_folder: Path, output_folder: Path, model_name: str = "model") -> bool:
        """Run 3DGS training."""
        impl = self.get_implementation()
        
        if impl is None:
            self.logger.error("No 3DGS implementation available")
            return False
        
        self.logger.info(f"Training with {impl} implementation")
        
        if impl == "original":
            return self._train_original(data_folder, output_folder, model_name)
        elif impl == "nerfstudio":
            return self._train_nerfstudio(data_folder, output_folder, model_name)
        elif impl == "gsplat":
            return self._train_gsplat(data_folder, output_folder, model_name)
        
        return False
    
    def _train_original(self, data_folder: Path, output_folder: Path, model_name: str) -> bool:
        """Train using original 3DGS implementation."""
        if not self.original_gs_path:
            self.logger.error("Original 3DGS path not set")
            return False
        
        train_script = self.original_gs_path / "train.py"
        model_output = output_folder / model_name
        
        cmd = [
            sys.executable, str(train_script),
            "-s", str(data_folder),
            "-m", str(model_output),
            "--iterations", str(self.config.iterations),
            "--densify_from_iter", str(self.config.densify_from_iter),
            "--densify_until_iter", str(self.config.densify_until_iter),
            "--densify_grad_threshold", str(self.config.densify_grad_threshold),
            "--position_lr_init", str(self.config.position_lr_init),
            "--position_lr_final", str(self.config.position_lr_final),
            "--feature_lr", str(self.config.feature_lr),
            "--opacity_lr", str(self.config.opacity_lr),
            "--scaling_lr", str(self.config.scaling_lr),
            "--rotation_lr", str(self.config.rotation_lr),
        ]
        
        if self.config.white_background:
            cmd.append("--white_background")
        
        if self.config.resolution > 0:
            cmd.extend(["--resolution", str(self.config.resolution)])
        
        # Add save iterations
        save_iters = ",".join(str(i) for i in self.config.save_iterations)
        cmd.extend(["--save_iterations", save_iters])
        
        test_iters = ",".join(str(i) for i in self.config.test_iterations)
        cmd.extend(["--test_iterations", test_iters])
        
        # Set GPU
        env = {"CUDA_VISIBLE_DEVICES": str(self.config.gpu_id)}
        
        self.logger.info(f"Starting training: {self.config.iterations} iterations")
        
        # Run training (can take hours)
        result = run_command(cmd, cwd=str(self.original_gs_path), env=env, 
                           logger=self.logger, timeout=86400)  # 24h timeout
        
        if result.returncode != 0:
            self.logger.error("Training failed")
            return False
        
        # Verify output
        final_ply = model_output / "point_cloud" / f"iteration_{self.config.iterations}" / "point_cloud.ply"
        if not final_ply.exists():
            # Check for last save iteration
            for save_iter in reversed(self.config.save_iterations):
                check_ply = model_output / "point_cloud" / f"iteration_{save_iter}" / "point_cloud.ply"
                if check_ply.exists():
                    self.logger.info(f"Training completed at iteration {save_iter}")
                    return True
            
            self.logger.error("No output point cloud found")
            return False
        
        self.logger.info("Training complete")
        return True
    
    def _train_nerfstudio(self, data_folder: Path, output_folder: Path, model_name: str) -> bool:
        """Train using nerfstudio's splatfacto."""
        # Detect data format
        input_info = detect_input_format(data_folder)
        
        # nerfstudio needs specific data format
        if input_info["format"] == "transforms_json":
            data_parser = "nerfstudio-data"
        elif input_info["format"] == "colmap":
            data_parser = "colmap"
        else:
            self.logger.error(f"Unsupported data format for nerfstudio: {input_info['format']}")
            return False
        
        cmd = [
            "ns-train", self.config.nerfstudio_method,
            "--data", str(data_folder),
            "--output-dir", str(output_folder),
            "--experiment-name", model_name,
            "--max-num-iterations", str(self.config.iterations),
            "--pipeline.datamanager.dataparser", data_parser,
        ]
        
        if self.config.white_background:
            cmd.extend(["--pipeline.model.background-color", "white"])
        
        env = {"CUDA_VISIBLE_DEVICES": str(self.config.gpu_id)}
        
        self.logger.info(f"Starting nerfstudio training: {self.config.iterations} iterations")
        
        result = run_command(cmd, env=env, logger=self.logger, timeout=86400)
        
        if result.returncode != 0:
            self.logger.error("Nerfstudio training failed")
            return False
        
        self.logger.info("Nerfstudio training complete")
        return True
    
    def _train_gsplat(self, data_folder: Path, output_folder: Path, model_name: str) -> bool:
        """Train using gsplat library."""
        # gsplat requires custom training script
        # This is a placeholder - gsplat is more of a library than a training pipeline
        self.logger.warning("gsplat direct training not implemented - use original 3DGS or nerfstudio")
        return False


# =============================================================================
# PIPELINE MANAGER
# =============================================================================

@dataclass
class JobStatus:
    """Status of a pipeline job."""
    folder: str
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    colmap_status: str = "pending"
    training_status: str = "pending"
    error_message: str = ""
    retries: int = 0


class PipelineManager:
    """Manages the full processing pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = setup_logging(config)
        
        # Initialize processors
        self.colmap = ColmapProcessor(config.colmap, self.logger)
        self.trainer = GaussianSplatTrainer(config.training, self.logger)
        
        # Job tracking
        self.jobs: Dict[str, JobStatus] = {}
        self.completed_count = 0
        self.failed_count = 0
        
        # Load checkpoint if exists
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load checkpoint from previous run."""
        checkpoint_path = Path(self.config.output_base) / self.config.checkpoint_file
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    data = json.load(f)
                    for folder, status_dict in data.get("jobs", {}).items():
                        self.jobs[folder] = JobStatus(**status_dict)
                self.logger.info(f"Loaded checkpoint with {len(self.jobs)} jobs")
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint: {e}")
    
    def _save_checkpoint(self):
        """Save current progress to checkpoint."""
        if not self.config.save_checkpoints:
            return
        
        checkpoint_path = Path(self.config.output_base) / self.config.checkpoint_file
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "jobs": {folder: asdict(status) for folder, status in self.jobs.items()}
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_folder(self, folder: Path):
        """Add a folder to the processing queue."""
        folder_str = str(folder.resolve())
        
        if folder_str in self.jobs:
            if self.jobs[folder_str].status == "completed":
                self.logger.info(f"Skipping already completed: {folder.name}")
                return
            elif self.jobs[folder_str].status == "failed" and not self.config.retry_failed:
                self.logger.info(f"Skipping failed job: {folder.name}")
                return
        
        # Validate folder
        input_info = detect_input_format(folder)
        if not input_info["has_images"]:
            self.logger.warning(f"No images found in {folder}, skipping")
            return
        
        self.jobs[folder_str] = JobStatus(folder=folder_str)
        self.logger.info(f"Added to queue: {folder.name} ({input_info['image_count']} images)")
    
    def add_folders_from_path(self, base_path: Path, recursive: bool = False):
        """Add all valid folders from a base path."""
        if not base_path.exists():
            self.logger.error(f"Path does not exist: {base_path}")
            return
        
        if base_path.is_file():
            self.logger.error(f"Expected directory, got file: {base_path}")
            return
        
        # Check if base_path itself contains images
        info = detect_input_format(base_path)
        if info["has_images"]:
            self.add_folder(base_path)
            return
        
        # Otherwise, scan subdirectories
        for item in base_path.iterdir():
            if item.is_dir():
                info = detect_input_format(item)
                if info["has_images"]:
                    self.add_folder(item)
                elif recursive:
                    self.add_folders_from_path(item, recursive=True)
    
    def process_folder(self, folder_path: str) -> bool:
        """Process a single folder through the pipeline."""
        folder = Path(folder_path)
        job = self.jobs[folder_path]
        
        job.status = "running"
        job.start_time = datetime.now().isoformat()
        self._save_checkpoint()
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Processing: {folder.name}")
        self.logger.info(f"{'='*60}")
        
        try:
            # Detect input format
            input_info = detect_input_format(folder)
            self.logger.info(f"Input format: {input_info['format']}")
            
            # Prepare output folder
            output_folder = Path(self.config.output_base) / folder.name
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Copy/link images to output
            colmap_folder = output_folder / "colmap"
            colmap_folder.mkdir(exist_ok=True)
            
            # Step 1: COLMAP processing (if needed)
            needs_colmap = (
                self.config.colmap.enabled and 
                input_info["format"] == "images_only"
            )
            
            skip_colmap = (
                self.config.colmap.skip_if_transforms_exist and
                input_info["has_transforms_json"]
            )
            
            if needs_colmap and not skip_colmap:
                job.colmap_status = "running"
                self._save_checkpoint()
                
                # Link images to colmap folder
                colmap_images = colmap_folder / "images"
                if not colmap_images.exists():
                    if input_info["image_folder"]:
                        colmap_images.symlink_to(input_info["image_folder"].resolve())
                    else:
                        colmap_images.mkdir()
                        for img in folder.glob("*.png"):
                            (colmap_images / img.name).symlink_to(img.resolve())
                        for img in folder.glob("*.jpg"):
                            (colmap_images / img.name).symlink_to(img.resolve())
                
                success = self.colmap.process(colmap_folder, colmap_folder)
                
                if not success:
                    job.colmap_status = "failed"
                    raise Exception("COLMAP processing failed")
                
                job.colmap_status = "completed"
                
                # Use COLMAP output for training
                training_data_folder = colmap_folder
            else:
                job.colmap_status = "skipped"
                training_data_folder = folder
                
                if input_info["has_transforms_json"]:
                    self.logger.info("Using existing transforms.json")
                elif input_info["has_colmap_sparse"]:
                    self.logger.info("Using existing COLMAP data")
            
            self._save_checkpoint()
            
            # Step 2: 3DGS Training
            job.training_status = "running"
            self._save_checkpoint()
            
            success = self.trainer.train(
                training_data_folder,
                output_folder,
                model_name="splat"
            )
            
            if not success:
                job.training_status = "failed"
                raise Exception("3DGS training failed")
            
            job.training_status = "completed"
            job.status = "completed"
            job.end_time = datetime.now().isoformat()
            self.completed_count += 1
            
            self.logger.info(f"Completed: {folder.name}")
            self._save_checkpoint()
            return True
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = datetime.now().isoformat()
            job.retries += 1
            self.failed_count += 1
            
            self.logger.error(f"Failed: {folder.name} - {e}")
            self._save_checkpoint()
            return False
    
    def run(self):
        """Run the pipeline on all queued folders."""
        pending_jobs = [
            folder for folder, job in self.jobs.items()
            if job.status in ("pending", "failed") and 
               (job.status != "failed" or job.retries < self.config.max_retries)
        ]
        
        if not pending_jobs:
            self.logger.info("No jobs to process")
            return
        
        self.logger.info(f"Starting pipeline with {len(pending_jobs)} jobs")
        
        # Check prerequisites
        if not self.trainer.is_available():
            self.logger.error("No 3DGS training implementation available!")
            self.logger.error("Install one of: original 3DGS, nerfstudio, or gsplat")
            return
        
        self.logger.info(f"Using {self.trainer.get_implementation()} for training")
        
        # GPU info
        gpu_info = check_gpu_memory()
        if gpu_info.get("available"):
            for gpu in gpu_info.get("gpus", []):
                self.logger.info(f"GPU {gpu['index']}: {gpu['name']} - {gpu['total_memory_gb']:.1f}GB")
        
        start_time = datetime.now()
        
        # Process sequentially (GPU memory constraint)
        for i, folder in enumerate(pending_jobs):
            self.logger.info(f"Progress: {i+1}/{len(pending_jobs)}")
            self.process_folder(folder)
        
        elapsed = datetime.now() - start_time
        
        # Summary
        self.logger.info(f"{'='*60}")
        self.logger.info("Pipeline Complete")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total time: {elapsed}")
        self.logger.info(f"Completed: {self.completed_count}")
        self.logger.info(f"Failed: {self.failed_count}")
        
        # List failed jobs
        failed = [f for f, j in self.jobs.items() if j.status == "failed"]
        if failed:
            self.logger.info("Failed jobs:")
            for f in failed:
                self.logger.info(f"  - {Path(f).name}: {self.jobs[f].error_message}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a summary report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_jobs": len(self.jobs),
                "completed": sum(1 for j in self.jobs.values() if j.status == "completed"),
                "failed": sum(1 for j in self.jobs.values() if j.status == "failed"),
                "pending": sum(1 for j in self.jobs.values() if j.status == "pending"),
            },
            "jobs": {}
        }
        
        for folder, job in self.jobs.items():
            report["jobs"][Path(folder).name] = asdict(job)
        
        return report


# =============================================================================
# CLI INTERFACE
# =============================================================================

def create_default_config(output_path: str):
    """Create a default configuration file."""
    config = PipelineConfig()
    config.input_folders = ["/path/to/your/captures"]
    config.output_base = "/path/to/output"
    
    config_dict = {
        "input_folders": config.input_folders,
        "output_base": config.output_base,
        "colmap": asdict(config.colmap),
        "training": asdict(config.training),
        "max_concurrent": config.max_concurrent,
        "retry_failed": config.retry_failed,
        "max_retries": config.max_retries,
        "log_file": config.log_file,
        "verbose": config.verbose,
        "save_checkpoints": config.save_checkpoints,
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created default config at: {output_path}")


def load_config(config_path: str) -> PipelineConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    config = PipelineConfig()
    config.input_folders = data.get("input_folders", [])
    config.output_base = data.get("output_base", "./gs_output")
    
    if "colmap" in data:
        config.colmap = ColmapConfig(**data["colmap"])
    
    if "training" in data:
        config.training = TrainingConfig(**data["training"])
    
    config.max_concurrent = data.get("max_concurrent", 1)
    config.retry_failed = data.get("retry_failed", True)
    config.max_retries = data.get("max_retries", 2)
    config.log_file = data.get("log_file", "pipeline.log")
    config.verbose = data.get("verbose", True)
    config.save_checkpoints = data.get("save_checkpoints", True)
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Gaussian Splatting Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process folders with default settings
  python gs_training_pipeline.py --input ./captures --output ./trained
  
  # Use configuration file
  python gs_training_pipeline.py --config pipeline_config.yaml
  
  # Create default configuration
  python gs_training_pipeline.py --create-config my_config.yaml
  
  # Process multiple specific folders
  python gs_training_pipeline.py --input ./obj1 ./obj2 ./obj3 --output ./trained
        """
    )
    
    parser.add_argument("--input", "-i", nargs="+", help="Input folder(s) containing captures")
    parser.add_argument("--output", "-o", help="Output base directory")
    parser.add_argument("--config", "-c", help="Path to YAML configuration file")
    parser.add_argument("--create-config", help="Create a default configuration file")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan input folders recursively")
    parser.add_argument("--implementation", choices=["original", "nerfstudio", "gsplat"],
                       help="3DGS implementation to use")
    parser.add_argument("--iterations", type=int, help="Training iterations")
    parser.add_argument("--skip-colmap", action="store_true", help="Skip COLMAP processing")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--report", action="store_true", help="Generate report only (no processing)")
    
    args = parser.parse_args()
    
    # Create default config
    if args.create_config:
        create_default_config(args.create_config)
        return
    
    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = PipelineConfig()
    
    # Override with CLI arguments
    if args.input:
        config.input_folders = args.input
    if args.output:
        config.output_base = args.output
    if args.implementation:
        config.training.implementation = args.implementation
    if args.iterations:
        config.training.iterations = args.iterations
    if args.skip_colmap:
        config.colmap.enabled = False
    if args.gpu is not None:
        config.training.gpu_id = args.gpu
        config.colmap.gpu_index = str(args.gpu)
    
    # Validate
    if not config.input_folders:
        parser.error("No input folders specified. Use --input or --config")
    
    # Initialize pipeline
    pipeline = PipelineManager(config)
    
    # Add folders
    for folder_path in config.input_folders:
        pipeline.add_folders_from_path(Path(folder_path), recursive=args.recursive)
    
    if args.report:
        # Generate report only
        report = pipeline.generate_report()
        print(json.dumps(report, indent=2))
        return
    
    # Run pipeline
    pipeline.run()
    
    # Save final report
    report = pipeline.generate_report()
    report_path = Path(config.output_base) / "pipeline_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
