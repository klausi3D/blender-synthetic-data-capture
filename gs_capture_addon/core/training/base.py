"""
Abstract base class for training backends.
Defines interface for all training framework integrations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
from enum import Enum


class TrainingStatus(Enum):
    """Training process status."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingConfig:
    """Configuration for training run."""

    # Paths
    data_path: str = ""
    output_path: str = ""

    # Training parameters
    iterations: int = 30000
    save_iterations: List[int] = field(default_factory=lambda: [7000, 15000, 30000])
    test_iterations: List[int] = field(default_factory=lambda: [7000, 15000, 30000])

    # Appearance
    white_background: bool = True
    resolution: int = -1  # -1 = auto

    # Densification (3DGS specific)
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densification_interval: int = 100

    # Performance
    gpu_id: int = 0

    # Extra arguments
    extra_args: List[str] = field(default_factory=list)


@dataclass
class TrainingProgress:
    """Training progress information."""

    status: TrainingStatus = TrainingStatus.IDLE
    iteration: int = 0
    total_iterations: int = 30000
    loss: float = 0.0
    psnr: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    message: str = ""
    error: Optional[str] = None

    @property
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.total_iterations <= 0:
            return 0.0
        return (self.iteration / self.total_iterations) * 100

    @property
    def eta_formatted(self) -> str:
        """Get ETA as formatted string."""
        if self.eta_seconds <= 0:
            return "Unknown"

        hours = int(self.eta_seconds // 3600)
        minutes = int((self.eta_seconds % 3600) // 60)
        seconds = int(self.eta_seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"


class TrainingBackend(ABC):
    """Abstract base class for training backends."""

    # Override in subclasses
    name: str = "Base Backend"
    description: str = "Abstract training backend"
    website: str = ""
    install_instructions: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is installed and available.

        Returns:
            True if backend can be used
        """
        pass

    @abstractmethod
    def get_install_path(self) -> Optional[str]:
        """Get installation path if applicable.

        Returns:
            Path string or None
        """
        pass

    @abstractmethod
    def validate_data(self, data_path: str) -> tuple:
        """Validate that data path contains required files.

        Args:
            data_path: Path to training data

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass

    @abstractmethod
    def get_command(self, config: TrainingConfig) -> List[str]:
        """Build command line for training.

        Args:
            config: Training configuration

        Returns:
            List of command arguments
        """
        pass

    @abstractmethod
    def parse_output(self, line: str) -> Optional[TrainingProgress]:
        """Parse a line of training output.

        Args:
            line: Output line from training process

        Returns:
            TrainingProgress if line contains progress info, else None
        """
        pass

    def get_default_config(self) -> TrainingConfig:
        """Get default training configuration for this backend.

        Returns:
            TrainingConfig with defaults
        """
        return TrainingConfig()

    def get_output_files(self, output_path: str, iteration: int) -> Dict[str, str]:
        """Get paths to output files for given iteration.

        Args:
            output_path: Training output directory
            iteration: Training iteration

        Returns:
            Dict with file type -> path mappings
        """
        return {}

    def get_final_model_path(self, output_path: str) -> Optional[str]:
        """Get path to final trained model.

        Args:
            output_path: Training output directory

        Returns:
            Path to model file or None
        """
        return None

    def get_model_path(self, output_path: str) -> Optional[str]:
        """Get path to trained model for downstream import workflows.

        Backends can override this when they have a preferred model file.
        The default keeps backward compatibility with existing implementations.

        Args:
            output_path: Training output directory

        Returns:
            Path to model file or None
        """
        return self.get_final_model_path(output_path)

    def cleanup(self, output_path: str) -> None:
        """Clean up temporary files after training.

        Args:
            output_path: Training output directory
        """
        pass

    def estimate_training_time(self, config: TrainingConfig) -> float:
        """Estimate training time in seconds.

        Args:
            config: Training configuration

        Returns:
            Estimated seconds
        """
        # Very rough estimate: ~0.1 seconds per iteration on RTX 3090
        return config.iterations * 0.1

    def estimate_vram_usage(self, config: TrainingConfig) -> float:
        """Estimate VRAM usage in GB.

        Args:
            config: Training configuration

        Returns:
            Estimated VRAM in GB
        """
        # Rough estimate
        return 8.0  # Base 3DGS uses ~8GB
