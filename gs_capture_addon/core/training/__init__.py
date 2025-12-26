"""
Training backend integration for 3DGS/NeRF frameworks.

This module provides:
- Abstract backend interface for training frameworks
- Implementations for 3DGS, Nerfstudio, and gsplat
- Subprocess management with progress tracking
- Real-time output parsing and status updates

Example usage:
    from .core.training import get_available_backends, start_training

    backends = get_available_backends()
    if 'gaussian_splatting' in backends:
        backend = backends['gaussian_splatting']
        process = start_training(backend, config, progress_callback)
"""

from .base import TrainingBackend, TrainingConfig, TrainingProgress, TrainingStatus
from .gaussian_splatting import GaussianSplattingBackend
from .nerfstudio import NerfstudioBackend
from .gsplat import GsplatBackend
from .process import TrainingProcess, get_running_process, start_training, stop_training

# Available backends (lowercase IDs to match properties.py)
BACKENDS = {
    'gaussian_splatting': GaussianSplattingBackend,
    'nerfstudio': NerfstudioBackend,
    'gsplat': GsplatBackend,
}


def get_backend(backend_id: str) -> TrainingBackend:
    """Get a training backend instance.

    Args:
        backend_id: Backend identifier (lowercase)

    Returns:
        TrainingBackend instance

    Raises:
        ValueError: If backend_id is not recognized
    """
    backend_class = BACKENDS.get(backend_id)
    if backend_class is None:
        raise ValueError(f"Unknown backend: {backend_id}")
    return backend_class()


def get_available_backends() -> dict:
    """Get dictionary of available (installed) backends.

    Returns:
        Dict mapping backend_id to TrainingBackend instance
        Only includes backends that are installed and available
    """
    available = {}
    for backend_id, backend_class in BACKENDS.items():
        backend = backend_class()
        if backend.is_available():
            available[backend_id] = backend
    return available


def get_all_backends() -> dict:
    """Get dictionary of all backends (installed or not).

    Returns:
        Dict mapping backend_id to TrainingBackend instance
    """
    return {bid: bcls() for bid, bcls in BACKENDS.items()}


def get_backend_enum_items():
    """Get backend items for Blender EnumProperty.

    Returns:
        List of (id, name, description) tuples
    """
    items = []
    for backend_id, backend_class in BACKENDS.items():
        backend = backend_class()
        status = "Available" if backend.is_available() else "Not Installed"
        items.append((
            backend_id,
            backend.name,
            f"{backend.description} ({status})"
        ))
    return items


__all__ = [
    'TrainingBackend',
    'TrainingConfig',
    'TrainingProgress',
    'TrainingStatus',
    'TrainingProcess',
    'GaussianSplattingBackend',
    'NerfstudioBackend',
    'GsplatBackend',
    'get_backend',
    'get_available_backends',
    'get_all_backends',
    'get_backend_enum_items',
    'get_running_process',
    'start_training',
    'stop_training',
    'BACKENDS',
]
