"""
Training backend integration for 3DGS/NeRF frameworks.

This module provides:
- Abstract backend interface for training frameworks
- Implementations for 3DGS, Nerfstudio, and gsplat
- Custom backend support via YAML/JSON configuration
- Subprocess management with progress tracking
- Real-time output parsing and status updates

Example usage:
    from .core.training import get_available_backends, start_training

    backends = get_available_backends()
    if 'gaussian_splatting' in backends:
        backend = backends['gaussian_splatting']
        process = start_training(backend, config, progress_callback)

Custom backends:
    Place YAML/JSON config files in the addon's custom_backends/ folder.
    See example_backend.yaml for the configuration format.
"""

from .base import TrainingBackend, TrainingConfig, TrainingProgress, TrainingStatus
from .gaussian_splatting import GaussianSplattingBackend
from .gs_lightning import GSLightningBackend
from .nerfstudio import NerfstudioBackend
from .gsplat import GsplatBackend
from .process import TrainingProcess, get_running_process, start_training, stop_training
from .custom_backend import (
    CustomBackend,
    load_custom_backends,
    get_custom_backend,
    reload_custom_backends,
    get_custom_backends_dir,
    validate_backend_config,
)

# Built-in backends (lowercase IDs to match properties.py)
BUILTIN_BACKENDS = {
    'gaussian_splatting': GaussianSplattingBackend,
    'gs_lightning': GSLightningBackend,
    'nerfstudio': NerfstudioBackend,
    'gsplat': GsplatBackend,
}

# For backwards compatibility
BACKENDS = BUILTIN_BACKENDS


def get_backend(backend_id: str) -> TrainingBackend:
    """Get a training backend instance.

    Args:
        backend_id: Backend identifier (lowercase)

    Returns:
        TrainingBackend instance

    Raises:
        ValueError: If backend_id is not recognized
    """
    # Check built-in backends first
    backend_class = BUILTIN_BACKENDS.get(backend_id)
    if backend_class is not None:
        return backend_class()

    # Check custom backends
    custom_backends = load_custom_backends()
    if backend_id in custom_backends:
        return custom_backends[backend_id]

    raise ValueError(f"Unknown backend: {backend_id}")


def get_available_backends() -> dict:
    """Get dictionary of available (installed) backends.

    Returns:
        Dict mapping backend_id to TrainingBackend instance
        Only includes backends that are installed and available
    """
    available = {}

    # Check built-in backends
    for backend_id, backend_class in BUILTIN_BACKENDS.items():
        backend = backend_class()
        if backend.is_available():
            available[backend_id] = backend

    # Check custom backends
    custom_backends = load_custom_backends()
    for backend_id, backend in custom_backends.items():
        if backend.is_available():
            available[backend_id] = backend

    return available


def get_all_backends() -> dict:
    """Get dictionary of all backends (installed or not).

    Returns:
        Dict mapping backend_id to TrainingBackend instance
        Includes both built-in and custom backends
    """
    all_backends = {bid: bcls() for bid, bcls in BUILTIN_BACKENDS.items()}

    # Add custom backends
    custom_backends = load_custom_backends()
    all_backends.update(custom_backends)

    return all_backends


def get_builtin_backends() -> dict:
    """Get dictionary of built-in backends only.

    Returns:
        Dict mapping backend_id to TrainingBackend instance
    """
    return {bid: bcls() for bid, bcls in BUILTIN_BACKENDS.items()}


def get_backend_enum_items():
    """Get backend items for Blender EnumProperty.

    Returns:
        List of (id, name, description) tuples
        Includes both built-in and custom backends
    """
    items = []

    # Add built-in backends
    for backend_id, backend_class in BUILTIN_BACKENDS.items():
        backend = backend_class()
        status = "Available" if backend.is_available() else "Not Installed"
        items.append((
            backend_id,
            backend.name,
            f"{backend.description} ({status})"
        ))

    # Add custom backends
    custom_backends = load_custom_backends()
    for backend_id, backend in custom_backends.items():
        status = "Available" if backend.is_available() else "Not Installed"
        items.append((
            backend_id,
            f"{backend.name} (Custom)",
            f"{backend.description} ({status})"
        ))

    return items


def get_custom_backend_enum_items():
    """Get only custom backend items for Blender EnumProperty.

    Returns:
        List of (id, name, description) tuples for custom backends only
    """
    items = []
    custom_backends = load_custom_backends()
    for backend_id, backend in custom_backends.items():
        status = "Available" if backend.is_available() else "Not Installed"
        items.append((
            backend_id,
            backend.name,
            f"{backend.description} ({status})"
        ))
    return items


__all__ = [
    # Base classes
    'TrainingBackend',
    'TrainingConfig',
    'TrainingProgress',
    'TrainingStatus',
    'TrainingProcess',
    # Built-in backends
    'GaussianSplattingBackend',
    'GSLightningBackend',
    'NerfstudioBackend',
    'GsplatBackend',
    # Custom backend support
    'CustomBackend',
    'load_custom_backends',
    'get_custom_backend',
    'reload_custom_backends',
    'get_custom_backends_dir',
    'validate_backend_config',
    # Backend access functions
    'get_backend',
    'get_available_backends',
    'get_all_backends',
    'get_builtin_backends',
    'get_backend_enum_items',
    'get_custom_backend_enum_items',
    # Process management
    'get_running_process',
    'start_training',
    'stop_training',
    # Registry (for backwards compatibility)
    'BACKENDS',
    'BUILTIN_BACKENDS',
]
