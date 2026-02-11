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

import os
import re
from typing import Optional

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
from ...utils.paths import normalize_path

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


def _score_model_candidate(path: str) -> tuple:
    """Rank candidate model paths for fallback discovery."""
    path_lower = path.lower().replace("\\", "/")
    file_name = os.path.basename(path_lower)

    score = 0
    if file_name == "point_cloud.ply":
        score += 100
    if "/point_cloud/" in path_lower:
        score += 40
    if "/exports/" in path_lower:
        score += 30
    if "splat" in file_name:
        score += 15
    if "final" in file_name:
        score += 10

    iter_match = re.search(r"iteration[_-](\d+)", path_lower)
    iteration = int(iter_match.group(1)) if iter_match else -1

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0

    return score, iteration, mtime


def find_model_path_with_fallback(backend: Optional[TrainingBackend], output_path: str) -> Optional[str]:
    """Find a trained .ply by backend preference, then recursive fallback search.

    Args:
        backend: Selected backend instance (or None)
        output_path: Output directory from training settings/process

    Returns:
        Absolute normalized path to the selected .ply file, or None
    """
    normalized_output = normalize_path(output_path)
    if not normalized_output:
        return None

    backend_candidate = None
    if backend:
        try:
            backend_candidate = backend.get_model_path(normalized_output)
        except Exception:
            backend_candidate = None

    if backend_candidate:
        backend_candidate = normalize_path(backend_candidate)
        if os.path.isfile(backend_candidate) and backend_candidate.lower().endswith(".ply"):
            return backend_candidate

    if not os.path.isdir(normalized_output):
        return None

    search_roots = [normalized_output]
    if backend_candidate and os.path.isdir(backend_candidate):
        search_roots.insert(0, backend_candidate)

    candidates = []
    seen = set()

    for root in search_roots:
        normalized_root = normalize_path(root)
        if normalized_root in seen or not os.path.isdir(normalized_root):
            continue
        seen.add(normalized_root)

        for walk_root, _, files in os.walk(normalized_root):
            for file_name in files:
                if file_name.lower().endswith(".ply"):
                    candidates.append(os.path.join(walk_root, file_name))

    if not candidates:
        return None

    return max(candidates, key=_score_model_candidate)


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
    'find_model_path_with_fallback',
    # Process management
    'get_running_process',
    'start_training',
    'stop_training',
    # Registry (for backwards compatibility)
    'BACKENDS',
    'BUILTIN_BACKENDS',
]
