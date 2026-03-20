"""
Capture + Training Job Config Schema
=====================================
Dataclass-based schema for YAML job configs that drive the full pipeline:
  .blend file → multi-view capture → GS training → cleaned PLY output.

Usage:
    from tools.capture_config_schema import load_job_config
    job = load_job_config("jobs/my_job.yaml")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "CaptureExportsConfig", "CaptureConfig", "CaptureItemConfig",
    "TrainingConfig", "CleanupConfig", "JobConfig",
    "load_job_config", "job_config_to_dict", "_dict_to_dataclass",
]


# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CaptureExportsConfig:
    """Which auxiliary data to export alongside images."""
    colmap: bool = True
    colmap_binary: bool = False
    transforms_json: bool = True
    depth: bool = False
    normals: bool = False
    masks: bool = False
    mask_source: str = "ALPHA"  # ALPHA | OBJECT_INDEX


@dataclass
class CaptureConfig:
    """Settings for the multi-view capture phase."""
    cameras: int = 100
    distribution: str = "FIBONACCI"
    preset: Optional[str] = None  # e.g. "3dgs", "nerfstudio"
    engine: Optional[str] = None  # EEVEE | CYCLES (None = use preset/scene default)
    speed_preset: str = "BALANCED"
    resolution: Optional[List[int]] = None  # [width, height]
    min_elevation: Optional[float] = None
    max_elevation: Optional[float] = None
    focal_length: Optional[float] = None
    distance_multiplier: Optional[float] = None
    transparent_background: bool = True
    lighting_mode: Optional[str] = None
    material_mode: Optional[str] = None
    checkpoints: bool = True
    auto_resume: bool = True
    batch_mode: Optional[str] = None  # SCENE, COLLECTIONS, etc.
    exports: CaptureExportsConfig = field(default_factory=CaptureExportsConfig)


@dataclass
class CaptureItemConfig:
    """A single item to capture.

    Currently only ``type="collection"`` is supported by the headless
    capture script.  Per-item ``cameras`` overrides are reserved for
    future use and currently ignored (the global camera count applies).
    """
    type: str = "collection"
    name: str = ""
    cameras: Optional[int] = None  # reserved — not yet wired to runtime


@dataclass
class TrainingConfig:
    """Settings for the GS training phase."""
    enabled: bool = False
    backend: str = "original"  # original | nerfstudio | gsplat
    iterations: int = 30000
    save_iterations: List[int] = field(default_factory=lambda: [7000, 15000, 30000])
    test_iterations: Optional[List[int]] = None
    white_background: bool = True
    gpu_id: int = 0
    extra_args: Optional[str] = None

    # Paths to implementations (auto-detected if empty)
    gaussian_splatting_path: str = ""
    nerfstudio_path: str = ""
    gsplat_path: str = ""


@dataclass
class CleanupConfig:
    """Settings for the post-training cleanup phase."""
    enabled: bool = False
    proxy_hulls: bool = True
    hull_margin: float = 0.01
    max_removal_ratio: float = 0.70


@dataclass
class JobConfig:
    """Top-level job configuration for the full pipeline."""
    blender_path: str = "blender"
    scene: str = ""
    output_base: str = ""
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    items: List[CaptureItemConfig] = field(default_factory=list)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)

    # Pipeline control
    timeout_per_capture: int = 3600
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _dict_to_dataclass(cls, data: dict) -> Any:
    """Recursively populate a dataclass from a dict, ignoring unknown keys."""
    if not isinstance(data, dict):
        return data
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {}
    for k, v in data.items():
        if k not in field_names:
            continue
        f = next(f for f in dataclasses.fields(cls) if f.name == k)
        if dataclasses.is_dataclass(f.type):
            filtered[k] = _dict_to_dataclass(f.type, v)
        elif hasattr(f.type, "__origin__") and f.type.__origin__ is list:
            # Handle List[CaptureItemConfig] etc.
            args = getattr(f.type, "__args__", ())
            if args and dataclasses.is_dataclass(args[0]):
                filtered[k] = [_dict_to_dataclass(args[0], item)
                               if isinstance(item, dict) else item
                               for item in v]
            else:
                filtered[k] = v
        else:
            # Handle dataclass fields whose type annotation is a string
            # or a direct dataclass class
            actual_type = f.type if isinstance(f.type, type) else None
            if actual_type and dataclasses.is_dataclass(actual_type) and isinstance(v, dict):
                filtered[k] = _dict_to_dataclass(actual_type, v)
            else:
                filtered[k] = v
    return cls(**filtered)


def load_job_config(path: str) -> JobConfig:
    """Load a YAML (or JSON) job config file into a JobConfig dataclass."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")

    try:
        import yaml
        data = yaml.safe_load(text) or {}
    except ImportError:
        import json
        data = json.loads(text)

    return _dict_to_dataclass(JobConfig, data)


def job_config_to_dict(job: JobConfig) -> dict:
    """Serialize a JobConfig back to a plain dict (for YAML/JSON export)."""
    import dataclasses
    return dataclasses.asdict(job)
