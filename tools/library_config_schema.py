"""
Library Batch Capture Config Schema
====================================
Dataclass schema for YAML configs that drive library-level batch processing:
scan a folder of .blend files → introspect → capture → train → cleanup.

Usage:
    from tools.library_config_schema import load_library_config
    config = load_library_config("jobs/my_library.yaml")
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from tools.capture_config_schema import (
    CaptureConfig,
    TrainingConfig,
    CleanupConfig,
    _dict_to_dataclass,
)


# ---------------------------------------------------------------------------
# Library-specific dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LibraryFilterConfig:
    """Filtering rules for .blend files and collections."""
    include_files: List[str] = field(default_factory=lambda: ["*.blend"])
    exclude_files: List[str] = field(default_factory=list)
    include_collections: List[str] = field(default_factory=list)  # empty = all
    exclude_collections: List[str] = field(default_factory=list)
    min_mesh_count: int = 1


@dataclass
class LibraryConfig:
    """Top-level config for library batch processing."""
    # Source
    library_path: str = ""
    blend_files: List[str] = field(default_factory=list)  # explicit file list

    # Output
    output_base: str = ""

    # Blender
    blender_path: str = "blender"

    # Behavior
    granularity: str = "COLLECTIONS"  # COLLECTIONS | SCENE | EACH_OBJECT
    skip_existing: bool = True
    recursive: bool = True
    adaptive: bool = False
    adaptive_quality_preset: str = "AUTO"

    # Filtering
    filters: LibraryFilterConfig = field(default_factory=LibraryFilterConfig)

    # Capture defaults (passed through to generated pipeline configs)
    capture: CaptureConfig = field(default_factory=CaptureConfig)

    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Cleanup
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)

    # Execution
    max_workers: int = 1
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    timeout_per_file: int = 7200


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_library_config(path: str) -> LibraryConfig:
    """Load a YAML (or JSON) library config file."""
    config_path = Path(path)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: Cannot read config file '{path}': {exc}", file=sys.stderr)
        raise SystemExit(1)
    try:
        import yaml
        data = yaml.safe_load(text) or {}
    except ImportError:
        pass
    except Exception as exc:
        print(f"ERROR: Failed to parse YAML config '{path}': {exc}", file=sys.stderr)
        raise SystemExit(1)
    else:
        return _dict_to_dataclass(LibraryConfig, data)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Failed to parse config '{path}': {exc}", file=sys.stderr)
        raise SystemExit(1)
    return _dict_to_dataclass(LibraryConfig, data)
