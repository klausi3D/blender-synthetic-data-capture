"""Helpers for Blender asset-library batch discovery and output paths."""

from __future__ import annotations

import os
from pathlib import Path

import bpy


NO_ASSET_LIBRARY_ID = "__NONE__"
_FALLBACK_LIBRARY_ITEM = (
    NO_ASSET_LIBRARY_ID,
    "No Asset Libraries Found",
    "Add asset libraries in Blender Preferences > File Paths",
)


def list_asset_libraries(context=None) -> list[dict]:
    """Return configured asset libraries with normalized absolute paths."""
    ctx = context or getattr(bpy, "context", None)
    if ctx is None:
        return []

    preferences = getattr(ctx, "preferences", None)
    if preferences is None:
        return []

    filepaths = getattr(preferences, "filepaths", None)
    libraries = getattr(filepaths, "asset_libraries", None)
    if libraries is None:
        return []

    result = []
    for library in libraries:
        name = str(getattr(library, "name", "")).strip()
        if not name:
            continue
        raw_path = str(getattr(library, "path", "")).strip()
        resolved = bpy.path.abspath(raw_path) if raw_path else ""
        normalized = os.path.normpath(resolved) if resolved else ""
        result.append(
            {
                "name": name,
                "path": normalized,
            }
        )
    return result


def asset_library_enum_items(context=None):
    """Build EnumProperty items from configured asset libraries."""
    libraries = list_asset_libraries(context=context)
    if not libraries:
        return [_FALLBACK_LIBRARY_ITEM]

    items = []
    for library in libraries:
        label = library["name"]
        description = library["path"] or "No path configured"
        items.append((label, label, description))
    return items


def resolve_asset_library_path(library_name: str, context=None):
    """Resolve selected asset-library name to a valid directory path."""
    if not library_name or library_name == NO_ASSET_LIBRARY_ID:
        return "", "No asset library selected"

    for library in list_asset_libraries(context=context):
        if library["name"] != library_name:
            continue
        library_path = library["path"]
        if not library_path:
            return "", f"Asset library '{library_name}' has no path configured"
        if not os.path.isdir(library_path):
            return "", f"Asset library '{library_name}' path does not exist: {library_path}"
        return library_path, ""

    return "", f"Asset library '{library_name}' was not found in Blender preferences"


def discover_asset_library_blend_files(
    library_root: str,
    recursive: bool = True,
    filename_filter: str = "",
) -> list[str]:
    """Discover .blend files from an asset library root."""
    root = Path(library_root)
    if not root.is_dir():
        return []

    include_filter = str(filename_filter or "").strip().lower()
    pattern = "**/*.blend" if recursive else "*.blend"
    files = []
    for path in root.glob(pattern):
        if not path.is_file():
            continue
        name = path.name.lower()
        if include_filter and include_filter not in name:
            continue
        files.append(str(path.resolve()))

    return sorted(set(files), key=lambda value: value.lower())


def make_output_subfolder_name(
    blend_file_path: str,
    library_root: str,
    used_names: set[str] | None = None,
    fallback_index: int = 0,
) -> str:
    """Create a stable, filesystem-safe per-file output folder name."""
    blend_path = Path(blend_file_path)
    root_path = Path(library_root)
    stem_parts = []

    try:
        relative = blend_path.resolve().relative_to(root_path.resolve())
        relative_parts = list(relative.parts)
        if relative_parts:
            relative_parts[-1] = Path(relative_parts[-1]).stem
        stem_parts = relative_parts
    except Exception:
        stem_parts = [blend_path.stem]

    cleaned_parts = [_clean_path_component(part) for part in stem_parts if part]
    base_name = "__".join(part for part in cleaned_parts if part) or f"asset_{fallback_index:04d}"

    if used_names is None:
        return base_name

    candidate = base_name
    suffix = 2
    while candidate in used_names:
        candidate = f"{base_name}__{suffix:02d}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def relative_blend_path(blend_file_path: str, library_root: str) -> str:
    """Return blend file path relative to library root when possible."""
    try:
        relative = Path(blend_file_path).resolve().relative_to(Path(library_root).resolve())
        return str(relative)
    except Exception:
        return os.path.basename(blend_file_path)


def make_safe_output_name(value: str, fallback: str = "item") -> str:
    """Return a filesystem-safe folder name for capture outputs."""
    cleaned = _clean_path_component(value)
    return cleaned or fallback


def _clean_path_component(value: str) -> str:
    """Normalize folder-name segments to portable ASCII-safe text."""
    cleaned = []
    for char in str(value):
        if ord(char) < 128 and (char.isalnum() or char in {"-", "_", "."}):
            cleaned.append(char)
        else:
            cleaned.append("_")

    normalized = "".join(cleaned).strip(" ._")
    return normalized or "item"
