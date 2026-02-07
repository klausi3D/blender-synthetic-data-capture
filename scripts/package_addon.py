#!/usr/bin/env python3
"""
Package the GS Capture addon as a Blender-installable zip.

Usage:
  python scripts/package_addon.py
  python scripts/package_addon.py --out dist/gs_capture_addon-2.2.1.zip
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path


EXCLUDE_DIRS = {"__pycache__", ".git", ".idea", ".vscode"}
EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".blend",
    ".blend1",
    ".blend2",
    ".tmp",
    ".temp",
    ".log",
}


def read_version(init_path: Path) -> str:
    text = init_path.read_text(encoding="utf-8")
    match = re.search(r'"version"\\s*:\\s*\\((\\d+),\\s*(\\d+),\\s*(\\d+)\\)', text)
    if not match:
        raise RuntimeError(f"Could not find version tuple in {init_path}")
    major, minor, patch = match.groups()
    return f"{major}.{minor}.{patch}"


def should_exclude(path: Path) -> bool:
    if path.suffix in EXCLUDE_EXTS:
        return True
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    return False


def package_addon(root: Path, out_path: Path) -> None:
    addon_dir = root / "gs_capture_addon"
    shim = root / "gs_capture_addon.py"

    if not addon_dir.is_dir():
        raise RuntimeError(f"Addon directory not found: {addon_dir}")
    if not shim.is_file():
        raise RuntimeError(f"Addon shim not found: {shim}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(shim, "gs_capture_addon.py")
        for path in addon_dir.rglob("*"):
            if path.is_dir() or should_exclude(path):
                continue
            arcname = path.relative_to(root).as_posix()
            zf.write(path, arcname)

    print(f"Created {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the GS Capture addon")
    parser.add_argument("--out", help="Output zip path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    version = read_version(root / "gs_capture_addon" / "__init__.py")
    default_out = root / "dist" / f"gs_capture_addon-{version}.zip"
    out_path = Path(args.out) if args.out else default_out

    package_addon(root, out_path)


if __name__ == "__main__":
    main()
