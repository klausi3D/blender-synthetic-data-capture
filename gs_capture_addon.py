# SPDX-License-Identifier: MIT
# Compatibility shim for the GS Capture addon entry point.

"""
Compatibility shim for the legacy single-file addon entry point.

This module delegates to the package `gs_capture_addon` to avoid duplicate
class definitions while keeping existing entry points working.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_PKG_DIR = os.path.join(os.path.dirname(__file__), "gs_capture_addon")
_PKG_INIT = os.path.join(_PKG_DIR, "__init__.py")

if not os.path.isfile(_PKG_INIT):
    raise ImportError(f"gs_capture_addon package not found at {_PKG_INIT}")

# Make this module behave like the package so relative imports work.
__path__ = [_PKG_DIR]

_spec = importlib.util.spec_from_file_location(
    __name__,
    _PKG_INIT,
    submodule_search_locations=[_PKG_DIR],
)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load gs_capture_addon package spec")

_spec.loader.exec_module(sys.modules[__name__])

# Re-export addon API for Blender.
bl_info = globals()["bl_info"]
register = globals()["register"]
unregister = globals()["unregister"]
