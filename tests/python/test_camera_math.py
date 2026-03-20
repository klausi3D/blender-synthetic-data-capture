#!/usr/bin/env python3
"""Unit tests for camera math functions in gs_capture_addon/core/camera.py.

These tests cover the pure math functions (sphere sampling, ring generation,
hemisphere points) which don't require Blender context. We mock mathutils.Vector
as a simple tuple-like class so tests run without Blender.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ── Mock mathutils.Vector so camera.py can import without Blender ────────

class _MockVector:
    """Minimal Vector mock matching the subset used by camera math functions."""
    __slots__ = ("_data",)

    def __init__(self, xyz):
        self._data = tuple(float(v) for v in xyz)

    @property
    def x(self):
        return self._data[0]

    @property
    def y(self):
        return self._data[1]

    @property
    def z(self):
        return self._data[2]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def __neg__(self):
        return _MockVector((-v for v in self._data))

    def __add__(self, other):
        return _MockVector((a + b for a, b in zip(self._data, other._data)))

    def __repr__(self):
        return f"Vector({self._data})"

    @property
    def length(self):
        return math.sqrt(sum(v * v for v in self._data))

    def normalized(self):
        lng = self.length
        if lng < 1e-12:
            return _MockVector((0, 0, 0))
        return _MockVector((v / lng for v in self._data))


# Install mocks
mathutils_mock = types.ModuleType("mathutils")
mathutils_mock.Vector = _MockVector
sys.modules["mathutils"] = mathutils_mock

bpy_mock = types.ModuleType("bpy")
sys.modules["bpy"] = bpy_mock

# Now load camera module
MODULE_PATH = ROOT / "gs_capture_addon" / "core" / "camera.py"
spec = importlib.util.spec_from_file_location("camera", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module spec for {MODULE_PATH}")
camera = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera)


class _Failures:
    """Simple test framework (no pytest dependency)."""
    def __init__(self):
        self.failures = []
        self.passed = 0
        self.total = 0

    def check(self, name: str, condition: bool, detail: str = ""):
        self.total += 1
        if condition:
            self.passed += 1
        else:
            msg = f"FAIL: {name}"
            if detail:
                msg += f" — {detail}"
            self.failures.append(msg)
            print(f"  {msg}")

    def report(self) -> int:
        if self.failures:
            print(f"\n{len(self.failures)} of {self.total} checks FAILED:")
            for f in self.failures:
                print(f"  - {f}")
            return 1
        print(f"\nAll {self.total} checks passed.")
        return 0


# ── Helper ───────────────────────────────────────────────────────────────

def _vec_length(v):
    return math.sqrt(sum(c * c for c in v))


# ── Tests ────────────────────────────────────────────────────────────────


def test_fibonacci_sphere_points(F: _Failures) -> None:
    """Test Fibonacci sphere sampling."""
    for n in (1, 10, 50):
        pts = camera.fibonacci_sphere_points(n)
        F.check(f"fib({n}) count", len(pts) == n)

    # All points should be on unit sphere
    pts = camera.fibonacci_sphere_points(30)
    for i, p in enumerate(pts):
        lng = _vec_length(p)
        F.check(f"fib unit length [{i}]", abs(lng - 1.0) < 1e-6, f"length={lng}")

    # Distribution: z values should span from ~1 to ~-1
    z_vals = [p.z for p in pts]
    F.check("fib z range top", max(z_vals) > 0.9)
    F.check("fib z range bottom", min(z_vals) < -0.9)


def test_generate_ring_points(F: _Failures) -> None:
    """Test ring point generation."""
    pts = camera.generate_ring_points(8, elevation_deg=0)
    F.check("ring count", len(pts) == 8)

    # At equator (elev=0), z should be ~0
    for p in pts:
        F.check("ring equator z~0", abs(p.z) < 1e-6, f"z={p.z}")

    # At elevation 45, z should be sin(45) ≈ 0.707
    pts45 = camera.generate_ring_points(4, elevation_deg=45)
    expected_z = math.sin(math.radians(45))
    for p in pts45:
        F.check("ring elev45 z", abs(p.z - expected_z) < 1e-6)

    # All points on unit sphere
    for p in pts45:
        F.check("ring unit length", abs(_vec_length(p) - 1.0) < 1e-6)


def test_generate_multi_ring_points(F: _Failures) -> None:
    """Test multi-ring distribution."""
    pts = camera.generate_multi_ring_points(20, 3, -45, 45)
    F.check("multi-ring count", len(pts) == 20)

    # All on unit sphere
    for p in pts:
        F.check("multi-ring unit length", abs(_vec_length(p) - 1.0) < 1e-6)

    # Elevation bounds respected
    min_z = math.sin(math.radians(-45))
    max_z = math.sin(math.radians(45))
    for p in pts:
        F.check("multi-ring z bounds", min_z - 0.01 <= p.z <= max_z + 0.01)

    # Edge case: zero cameras
    pts0 = camera.generate_multi_ring_points(0, 3, -45, 45)
    F.check("multi-ring zero count", len(pts0) == 0)


def test_generate_hemisphere_points(F: _Failures) -> None:
    """Test hemisphere point generation."""
    top = camera.generate_hemisphere_points(20, top=True)
    F.check("hemisphere top count <= requested", len(top) <= 20)
    F.check("hemisphere top count > 0", len(top) > 0)
    for p in top:
        F.check("hemisphere top z>0", p.z > 0)

    bottom = camera.generate_hemisphere_points(20, top=False)
    F.check("hemisphere bottom count > 0", len(bottom) > 0)
    for p in bottom:
        F.check("hemisphere bottom z<0", p.z < 0)


def test_generate_camera_positions(F: _Failures) -> None:
    """Test the main dispatch function for all distribution types."""
    # FIBONACCI
    fib = camera.generate_camera_positions("FIBONACCI", 30)
    F.check("FIBONACCI returns points", len(fib) > 0)

    # HEMISPHERE_TOP
    top = camera.generate_camera_positions("HEMISPHERE_TOP", 20)
    F.check("HEMISPHERE_TOP returns points", len(top) > 0)

    # HEMISPHERE_BOTTOM
    bot = camera.generate_camera_positions("HEMISPHERE_BOTTOM", 20)
    F.check("HEMISPHERE_BOTTOM returns points", len(bot) > 0)

    # RING
    ring = camera.generate_camera_positions("RING", 12)
    F.check("RING count", len(ring) == 12)

    # MULTI_RING
    mr = camera.generate_camera_positions("MULTI_RING", 24, ring_count=4)
    F.check("MULTI_RING count", len(mr) == 24)

    # Unknown defaults to fibonacci
    unk = camera.generate_camera_positions("UNKNOWN", 10)
    F.check("unknown fallback returns points", len(unk) == 10)


def test_apply_hotspot_bias(F: _Failures) -> None:
    """Test hotspot-biased camera placement."""
    base_pts = camera.fibonacci_sphere_points(20)
    hotspots = [(_MockVector((1, 0, 0)), 1.0)]

    # Strength 0 → no change
    result0 = camera.apply_hotspot_bias(list(base_pts), hotspots, 0.0)
    F.check("bias 0 no extra points", len(result0) == len(base_pts))

    # Positive strength → adds extra points
    result = camera.apply_hotspot_bias(list(base_pts), hotspots, 1.0)
    F.check("bias adds points", len(result) > len(base_pts))

    # Empty hotspots → no change
    result_empty = camera.apply_hotspot_bias(list(base_pts), [], 1.0)
    F.check("empty hotspots no change", len(result_empty) == len(base_pts))


def main() -> int:
    F = _Failures()

    test_fibonacci_sphere_points(F)
    test_generate_ring_points(F)
    test_generate_multi_ring_points(F)
    test_generate_hemisphere_points(F)
    test_generate_camera_positions(F)
    test_apply_hotspot_bias(F)

    print()
    return F.report()


if __name__ == "__main__":
    raise SystemExit(main())
