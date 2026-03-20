"""Realistic 3D Gaussian Splatting PLY generator for cleanup tests.

Generates binary or ASCII PLY files with full 3DGS property sets
(14 or 17 properties per vertex). All fixtures are generated at runtime
to avoid committing large files to the repository.

No external dependencies — uses only Python stdlib (struct, math, random).
Inspired by klausi3D/godotGS test_data generators.
"""

from __future__ import annotations

import json
import math
import os
import random
import struct


# ── PLY property definitions ────────────────────────────────────────────

PROPERTIES_WITH_NORMALS = [
    "x", "y", "z",
    "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]

PROPERTIES_WITHOUT_NORMALS = [
    "x", "y", "z",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]


# ── Math helpers ────────────────────────────────────────────────────────

def _logit(p):
    """Convert linear probability to logit space: log(p / (1-p))."""
    p = max(1e-6, min(1.0 - 1e-6, p))
    return math.log(p / (1.0 - p))


def _normalize_quaternion(q):
    """Normalize a 4-tuple quaternion to unit length."""
    length = math.sqrt(sum(v * v for v in q))
    if length < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(v / length for v in q)


def _fibonacci_sphere_points(n, rng, radius, noise_amplitude, clamp):
    """Generate n points on a sphere using Fibonacci spiral sampling.

    Returns list of (x, y, z) tuples.
    """
    phi = math.pi * (3.0 - math.sqrt(5.0))  # Golden angle
    points = []

    for i in range(n):
        z = 1.0 - (i / float(n - 1)) * 2.0 if n > 1 else 0.0
        r = math.sqrt(1.0 - z * z)
        theta = phi * i

        x = math.cos(theta) * r * radius
        y = math.sin(theta) * r * radius
        z_pos = z * radius

        # Add uniform noise
        x += rng.uniform(-noise_amplitude, noise_amplitude)
        y += rng.uniform(-noise_amplitude, noise_amplitude)
        z_pos += rng.uniform(-noise_amplitude, noise_amplitude)

        # Clamp to ensure inside hull
        x = max(-clamp, min(clamp, x))
        y = max(-clamp, min(clamp, y))
        z_pos = max(-clamp, min(clamp, z_pos))

        points.append((x, y, z_pos))

    return points


def _generate_outlier_points(n, rng, outlier_min, outlier_max, inner_range=5.0):
    """Generate n outlier points guaranteed to be outside a cube hull.

    Each outlier has at least one coordinate with absolute value > outlier_min,
    which exceeds the hull boundary. Other coordinates are random in [-inner_range, inner_range].
    """
    points = []
    for _ in range(n):
        coords = [
            rng.uniform(-inner_range, inner_range),
            rng.uniform(-inner_range, inner_range),
            rng.uniform(-inner_range, inner_range),
        ]
        # Pick a random axis and force it outside
        axis = rng.randint(0, 2)
        sign = 1.0 if rng.random() > 0.5 else -1.0
        coords[axis] = sign * rng.uniform(outlier_min, outlier_max)

        points.append(tuple(coords))

    return points


def _make_vertex(rng, x, y, z, include_normals):
    """Create a full vertex tuple with realistic 3DGS property values."""
    # Normal: radial direction from origin (or random if near origin)
    length = math.sqrt(x * x + y * y + z * z)
    if length > 1e-6:
        nx, ny, nz = x / length, y / length, z / length
    else:
        nx, ny, nz = 0.0, 0.0, 1.0

    # SH DC coefficients (realistic range)
    f_dc_0 = rng.uniform(-0.5, 0.5)
    f_dc_1 = rng.uniform(-0.5, 0.5)
    f_dc_2 = rng.uniform(-0.5, 0.5)

    # Opacity in logit space (linear p in [0.3, 0.99])
    opacity = _logit(rng.uniform(0.3, 0.99))

    # Scale in log space (linear s in [0.001, 0.05])
    scale_0 = math.log(rng.uniform(0.001, 0.05))
    scale_1 = math.log(rng.uniform(0.001, 0.05))
    scale_2 = math.log(rng.uniform(0.001, 0.05))

    # Rotation quaternion (normalized random)
    rot = _normalize_quaternion((
        rng.uniform(-1, 1),
        rng.uniform(-1, 1),
        rng.uniform(-1, 1),
        rng.uniform(-1, 1),
    ))

    if include_normals:
        return (x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2,
                opacity, scale_0, scale_1, scale_2,
                rot[0], rot[1], rot[2], rot[3])
    else:
        return (x, y, z, f_dc_0, f_dc_1, f_dc_2,
                opacity, scale_0, scale_1, scale_2,
                rot[0], rot[1], rot[2], rot[3])


# ── Public API ──────────────────────────────────────────────────────────

def generate_sphere_with_outliers(
    output_path,
    *,
    fmt="binary_little_endian",
    sphere_count=1000,
    outlier_count=200,
    sphere_radius=5.0,
    outlier_min=7.0,
    outlier_max=15.0,
    seed=42,
    include_normals=True,
):
    """Generate a PLY file with sphere points + outliers for cleanup testing.

    Sphere points are distributed on a sphere surface with small noise,
    clamped to stay inside a cube hull at ±6. Outlier points are placed
    with at least one coordinate exceeding the hull boundary.

    Args:
        output_path: Path to write the PLY file.
        fmt: PLY format — "binary_little_endian" or "ascii".
        sphere_count: Number of in-bounds sphere points.
        outlier_count: Number of out-of-bounds outlier points.
        sphere_radius: Radius of the sphere distribution.
        outlier_min: Minimum outlier coordinate magnitude on forced axis.
        outlier_max: Maximum outlier coordinate magnitude on forced axis.
        seed: RNG seed for deterministic output.
        include_normals: If True, include nx/ny/nz (17 props), else 14 props.

    Returns:
        dict with metadata: total_points, sphere_count, outlier_count,
        property_count, bytes_per_vertex (binary only), format.
    """
    rng = random.Random(seed)

    properties = PROPERTIES_WITH_NORMALS if include_normals else PROPERTIES_WITHOUT_NORMALS
    prop_count = len(properties)

    # Generate point positions
    sphere_pts = _fibonacci_sphere_points(
        sphere_count, rng, sphere_radius, noise_amplitude=0.3, clamp=5.8,
    )
    outlier_pts = _generate_outlier_points(
        outlier_count, rng, outlier_min, outlier_max,
    )

    # Build vertex data — sphere first, then outliers
    vertices = []
    for x, y, z in sphere_pts:
        vertices.append(_make_vertex(rng, x, y, z, include_normals))
    for x, y, z in outlier_pts:
        vertices.append(_make_vertex(rng, x, y, z, include_normals))

    total = len(vertices)

    # Write PLY file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    header_lines = [
        "ply",
        f"format {fmt} 1.0",
        f"element vertex {total}",
    ]
    for prop_name in properties:
        header_lines.append(f"property float {prop_name}")
    header_lines.append("end_header")
    header_text = "\n".join(header_lines) + "\n"

    if fmt == "binary_little_endian":
        struct_fmt = f"<{prop_count}f"
        with open(output_path, "wb") as f:
            f.write(header_text.encode("ascii"))
            for vertex in vertices:
                f.write(struct.pack(struct_fmt, *vertex))
    elif fmt == "ascii":
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header_text)
            for vertex in vertices:
                f.write(" ".join(f"{v:.8g}" for v in vertex) + "\n")
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return {
        "total_points": total,
        "sphere_count": sphere_count,
        "outlier_count": outlier_count,
        "property_count": prop_count,
        "bytes_per_vertex": prop_count * 4 if fmt == "binary_little_endian" else None,
        "format": fmt,
    }


def write_cube_proxy_hull(output_path, half_extent=6.0):
    """Write a proxy_hulls.json cube centered at origin.

    The cube extends from -half_extent to +half_extent on all axes.
    Uses the standard plane convention: [nx, ny, nz, d] where a point
    is inside when nx*x + ny*y + nz*z + d <= 0.
    """
    he = half_extent
    payload = {
        "schema_version": 1,
        "coordinate_system": "BLENDER_WORLD",
        "object_count": 1,
        "global_bounds": {
            "min": [-he, -he, -he],
            "max": [he, he, he],
        },
        "objects": [
            {
                "name": "TestCube",
                "planes": [
                    [1.0, 0.0, 0.0, -he],
                    [-1.0, 0.0, 0.0, -he],
                    [0.0, 1.0, 0.0, -he],
                    [0.0, -1.0, 0.0, -he],
                    [0.0, 0.0, 1.0, -he],
                    [0.0, 0.0, -1.0, -he],
                ],
            }
        ],
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
