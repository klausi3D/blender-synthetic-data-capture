#!/usr/bin/env python3
"""Sanity tests for proxy-hull splat cleanup core module."""

from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "gs_capture_addon" / "core" / "splat_cleanup.py"

spec = importlib.util.spec_from_file_location("splat_cleanup_core", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module spec for {MODULE_PATH}")

splat_cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(splat_cleanup)


def write_proxy_hulls(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "coordinate_system": "BLENDER_WORLD",
        "object_count": 1,
        "global_bounds": {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        "objects": [
            {
                "name": "Cube",
                "planes": [
                    [1.0, 0.0, 0.0, -1.0],
                    [-1.0, 0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0, -1.0],
                    [0.0, -1.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0, -1.0],
                    [0.0, 0.0, -1.0, -1.0],
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError("Unexpected EOF in PLY header")
            decoded = line.decode("ascii").strip()
            if decoded.startswith("element vertex "):
                return int(decoded.split()[-1])
            if decoded == "end_header":
                break
    raise RuntimeError("No vertex element in PLY header")


def read_ascii_xyz_points(path: Path) -> list[tuple[float, float, float]]:
    with path.open("r", encoding="ascii") as handle:
        in_body = False
        points = []
        for raw_line in handle:
            line = raw_line.strip()
            if not in_body:
                if line == "end_header":
                    in_body = True
                continue
            if not line:
                continue
            parts = line.split()
            points.append((float(parts[0]), float(parts[1]), float(parts[2])))
        return points


def read_binary_xyz_points(path: Path) -> list[tuple[float, float, float]]:
    with path.open("rb") as handle:
        vertex_count = None
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError("Unexpected EOF in binary PLY header")
            decoded = line.decode("ascii").strip()
            if decoded.startswith("element vertex "):
                vertex_count = int(decoded.split()[-1])
            if decoded == "end_header":
                break
        if vertex_count is None:
            raise RuntimeError("No vertex count in binary PLY header")
        points = []
        for _ in range(vertex_count):
            row = handle.read(16)  # x, y, z, opacity
            if len(row) != 16:
                raise RuntimeError("Unexpected EOF in binary PLY vertex rows")
            x, y, z, _opacity = struct.unpack("<ffff", row)
            points.append((x, y, z))
        return points


def test_ascii_cleanup(tmpdir: Path) -> None:
    proxy_hull_path = tmpdir / "capture" / "cleanup" / "proxy_hulls.json"
    write_proxy_hulls(proxy_hull_path)

    input_ply = tmpdir / "model_ascii.ply"
    input_ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 5",
                "property float x",
                "property float y",
                "property float z",
                "property float opacity",
                "property uchar red",
                "end_header",
                "0.0 0.0 0.0 0.50 120",
                "0.6 0.1 0.2 0.20 10",
                "1.8 0.0 0.0 0.10 255",
                "0.2 -0.4 0.7 0.90 80",
                "0.0 0.0 -1.6 0.30 30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_ply = tmpdir / "model_ascii.cleaned.ply"
    report_path = tmpdir / "cleanup_report_ascii.json"

    report = splat_cleanup.run_proxy_hull_cleanup(
        model_path=str(input_ply),
        proxy_hull_path=str(proxy_hull_path),
        hull_margin=0.0,
        max_removal_ratio=0.9,
        output_path=str(output_ply),
        report_path=str(report_path),
    )

    assert report["removed_points"] == 2
    assert report["kept_points"] == 3
    assert output_ply.exists()
    assert report_path.exists()
    assert read_vertex_count(output_ply) == 3
    kept_points = read_ascii_xyz_points(output_ply)
    assert (1.8, 0.0, 0.0) not in kept_points
    assert (0.0, 0.0, -1.6) not in kept_points
    assert (0.0, 0.0, 0.0) in kept_points


def test_binary_cleanup(tmpdir: Path) -> None:
    proxy_hull_path = tmpdir / "capture" / "cleanup" / "proxy_hulls.json"
    write_proxy_hulls(proxy_hull_path)

    input_ply = tmpdir / "model_binary.ply"
    rows = [
        (0.0, 0.0, 0.0, 0.5),
        (0.5, 0.5, 0.5, 0.3),
        (1.4, 0.0, 0.0, 0.2),
        (0.0, 0.0, -1.3, 0.1),
    ]
    with input_ply.open("wb") as handle:
        handle.write(
            "\n".join(
                [
                    "ply",
                    "format binary_little_endian 1.0",
                    "element vertex 4",
                    "property float x",
                    "property float y",
                    "property float z",
                    "property float opacity",
                    "end_header",
                    "",
                ]
            ).encode("ascii")
        )
        for row in rows:
            handle.write(struct.pack("<ffff", *row))

    output_ply = tmpdir / "model_binary.cleaned.ply"
    report = splat_cleanup.run_proxy_hull_cleanup(
        model_path=str(input_ply),
        proxy_hull_path=str(proxy_hull_path),
        hull_margin=0.0,
        max_removal_ratio=0.9,
        output_path=str(output_ply),
        report_path=str(tmpdir / "cleanup_report_binary.json"),
    )

    assert report["removed_points"] == 2
    assert report["kept_points"] == 2
    assert output_ply.exists()
    assert read_vertex_count(output_ply) == 2
    kept_points = read_binary_xyz_points(output_ply)
    rounded = {(round(x, 3), round(y, 3), round(z, 3)) for x, y, z in kept_points}
    assert rounded == {(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)}


def test_removal_ratio_guard(tmpdir: Path) -> None:
    proxy_hull_path = tmpdir / "capture" / "cleanup" / "proxy_hulls.json"
    write_proxy_hulls(proxy_hull_path)

    input_ply = tmpdir / "guard_case.ply"
    input_ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0.0 0.0 0.0",
                "1.4 0.0 0.0",
                "0.0 1.5 0.0",
                "0.0 0.0 1.6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        splat_cleanup.run_proxy_hull_cleanup(
            model_path=str(input_ply),
            proxy_hull_path=str(proxy_hull_path),
            hull_margin=0.0,
            max_removal_ratio=0.2,
            output_path=str(tmpdir / "guard_case.cleaned.ply"),
            report_path=str(tmpdir / "cleanup_report_guard.json"),
        )
    except RuntimeError:
        return
    raise AssertionError("Expected RuntimeError for removal ratio guard")


def test_non_vertex_element_rejected(tmpdir: Path) -> None:
    proxy_hull_path = tmpdir / "capture" / "cleanup" / "proxy_hulls.json"
    write_proxy_hulls(proxy_hull_path)

    input_ply = tmpdir / "mesh_like.ply"
    input_ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "3 0 1 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        splat_cleanup.run_proxy_hull_cleanup(
            model_path=str(input_ply),
            proxy_hull_path=str(proxy_hull_path),
            output_path=str(tmpdir / "mesh_like.cleaned.ply"),
            report_path=str(tmpdir / "cleanup_report_mesh.json"),
        )
    except splat_cleanup.PlyFormatError:
        return
    raise AssertionError("Expected PlyFormatError for non-vertex elements")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="gs_splat_cleanup_test_") as temp_dir:
        tmpdir = Path(temp_dir)
        test_ascii_cleanup(tmpdir)
        test_binary_cleanup(tmpdir)
        test_removal_ratio_guard(tmpdir)
        test_non_vertex_element_rejected(tmpdir)

    print("OK: splat cleanup core tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
