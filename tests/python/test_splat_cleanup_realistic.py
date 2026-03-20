#!/usr/bin/env python3
"""Realistic PLY cleanup tests using runtime-generated fixtures.

Tests splat cleanup on PLY files with full 3DGS property sets (14-17
properties per vertex) to verify the cleanup works on real-world data,
not just toy 4-vertex files.

All PLY fixtures are generated at runtime — nothing committed to the repo.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ── Load modules via importlib ──────────────────────────────────────────

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


splat_cleanup = _load_module(
    "splat_cleanup",
    ROOT / "gs_capture_addon" / "core" / "splat_cleanup.py",
)

generate_ply = _load_module(
    "generate_realistic_ply",
    ROOT / "tests" / "python" / "generate_realistic_ply.py",
)


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


# ── Helpers ─────────────────────────────────────────────────────────────

def _read_ply_vertex_count(path):
    """Parse the vertex count from a PLY header."""
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            decoded = line.decode("ascii", errors="replace").strip()
            if decoded.startswith("element vertex "):
                return int(decoded.split()[-1])
            if decoded == "end_header":
                break
    return -1


def _read_ply_header_end_offset(path):
    """Return the byte offset just after 'end_header\\n'."""
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.strip() == b"end_header":
                return f.tell()
    return -1


def _read_binary_data_section(path):
    """Return the raw bytes after the PLY header."""
    offset = _read_ply_header_end_offset(path)
    if offset < 0:
        return b""
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read()


# ── Tests ───────────────────────────────────────────────────────────────

def test_binary_sphere_cleanup(F, tmpdir):
    """Test cleanup on a realistic binary PLY with 17 properties."""
    ply_path = str(tmpdir / "sphere_binary.ply")
    hull_path = str(tmpdir / "cleanup" / "proxy_hulls.json")
    output_path = str(tmpdir / "sphere_binary.cleaned.ply")
    report_path = str(tmpdir / "report_binary.json")

    meta = generate_ply.generate_sphere_with_outliers(
        ply_path,
        fmt="binary_little_endian",
        sphere_count=1000,
        outlier_count=200,
        seed=42,
    )
    generate_ply.write_cube_proxy_hull(hull_path, half_extent=6.0)

    report = splat_cleanup.run_proxy_hull_cleanup(
        model_path=ply_path,
        proxy_hull_path=hull_path,
        hull_margin=0.0,
        max_removal_ratio=0.9,
        output_path=output_path,
        report_path=report_path,
    )

    F.check("binary: removed outliers", report["removed_points"] == 200,
            f"removed={report['removed_points']}, expected=200")
    F.check("binary: kept sphere points", report["kept_points"] == 1000,
            f"kept={report['kept_points']}, expected=1000")
    F.check("binary: output exists", os.path.isfile(output_path))
    F.check("binary: output vertex count",
            _read_ply_vertex_count(output_path) == 1000)

    # Verify file size: header + 1000 vertices * 68 bytes each
    header_end = _read_ply_header_end_offset(output_path)
    file_size = os.path.getsize(output_path)
    expected_data = 1000 * meta["bytes_per_vertex"]
    F.check("binary: output file size correct",
            file_size == header_end + expected_data,
            f"file_size={file_size}, header_end={header_end}, expected_data={expected_data}")


def test_ascii_sphere_cleanup(F, tmpdir):
    """Test cleanup on a realistic ASCII PLY with 17 properties."""
    ply_path = str(tmpdir / "sphere_ascii.ply")
    hull_path = str(tmpdir / "cleanup" / "proxy_hulls.json")
    output_path = str(tmpdir / "sphere_ascii.cleaned.ply")
    report_path = str(tmpdir / "report_ascii.json")

    generate_ply.generate_sphere_with_outliers(
        ply_path,
        fmt="ascii",
        sphere_count=200,
        outlier_count=50,
        seed=42,
    )
    generate_ply.write_cube_proxy_hull(hull_path, half_extent=6.0)

    report = splat_cleanup.run_proxy_hull_cleanup(
        model_path=ply_path,
        proxy_hull_path=hull_path,
        hull_margin=0.0,
        max_removal_ratio=0.9,
        output_path=output_path,
        report_path=report_path,
    )

    F.check("ascii: removed outliers", report["removed_points"] == 50,
            f"removed={report['removed_points']}, expected=50")
    F.check("ascii: kept sphere points", report["kept_points"] == 200,
            f"kept={report['kept_points']}, expected=200")

    # Read output and verify each data row has 17 tokens
    with open(output_path, "r", encoding="utf-8") as f:
        in_header = True
        data_rows = 0
        for line in f:
            stripped = line.strip()
            if in_header:
                if stripped == "end_header":
                    in_header = False
                continue
            if stripped:
                tokens = stripped.split()
                F.check(f"ascii row {data_rows}: 17 tokens",
                        len(tokens) == 17,
                        f"got {len(tokens)}")
                data_rows += 1

    F.check("ascii: correct row count", data_rows == 200,
            f"rows={data_rows}")


def test_all_properties_preserved_binary(F, tmpdir):
    """Verify byte-for-byte property preservation when no points are removed."""
    ply_path = str(tmpdir / "preserve_input.ply")
    hull_path = str(tmpdir / "cleanup" / "proxy_hulls.json")
    output_path = str(tmpdir / "preserve_output.ply")

    generate_ply.generate_sphere_with_outliers(
        ply_path,
        fmt="binary_little_endian",
        sphere_count=50,
        outlier_count=0,
        seed=99,
    )
    # Hull at ±6 encompasses all sphere points (clamped to ±5.8)
    generate_ply.write_cube_proxy_hull(hull_path, half_extent=6.0)

    report = splat_cleanup.run_proxy_hull_cleanup(
        model_path=ply_path,
        proxy_hull_path=hull_path,
        hull_margin=0.0,
        max_removal_ratio=0.9,
        output_path=output_path,
    )

    F.check("preserve: zero removed", report["removed_points"] == 0)
    F.check("preserve: all kept", report["kept_points"] == 50)

    # Compare raw data sections byte-for-byte
    input_data = _read_binary_data_section(ply_path)
    output_data = _read_binary_data_section(output_path)
    F.check("preserve: data sections identical", input_data == output_data,
            f"input_len={len(input_data)}, output_len={len(output_data)}")


def test_hull_margin_effect(F, tmpdir):
    """Test that hull margin changes cleanup behavior."""
    hull_path = str(tmpdir / "cleanup" / "proxy_hulls.json")
    generate_ply.write_cube_proxy_hull(hull_path, half_extent=6.0)

    # Generate outliers at coordinate magnitude 6.5 — between hull (6.0)
    # and hull+margin(1.0) = 7.0
    ply_path = str(tmpdir / "margin_test.ply")
    generate_ply.generate_sphere_with_outliers(
        ply_path,
        fmt="binary_little_endian",
        sphere_count=100,
        outlier_count=50,
        outlier_min=6.5,
        outlier_max=6.5,
        seed=77,
    )

    # margin=0: outliers at 6.5 are outside the ±6 hull → removed
    output_tight = str(tmpdir / "margin_tight.cleaned.ply")
    report_tight = splat_cleanup.run_proxy_hull_cleanup(
        model_path=ply_path,
        proxy_hull_path=hull_path,
        hull_margin=0.0,
        max_removal_ratio=0.9,
        output_path=output_tight,
    )

    # margin=1.0: hull extends to ±7, outliers at 6.5 are now inside → kept
    output_wide = str(tmpdir / "margin_wide.cleaned.ply")
    report_wide = splat_cleanup.run_proxy_hull_cleanup(
        model_path=ply_path,
        proxy_hull_path=hull_path,
        hull_margin=1.0,
        max_removal_ratio=0.9,
        output_path=output_wide,
    )

    F.check("margin: tight removes more",
            report_tight["removed_points"] > report_wide["removed_points"],
            f"tight_removed={report_tight['removed_points']}, "
            f"wide_removed={report_wide['removed_points']}")
    F.check("margin: tight removes outliers",
            report_tight["removed_points"] == 50,
            f"removed={report_tight['removed_points']}")
    F.check("margin: wide keeps outliers",
            report_wide["removed_points"] == 0,
            f"removed={report_wide['removed_points']}")


def test_removal_ratio_guard_realistic(F, tmpdir):
    """Test safety guard with realistic data where most points are outliers."""
    ply_path = str(tmpdir / "guard_test.ply")
    hull_path = str(tmpdir / "cleanup" / "proxy_hulls.json")

    generate_ply.generate_sphere_with_outliers(
        ply_path,
        fmt="binary_little_endian",
        sphere_count=200,
        outlier_count=800,
        seed=55,
    )
    generate_ply.write_cube_proxy_hull(hull_path, half_extent=6.0)

    # 800/1000 = 80% removal, but limit is 50% → should raise
    try:
        splat_cleanup.run_proxy_hull_cleanup(
            model_path=ply_path,
            proxy_hull_path=hull_path,
            hull_margin=0.0,
            max_removal_ratio=0.5,
            output_path=str(tmpdir / "guard_output.ply"),
        )
        F.check("guard: RuntimeError raised", False, "no exception was raised")
    except RuntimeError as e:
        F.check("guard: RuntimeError raised", True)
        F.check("guard: error mentions ratio",
                "removal ratio" in str(e).lower() or "safety limit" in str(e).lower(),
                str(e))


def test_report_statistics(F, tmpdir):
    """Verify the JSON cleanup report has correct statistics."""
    ply_path = str(tmpdir / "report_test.ply")
    hull_path = str(tmpdir / "cleanup" / "proxy_hulls.json")
    output_path = str(tmpdir / "report_test.cleaned.ply")
    report_path = str(tmpdir / "cleanup_report.json")

    generate_ply.generate_sphere_with_outliers(
        ply_path,
        fmt="binary_little_endian",
        sphere_count=1000,
        outlier_count=200,
        seed=42,
    )
    generate_ply.write_cube_proxy_hull(hull_path, half_extent=6.0)

    splat_cleanup.run_proxy_hull_cleanup(
        model_path=ply_path,
        proxy_hull_path=hull_path,
        hull_margin=0.0,
        max_removal_ratio=0.9,
        output_path=output_path,
        report_path=report_path,
    )

    F.check("report: file exists", os.path.isfile(report_path))

    with open(report_path, "r", encoding="utf-8") as f:
        rpt = json.load(f)

    F.check("report: schema_version", rpt.get("schema_version") == 1)
    F.check("report: mode", rpt.get("mode") == "PROXY_HULL")
    F.check("report: total_points", rpt.get("total_points") == 1200)
    F.check("report: kept_points", rpt.get("kept_points") == 1000)
    F.check("report: removed_points", rpt.get("removed_points") == 200)
    F.check("report: format", rpt.get("format") == "binary_little_endian")
    F.check("report: success", rpt.get("success") is True)

    # removal_ratio should be ~0.1667
    ratio = rpt.get("removal_ratio", -1)
    expected_ratio = 200 / 1200
    F.check("report: removal_ratio",
            abs(ratio - expected_ratio) < 0.001,
            f"ratio={ratio}, expected={expected_ratio:.4f}")


def test_14_property_format(F, tmpdir):
    """Test cleanup on a PLY with 14 properties (no normals)."""
    ply_path = str(tmpdir / "no_normals.ply")
    hull_path = str(tmpdir / "cleanup" / "proxy_hulls.json")
    output_path = str(tmpdir / "no_normals.cleaned.ply")

    meta = generate_ply.generate_sphere_with_outliers(
        ply_path,
        fmt="binary_little_endian",
        sphere_count=500,
        outlier_count=100,
        include_normals=False,
        seed=33,
    )
    generate_ply.write_cube_proxy_hull(hull_path, half_extent=6.0)

    F.check("14-prop: 14 properties", meta["property_count"] == 14)
    F.check("14-prop: 56 bytes/vertex", meta["bytes_per_vertex"] == 56)

    report = splat_cleanup.run_proxy_hull_cleanup(
        model_path=ply_path,
        proxy_hull_path=hull_path,
        hull_margin=0.0,
        max_removal_ratio=0.9,
        output_path=output_path,
    )

    F.check("14-prop: removed outliers", report["removed_points"] == 100,
            f"removed={report['removed_points']}")
    F.check("14-prop: kept sphere", report["kept_points"] == 500,
            f"kept={report['kept_points']}")
    F.check("14-prop: output vertex count",
            _read_ply_vertex_count(output_path) == 500)

    # Verify output data size
    header_end = _read_ply_header_end_offset(output_path)
    file_size = os.path.getsize(output_path)
    expected_data = 500 * 56
    F.check("14-prop: output file size",
            file_size == header_end + expected_data,
            f"file_size={file_size}, expected={header_end + expected_data}")


def main() -> int:
    F = _Failures()

    with tempfile.TemporaryDirectory(prefix="gs_realistic_cleanup_") as temp_dir:
        tmpdir = Path(temp_dir)
        test_binary_sphere_cleanup(F, tmpdir)
        test_ascii_sphere_cleanup(F, tmpdir)
        test_all_properties_preserved_binary(F, tmpdir)
        test_hull_margin_effect(F, tmpdir)
        test_removal_ratio_guard_realistic(F, tmpdir)
        test_report_statistics(F, tmpdir)
        test_14_property_format(F, tmpdir)

    print()
    return F.report()


if __name__ == "__main__":
    raise SystemExit(main())
