"""Proxy-hull cleanup for trained Gaussian splat PLY files."""

from __future__ import annotations

import json
import os
import struct
from datetime import datetime, timezone

PROXY_HULL_RELATIVE_PATH = os.path.join("cleanup", "proxy_hulls.json")

_PLY_SCALAR_TO_STRUCT = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


class PlyFormatError(ValueError):
    """Raised for unsupported or malformed PLY files."""


def default_proxy_hull_path(training_data_path):
    """Get default proxy hull artifact path for a captured dataset."""
    return os.path.join(training_data_path, PROXY_HULL_RELATIVE_PATH)


def default_cleanup_report_path(training_output_path):
    """Get default cleanup report output path."""
    return os.path.join(training_output_path, "cleanup_report.json")


def default_cleaned_model_path(model_path):
    """Get default cleaned model path for a source PLY file."""
    if model_path.lower().endswith(".ply"):
        return f"{model_path[:-4]}.cleaned.ply"
    return f"{model_path}.cleaned.ply"


def _normalize_existing_path(path, description):
    normalized = os.path.abspath(os.path.normpath(os.path.expanduser(path)))
    if not os.path.exists(normalized):
        raise FileNotFoundError(f"{description} does not exist: {normalized}")
    return normalized


def _read_ply_header(handle):
    raw_header_lines = []
    header_lines = []

    while True:
        raw_line = handle.readline()
        if not raw_line:
            raise PlyFormatError("Invalid PLY: missing end_header")
        raw_header_lines.append(raw_line)
        try:
            line = raw_line.decode("ascii").rstrip("\r\n")
        except UnicodeDecodeError as exc:
            raise PlyFormatError(f"Invalid PLY header encoding: {exc}") from exc
        header_lines.append(line)
        if line == "end_header":
            break

    if not header_lines or header_lines[0] != "ply":
        raise PlyFormatError("Invalid PLY header: missing 'ply' signature")

    fmt = None
    elements = []
    current_element = None

    for line_index, line in enumerate(header_lines[1:], start=1):
        if not line or line.startswith("comment") or line.startswith("obj_info"):
            continue

        parts = line.split()
        keyword = parts[0]

        if keyword == "format":
            if len(parts) < 2:
                raise PlyFormatError("Invalid format line in PLY header")
            fmt = parts[1]
            continue

        if keyword == "element":
            if len(parts) != 3:
                raise PlyFormatError(f"Invalid element line: {line}")
            try:
                count = int(parts[2])
            except ValueError as exc:
                raise PlyFormatError(f"Invalid element count in line: {line}") from exc
            current_element = {
                "name": parts[1],
                "count": count,
                "properties": [],
                "line_index": line_index,
            }
            elements.append(current_element)
            continue

        if keyword == "property":
            if current_element is None:
                raise PlyFormatError("Property defined before any element")
            if len(parts) < 3:
                raise PlyFormatError(f"Invalid property line: {line}")

            if parts[1] == "list":
                if len(parts) != 5:
                    raise PlyFormatError(f"Invalid list property line: {line}")
                current_element["properties"].append(
                    {
                        "kind": "list",
                        "count_type": parts[2],
                        "item_type": parts[3],
                        "name": parts[4],
                    }
                )
            else:
                current_element["properties"].append(
                    {
                        "kind": "scalar",
                        "type": parts[1],
                        "name": parts[2],
                    }
                )

    if fmt not in {"ascii", "binary_little_endian"}:
        raise PlyFormatError(f"Unsupported PLY format: {fmt}")

    vertex_indices = [index for index, element in enumerate(elements) if element["name"] == "vertex"]
    if len(vertex_indices) != 1:
        raise PlyFormatError("PLY must contain exactly one 'vertex' element")
    vertex_element_index = vertex_indices[0]
    vertex_element = elements[vertex_element_index]

    # Cleanup rewrites only the vertex element count and data.
    # To avoid creating invalid references, require all other elements to be empty.
    non_vertex_with_data = [
        element["name"]
        for element in elements
        if element["name"] != "vertex" and element["count"] > 0
    ]
    if non_vertex_with_data:
        joined = ", ".join(non_vertex_with_data)
        raise PlyFormatError(
            "Cleanup only supports point-cloud PLY files with vertex data only. "
            f"Unsupported elements with data: {joined}"
        )

    property_names = [prop["name"] for prop in vertex_element["properties"] if prop["kind"] == "scalar"]
    for axis in ("x", "y", "z"):
        if axis not in property_names:
            raise PlyFormatError(f"Vertex property '{axis}' is required for cleanup")

    return {
        "format": fmt,
        "header_lines": header_lines,
        "elements": elements,
        "vertex_element_index": vertex_element_index,
    }


def _load_proxy_hulls(proxy_hull_path):
    with open(proxy_hull_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    objects = payload.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("Proxy hull artifact has no objects")

    return payload


def _point_inside_any_hull(point_xyz, hull_payload, margin):
    bounds = hull_payload.get("global_bounds")
    if isinstance(bounds, dict):
        min_bound = bounds.get("min")
        max_bound = bounds.get("max")
        if (
            isinstance(min_bound, list)
            and isinstance(max_bound, list)
            and len(min_bound) == 3
            and len(max_bound) == 3
        ):
            if (
                point_xyz[0] < (float(min_bound[0]) - margin)
                or point_xyz[1] < (float(min_bound[1]) - margin)
                or point_xyz[2] < (float(min_bound[2]) - margin)
                or point_xyz[0] > (float(max_bound[0]) + margin)
                or point_xyz[1] > (float(max_bound[1]) + margin)
                or point_xyz[2] > (float(max_bound[2]) + margin)
            ):
                return False

    for obj in hull_payload.get("objects", []):
        planes = obj.get("planes", [])
        if not planes:
            continue
        inside = True
        for plane in planes:
            if not isinstance(plane, list) or len(plane) != 4:
                inside = False
                break
            nx, ny, nz, d = float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3])
            signed_distance = (nx * point_xyz[0]) + (ny * point_xyz[1]) + (nz * point_xyz[2]) + d
            if signed_distance > (margin + 1e-9):
                inside = False
                break
        if inside:
            return True

    return False


def _build_vertex_binary_reader(vertex_element):
    if not vertex_element["properties"]:
        raise PlyFormatError("Vertex element has no properties")

    format_codes = []
    x_index = y_index = z_index = None

    for index, prop in enumerate(vertex_element["properties"]):
        if prop["kind"] != "scalar":
            raise PlyFormatError("List properties in vertex element are not supported")
        scalar_type = prop["type"]
        if scalar_type not in _PLY_SCALAR_TO_STRUCT:
            raise PlyFormatError(f"Unsupported vertex scalar type: {scalar_type}")
        format_codes.append(_PLY_SCALAR_TO_STRUCT[scalar_type])
        if prop["name"] == "x":
            x_index = index
        elif prop["name"] == "y":
            y_index = index
        elif prop["name"] == "z":
            z_index = index

    if x_index is None or y_index is None or z_index is None:
        raise PlyFormatError("Vertex properties x/y/z are required")

    fmt = "<" + "".join(format_codes)
    return {
        "fmt": fmt,
        "row_size": struct.calcsize(fmt),
        "x_index": x_index,
        "y_index": y_index,
        "z_index": z_index,
    }


def _filter_ascii_vertex_rows(handle, vertex_element, hull_payload, margin):
    vertex_count = vertex_element["count"]

    scalar_names = [
        prop["name"]
        for prop in vertex_element["properties"]
        if prop["kind"] == "scalar"
    ]

    try:
        x_index = scalar_names.index("x")
        y_index = scalar_names.index("y")
        z_index = scalar_names.index("z")
    except ValueError as exc:
        raise PlyFormatError("Vertex properties x/y/z are required") from exc

    max_index = max(x_index, y_index, z_index)
    kept_rows = []
    removed = 0

    for _ in range(vertex_count):
        row = handle.readline()
        if not row:
            raise PlyFormatError("Unexpected EOF while reading ASCII vertex rows")

        line = row.decode("utf-8", errors="replace")
        tokens = line.strip().split()
        if len(tokens) <= max_index:
            raise PlyFormatError("Malformed vertex row in ASCII PLY")

        try:
            xyz = (float(tokens[x_index]), float(tokens[y_index]), float(tokens[z_index]))
        except ValueError as exc:
            raise PlyFormatError("Failed to parse vertex coordinates in ASCII PLY") from exc

        if _point_inside_any_hull(xyz, hull_payload, margin):
            kept_rows.append(line)
        else:
            removed += 1

    return kept_rows, removed


def _filter_binary_vertex_rows(handle, vertex_element, hull_payload, margin):
    vertex_count = vertex_element["count"]
    reader = _build_vertex_binary_reader(vertex_element)

    kept_rows = []
    removed = 0

    for _ in range(vertex_count):
        row = handle.read(reader["row_size"])
        if len(row) != reader["row_size"]:
            raise PlyFormatError("Unexpected EOF while reading binary vertex rows")

        values = struct.unpack(reader["fmt"], row)
        xyz = (
            float(values[reader["x_index"]]),
            float(values[reader["y_index"]]),
            float(values[reader["z_index"]]),
        )

        if _point_inside_any_hull(xyz, hull_payload, margin):
            kept_rows.append(row)
        else:
            removed += 1

    return kept_rows, removed


def _update_vertex_count_in_header(header_info, kept_count):
    lines = list(header_info["header_lines"])
    vertex_element = header_info["elements"][header_info["vertex_element_index"]]
    lines[vertex_element["line_index"]] = f"element vertex {kept_count}"
    return "\n".join(lines) + "\n"


def _write_ascii_ply(output_path, header_text, kept_rows):
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(header_text)
        for row in kept_rows:
            if row.endswith("\n"):
                handle.write(row)
            else:
                handle.write(f"{row}\n")


def _write_binary_ply(output_path, header_text, kept_rows):
    with open(output_path, "wb") as handle:
        handle.write(header_text.encode("ascii"))
        for row in kept_rows:
            handle.write(row)


def _write_cleanup_report(report_path, payload):
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_proxy_hull_cleanup(
    model_path,
    proxy_hull_path,
    hull_margin=0.01,
    max_removal_ratio=0.70,
    output_path=None,
    report_path=None,
):
    """Filter a trained splat PLY by proxy-hull inclusion and write report outputs."""
    normalized_model = _normalize_existing_path(model_path, "Model file")
    normalized_hulls = _normalize_existing_path(proxy_hull_path, "Proxy hull artifact")

    if not normalized_model.lower().endswith(".ply"):
        raise ValueError(f"Model file must be a .ply file: {normalized_model}")

    margin = float(hull_margin)
    if margin < 0.0:
        raise ValueError("hull_margin must be >= 0")

    removal_limit = float(max_removal_ratio)
    if removal_limit <= 0.0 or removal_limit >= 1.0:
        raise ValueError("max_removal_ratio must be between 0 and 1")

    hull_payload = _load_proxy_hulls(normalized_hulls)

    if output_path is None:
        normalized_output = default_cleaned_model_path(normalized_model)
    else:
        normalized_output = os.path.abspath(os.path.normpath(os.path.expanduser(output_path)))

    output_dir = os.path.dirname(normalized_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(normalized_model, "rb") as handle:
        header_info = _read_ply_header(handle)
        vertex_element = header_info["elements"][header_info["vertex_element_index"]]
        total_points = int(vertex_element["count"])

        if total_points <= 0:
            raise PlyFormatError("PLY has no vertices to clean")

        if header_info["format"] == "ascii":
            kept_rows, removed_points = _filter_ascii_vertex_rows(
                handle,
                vertex_element,
                hull_payload,
                margin,
            )
        else:
            kept_rows, removed_points = _filter_binary_vertex_rows(
                handle,
                vertex_element,
                hull_payload,
                margin,
            )

    kept_points = len(kept_rows)
    removal_ratio = float(removed_points) / float(total_points) if total_points > 0 else 0.0
    if removal_ratio > removal_limit:
        raise RuntimeError(
            "Cleanup aborted because removal ratio exceeded safety limit: "
            f"removed={removed_points}/{total_points} ({removal_ratio:.3f}), "
            f"limit={removal_limit:.3f}"
        )

    header_text = _update_vertex_count_in_header(header_info, kept_points)
    if header_info["format"] == "ascii":
        _write_ascii_ply(normalized_output, header_text, kept_rows)
    else:
        _write_binary_ply(normalized_output, header_text, kept_rows)

    if not os.path.exists(normalized_output) or os.path.getsize(normalized_output) <= 0:
        raise RuntimeError(f"Cleanup output was not written correctly: {normalized_output}")

    if report_path is None:
        report_output_path = default_cleanup_report_path(os.path.dirname(normalized_output))
    else:
        report_output_path = os.path.abspath(os.path.normpath(os.path.expanduser(report_path)))

    report_payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "PROXY_HULL",
        "input_model_path": normalized_model,
        "output_model_path": normalized_output,
        "proxy_hull_path": normalized_hulls,
        "format": header_info["format"],
        "hull_margin": margin,
        "max_removal_ratio": removal_limit,
        "total_points": total_points,
        "kept_points": kept_points,
        "removed_points": removed_points,
        "removal_ratio": removal_ratio,
        "success": True,
    }

    _write_cleanup_report(report_output_path, report_payload)
    report_payload["report_path"] = report_output_path
    return report_payload
