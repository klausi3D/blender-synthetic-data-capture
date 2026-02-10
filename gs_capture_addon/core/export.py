"""
Export functions for COLMAP format, transforms.json, depth maps, normals, and masks.
Handles conversion from Blender coordinate system to various output formats.
"""

import bpy
import os
import math
import json
import random
import struct
import zlib
import numpy as np
from mathutils import Vector, Matrix

from ..utils.paths import validate_path_length


def _ensure_file_written(filepath, description="file"):
    """Verify a file exists and is non-empty after writing."""
    try:
        if not os.path.exists(filepath):
            raise IOError(f"{description} was not written: {filepath}")
        if os.path.getsize(filepath) <= 0:
            raise IOError(f"{description} is empty: {filepath}")
    except OSError as e:
        raise IOError(f"Failed to verify {description} '{filepath}': {e}") from e


def _ensure_path_length(filepath, description="file"):
    """Validate path length before writing on Windows."""
    is_valid, _, error = validate_path_length(filepath)
    if not is_valid:
        raise IOError(f"{description} path is too long for Windows. {error}")


def get_compositor_tree(scene, create=False):
    """Get compositor node tree for both Blender 4.x and 5.x APIs."""
    if create and hasattr(scene, 'use_nodes'):
        try:
            scene.use_nodes = True
        except Exception:
            pass

    # Blender 4.x path.
    tree = getattr(scene, 'node_tree', None)
    if tree is not None:
        return tree

    # Blender 5.x path.
    if hasattr(scene, 'compositing_node_group'):
        tree = scene.compositing_node_group
        if tree is None and create:
            tree = bpy.data.node_groups.new("GS_Compositor_NodeTree", "CompositorNodeTree")
            scene.compositing_node_group = tree
        return tree

    return None


def get_or_create_render_layers_node(tree):
    """Get the compositor Render Layers node, creating it if missing."""
    for node in tree.nodes:
        if node.type == 'R_LAYERS':
            return node
    node = tree.nodes.new('CompositorNodeRLayers')
    node.location = (0, 0)
    return node


def find_socket_by_names(sockets, names):
    """Find a node socket by matching either socket name or identifier."""
    wanted = {str(name).lower() for name in names}
    for socket in sockets:
        socket_name = str(getattr(socket, 'name', '')).lower()
        socket_id = str(getattr(socket, 'identifier', '')).lower()
        if socket_name in wanted or socket_id in wanted:
            return socket
    return None


def _first_non_custom_input(node):
    for socket in node.inputs:
        if getattr(socket, 'type', None) != 'CUSTOM':
            return socket
    if node.inputs:
        return node.inputs[0]
    return None


def configure_output_file_node(
    node,
    output_dir,
    basename,
    file_format,
    color_mode=None,
    color_depth=None,
    slot_name="Image",
    socket_type="RGBA",
):
    """Configure a compositor file output node across Blender versions.

    Returns:
        input socket to connect render/compositor data into.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Blender 4.x API.
    if hasattr(node, 'base_path') and hasattr(node, 'file_slots'):
        node.base_path = output_dir
        if len(node.file_slots) == 0:
            node.file_slots.new(slot_name)
        node.file_slots[0].path = basename
        node.format.file_format = file_format
        if color_mode and hasattr(node.format, 'color_mode'):
            node.format.color_mode = color_mode
        if color_depth and hasattr(node.format, 'color_depth'):
            node.format.color_depth = color_depth
        return _first_non_custom_input(node)

    # Blender 5.x API.
    if hasattr(node, 'directory'):
        node.directory = output_dir
    if hasattr(node, 'file_name'):
        node.file_name = basename

    item = None
    if hasattr(node, 'file_output_items'):
        for candidate in node.file_output_items:
            if str(getattr(candidate, 'name', '')).lower() == slot_name.lower():
                item = candidate
                break
        if item is None:
            try:
                item = node.file_output_items.new(socket_type, slot_name)
            except Exception:
                item = node.file_output_items.new('RGBA', slot_name)

    format_target = getattr(node, 'format', None)
    if item is not None and hasattr(item, 'override_node_format'):
        try:
            item.override_node_format = True
            format_target = item.format
        except Exception:
            format_target = getattr(node, 'format', None)

    if format_target is not None and hasattr(format_target, 'file_format'):
        format_target.file_format = file_format
        if color_mode and hasattr(format_target, 'color_mode'):
            format_target.color_mode = color_mode
        if color_depth and hasattr(format_target, 'color_depth'):
            format_target.color_depth = color_depth

    input_socket = find_socket_by_names(node.inputs, {slot_name})
    if input_socket is None:
        input_socket = _first_non_custom_input(node)
    return input_socket


def set_output_file_basename(node, basename):
    """Set output filename stem for a compositor file output node."""
    if hasattr(node, 'file_slots'):
        if len(node.file_slots) == 0:
            node.file_slots.new("Image")
        node.file_slots[0].path = basename
        return
    if hasattr(node, 'file_name'):
        node.file_name = basename


def image_has_pixels(image):
    """Check whether a Blender image has accessible pixel data."""
    if image is None:
        return False
    size = getattr(image, 'size', None)
    if not size or len(size) < 2:
        return False
    if size[0] <= 0 or size[1] <= 0:
        return False
    try:
        return len(image.pixels) > 0
    except Exception:
        return True


def get_image_extension(file_format: str) -> str:
    """Get file extension for a Blender image format.

    Args:
        file_format: Blender image format identifier (e.g., 'PNG', 'JPEG')

    Returns:
        Lowercase file extension without leading dot.
    """
    format_to_ext = {
        'PNG': 'png',
        'JPEG': 'jpg',
        'OPEN_EXR': 'exr',
        'OPEN_EXR_MULTILAYER': 'exr',
        'TARGA': 'tga',
        'TARGA_RAW': 'tga',
        'BMP': 'bmp',
        'TIFF': 'tiff',
    }
    return format_to_ext.get(file_format, 'png')


def get_camera_intrinsics(camera, image_width, image_height):
    """Calculate camera intrinsic parameters.

    Args:
        camera: Blender camera object
        image_width: Output image width in pixels
        image_height: Output image height in pixels

    Returns:
        dict: Camera intrinsics (fx, fy, cx, cy, focal_mm, sensor_width)
    """
    cam_data = camera.data
    sensor_width = cam_data.sensor_width
    focal_mm = cam_data.lens

    # Calculate focal length in pixels
    focal_px = (focal_mm / sensor_width) * image_width

    # Principal point (image center)
    cx = image_width / 2
    cy = image_height / 2

    return {
        'fx': focal_px,
        'fy': focal_px,
        'cx': cx,
        'cy': cy,
        'focal_mm': focal_mm,
        'sensor_width': sensor_width,
    }


def blender_to_colmap_matrix(blender_matrix):
    """Convert Blender camera matrix to COLMAP convention.

    Blender: Y-forward, Z-up, camera looks down -Z with Y up
    COLMAP/OpenCV: Z-forward, -Y-up, camera looks down +Z with -Y up

    Args:
        blender_matrix: Blender camera world matrix

    Returns:
        tuple: (rotation quaternion, translation vector)
    """
    # Coordinate system conversion matrix
    blender_to_colmap = Matrix((
        (1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (0, 0, 0, 1)
    ))

    # Apply conversion and invert for world-to-camera
    mat_colmap = blender_to_colmap @ blender_matrix.inverted()

    # Extract rotation as quaternion
    rot = mat_colmap.to_quaternion()

    # Extract translation
    trans = mat_colmap.translation

    return rot, trans


def export_colmap_cameras(cameras, output_path, image_width, image_height, image_ext="png"):
    """Export camera parameters in COLMAP format.

    Creates cameras.txt, images.txt, and points3D.txt files
    compatible with COLMAP and LichtFeld Studio.

    Args:
        cameras: List of Blender camera objects
        output_path: Directory to write COLMAP files
        image_width: Image width in pixels
        image_height: Image height in pixels
        image_ext: Image file extension (without dot)

    Returns:
        tuple: (success: bool, warning_message: str or None)
    """
    if not cameras:
        return False, "COLMAP export skipped: no cameras available."

    _ensure_path_length(output_path, "COLMAP output directory")
    os.makedirs(output_path, exist_ok=True)

    # cameras.txt - camera intrinsics
    cameras_file = os.path.join(output_path, "cameras.txt")
    try:
        _ensure_path_length(cameras_file, "COLMAP cameras file")
        with open(cameras_file, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: 1\n")

            # Get intrinsics from first camera (assumes all same)
            intrinsics = get_camera_intrinsics(cameras[0], image_width, image_height)

            # PINHOLE model: fx, fy, cx, cy
            f.write(f"1 PINHOLE {image_width} {image_height} "
                    f"{intrinsics['fx']:.6f} {intrinsics['fy']:.6f} "
                    f"{intrinsics['cx']:.6f} {intrinsics['cy']:.6f}\n")
        _ensure_file_written(cameras_file, "COLMAP cameras file")
    except Exception as e:
        raise IOError(f"Error writing COLMAP cameras file '{cameras_file}': {e}") from e

    # images.txt - camera extrinsics
    # Note: LichtFeld Studio requires even number of non-comment lines
    # Each image needs: pose line + points line
    images_file = os.path.join(output_path, "images.txt")
    try:
        _ensure_path_length(images_file, "COLMAP images file")
        with open(images_file, 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

            for i, cam in enumerate(cameras):
                rot, trans = blender_to_colmap_matrix(cam.matrix_world)

                image_name = f"image_{i:04d}.{image_ext}"

                f.write(f"{i + 1} {rot.w:.6f} {rot.x:.6f} {rot.y:.6f} {rot.z:.6f} ")
                f.write(f"{trans.x:.6f} {trans.y:.6f} {trans.z:.6f} 1 {image_name}\n")
                # Use single space for points line - prevents removal as "empty"
                f.write(" \n")
        _ensure_file_written(images_file, "COLMAP images file")
    except Exception as e:
        raise IOError(f"Error writing COLMAP images file '{images_file}': {e}") from e

    # points3D.txt - generate initial points
    points_file = os.path.join(output_path, "points3D.txt")
    _ensure_path_length(points_file, "COLMAP points file")
    _, points_warning = generate_initial_points(cameras, points_file)
    if points_warning:
        return True, points_warning
    return True, None


def generate_initial_points(cameras, output_file, num_points=5000):
    """Generate initial 3D points for Gaussian Splatting training.

    For synthetic Blender captures, generates random points in the
    scene volume since we don't have COLMAP feature matching.

    Args:
        cameras: List of Blender camera objects
        output_file: Path to output points3D.txt
        num_points: Number of points to generate

    Returns:
        tuple: (success: bool, warning_message: str or None)
    """
    if not cameras:
        return False, "COLMAP points export skipped: no cameras available."

    _ensure_path_length(output_file, "COLMAP points file")

    # Use local random instance to avoid polluting global random state
    rng = random.Random(42)  # Reproducible

    # Calculate scene bounds from camera positions
    cam_positions = []
    cam_targets = []

    for cam in cameras:
        pos = cam.matrix_world.translation
        cam_positions.append(pos)

        # Camera looks along -Z in local space
        forward = cam.matrix_world.to_3x3() @ Vector((0, 0, -1))
        forward.normalize()
        dist_to_origin = pos.length
        target = pos + forward * dist_to_origin
        cam_targets.append(target)

    if not cam_positions:
        center = Vector((0, 0, 0))
        radius = 5.0
    else:
        # Camera center
        cam_center = sum(cam_positions, Vector()) / len(cam_positions)

        # Target center (where cameras are looking)
        target_center = sum(cam_targets, Vector()) / len(cam_targets)

        # Max distance from camera center
        max_cam_dist = max((p - cam_center).length for p in cam_positions)

        # Center points at target, radius proportional to camera spread
        center = target_center
        radius = max_cam_dist * 0.4

    # Generate points in sphere
    points = []
    for _ in range(num_points):
        # Random point in unit sphere
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            z = rng.uniform(-1, 1)
            if x * x + y * y + z * z <= 1:
                break

        # Scale and translate
        px = center.x + x * radius
        py = center.y + y * radius
        pz = center.z + z * radius

        # Random color
        r = rng.randint(128, 255)
        g = rng.randint(128, 255)
        b = rng.randint(128, 255)

        points.append((px, py, pz, r, g, b))

    # Write points3D.txt
    try:
        with open(output_file, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

            for i, (px, py, pz, r, g, b) in enumerate(points):
                f.write(f"{i + 1} {px:.6f} {py:.6f} {pz:.6f} {r} {g} {b} 0.0\n")
        _ensure_file_written(output_file, "COLMAP points file")
    except Exception as e:
        raise IOError(f"Error writing COLMAP points file '{output_file}': {e}") from e

    return True, None


def export_transforms_json(cameras, output_path, image_width, image_height,
                            aabb_scale=16, include_depth=False, include_masks=False,
                            image_ext="png", depth_ext="png", mask_ext="png",
                            mask_format="STANDARD"):
    """Export camera data in NeRF/3DGS transforms.json format.

    Args:
        cameras: List of Blender camera objects
        output_path: Directory to write transforms.json
        image_width: Image width in pixels
        image_height: Image height in pixels
        aabb_scale: AABB scale for NeRF (default 16)
        include_depth: Whether depth maps are included
        include_masks: Whether masks are included
        image_ext: Image file extension (without dot)
        depth_ext: Depth file extension (without dot)
        mask_ext: Mask file extension (without dot)
        mask_format: Mask naming format ('STANDARD' or 'GSL')
    """
    if not cameras:
        return

    # Get camera parameters
    intrinsics = get_camera_intrinsics(cameras[0], image_width, image_height)

    # Calculate camera angle (used by some NeRF implementations)
    camera_angle_x = 2 * math.atan(image_width / (2 * intrinsics['fx']))
    camera_angle_y = 2 * math.atan(image_height / (2 * intrinsics['fy']))

    # Build frames list
    frames = []
    for i, cam in enumerate(cameras):
        # Get world matrix (Blender convention)
        mat = cam.matrix_world

        # Convert to NeRF convention (same as OpenGL)
        # Blender: Z-up, -Y forward (camera looks -Z)
        # OpenGL/NeRF: Y-up, -Z forward

        # Flip Y and Z for OpenGL convention
        convert = Matrix((
            (1, 0, 0, 0),
            (0, 0, 1, 0),
            (0, -1, 0, 0),
            (0, 0, 0, 1)
        ))

        mat_nerf = convert @ mat

        image_name = f"image_{i:04d}.{image_ext}"
        frame = {
            "file_path": f"./images/{image_name}",
            "transform_matrix": [[mat_nerf[row][col] for col in range(4)] for row in range(4)],
        }

        if include_depth:
            frame["depth_file_path"] = f"./depth/depth_{i:04d}.{depth_ext}"

        if include_masks:
            if mask_format == 'GSL':
                mask_name = f"{image_name}.{mask_ext}"
            else:
                mask_name = f"mask_{i:04d}.{mask_ext}"
            frame["mask_file_path"] = f"./masks/{mask_name}"

        frames.append(frame)

    # Build output dictionary
    output = {
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        "fl_x": intrinsics['fx'],
        "fl_y": intrinsics['fy'],
        "cx": intrinsics['cx'],
        "cy": intrinsics['cy'],
        "w": image_width,
        "h": image_height,
        "aabb_scale": aabb_scale,
        "frames": frames,
    }

    # Write JSON
    json_path = os.path.join(output_path, "transforms.json")
    try:
        _ensure_path_length(json_path, "transforms.json")
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)
        _ensure_file_written(json_path, "transforms.json")
    except Exception as e:
        raise IOError(f"Error writing transforms file '{json_path}': {e}") from e


def save_depth_map(render_result, output_path, normalize=True, format='PNG'):
    """Save depth map from render result.

    Extracts the Z pass from the render and saves it as an image.

    Args:
        render_result: Blender render result with Z pass
        output_path: Path to save depth image
        normalize: Whether to normalize depth to 0-1 range
        format: Output format ('PNG', 'EXR')
    """
    _ensure_path_length(output_path, "depth map")
    # Get the Z pass
    try:
        # Access via compositor
        viewer = bpy.data.images.get('Viewer Node')
        if image_has_pixels(viewer):
            # Get pixel data
            width, height = viewer.size
            pixels = np.array(viewer.pixels[:]).reshape(height, width, 4)

            if normalize:
                # Normalize depth values
                depth = pixels[:, :, 0]  # Z is typically in first channel
                valid_mask = depth < 1e10  # Ignore infinite values
                if valid_mask.any():
                    min_d = depth[valid_mask].min()
                    max_d = depth[valid_mask].max()
                    if max_d > min_d:
                        depth = (depth - min_d) / (max_d - min_d)
                    depth[~valid_mask] = 1.0
                    pixels[:, :, 0] = depth
                    pixels[:, :, 1] = depth
                    pixels[:, :, 2] = depth

            # Create new image for saving
            depth_img = bpy.data.images.new("depth_temp", width, height, alpha=False)
            depth_img.pixels = pixels.flatten().tolist()
            depth_img.filepath_raw = output_path
            depth_img.file_format = format
            depth_img.save()
            bpy.data.images.remove(depth_img)
            _ensure_file_written(output_path, "depth map")
            return True
    except Exception as e:
        raise RuntimeError(f"Error saving depth map '{output_path}': {e}") from e


def save_depth_from_z_buffer(context, output_path, camera, near_clip=0.1, far_clip=100.0):
    """Save depth map using compositor Z buffer.

    Sets up compositor nodes to extract and save depth.

    Args:
        context: Blender context
        output_path: Path to save depth image
        camera: Active camera for depth calculation
        near_clip: Near clipping plane
        far_clip: Far clipping plane

    Returns:
        bool: Success status
    """
    _ensure_path_length(output_path, "depth map")
    scene = context.scene

    # Enable compositor and Z pass
    view_layer = context.view_layer
    view_layer.use_pass_z = True

    tree = get_compositor_tree(scene, create=True)
    if tree is None:
        return False

    # Clear existing nodes
    for node in tree.nodes:
        if node.name.startswith("GS_Depth_"):
            tree.nodes.remove(node)

    # Get render layers node
    render_layers = get_or_create_render_layers_node(tree)
    depth_socket = find_socket_by_names(render_layers.outputs, {"depth", "z"})
    if depth_socket is None:
        return False

    # Create normalize node
    normalize = tree.nodes.new('CompositorNodeNormalize')
    normalize.name = "GS_Depth_Normalize"

    # Create file output node for depth
    file_output = tree.nodes.new('CompositorNodeOutputFile')
    file_output.name = "GS_Depth_Output"
    output_socket = configure_output_file_node(
        file_output,
        os.path.dirname(output_path),
        os.path.splitext(os.path.basename(output_path))[0],
        file_format='PNG',
        color_mode='BW',
        color_depth='16',
        slot_name='Depth',
        socket_type='FLOAT',
    )
    if output_socket is None:
        return False

    # Connect nodes
    tree.links.new(depth_socket, normalize.inputs[0])
    tree.links.new(normalize.outputs[0], output_socket)

    return True


def save_normal_map(context, output_path):
    """Save world-space normal map using compositor.

    Args:
        context: Blender context
        output_path: Path to save normal image

    Returns:
        bool: Success status
    """
    _ensure_path_length(output_path, "normal map")
    scene = context.scene

    # Enable normal pass
    view_layer = context.view_layer
    view_layer.use_pass_normal = True

    tree = get_compositor_tree(scene, create=True)
    if tree is None:
        return False
    render_layers = get_or_create_render_layers_node(tree)
    normal_socket = find_socket_by_names(render_layers.outputs, {"normal"})
    if normal_socket is None:
        return False

    # Create file output for normals
    file_output = tree.nodes.new('CompositorNodeOutputFile')
    file_output.name = "GS_Normal_Output"
    output_socket = configure_output_file_node(
        file_output,
        os.path.dirname(output_path),
        os.path.splitext(os.path.basename(output_path))[0],
        file_format='OPEN_EXR',  # EXR for full precision
        color_mode='RGB',
        slot_name='Normal',
        socket_type='VECTOR',
    )
    if output_socket is None:
        return False

    # Connect normal output
    tree.links.new(normal_socket, output_socket)

    return True


def save_object_mask(context, output_path, target_objects):
    """Save object mask (binary mask of target objects).

    Uses Object Index pass to create masks. Creates ID mask nodes for
    each target object and combines them to capture all objects.

    Args:
        context: Blender context
        output_path: Path to save mask image
        target_objects: List of objects to include in mask

    Returns:
        bool: Success status
    """
    _ensure_path_length(output_path, "mask image")
    scene = context.scene

    if not target_objects:
        return False

    # Assign object indices
    for i, obj in enumerate(target_objects):
        obj.pass_index = i + 1

    # Enable object index pass
    view_layer = context.view_layer
    view_layer.use_pass_object_index = True

    tree = get_compositor_tree(scene, create=True)
    if tree is None:
        return False
    render_layers = get_or_create_render_layers_node(tree)

    # Create ID Mask nodes for each object index
    index_output = None
    wanted = {'indexob', 'object index'}
    for socket in render_layers.outputs:
        socket_name = str(getattr(socket, 'name', '')).lower()
        socket_id = str(getattr(socket, 'identifier', '')).lower()
        if socket_name in wanted or socket_id in wanted:
            index_output = socket
            break
    if index_output is None:
        return False

    id_masks = []
    for i in range(len(target_objects)):
        id_mask = tree.nodes.new('CompositorNodeIDMask')
        id_mask.name = f"GS_ID_Mask_{i}"
        # Blender 4.x uses node properties; Blender 5.x uses input sockets.
        if hasattr(id_mask, 'index'):
            id_mask.index = i + 1  # Object indices start at 1
            if hasattr(id_mask, 'use_antialiasing'):
                id_mask.use_antialiasing = True
        else:
            index_input = find_socket_by_names(id_mask.inputs, {"index"})
            if index_input is not None and hasattr(index_input, 'default_value'):
                index_input.default_value = i + 1
            aa_input = find_socket_by_names(id_mask.inputs, {"anti-alias", "anti alias", "anti_alias"})
            if aa_input is not None and hasattr(aa_input, 'default_value'):
                aa_input.default_value = True
        tree.links.new(index_output, id_mask.inputs[0])
        id_masks.append(id_mask)

    # Combine masks using Maximum math nodes
    if len(id_masks) == 1:
        # Only one object, use its mask directly
        combined_output = id_masks[0].outputs[0]
    else:
        # Combine multiple masks with Maximum nodes
        combined_output = id_masks[0].outputs[0]
        for i in range(1, len(id_masks)):
            math_node = tree.nodes.new('CompositorNodeMath')
            math_node.name = f"GS_Mask_Combine_{i}"
            math_node.operation = 'MAXIMUM'
            tree.links.new(combined_output, math_node.inputs[0])
            tree.links.new(id_masks[i].outputs[0], math_node.inputs[1])
            combined_output = math_node.outputs[0]

    # Create file output for mask
    file_output = tree.nodes.new('CompositorNodeOutputFile')
    file_output.name = "GS_Mask_Output"
    output_socket = configure_output_file_node(
        file_output,
        os.path.dirname(output_path),
        os.path.splitext(os.path.basename(output_path))[0],
        file_format='PNG',
        color_mode='BW',
        slot_name='Mask',
        socket_type='FLOAT',
    )
    if output_socket is None:
        return False

    # Connect combined mask to output
    tree.links.new(combined_output, output_socket)

    return True


def cleanup_compositor_nodes(context):
    """Remove GS Capture compositor nodes.

    Args:
        context: Blender context
    """
    tree = get_compositor_tree(context.scene, create=False)
    if tree is None:
        return

    nodes_to_remove = [node for node in tree.nodes if node.name.startswith("GS_")]

    for node in nodes_to_remove:
        tree.nodes.remove(node)


def _write_grayscale_png(filepath, data):
    """Write a single-channel grayscale PNG using pure Python.

    Args:
        filepath: Output file path
        data: 2D numpy array (height, width) with values 0-255

    Returns:
        bool: Success status
    """
    _ensure_path_length(filepath, "grayscale PNG")
    height, width = data.shape

    def make_chunk(chunk_type, chunk_data):
        """Create a PNG chunk with CRC."""
        chunk = chunk_type + chunk_data
        crc = zlib.crc32(chunk) & 0xffffffff
        return struct.pack('>I', len(chunk_data)) + chunk + struct.pack('>I', crc)

    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'

    # IHDR chunk: width, height, bit_depth, color_type, compression, filter, interlace
    # color_type 0 = grayscale
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0)
    ihdr = make_chunk(b'IHDR', ihdr_data)

    # IDAT chunk: compressed image data
    # Each row has a filter byte (0 = no filter) followed by pixel data
    raw_data = b''
    for row in range(height):
        raw_data += b'\x00'  # No filter
        raw_data += data[height - 1 - row, :].astype(np.uint8).tobytes()  # Flip Y

    compressed = zlib.compress(raw_data, 9)
    idat = make_chunk(b'IDAT', compressed)

    # IEND chunk
    iend = make_chunk(b'IEND', b'')

    # Write PNG file
    try:
        with open(filepath, 'wb') as f:
            f.write(signature + ihdr + idat + iend)
        _ensure_file_written(filepath, "grayscale PNG")
    except Exception as e:
        raise IOError(f"Error writing grayscale PNG '{filepath}': {e}") from e

    return True


def _convert_to_grayscale(filepath):
    """Convert RGB PNG to single-channel grayscale PNG.

    GS-Lightning requires single-channel masks. Blender always saves RGB/RGBA,
    so we need to convert after saving.

    Args:
        filepath: Path to PNG file to convert in-place

    Returns:
        bool: Success status
    """
    _ensure_path_length(filepath, "grayscale mask")
    # Try PIL first (cleanest solution)
    try:
        from PIL import Image
        try:
            img = Image.open(filepath)
            gray = img.convert('L')
            gray.save(filepath)
            _ensure_file_written(filepath, "grayscale mask")
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to convert mask to grayscale using PIL: {e}") from e
    except ImportError:
        pass  # PIL not available, use fallback

    # Fallback: Read PNG and rewrite as grayscale using pure Python
    try:
        # Load the image we just saved via Blender
        img = bpy.data.images.load(filepath)
        width, height = img.size

        if width == 0 or height == 0:
            bpy.data.images.remove(img)
            return False

        # Get pixel data (RGBA float)
        pixels = np.array(img.pixels[:]).reshape(height, width, 4)

        # Extract just the first channel (R=G=B for our mask)
        # Convert from 0-1 float to 0-255 uint8
        gray_data = (pixels[:, :, 0] * 255).astype(np.uint8)

        # Remove Blender image before overwriting file
        bpy.data.images.remove(img)

        # Write grayscale PNG
        _write_grayscale_png(filepath, gray_data)
        _ensure_file_written(filepath, "grayscale mask")
        return True

    except Exception as e:
        raise RuntimeError(f"Error converting to grayscale '{filepath}': {e}") from e


def extract_alpha_mask(image_path, mask_path):
    """Extract binary mask from RGBA image's alpha channel.

    Creates a mask where:
    - White (255) = object (alpha > 0)
    - Black (0) = background (alpha == 0)

    Args:
        image_path: Path to source RGBA image
        mask_path: Path to save binary mask

    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    _ensure_path_length(mask_path, "mask image")
    try:
        # Load the rendered image
        img = bpy.data.images.load(image_path)

        # Note: In Blender 5.0, has_data may report False even when data is accessible
        # Check size instead - if both dimensions are 0, there's truly no data
        width, height = img.size
        if width == 0 or height == 0:
            bpy.data.images.remove(img)
            return False, f"Image {image_path} has no valid dimensions"
        pixels = np.array(img.pixels[:]).reshape(height, width, 4)

        # Extract alpha channel and convert to binary mask (0.0 or 1.0)
        alpha = pixels[:, :, 3]
        mask = (alpha > 0.01).astype(np.float32)

        # Create grayscale mask image and save
        # Blender requires RGBA internally but we save as grayscale
        mask_img = bpy.data.images.new("gs_temp_mask", width, height, alpha=False, float_buffer=False)

        # Set all channels to same mask value (grayscale)
        mask_rgba = np.zeros((height, width, 4), dtype=np.float32)
        mask_rgba[:, :, 0] = mask
        mask_rgba[:, :, 1] = mask
        mask_rgba[:, :, 2] = mask
        mask_rgba[:, :, 3] = 1.0

        mask_img.pixels = mask_rgba.flatten().tolist()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)

        # Save as PNG - Blender saves RGB but we'll convert to grayscale after
        mask_img.filepath_raw = mask_path
        mask_img.file_format = 'PNG'
        mask_img.save()
        _ensure_file_written(mask_path, "mask image")

        # Cleanup Blender image
        bpy.data.images.remove(mask_img)
        bpy.data.images.remove(img)

        # Convert to true grayscale using subprocess (cv2/PIL might not be in Blender)
        # This ensures single-channel output for GS-Lightning
        _convert_to_grayscale(mask_path)
        _ensure_file_written(mask_path, "mask image")

        return True, None

    except Exception as e:
        raise RuntimeError(f"Error extracting alpha mask: {e}") from e


def save_alpha_mask_from_render(mask_path):
    """Save binary mask from current render result's alpha channel.

    Extracts the alpha channel from the Render Result and saves as mask.

    Args:
        mask_path: Path to save binary mask

    Returns:
        bool: Success status
    """
    _ensure_path_length(mask_path, "mask image")
    try:
        render_result = bpy.data.images.get('Render Result')
        if not image_has_pixels(render_result):
            return False

        width, height = render_result.size
        pixels = np.array(render_result.pixels[:]).reshape(height, width, 4)

        # Extract alpha channel and convert to binary mask
        alpha = pixels[:, :, 3]
        mask = (alpha > 0.01).astype(np.float32)

        # Create mask image
        mask_img = bpy.data.images.new("gs_temp_mask", width, height, alpha=False)

        mask_rgba = np.zeros((height, width, 4), dtype=np.float32)
        mask_rgba[:, :, 0] = mask
        mask_rgba[:, :, 1] = mask
        mask_rgba[:, :, 2] = mask
        mask_rgba[:, :, 3] = 1.0

        mask_img.pixels = mask_rgba.flatten().tolist()

        # Ensure directory exists
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)

        mask_img.filepath_raw = mask_path
        mask_img.file_format = 'PNG'
        mask_img.save()
        _ensure_file_written(mask_path, "mask image")

        bpy.data.images.remove(mask_img)
        return True

    except Exception as e:
        raise RuntimeError(f"Error saving alpha mask '{mask_path}': {e}") from e
