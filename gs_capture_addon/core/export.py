"""
Export functions for COLMAP format, transforms.json, depth maps, normals, and masks.
Handles conversion from Blender coordinate system to various output formats.
"""

import bpy
import os
import math
import json
import random
import numpy as np
from mathutils import Vector, Matrix


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


def export_colmap_cameras(cameras, output_path, image_width, image_height):
    """Export camera parameters in COLMAP format.

    Creates cameras.txt, images.txt, and points3D.txt files
    compatible with COLMAP and LichtFeld Studio.

    Args:
        cameras: List of Blender camera objects
        output_path: Directory to write COLMAP files
        image_width: Image width in pixels
        image_height: Image height in pixels
    """
    os.makedirs(output_path, exist_ok=True)

    # cameras.txt - camera intrinsics
    cameras_file = os.path.join(output_path, "cameras.txt")
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

    # images.txt - camera extrinsics
    # Note: LichtFeld Studio requires even number of non-comment lines
    # Each image needs: pose line + points line
    images_file = os.path.join(output_path, "images.txt")
    with open(images_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for i, cam in enumerate(cameras):
            rot, trans = blender_to_colmap_matrix(cam.matrix_world)

            image_name = f"image_{i:04d}.png"

            f.write(f"{i + 1} {rot.w:.6f} {rot.x:.6f} {rot.y:.6f} {rot.z:.6f} ")
            f.write(f"{trans.x:.6f} {trans.y:.6f} {trans.z:.6f} 1 {image_name}\n")
            # Use single space for points line - prevents removal as "empty"
            f.write(" \n")

    # points3D.txt - generate initial points
    points_file = os.path.join(output_path, "points3D.txt")
    generate_initial_points(cameras, points_file)


def generate_initial_points(cameras, output_file, num_points=5000):
    """Generate initial 3D points for Gaussian Splatting training.

    For synthetic Blender captures, generates random points in the
    scene volume since we don't have COLMAP feature matching.

    Args:
        cameras: List of Blender camera objects
        output_file: Path to output points3D.txt
        num_points: Number of points to generate
    """
    random.seed(42)  # Reproducible

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
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            z = random.uniform(-1, 1)
            if x * x + y * y + z * z <= 1:
                break

        # Scale and translate
        px = center.x + x * radius
        py = center.y + y * radius
        pz = center.z + z * radius

        # Random color
        r = random.randint(128, 255)
        g = random.randint(128, 255)
        b = random.randint(128, 255)

        points.append((px, py, pz, r, g, b))

    # Write points3D.txt
    with open(output_file, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        for i, (px, py, pz, r, g, b) in enumerate(points):
            f.write(f"{i + 1} {px:.6f} {py:.6f} {pz:.6f} {r} {g} {b} 0.0\n")


def export_transforms_json(cameras, output_path, image_width, image_height,
                            aabb_scale=16, include_depth=False, include_masks=False):
    """Export camera data in NeRF/3DGS transforms.json format.

    Args:
        cameras: List of Blender camera objects
        output_path: Directory to write transforms.json
        image_width: Image width in pixels
        image_height: Image height in pixels
        aabb_scale: AABB scale for NeRF (default 16)
        include_depth: Whether depth maps are included
        include_masks: Whether masks are included
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

        frame = {
            "file_path": f"./images/image_{i:04d}",
            "transform_matrix": [[mat_nerf[row][col] for col in range(4)] for row in range(4)],
        }

        if include_depth:
            frame["depth_file_path"] = f"./depth/depth_{i:04d}"

        if include_masks:
            frame["mask_file_path"] = f"./masks/mask_{i:04d}"

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
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)


def save_depth_map(render_result, output_path, normalize=True, format='PNG'):
    """Save depth map from render result.

    Extracts the Z pass from the render and saves it as an image.

    Args:
        render_result: Blender render result with Z pass
        output_path: Path to save depth image
        normalize: Whether to normalize depth to 0-1 range
        format: Output format ('PNG', 'EXR')
    """
    # Get the Z pass
    try:
        # Access via compositor
        viewer = bpy.data.images.get('Viewer Node')
        if viewer and viewer.has_data:
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
            return True
    except Exception as e:
        print(f"Error saving depth map: {e}")
        return False


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
    scene = context.scene

    # Enable compositor and Z pass
    scene.use_nodes = True
    scene.view_layers["ViewLayer"].use_pass_z = True

    tree = scene.node_tree

    # Clear existing nodes
    for node in tree.nodes:
        if node.name.startswith("GS_Depth_"):
            tree.nodes.remove(node)

    # Get render layers node
    render_layers = None
    for node in tree.nodes:
        if node.type == 'R_LAYERS':
            render_layers = node
            break

    if not render_layers:
        render_layers = tree.nodes.new('CompositorNodeRLayers')

    # Create normalize node
    normalize = tree.nodes.new('CompositorNodeNormalize')
    normalize.name = "GS_Depth_Normalize"

    # Create file output node for depth
    file_output = tree.nodes.new('CompositorNodeOutputFile')
    file_output.name = "GS_Depth_Output"
    file_output.base_path = os.path.dirname(output_path)
    file_output.file_slots[0].path = os.path.basename(output_path).replace('.png', '')
    file_output.format.file_format = 'PNG'
    file_output.format.color_mode = 'BW'
    file_output.format.color_depth = '16'

    # Connect nodes
    tree.links.new(render_layers.outputs['Depth'], normalize.inputs[0])
    tree.links.new(normalize.outputs[0], file_output.inputs[0])

    return True


def save_normal_map(context, output_path):
    """Save world-space normal map using compositor.

    Args:
        context: Blender context
        output_path: Path to save normal image

    Returns:
        bool: Success status
    """
    scene = context.scene

    # Enable normal pass
    scene.use_nodes = True
    scene.view_layers["ViewLayer"].use_pass_normal = True

    tree = scene.node_tree

    # Get render layers node
    render_layers = None
    for node in tree.nodes:
        if node.type == 'R_LAYERS':
            render_layers = node
            break

    if not render_layers:
        return False

    # Create file output for normals
    file_output = tree.nodes.new('CompositorNodeOutputFile')
    file_output.name = "GS_Normal_Output"
    file_output.base_path = os.path.dirname(output_path)
    file_output.file_slots[0].path = os.path.basename(output_path).replace('.png', '')
    file_output.format.file_format = 'OPEN_EXR'  # EXR for full precision
    file_output.format.color_mode = 'RGB'

    # Connect normal output
    tree.links.new(render_layers.outputs['Normal'], file_output.inputs[0])

    return True


def save_object_mask(context, output_path, target_objects):
    """Save object mask (binary mask of target objects).

    Uses Object Index pass to create masks.

    Args:
        context: Blender context
        output_path: Path to save mask image
        target_objects: List of objects to include in mask

    Returns:
        bool: Success status
    """
    scene = context.scene

    # Assign object indices
    for i, obj in enumerate(target_objects):
        obj.pass_index = i + 1

    # Enable object index pass
    scene.use_nodes = True
    scene.view_layers["ViewLayer"].use_pass_object_index = True

    tree = scene.node_tree

    # Get render layers node
    render_layers = None
    for node in tree.nodes:
        if node.type == 'R_LAYERS':
            render_layers = node
            break

    if not render_layers:
        return False

    # Create ID Mask node to extract objects with index > 0
    id_mask = tree.nodes.new('CompositorNodeIDMask')
    id_mask.name = "GS_ID_Mask"
    id_mask.index = 1  # Objects with index >= 1
    id_mask.use_antialiasing = True

    # Create file output for mask
    file_output = tree.nodes.new('CompositorNodeOutputFile')
    file_output.name = "GS_Mask_Output"
    file_output.base_path = os.path.dirname(output_path)
    file_output.file_slots[0].path = os.path.basename(output_path).replace('.png', '')
    file_output.format.file_format = 'PNG'
    file_output.format.color_mode = 'BW'

    # Connect nodes
    tree.links.new(render_layers.outputs['IndexOB'], id_mask.inputs[0])
    tree.links.new(id_mask.outputs[0], file_output.inputs[0])

    return True


def cleanup_compositor_nodes(context):
    """Remove GS Capture compositor nodes.

    Args:
        context: Blender context
    """
    if not context.scene.use_nodes:
        return

    tree = context.scene.node_tree
    nodes_to_remove = [node for node in tree.nodes if node.name.startswith("GS_")]

    for node in nodes_to_remove:
        tree.nodes.remove(node)
