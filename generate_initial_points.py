#!/usr/bin/env python3
"""
Generate initial 3D points for Gaussian Splatting training from synthetic Blender captures.
Since we don't have COLMAP feature matching for synthetic data, we generate random points
in the scene volume that cameras are looking at.
"""

import json
import numpy as np
import os
import sys

def load_transforms(transforms_path):
    """Load transforms.json and extract camera positions."""
    with open(transforms_path, 'r') as f:
        data = json.load(f)

    camera_positions = []
    camera_targets = []

    for frame in data['frames']:
        matrix = np.array(frame['transform_matrix'])
        # Camera position is the translation column
        pos = matrix[:3, 3]
        camera_positions.append(pos)

        # Camera looks along -Z in camera space, so forward vector is -Z column
        forward = -matrix[:3, 2]
        # Approximate target point (assume looking at origin or along forward)
        target = pos + forward * np.linalg.norm(pos)  # Point towards center
        camera_targets.append(target)

    return np.array(camera_positions), np.array(camera_targets)

def generate_points_in_volume(center, radius, num_points=5000, seed=42):
    """Generate random points within a sphere."""
    np.random.seed(seed)

    # Generate points in a sphere
    points = []
    while len(points) < num_points:
        # Random points in cube
        p = (np.random.rand(3) - 0.5) * 2 * radius + center
        # Keep if within sphere
        if np.linalg.norm(p - center) <= radius:
            points.append(p)

    return np.array(points[:num_points])

def generate_points_on_surface(center, radius, num_points=2000, seed=42):
    """Generate random points on a sphere surface (good for object surfaces)."""
    np.random.seed(seed)

    # Fibonacci sphere for even distribution
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2
        r = np.sqrt(1 - y * y)
        theta = phi * i

        x = np.cos(theta) * r
        z = np.sin(theta) * r

        point = center + np.array([x, y, z]) * radius * 0.8  # Slightly inside
        points.append(point)

    return np.array(points)

def write_colmap_points(points, output_path, colors=None):
    """Write points3D.txt in COLMAP format."""
    with open(output_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")

        for i, point in enumerate(points):
            x, y, z = point
            if colors is not None:
                r, g, b = colors[i]
            else:
                # Default gray color
                r, g, b = 128, 128, 128

            # POINT3D_ID X Y Z R G B ERROR TRACK[]
            # Error = 0, no tracks for synthetic data
            f.write(f"{i+1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0\n")

def write_colmap_points_binary(points, output_path, colors=None):
    """Write points3D.bin in COLMAP binary format."""
    import struct

    with open(output_path, 'wb') as f:
        # Number of points
        f.write(struct.pack('<Q', len(points)))

        for i, point in enumerate(points):
            x, y, z = point
            if colors is not None:
                r, g, b = colors[i]
            else:
                r, g, b = 128, 128, 128

            # POINT3D_ID (uint64), X Y Z (double), R G B (uint8), ERROR (double), TRACK_LENGTH (uint64)
            f.write(struct.pack('<Q', i + 1))  # point_id
            f.write(struct.pack('<ddd', x, y, z))  # xyz
            f.write(struct.pack('<BBB', r, g, b))  # rgb
            f.write(struct.pack('<d', 0.0))  # error
            f.write(struct.pack('<Q', 0))  # track_length (no tracks)

def main():
    if len(sys.argv) < 2:
        capture_dir = r"C:\Projects\GS_Blender\capture"
    else:
        capture_dir = sys.argv[1]

    transforms_path = os.path.join(capture_dir, "transforms.json")
    points_txt_path = os.path.join(capture_dir, "sparse", "0", "points3D.txt")
    points_bin_path = os.path.join(capture_dir, "sparse", "0", "points3D.bin")

    if not os.path.exists(transforms_path):
        print(f"Error: transforms.json not found at {transforms_path}")
        sys.exit(1)

    print(f"Loading transforms from {transforms_path}")
    camera_positions, camera_targets = load_transforms(transforms_path)

    print(f"Found {len(camera_positions)} cameras")

    # Calculate scene bounds from camera positions
    cam_center = np.mean(camera_positions, axis=0)
    cam_distances = np.linalg.norm(camera_positions - cam_center, axis=1)
    max_cam_dist = np.max(cam_distances)

    print(f"Camera center: {cam_center}")
    print(f"Max camera distance: {max_cam_dist}")

    # Assume object is at origin (typical for Blender captures)
    # Generate points in a sphere centered at origin
    object_center = np.array([0.0, 0.0, 0.0])
    object_radius = max_cam_dist * 0.3  # Object is roughly 30% of camera distance

    print(f"Generating points around center {object_center} with radius {object_radius}")

    # Generate mix of volume and surface points
    num_volume_points = 3000
    num_surface_points = 2000

    volume_points = generate_points_in_volume(object_center, object_radius, num_volume_points)
    surface_points = generate_points_on_surface(object_center, object_radius, num_surface_points)

    all_points = np.vstack([volume_points, surface_points])

    # Generate random colors (will be refined during training)
    np.random.seed(42)
    colors = np.random.randint(100, 200, size=(len(all_points), 3))

    print(f"Generated {len(all_points)} initial points")

    # Write text format
    print(f"Writing points to {points_txt_path}")
    write_colmap_points(all_points, points_txt_path, colors)

    # Also write binary format (some tools prefer this)
    print(f"Writing binary points to {points_bin_path}")
    write_colmap_points_binary(all_points, points_bin_path, colors)

    print("Done! Initial point cloud generated.")
    print(f"\nTo train with LichtFeld Studio:")
    print(f'  cd C:\\Projects\\GS_Tools\\LichtFeld-Studio\\build')
    print(f'  .\\LichtFeld-Studio.exe -d "{capture_dir}" -o "{capture_dir}\\output" --strategy mcmc -i 30000')

if __name__ == "__main__":
    main()
