"""Folder structure templates and validation for different training backends."""

import os
from typing import Dict, List, Tuple

# Folder structure requirements per backend
FOLDER_STRUCTURES: Dict[str, dict] = {
    'gaussian_splatting': {
        'name': '3D Gaussian Splatting',
        'dirs': ['images', 'sparse/0'],
        'files': {
            'sparse/0/cameras.txt': 'required',
            'sparse/0/images.txt': 'required',
            'sparse/0/points3D.txt': 'required',
        },
        'alt_files': {
            'sparse/0/cameras.bin': 'sparse/0/cameras.txt',
            'sparse/0/images.bin': 'sparse/0/images.txt',
            'sparse/0/points3D.bin': 'sparse/0/points3D.txt',
        },
    },
    'gs_lightning': {
        'name': 'GS-Lightning',
        'dirs': ['images', 'sparse/0'],
        'optional_dirs': ['masks'],
        'files': {
            'sparse/0/cameras.txt': 'required',
            'sparse/0/images.txt': 'required',
            'sparse/0/points3D.txt': 'required',
        },
        'alt_files': {
            'sparse/0/cameras.bin': 'sparse/0/cameras.txt',
            'sparse/0/images.bin': 'sparse/0/images.txt',
            'sparse/0/points3D.bin': 'sparse/0/points3D.txt',
        },
        'mask_format': '{image_name}.png',  # e.g., image_0001.png.png
    },
    'nerfstudio': {
        'name': 'Nerfstudio',
        'dirs': ['images'],
        'files': {
            'transforms.json': 'required',
        },
    },
    'gsplat': {
        'name': 'gsplat',
        'dirs': ['images'],
        'files': {
            'transforms.json': 'optional',
        },
        'optional_dirs': ['sparse/0'],
    },
}

# Recommended export settings per backend
EXPORT_SETTINGS: Dict[str, dict] = {
    'gaussian_splatting': {
        'export_colmap': True,
        'export_transforms_json': False,
        'export_masks': False,
    },
    'gs_lightning': {
        'export_colmap': True,
        'export_transforms_json': False,
        'export_masks': True,
        'mask_format': 'GSL',
    },
    'nerfstudio': {
        'export_colmap': False,
        'export_transforms_json': True,
        'export_masks': False,
    },
    'gsplat': {
        'export_colmap': True,
        'export_transforms_json': True,
        'export_masks': False,
    },
}


def validate_structure(data_path: str, backend_id: str) -> Tuple[bool, List[str], List[str]]:
    """Validate folder structure for a specific backend.

    Args:
        data_path: Path to the training data directory
        backend_id: ID of the training backend

    Returns:
        tuple: (is_valid, missing_required, missing_optional)
    """
    structure = FOLDER_STRUCTURES.get(backend_id, FOLDER_STRUCTURES['gaussian_splatting'])

    missing_required = []
    missing_optional = []

    # Check required directories
    for dir_path in structure.get('dirs', []):
        full_path = os.path.join(data_path, dir_path)
        if not os.path.exists(full_path):
            missing_required.append(f"Directory: {dir_path}")

    # Check optional directories
    for dir_path in structure.get('optional_dirs', []):
        full_path = os.path.join(data_path, dir_path)
        if not os.path.exists(full_path):
            missing_optional.append(f"Directory: {dir_path}")

    # Check required/optional files (with alternatives)
    alt_files = structure.get('alt_files', {})

    for file_path, requirement in structure.get('files', {}).items():
        full_path = os.path.join(data_path, file_path)

        if not os.path.exists(full_path):
            # Check for alternative (binary format)
            alt_path = None
            for alt, orig in alt_files.items():
                if orig == file_path:
                    alt_path = os.path.join(data_path, alt)
                    break

            if alt_path and os.path.exists(alt_path):
                continue  # Alternative exists, OK

            if requirement == 'required':
                missing_required.append(f"File: {file_path}")
            else:
                missing_optional.append(f"File: {file_path}")

    return len(missing_required) == 0, missing_required, missing_optional


def get_export_settings(backend_id: str) -> dict:
    """Get recommended export settings for a backend.

    Args:
        backend_id: ID of the training backend

    Returns:
        dict: Recommended export settings
    """
    return EXPORT_SETTINGS.get(backend_id, EXPORT_SETTINGS['gaussian_splatting']).copy()


def check_mask_naming(data_path: str, backend_id: str) -> Tuple[bool, str]:
    """Check if mask files follow the expected naming convention.

    Args:
        data_path: Path to the training data directory
        backend_id: ID of the training backend

    Returns:
        tuple: (is_correct, message)
    """
    structure = FOLDER_STRUCTURES.get(backend_id)
    if not structure or 'mask_format' not in structure:
        return True, "Backend does not use masks"

    masks_dir = os.path.join(data_path, "masks")
    images_dir = os.path.join(data_path, "images")

    if not os.path.exists(masks_dir):
        return True, "No masks directory found"

    if not os.path.exists(images_dir):
        return False, "Images directory not found"

    mask_files = set(os.listdir(masks_dir))
    image_files = os.listdir(images_dir)

    if not mask_files:
        return True, "Masks directory is empty"

    # For GS-Lightning, masks should be named {image_name}.png
    # e.g., for image_0001.png -> image_0001.png.png
    matching = 0
    for img_file in image_files:
        expected_mask = f"{img_file}.png"
        if expected_mask in mask_files:
            matching += 1

    if matching == 0:
        return False, f"No masks match expected naming. Expected: {{image_name}}.png (e.g., image_0001.png.png)"

    if matching < len(image_files):
        return True, f"Partial mask coverage: {matching}/{len(image_files)} images have masks"

    return True, f"All {matching} images have correctly named masks"


def count_images(data_path: str) -> int:
    """Count the number of training images.

    Args:
        data_path: Path to the training data directory

    Returns:
        int: Number of images found
    """
    images_dir = os.path.join(data_path, "images")
    if not os.path.exists(images_dir):
        return 0

    valid_extensions = {'.png', '.jpg', '.jpeg', '.exr', '.tiff', '.tif'}
    count = 0

    for filename in os.listdir(images_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            count += 1

    return count


def get_structure_description(backend_id: str) -> str:
    """Get a human-readable description of the required folder structure.

    Args:
        backend_id: ID of the training backend

    Returns:
        str: Description of folder structure
    """
    structure = FOLDER_STRUCTURES.get(backend_id, FOLDER_STRUCTURES['gaussian_splatting'])

    lines = [f"Required structure for {structure.get('name', backend_id)}:", ""]
    lines.append("data_path/")

    for dir_path in structure.get('dirs', []):
        lines.append(f"  {dir_path}/")

    for dir_path in structure.get('optional_dirs', []):
        lines.append(f"  {dir_path}/ (optional)")

    for file_path, req in structure.get('files', {}).items():
        prefix = "" if req == "required" else " (optional)"
        lines.append(f"  {file_path}{prefix}")

    return "\n".join(lines)
