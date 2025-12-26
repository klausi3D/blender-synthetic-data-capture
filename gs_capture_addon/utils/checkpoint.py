"""
Checkpoint/resume system for long capture sessions.
Allows resuming interrupted renders without starting over.
"""

import os
import json
from datetime import datetime


def get_checkpoint_path(output_path):
    """Get the checkpoint file path for a capture session.

    Args:
        output_path: Base output directory

    Returns:
        str: Path to checkpoint JSON file
    """
    return os.path.join(output_path, ".gs_capture_checkpoint.json")


def save_checkpoint(output_path, checkpoint_data):
    """Save capture checkpoint to disk.

    Args:
        output_path: Base output directory
        checkpoint_data: Dictionary with checkpoint state:
            - current_index: Current camera index
            - total_cameras: Total number of cameras
            - cameras_data: List of camera transforms/data
            - settings_hash: Hash of settings for validation
            - started_at: ISO timestamp
            - updated_at: ISO timestamp
            - completed_images: List of completed image indices
    """
    checkpoint_path = get_checkpoint_path(output_path)

    # Update timestamp
    checkpoint_data['updated_at'] = datetime.now().isoformat()

    # Write atomically (write to temp, then rename)
    temp_path = checkpoint_path + ".tmp"
    with open(temp_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

    # Rename for atomic write
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    os.rename(temp_path, checkpoint_path)


def load_checkpoint(output_path):
    """Load checkpoint from disk if it exists.

    Args:
        output_path: Base output directory

    Returns:
        dict or None: Checkpoint data if exists, None otherwise
    """
    checkpoint_path = get_checkpoint_path(output_path)

    if not os.path.exists(checkpoint_path):
        return None

    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)

        # Validate checkpoint has required fields
        required = ['current_index', 'total_cameras', 'completed_images']
        for field in required:
            if field not in data:
                print(f"Invalid checkpoint: missing {field}")
                return None

        return data

    except (json.JSONDecodeError, IOError) as e:
        print(f"Failed to load checkpoint: {e}")
        return None


def clear_checkpoint(output_path):
    """Remove checkpoint file after successful completion.

    Args:
        output_path: Base output directory
    """
    checkpoint_path = get_checkpoint_path(output_path)

    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except IOError as e:
            print(f"Failed to remove checkpoint: {e}")


def create_checkpoint(current_index, total_cameras, cameras_data=None,
                      settings_hash=None, completed_images=None):
    """Create a new checkpoint dictionary.

    Args:
        current_index: Current camera index
        total_cameras: Total cameras to render
        cameras_data: Optional list of camera data for recreation
        settings_hash: Hash of capture settings for validation
        completed_images: List of completed image indices

    Returns:
        dict: Checkpoint data
    """
    return {
        'current_index': current_index,
        'total_cameras': total_cameras,
        'cameras_data': cameras_data or [],
        'settings_hash': settings_hash,
        'started_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'completed_images': completed_images or [],
        'version': '2.0.0',  # Checkpoint format version
    }


def get_missing_images(output_path, total_cameras, extension='png'):
    """Find which images are missing from an incomplete capture.

    Args:
        output_path: Base output directory
        total_cameras: Expected number of cameras
        extension: Image file extension

    Returns:
        list: List of missing image indices
    """
    images_path = os.path.join(output_path, "images")

    if not os.path.exists(images_path):
        return list(range(total_cameras))

    missing = []
    for i in range(total_cameras):
        image_name = f"image_{i:04d}.{extension}"
        image_path = os.path.join(images_path, image_name)
        if not os.path.exists(image_path):
            missing.append(i)

    return missing


def validate_checkpoint(output_path, settings_hash):
    """Validate that a checkpoint matches current settings.

    Args:
        output_path: Base output directory
        settings_hash: Hash of current capture settings

    Returns:
        tuple: (is_valid, checkpoint_data or None, error_message or None)
    """
    checkpoint = load_checkpoint(output_path)

    if checkpoint is None:
        return False, None, "No checkpoint found"

    # Check settings hash if present
    if checkpoint.get('settings_hash') and settings_hash:
        if checkpoint['settings_hash'] != settings_hash:
            return False, checkpoint, "Settings have changed since checkpoint"

    # Verify some completed images still exist
    images_path = os.path.join(output_path, "images")
    if checkpoint.get('completed_images'):
        for idx in checkpoint['completed_images'][:5]:  # Check first 5
            expected_path = os.path.join(images_path, f"image_{idx:04d}.png")
            if not os.path.exists(expected_path):
                return False, checkpoint, "Some completed images are missing"

    return True, checkpoint, None


def calculate_settings_hash(settings, scene=None):
    """Calculate a hash of capture settings for checkpoint validation.

    Args:
        settings: GSCaptureSettings
        scene: Blender scene (optional, for render settings)

    Returns:
        str: Hash string
    """
    import hashlib

    # Get resolution from Blender's render settings if scene provided
    if scene:
        res_x = scene.render.resolution_x
        res_y = scene.render.resolution_y
    else:
        res_x = 1920  # Fallback
        res_y = 1080

    # Create string from relevant settings
    settings_str = (
        f"{settings.camera_count}|"
        f"{settings.camera_distribution}|"
        f"{res_x}x{res_y}|"
        f"{settings.min_elevation}|{settings.max_elevation}|"
        f"{settings.camera_distance_mode}|{settings.camera_distance_multiplier}|"
        f"{settings.focal_length}"
    )

    return hashlib.md5(settings_str.encode()).hexdigest()[:16]
