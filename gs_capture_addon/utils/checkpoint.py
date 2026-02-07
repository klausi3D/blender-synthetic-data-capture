"""
Checkpoint/resume system for long capture sessions.
Allows resuming interrupted renders without starting over.
"""

import os
import json
from datetime import datetime

from .paths import validate_path_length

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
            - settings_hash_legacy: Legacy hash for backward compatibility
            - started_at: ISO timestamp
            - updated_at: ISO timestamp
            - completed_images: List of completed image indices

    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    checkpoint_path = get_checkpoint_path(output_path)
    is_valid, _, error = validate_path_length(checkpoint_path)
    if not is_valid:
        return False, f"Checkpoint path is too long for Windows. {error}"

    # Update timestamp
    checkpoint_data['updated_at'] = datetime.now().isoformat()

    # Write atomically (write to temp, then replace)
    temp_path = checkpoint_path + ".tmp"
    is_valid, _, error = validate_path_length(temp_path)
    if not is_valid:
        return False, f"Checkpoint temp path is too long for Windows. {error}"
    try:
        with open(temp_path, 'w') as f:
            json.dump(checkpoint_data, f)  # No indent for speed
        os.replace(temp_path, checkpoint_path)  # Atomic on all platforms
        return True, None
    except Exception as e:
        error_message = f"Failed to save checkpoint: {e}"
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False, error_message


def load_checkpoint(output_path):
    """Load checkpoint from disk if it exists.

    Args:
        output_path: Base output directory

    Returns:
        tuple: (checkpoint_data: dict or None, error_message: str or None)
    """
    checkpoint_path = get_checkpoint_path(output_path)

    if not os.path.exists(checkpoint_path):
        return None, None

    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)

        # Validate checkpoint has required fields
        required = ['current_index', 'total_cameras', 'completed_images']
        for field in required:
            if field not in data:
                return None, f"Invalid checkpoint: missing {field}"

        return data, None

    except (json.JSONDecodeError, IOError) as e:
        return None, f"Failed to load checkpoint: {e}"


def clear_checkpoint(output_path):
    """Remove checkpoint file after successful completion.

    Args:
        output_path: Base output directory

    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    checkpoint_path = get_checkpoint_path(output_path)

    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            return True, None
        except IOError as e:
            return False, f"Failed to remove checkpoint: {e}"
    return True, None


def create_checkpoint(current_index, total_cameras, cameras_data=None,
                      settings_hash=None, settings_hash_legacy=None,
                      completed_images=None):
    """Create a new checkpoint dictionary.

    Args:
        current_index: Current camera index
        total_cameras: Total cameras to render
        cameras_data: Optional list of camera data for recreation
        settings_hash: Hash of capture settings for validation
        settings_hash_legacy: Legacy settings hash for backward compatibility
        completed_images: List of completed image indices

    Returns:
        dict: Checkpoint data
    """
    checkpoint = {
        'current_index': current_index,
        'total_cameras': total_cameras,
        'cameras_data': cameras_data or [],
        'settings_hash': settings_hash,
        'started_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'completed_images': completed_images or [],
        'version': '2.0.0',  # Checkpoint format version
    }
    if settings_hash_legacy is not None:
        checkpoint['settings_hash_legacy'] = settings_hash_legacy
    return checkpoint


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


def validate_checkpoint(output_path, settings_hash, legacy_settings_hash=None):
    """Validate that a checkpoint matches current settings.

    Args:
        output_path: Base output directory
        settings_hash: Hash of current capture settings
        legacy_settings_hash: Legacy hash for backward compatibility

    Returns:
        tuple: (is_valid, checkpoint_data or None, error_message or None)
    """
    checkpoint, error = load_checkpoint(output_path)

    if checkpoint is None:
        if error:
            return False, None, error
        return False, None, "No checkpoint found"

    # Check settings hash if present
    if checkpoint.get('settings_hash') and settings_hash:
        if not settings_hash_matches(checkpoint, settings_hash, legacy_settings_hash):
            return False, checkpoint, "Settings have changed since checkpoint"

    # Verify some completed images still exist and are valid
    images_path = os.path.join(output_path, "images")
    if checkpoint.get('completed_images'):
        for idx in checkpoint['completed_images'][:5]:  # Check first 5
            expected_path = os.path.join(images_path, f"image_{idx:04d}.png")
            if not os.path.exists(expected_path):
                return False, checkpoint, "Some completed images are missing"
            if os.path.getsize(expected_path) == 0:
                return False, checkpoint, "Some completed images are corrupt (0 bytes)"

    return True, checkpoint, None


def settings_hash_matches(checkpoint, settings_hash, legacy_settings_hash=None):
    """Check whether a checkpoint settings hash matches current settings.

    Args:
        checkpoint: Loaded checkpoint dict
        settings_hash: Current full settings hash
        legacy_settings_hash: Optional legacy settings hash

    Returns:
        bool: True if hashes match (full or legacy), False otherwise
    """
    if not checkpoint or not settings_hash:
        return False

    candidates = {
        checkpoint.get('settings_hash'),
        checkpoint.get('settings_hash_legacy'),
        checkpoint.get('settings_hash_v1'),
    }
    candidates.discard(None)

    if settings_hash in candidates:
        return True
    if legacy_settings_hash and legacy_settings_hash in candidates:
        return True
    return False


def calculate_settings_hash(settings, scene=None, legacy=False):
    """Calculate a hash of capture settings for checkpoint validation.

    Args:
        settings: GSCaptureSettings
        scene: Blender scene (optional, for render settings)
        legacy: If True, compute legacy hash for backwards compatibility

    Returns:
        str: Hash string
    """
    import hashlib

    # Legacy hash (v1) for backward compatibility
    if legacy:
        # Get resolution from Blender's render settings if scene provided
        if scene:
            res_x = scene.render.resolution_x
            res_y = scene.render.resolution_y
        else:
            res_x = 1920  # Fallback
            res_y = 1080

        # Create string from relevant settings (legacy)
        settings_str = (
            f"{settings.camera_count}|"
            f"{settings.camera_distribution}|"
            f"{res_x}x{res_y}|"
            f"{settings.min_elevation}|{settings.max_elevation}|"
            f"{settings.camera_distance_mode}|{settings.camera_distance_multiplier}|"
            f"{settings.focal_length}"
        )

        return hashlib.md5(settings_str.encode()).hexdigest()[:16]

    # Full hash (v2) includes render settings + outputs affecting files
    if scene:
        rd = scene.render
        img_settings = rd.image_settings
        res_x = rd.resolution_x
        res_y = rd.resolution_y
        res_pct = getattr(rd, 'resolution_percentage', 100)
        render_engine = rd.engine
        file_format = img_settings.file_format
        color_mode = getattr(img_settings, 'color_mode', None)
        color_depth = getattr(img_settings, 'color_depth', None)
    else:
        rd = None
        img_settings = None
        res_x = 1920
        res_y = 1080
        res_pct = 100
        render_engine = None
        file_format = None
        color_mode = None
        color_depth = None

    # Sample settings (engine-specific)
    samples = None
    if render_engine == 'CYCLES' and scene and hasattr(scene, 'cycles'):
        samples = getattr(scene.cycles, 'samples', None)
    elif render_engine and 'EEVEE' in render_engine and scene and hasattr(scene, 'eevee'):
        samples = getattr(scene.eevee, 'taa_render_samples', None)
        if samples is None:
            samples = getattr(scene.eevee, 'taa_samples', None)

    # Effective transparency as used by capture
    transparent_formats = {'PNG', 'OPEN_EXR', 'OPEN_EXR_MULTILAYER'}
    wants_transparency = bool(getattr(settings, 'transparent_background', False))
    scene_transparent = rd.film_transparent if rd else False
    effective_transparency = (
        True if (wants_transparency and file_format in transparent_formats) else scene_transparent
    )

    # Build stable payload for hashing
    payload = {
        "version": 2,
        "camera": {
            "camera_count": getattr(settings, 'camera_count', None),
            "camera_distribution": getattr(settings, 'camera_distribution', None),
            "min_elevation": getattr(settings, 'min_elevation', None),
            "max_elevation": getattr(settings, 'max_elevation', None),
            "camera_distance_mode": getattr(settings, 'camera_distance_mode', None),
            "camera_distance": getattr(settings, 'camera_distance', None),
            "camera_distance_multiplier": getattr(settings, 'camera_distance_multiplier', None),
            "focal_length": getattr(settings, 'focal_length', None),
            "ring_count": getattr(settings, 'ring_count', None),
        },
        "render": {
            "engine": render_engine,
            "resolution": {
                "x": res_x,
                "y": res_y,
                "percentage": res_pct,
            },
            "file_format": file_format,
            "color_mode": color_mode,
            "color_depth": color_depth,
            "transparency": effective_transparency,
            "samples": samples,
        },
        "outputs": {
            "export_colmap": getattr(settings, 'export_colmap', None),
            "export_transforms_json": getattr(settings, 'export_transforms_json', None),
            "export_depth": getattr(settings, 'export_depth', None),
            "export_normals": getattr(settings, 'export_normals', None),
            "export_masks": getattr(settings, 'export_masks', None),
            "mask_source": getattr(settings, 'mask_source', None),
            "mask_format": getattr(settings, 'mask_format', None),
        },
        "capture": {
            "render_speed_preset": getattr(settings, 'render_speed_preset', None),
            "lighting_mode": getattr(settings, 'lighting_mode', None),
            "material_mode": getattr(settings, 'material_mode', None),
            "include_children": getattr(settings, 'include_children', None),
            "use_adaptive_capture": getattr(settings, 'use_adaptive_capture', None),
            "adaptive_quality_preset": getattr(settings, 'adaptive_quality_preset', None),
            "adaptive_use_hotspots": getattr(settings, 'adaptive_use_hotspots', None),
            "adaptive_hotspot_bias": getattr(settings, 'adaptive_hotspot_bias', None),
        },
        "flags": {
            "transparent_background": wants_transparency,
        },
    }

    settings_str = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.md5(settings_str.encode()).hexdigest()[:16]
