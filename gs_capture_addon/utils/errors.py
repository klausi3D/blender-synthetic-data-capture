"""User-friendly error messages for common training issues."""

ERROR_MESSAGES = {
    'missing_colmap': """COLMAP data not found.

The selected training backend requires COLMAP format data.
Please ensure "Export COLMAP Format" is enabled in Output settings before capturing.

Expected structure:
  {data_path}/
    images/
    sparse/0/
      cameras.txt (or .bin)
      images.txt (or .bin)
      points3D.txt (or .bin)
""",

    'missing_transforms': """transforms.json not found.

The selected training backend (Nerfstudio) requires transforms.json.
Please ensure "Export transforms.json" is enabled in Output settings before capturing.

Expected structure:
  {data_path}/
    images/
    transforms.json
""",

    'missing_images': """No training images found.

The images/ directory is empty or missing.
Please run a capture first to generate training images.
""",

    'backend_not_installed': """{backend_name} is not installed or configured.

Installation path: {install_path}

Please:
1. Install {backend_name} following the official guide
2. Set the correct path in Addon Preferences > GS Capture
""",

    'conda_env_not_found': """Conda environment not found: {env_name}

Please create the environment:
  conda create -n {env_name} python=3.10
  conda activate {env_name}
  # Install {backend_name} dependencies

Then verify the environment name in Addon Preferences.
""",

    'mask_naming_mismatch': """Mask files don't match expected naming.

GS-Lightning expects masks named: {{image_name}}.png
Example: for image_0001.png -> mask should be image_0001.png.png

Current masks don't follow this convention.
Enable "GS-Lightning" mask format in Output settings before capturing.
""",

    'insufficient_disk_space': """Insufficient disk space.

Free space: {free_gb:.1f} GB
Estimated need: {required_gb:.1f} GB

Please free up disk space or choose a different output location.
""",

    'invalid_data_path': """Invalid training data path.

Path: {data_path}
Error: {error}

Please select a valid directory containing your captured training data.
""",

    'invalid_output_path': """Invalid training output path.

Path: {output_path}
Error: {error}

Please select a valid directory for training output.
""",
}


def get_error_message(error_code: str, **kwargs) -> str:
    """Get a formatted error message.

    Args:
        error_code: Key for the error message template
        **kwargs: Values to format into the template

    Returns:
        str: Formatted error message
    """
    template = ERROR_MESSAGES.get(error_code, f"Unknown error: {error_code}")
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


# Error patterns and their suggestions for runtime errors
ERROR_SUGGESTIONS = {
    # CUDA/GPU errors
    'cuda out of memory': "Try reducing resolution or batch size. Close other GPU applications.",
    'cuda': "Check CUDA installation. Ensure GPU drivers are up to date.",
    'out of memory': "System memory exhausted. Reduce image count or resolution.",

    # File/Path errors
    'no such file': "File not found. Check paths are correct and files exist.",
    'permission denied': "Permission error. Run Blender as administrator or check folder permissions.",
    'file not found': "Required file missing. Ensure capture completed successfully.",

    # Python/Environment errors
    'modulenotfounderror': "Missing Python module. Activate correct conda environment.",
    'importerror': "Import failed. Check environment has all dependencies installed.",
    'no module named': "Module not installed. Run pip/conda install for the required package.",

    # Data format errors
    'invalid colmap': "COLMAP data corrupted. Re-run capture with COLMAP export enabled.",
    'transforms.json': "transforms.json invalid or missing. Re-capture with JSON export.",
    'no images': "No training images found. Ensure images directory contains valid files.",

    # Training errors
    'nan loss': "Training diverged. Try lower learning rate or check input data quality.",
    'iteration': "Training iteration error. Check if data format matches backend expectations.",

    # Process errors
    'timeout': "Process timed out. Training may need more time or resources.",
    'killed': "Process was killed. Check system resources and memory.",
    'segfault': "Segmentation fault. Check CUDA/GPU compatibility.",
}


def get_error_suggestion(error_message: str) -> str:
    """Get a suggestion based on error message patterns.

    Args:
        error_message: The raw error message string

    Returns:
        str: A helpful suggestion, or empty string if no match found
    """
    if not error_message:
        return ""

    error_lower = error_message.lower()

    for pattern, suggestion in ERROR_SUGGESTIONS.items():
        if pattern in error_lower:
            return suggestion

    # Default suggestions based on common patterns
    if 'error' in error_lower and 'path' in error_lower:
        return "Check that all file paths are correct and accessible."

    if 'failed' in error_lower:
        return "Check the training log for more details. Ensure backend is properly configured."

    return ""
