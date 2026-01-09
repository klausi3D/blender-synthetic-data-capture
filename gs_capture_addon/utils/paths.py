"""Path validation and normalization utilities for cross-platform compatibility."""

import os
import sys
import platform


def normalize_path(path: str) -> str:
    """Normalize a path for the current platform.

    Handles:
    - Blender relative paths (//)
    - Home directory expansion (~)
    - Forward/backward slashes
    - Trailing slashes
    """
    if not path:
        return ""

    # Try to use bpy for Blender paths, but handle case where not available
    try:
        import bpy
        path = bpy.path.abspath(path)
    except ImportError:
        pass

    # Expand home directory
    path = os.path.expanduser(path)

    # Normalize path separators and resolve relative components
    path = os.path.normpath(path)

    # Remove trailing separator (except for root)
    if len(path) > 1:
        path = path.rstrip(os.sep)

    return path


def validate_directory(path: str, must_exist: bool = False,
                       create: bool = False) -> tuple:
    """Validate a directory path.

    Args:
        path: Directory path to validate
        must_exist: If True, directory must already exist
        create: If True, create directory if it doesn't exist

    Returns:
        tuple: (is_valid, normalized_path, error_message)
    """
    if not path:
        return False, "", "Path is empty"

    normalized = normalize_path(path)

    # Check for invalid characters (Windows)
    if sys.platform == 'win32':
        invalid_chars = '<>"|?*'
        for i, char in enumerate(normalized):
            if char in invalid_chars:
                return False, normalized, f"Invalid character '{char}' in path"
            if char == ':' and i != 1:
                return False, normalized, "Invalid use of ':' in path"

    if must_exist and not os.path.exists(normalized):
        return False, normalized, f"Directory does not exist: {normalized}"

    if create and not os.path.exists(normalized):
        try:
            os.makedirs(normalized, exist_ok=True)
        except OSError as e:
            return False, normalized, f"Cannot create directory: {e}"

    return True, normalized, ""


def validate_file(path: str, must_exist: bool = True) -> tuple:
    """Validate a file path.

    Args:
        path: File path to validate
        must_exist: If True, file must exist

    Returns:
        tuple: (is_valid, normalized_path, error_message)
    """
    if not path:
        return False, "", "Path is empty"

    normalized = normalize_path(path)

    if must_exist and not os.path.isfile(normalized):
        return False, normalized, f"File does not exist: {normalized}"

    return True, normalized, ""


def get_conda_base() -> str:
    """Find conda base installation directory."""
    # Check common locations
    search_paths = []

    if sys.platform == 'win32':
        search_paths = [
            os.path.expanduser("~/miniconda3"),
            os.path.expanduser("~/anaconda3"),
            "C:/Miniconda3",
            "C:/Anaconda3",
            os.path.expandvars("%LOCALAPPDATA%/miniconda3"),
            os.path.expandvars("%LOCALAPPDATA%/anaconda3"),
        ]
    else:
        search_paths = [
            os.path.expanduser("~/miniconda3"),
            os.path.expanduser("~/anaconda3"),
            "/opt/conda",
            "/opt/miniconda3",
            "/opt/anaconda3",
        ]

    for path in search_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "envs")):
            return path

    return ""


def get_conda_python(env_name: str, conda_base: str = None) -> str:
    """Get Python executable path for a conda environment.

    Args:
        env_name: Name of the conda environment
        conda_base: Optional conda base path (auto-detected if not provided)

    Returns:
        Path to Python executable, or empty string if not found
    """
    if not env_name:
        return ""

    if not conda_base:
        conda_base = get_conda_base()

    if not conda_base:
        return ""

    if sys.platform == 'win32':
        python_path = os.path.join(conda_base, "envs", env_name, "python.exe")
    else:
        python_path = os.path.join(conda_base, "envs", env_name, "bin", "python")

    if os.path.exists(python_path):
        return python_path

    return ""


def get_conda_script(env_name: str, script_name: str, conda_base: str = None) -> str:
    """Get path to a script in a conda environment's Scripts/bin folder.

    Args:
        env_name: Name of the conda environment
        script_name: Name of the script (without extension on Unix)
        conda_base: Optional conda base path

    Returns:
        Path to script, or empty string if not found
    """
    if not env_name or not script_name:
        return ""

    if not conda_base:
        conda_base = get_conda_base()

    if not conda_base:
        return ""

    if sys.platform == 'win32':
        # Try with .exe extension first
        script_path = os.path.join(conda_base, "envs", env_name, "Scripts", f"{script_name}.exe")
        if os.path.exists(script_path):
            return script_path
        # Try without extension
        script_path = os.path.join(conda_base, "envs", env_name, "Scripts", script_name)
    else:
        script_path = os.path.join(conda_base, "envs", env_name, "bin", script_name)

    if os.path.exists(script_path):
        return script_path

    return ""


def get_conda_executable(conda_base: str = None) -> str:
    """Get path to the conda executable.

    Args:
        conda_base: Optional conda base path (auto-detected if not provided)

    Returns:
        Path to conda executable, or empty string if not found
    """
    import shutil

    if not conda_base:
        conda_base = get_conda_base()

    if conda_base:
        if sys.platform == 'win32':
            conda_exe = os.path.join(conda_base, "Scripts", "conda.exe")
        else:
            conda_exe = os.path.join(conda_base, "bin", "conda")

        if os.path.exists(conda_exe):
            return conda_exe

    # Fallback to PATH
    return shutil.which("conda") or ""


def check_disk_space(path: str, required_gb: float = 1.0) -> tuple:
    """Check if there's enough disk space at the given path.

    Args:
        path: Directory path to check
        required_gb: Required free space in GB

    Returns:
        tuple: (has_space, free_gb, error_message)
    """
    import shutil

    normalized = normalize_path(path)

    # Find the mount point / drive
    check_path = normalized
    while not os.path.exists(check_path) and check_path:
        check_path = os.path.dirname(check_path)

    if not check_path:
        check_path = os.getcwd()

    try:
        total, used, free = shutil.disk_usage(check_path)
        free_gb = free / (1024 ** 3)

        if free_gb < required_gb:
            return False, free_gb, f"Insufficient disk space: {free_gb:.1f}GB free, {required_gb:.1f}GB required"

        return True, free_gb, ""
    except Exception as e:
        return False, 0.0, f"Cannot check disk space: {e}"
