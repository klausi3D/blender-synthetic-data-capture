"""Custom training backend loaded from YAML/JSON configuration."""

import os
import re
import json
from typing import Optional, List, Dict, Any

from .base import TrainingBackend, TrainingConfig, TrainingProgress, TrainingStatus
from ...utils.paths import normalize_path, get_conda_base, get_conda_python, get_conda_script

# Try to import yaml, fall back gracefully
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class CustomBackend(TrainingBackend):
    """Training backend dynamically configured from YAML/JSON file."""

    def __init__(self, config_path: str):
        """Initialize from config file.

        Args:
            config_path: Path to YAML or JSON configuration file
        """
        self._config = self._load_config(config_path)
        self._config_path = config_path

        # Set base class attributes from config
        self.name = self._config.get('name', 'Custom Backend')
        self.description = self._config.get('description', '')
        self.website = self._config.get('website', '')
        self.install_instructions = self._generate_install_instructions()

        # Pre-compile output parsing patterns
        self._patterns = {}
        self._completion_patterns = []
        self._error_patterns = []
        self._compile_patterns()

    @property
    def backend_id(self) -> str:
        """Get unique backend identifier."""
        return self._config.get('id', os.path.splitext(os.path.basename(self._config_path))[0])

    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML or JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith(('.yaml', '.yml')):
                if not HAS_YAML:
                    raise ImportError("PyYAML is required to load YAML config files. Install with: pip install pyyaml")
                return yaml.safe_load(f)
            else:
                return json.load(f)

    def _compile_patterns(self):
        """Pre-compile regex patterns for output parsing."""
        parsing = self._config.get('output_parsing', {})

        # Compile main patterns
        for key in ['iteration', 'loss', 'psnr', 'progress_bar']:
            if key in parsing and 'pattern' in parsing[key]:
                try:
                    self._patterns[key] = re.compile(parsing[key]['pattern'])
                except re.error as e:
                    print(f"Invalid regex pattern for {key}: {e}")

        # Compile completion patterns
        completion = parsing.get('completion', {})
        for pattern in completion.get('patterns', []):
            try:
                self._completion_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                pass

        # Compile error patterns
        error_config = parsing.get('error', {})
        flags = 0 if error_config.get('case_sensitive', False) else re.IGNORECASE
        for pattern in error_config.get('patterns', ['error', 'exception', 'failed']):
            try:
                self._error_patterns.append(re.compile(pattern, flags))
            except re.error:
                pass

    def _generate_install_instructions(self) -> str:
        """Generate installation instructions from config."""
        custom_instructions = self._config.get('install_instructions', '')
        if custom_instructions:
            return custom_instructions

        return f"""To use {self.name}:

1. Visit: {self.website or 'the project repository'}
2. Follow the installation instructions
3. Set the installation path in Addon Preferences > GS Capture > Custom Backends
4. Configure the conda environment name if required
"""

    def is_available(self) -> bool:
        """Check if backend is installed based on config detection rules."""
        detection = self._config.get('detection', {})
        method = detection.get('method', 'file')

        install_path = self.get_install_path()

        if method == 'file':
            if not install_path:
                return False
            check_file = detection.get('check_file', '')
            if check_file:
                return os.path.exists(os.path.join(install_path, check_file))
            return os.path.exists(install_path)

        elif method == 'command':
            import shutil
            cmd = detection.get('check_command', '')
            return shutil.which(cmd) is not None

        elif method == 'python_import':
            try:
                module_name = detection.get('check_import', '')
                __import__(module_name)
                return True
            except ImportError:
                return False

        return False

    def get_install_path(self) -> Optional[str]:
        """Find installation path based on config."""
        path_config = self._config.get('install_path', {})

        # Check addon preferences first
        pref_key = path_config.get('preference_key')
        if pref_key:
            try:
                import bpy
                prefs = bpy.context.preferences.addons.get('gs_capture_addon')
                if prefs and hasattr(prefs.preferences, pref_key):
                    path = getattr(prefs.preferences, pref_key)
                    if path:
                        normalized = normalize_path(path)
                        if os.path.exists(normalized):
                            return normalized
            except Exception:
                pass

        # Check environment variable
        env_var = path_config.get('env_var')
        if env_var:
            env_path = os.environ.get(env_var)
            if env_path:
                normalized = normalize_path(env_path)
                if os.path.exists(normalized):
                    return normalized

        # Search default paths
        for path in path_config.get('search_paths', []):
            expanded = normalize_path(path)
            if os.path.exists(expanded):
                return expanded

        return None

    def validate_data(self, data_path: str) -> tuple:
        """Validate data format based on config requirements."""
        data_path = normalize_path(data_path)
        data_format = self._config.get('data_format', {})

        # Check required directories
        for dir_path in data_format.get('required_dirs', ['images']):
            full_path = os.path.join(data_path, dir_path)
            if not os.path.exists(full_path):
                return False, f"Missing required directory: {dir_path}"

        # Check required files
        required_files = data_format.get('required_files', [])
        alt_files = data_format.get('alternative_files', [])

        for i, req_file in enumerate(required_files):
            full_path = os.path.join(data_path, req_file)
            if not os.path.exists(full_path):
                # Check for alternative
                if i < len(alt_files):
                    alt_path = os.path.join(data_path, alt_files[i])
                    if os.path.exists(alt_path):
                        continue
                return False, f"Missing required file: {req_file}"

        # Count images
        images_dir = os.path.join(data_path, "images")
        if os.path.exists(images_dir):
            image_count = len([f for f in os.listdir(images_dir)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.exr'))])
            if image_count == 0:
                return False, "No images found in images/ directory"
            return True, f"Valid dataset with {image_count} images"

        return True, "Dataset structure appears valid"

    def get_command(self, config: TrainingConfig) -> List[str]:
        """Build command from template."""
        cmd_config = self._config.get('command', {})

        # Get Python executable
        python_exe = self._get_python_executable()
        if not python_exe:
            raise RuntimeError(f"Could not find Python executable for {self.name}")

        # Get script path
        install_path = self.get_install_path() or ''
        script = cmd_config.get('script', 'train.py')
        script_path = os.path.join(install_path, script) if install_path else script

        # Build save iterations string
        save_iters = ','.join(str(i) for i in config.save_iterations) if config.save_iterations else str(config.iterations)

        # Get flag values
        flags = cmd_config.get('flags', {})
        white_bg_flag = flags.get('white_background', '--white_background')
        white_bg = white_bg_flag if config.white_background else ''

        # Get template
        template = cmd_config.get('template', '{python} {script} --source {data_path} --model {output_path} --iterations {iterations}')

        # Build extra args string
        extra_args = ' '.join(config.extra_args) if config.extra_args else ''

        # Substitute placeholders
        cmd_str = template.format(
            python=python_exe,
            script=script_path,
            data_path=normalize_path(config.data_path),
            output_path=normalize_path(config.output_path),
            iterations=config.iterations,
            save_iterations=save_iters,
            white_background=white_bg,
            extra_args=extra_args,
            gpu_id=config.gpu_id if hasattr(config, 'gpu_id') else 0,
        )

        # Parse into list (handle quoted strings properly)
        import shlex
        try:
            return shlex.split(cmd_str)
        except ValueError:
            # Fallback for Windows paths with backslashes
            return cmd_str.split()

    def parse_output(self, line: str) -> Optional[TrainingProgress]:
        """Parse training output using config patterns."""
        if not line.strip():
            return None

        progress = TrainingProgress(status=TrainingStatus.RUNNING)
        found_something = False

        # Match iteration
        if 'iteration' in self._patterns:
            match = self._patterns['iteration'].search(line)
            if match:
                parsing = self._config['output_parsing']['iteration']
                group = parsing.get('group', 1)
                try:
                    progress.iteration = int(match.group(group))
                    found_something = True
                except (IndexError, ValueError):
                    pass

        # Match loss
        if 'loss' in self._patterns:
            match = self._patterns['loss'].search(line)
            if match:
                parsing = self._config['output_parsing']['loss']
                group = parsing.get('group', 1)
                try:
                    progress.loss = float(match.group(group))
                    found_something = True
                except (IndexError, ValueError):
                    pass

        # Match PSNR
        if 'psnr' in self._patterns:
            match = self._patterns['psnr'].search(line)
            if match:
                parsing = self._config['output_parsing']['psnr']
                group = parsing.get('group', 1)
                try:
                    progress.psnr = float(match.group(group))
                    found_something = True
                except (IndexError, ValueError):
                    pass

        # Check completion
        for pattern in self._completion_patterns:
            if pattern.search(line):
                progress.status = TrainingStatus.COMPLETED
                found_something = True
                break

        # Check errors (but don't override completion)
        if progress.status != TrainingStatus.COMPLETED:
            for pattern in self._error_patterns:
                if pattern.search(line):
                    progress.error = line.strip()
                    found_something = True
                    break

        if found_something:
            progress.message = line.strip()[:200]  # Limit message length
            return progress

        return None

    def get_final_model_path(self, output_path: str) -> Optional[str]:
        """Get path to the final trained model."""
        output_files = self._config.get('output_files', {})
        model_pattern = output_files.get('model_path', 'point_cloud/iteration_30000/point_cloud.ply')

        # Try to find the model
        output_path = normalize_path(output_path)
        model_path = os.path.join(output_path, model_pattern)

        if os.path.exists(model_path):
            return model_path

        # Try to find any PLY file
        for root, dirs, files in os.walk(output_path):
            for f in files:
                if f.endswith('.ply'):
                    return os.path.join(root, f)

        return None

    def _get_python_executable(self) -> Optional[str]:
        """Get Python executable from configured environment."""
        env_config = self._config.get('environment', {})
        env_type = env_config.get('type', 'conda')
        env_name = env_config.get('name', '')

        # Check preference key for environment name
        pref_key = env_config.get('preference_key')
        if pref_key:
            try:
                import bpy
                prefs = bpy.context.preferences.addons.get('gs_capture_addon')
                if prefs and hasattr(prefs.preferences, pref_key):
                    env_name = getattr(prefs.preferences, pref_key) or env_name
            except Exception:
                pass

        if env_type == 'conda' and env_name:
            python = get_conda_python(env_name)
            if python:
                return python

        # Fallback to system Python
        import shutil
        return shutil.which('python') or shutil.which('python3')

    def get_config(self) -> dict:
        """Get the raw configuration dictionary.

        Returns:
            dict: The loaded configuration
        """
        return self._config.copy()

    def get_config_path(self) -> str:
        """Get path to the configuration file.

        Returns:
            str: Path to the YAML/JSON config file
        """
        return self._config_path


# Cache for loaded custom backends
_custom_backends_cache: Dict[str, CustomBackend] = {}
_cache_timestamp: float = 0.0


def load_custom_backends(custom_dir: str = None, force_reload: bool = False) -> Dict[str, CustomBackend]:
    """Load all custom backend configurations from a directory.

    Args:
        custom_dir: Directory containing YAML/JSON config files.
                   Defaults to addon's custom_backends/ folder.
        force_reload: Force reload even if cached

    Returns:
        dict: Mapping of backend_id to CustomBackend instance
    """
    global _custom_backends_cache, _cache_timestamp

    if custom_dir is None:
        # Default to addon's custom_backends folder
        addon_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        custom_dir = os.path.join(addon_dir, "custom_backends")

    if not os.path.exists(custom_dir):
        return {}

    # Check if we need to reload based on directory modification time
    try:
        dir_mtime = os.path.getmtime(custom_dir)
        if not force_reload and _custom_backends_cache and dir_mtime <= _cache_timestamp:
            return _custom_backends_cache
    except OSError:
        pass

    backends = {}

    for filename in os.listdir(custom_dir):
        if filename.endswith(('.yaml', '.yml', '.json')):
            config_path = os.path.join(custom_dir, filename)
            try:
                backend = CustomBackend(config_path)
                backends[backend.backend_id] = backend
            except Exception as e:
                print(f"Failed to load custom backend {filename}: {e}")

    # Update cache
    _custom_backends_cache = backends
    _cache_timestamp = os.path.getmtime(custom_dir) if os.path.exists(custom_dir) else 0.0

    return backends


def get_custom_backend(backend_id: str, custom_dir: str = None) -> Optional[CustomBackend]:
    """Get a specific custom backend by ID.

    Args:
        backend_id: The unique backend identifier
        custom_dir: Optional custom directory path

    Returns:
        CustomBackend instance or None if not found
    """
    backends = load_custom_backends(custom_dir)
    return backends.get(backend_id)


def reload_custom_backends(custom_dir: str = None) -> Dict[str, CustomBackend]:
    """Force reload all custom backends.

    Args:
        custom_dir: Optional custom directory path

    Returns:
        dict: Mapping of backend_id to CustomBackend instance
    """
    return load_custom_backends(custom_dir, force_reload=True)


def get_custom_backends_dir() -> str:
    """Get the default custom backends directory path.

    Returns:
        str: Path to the custom_backends folder
    """
    addon_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(addon_dir, "custom_backends")


def validate_backend_config(config_path: str) -> tuple:
    """Validate a backend configuration file.

    Args:
        config_path: Path to the config file

    Returns:
        tuple: (is_valid, errors_list, warnings_list)
    """
    errors = []
    warnings = []

    # Check file exists
    if not os.path.exists(config_path):
        return False, ["Configuration file not found"], []

    # Try to load
    try:
        if config_path.endswith(('.yaml', '.yml')):
            if not HAS_YAML:
                return False, ["PyYAML not installed - cannot validate YAML files"], []
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
    except Exception as e:
        return False, [f"Failed to parse config: {e}"], []

    if not isinstance(config, dict):
        return False, ["Config must be a dictionary/object"], []

    # Check required fields
    if not config.get('id'):
        errors.append("Missing required field: 'id'")
    if not config.get('name'):
        errors.append("Missing required field: 'name'")

    # Check command section
    command = config.get('command', {})
    if not command.get('template') and not command.get('script'):
        warnings.append("No command template or script defined")

    # Check output parsing patterns
    parsing = config.get('output_parsing', {})
    for key in ['iteration', 'loss', 'psnr']:
        if key in parsing:
            pattern = parsing[key].get('pattern', '')
            if pattern:
                try:
                    re.compile(pattern)
                except re.error as e:
                    errors.append(f"Invalid regex pattern for '{key}': {e}")

    # Check completion patterns
    completion = parsing.get('completion', {})
    for i, pattern in enumerate(completion.get('patterns', [])):
        try:
            re.compile(pattern)
        except re.error as e:
            errors.append(f"Invalid completion pattern {i}: {e}")

    # Check error patterns
    error_config = parsing.get('error', {})
    for i, pattern in enumerate(error_config.get('patterns', [])):
        try:
            re.compile(pattern)
        except re.error as e:
            errors.append(f"Invalid error pattern {i}: {e}")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings
